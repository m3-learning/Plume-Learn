import numpy as np
import h5py
import matplotlib.pyplot as plt
import pandas as pd
import sys
sys.path.append('../../../src')
from plume_dynamic_analyzer.src.analyzer.PlumeDataset import plume_dataset
from AutoAlign import align_plumes
from viz import show_images




def add_csv_columns_to_h5(out_file, csv_file, total_images, frame_index_list):
    """
    Read a CSV file and add each column as a separate dataset in the H5 file.
    
    :param out_file: The h5py.File object to write to
    :param csv_file: Path to the CSV file
    :param total_images: Current total number of images/rows
    :return: Updated total_images count
    """
    # Read the CSV file
    labels = pd.read_csv(csv_file, index_col=0)
    if isinstance(frame_index_list, list):
        labels = labels[labels['time_index'] < np.max(frame_index_list)]
    
    # Process each column in the CSV file
    for column in labels.columns:
        column_data = labels[column]
        
        # Determine the datatype for the dataset
        if column_data.dtype == 'object':
            dtype = h5py.special_dtype(vlen=str)
        else:
            dtype = column_data.dtype
        
        # Check if the dataset already exists
        if column not in out_file:
            # Create a new dataset for this column
            dataset = out_file.create_dataset(column, 
                                              shape=(total_images + len(labels),),
                                              maxshape=(None,),
                                              dtype=dtype)
            # Fill the existing part with a default value or NaN
            if dtype == h5py.special_dtype(vlen=str):
                dataset[:total_images] = ''
            else:
                dataset[:total_images] = 0
        else:
            # Get the existing dataset
            dataset = out_file[column]
            # Resize the dataset
            dataset.resize((total_images + len(labels),))
        
        # Add the new data
        dataset[total_images:] = column_data.values
    
    return total_images + len(labels)


def add_h5_images_to_h5(out_file, h5_file, coords_file, coords_standard, total_images, frame_index_list, viz_sample=False):
    """
    Read images from an H5 file and add them to the output H5 file.
    
    :param out_file: The h5py.File object to write to
    :param h5_file: Path to the input H5 file
    :param total_images: Current total number of images
    :return: Updated total_images count, images_dataset
    """
    plume_ds = plume_dataset(file_path=h5_file, group_name='PLD_Plumes')
    images = plume_ds.load_plumes('1-SrRuO3')
    if isinstance(frame_index_list, list):
        images = images[:, frame_index_list]
    coords = np.load(coords_file)
    images = align_plumes(images, coords, coords_standard)
    if viz_sample:
        show_images(images[0, :64], img_per_row=16, img_height=1)
        plt.tight_layout()
        plt.show()
    
    images = images.reshape(images.shape[0]*images.shape[1], images.shape[2], images.shape[3])
    
    if 'images' not in out_file:
        images_dataset = out_file.create_dataset('images', shape=(0,) + images.shape[1:], 
                                                 maxshape=(None,) + images.shape[1:], 
                                                 dtype=images.dtype)
    else:
        images_dataset = out_file['images']
    
    images_dataset.resize(total_images + images.shape[0], axis=0)
    images_dataset[total_images:] = images
    
    return total_images + images.shape[0], images_dataset


def merge_h5_and_csv(h5_files, csv_files, coords_files, coords_standard, output_file, frame_index_list, viz_sample=False):
    with h5py.File(output_file, 'w') as out_file:
        total_images = 0
        
        # Process each pair of H5 and CSV files
        for h5_file, csv_file, coords_file in zip(h5_files, csv_files, coords_files):
            # Process H5 file (images)
            print(f'loading {h5_file}...')
            total_images_h5, _ = add_h5_images_to_h5(out_file, h5_file, coords_file, coords_standard, total_images, frame_index_list, viz_sample=viz_sample)
        
            # Process CSV file
            print(f'loading {csv_file}...')
            total_images_csv = add_csv_columns_to_h5(out_file, csv_file, total_images, frame_index_list)
            
            if total_images_h5 != total_images_csv:
                raise ValueError(f"Number of images ({total_images_h5}) does not match number of labels ({total_images_csv})")
            
        # Add metadata
        out_file.attrs['total_images'] = total_images_h5

    print(f"Merged {len(h5_files)} H5 files and {len(csv_files)} CSV files into {output_file}")
    print(f"Total images: {total_images}")



def _plot_histograms(array, title, labels=None, max_cols=5):
    
    n_features = array.shape[1]
    n_cols = min(n_features, max_cols)
    n_rows = (n_features - 1) // n_cols + 1

    fig, axes = plt.subplots(n_rows, n_cols, figsize=(7*n_cols, 5*n_rows))
    axes = axes.flatten() if n_features > 1 else [axes]
    # for ax in axes.flatten():
    #     if not ax.has_data():
    #         fig.delaxes(ax)
            
    if labels is None:
        labels = [f'Feature {i+1}' for i in range(n_features)]
        
    for i, (ax, label) in enumerate(zip(axes, labels)):
        if i < n_features:
            data = array[:, i]
            
        # Calculate statistics
        min_val = np.min(data)
        max_val = np.max(data)
        mean_val = np.mean(data)
        std_val = np.std(data)
        
        # Add statistics to the plot
        stats_text = f'Min: {min_val:.2f}\nMax: {max_val:.2f}\nMean: {mean_val:.2f}\nStd: {std_val:.2f}'
        ax.text(0.95, 0.95, stats_text,
                verticalalignment='top', horizontalalignment='right',
                transform=ax.transAxes,
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

        # Plot vertical lines for mean and std
        ax.axvline(mean_val, color='r', linestyle='dashed', linewidth=2, label='Mean')
        ax.axvline(mean_val - std_val, color='g', linestyle=':', linewidth=2, label='Mean ± Std')
        ax.axvline(mean_val + std_val, color='g', linestyle=':', linewidth=2)
        # ax.legend()
        
        ax.hist(data, bins=50, edgecolor='black')
        ax.set_title(f'{label}')
        ax.set_xlabel('Value')
        ax.set_ylabel('Frequency')
        
    for ax in axes.flatten():
        if not ax.has_data():
            fig.delaxes(ax)
            
    plt.tight_layout()
    plt.suptitle(f'{title} Labels Distribution', fontsize=16)
    plt.tight_layout()
    plt.show()

class EqualRangeNormalizer:
    def __init__(self, min_vals=None, max_vals=None):
        self.min_vals = min_vals
        self.max_vals = max_vals
        self.constant_columns = None

    def fit(self, array, label_keys=None, viz=False):
        if self.min_vals is not None or self.max_vals is not None:
            print("Warning: Normalizer is already fitted. Overwriting the previous values.")
        self.min_vals = np.min(array, axis=0)
        self.max_vals = np.max(array, axis=0)
        self.constant_columns = np.isclose(self.max_vals, self.min_vals)

        if viz:
            _plot_histograms(array, "Original", label_keys)

    def transform(self, array, label_keys=None, viz=False):
        if self.min_vals is None or self.max_vals is None:
            raise ValueError("Normalizer has not been fitted. Call 'fit' before using 'transform'.")

        normalized_labels = np.zeros_like(array, dtype=np.float32)
        for i in range(array.shape[1]):
            if self.constant_columns[i]:
                normalized_labels[:, i] = 0.5  # or any other value you prefer for constant columns
            else:
                normalized_labels[:, i] = (array[:, i] - self.min_vals[i]) / (self.max_vals[i] - self.min_vals[i])
                
        if viz:
            _plot_histograms(normalized_labels, "Normalized", label_keys)
    
        # Check for any remaining NaN values
        if np.any(np.isnan(normalized_labels)):
            nan_columns = np.where(np.any(np.isnan(normalized_labels), axis=0))[0]
            print(f"Warning: NaN values found in columns: {nan_columns}")
            print("This might be due to unexpected data ranges. Please check your input data.")

        return normalized_labels

    def inverse_transform(self, array, label_keys=None, viz=False):
        reconstructed_labels = array * (self.max_vals - self.min_vals) + self.min_vals
        if viz:
            _plot_histograms(reconstructed_labels, "Reconstructed", label_keys)
        return reconstructed_labels


def make_dataset(target_file, input_files, df_condition, selected_frame, growth_name_dict, normalize_labels=False):

    selected_frame = (2, 36)
    growth_names = list(growth_name_dict.keys())

    length = 0
    for file in input_files:
        plume_ds = plume_dataset(file_path=file, group_name='PLD_Plumes')
        keys = plume_ds.dataset_names()
        plumes = plume_ds.load_plumes('1-SrRuO3')
        length += len(plumes)
    print(plumes.shape, plumes.dtype, np.min(plumes), np.max(plumes), length)

    with h5py.File(target_file, 'w') as f:
        f.create_dataset('plumes', shape=(length, 34, 250, 400), dtype=np.uint8)
        f.create_dataset('growth_rate(angstrom_per_pulse)', shape=(length, 1), dtype=np.float32)
        f.create_dataset('growth_rate(nm_per_min)', shape=(length, 1), dtype=np.float32)
        f.create_dataset('Pressure (mTorr)', shape=(length, 1), dtype=np.float32)
        f.create_dataset('Fluence (J/cm2)', shape=(length, 1), dtype=np.float32)
        f.create_dataset('labels', shape=(length, 3), dtype=np.float32)
        f.create_dataset('growth_name', shape=(length, 1), dtype=np.uint8)

        index = 0
        for growth, file in zip(growth_names, input_files):
            print(file)
            plume_ds = plume_dataset(file_path=file, group_name='PLD_Plumes')
            plumes = plume_ds.load_plumes('1-SrRuO3')[:, selected_frame[0]:selected_frame[1]]
            f['plumes'][index:index+len(plumes)] = plumes
            print(len(plumes))

            f['growth_rate(angstrom_per_pulse)'][index:index+len(plumes)] = df_condition[df_condition['Growth'] == growth]['Growth rate (Å/pulse)'].values[0]
            f['growth_rate(nm_per_min)'][index:index+len(plumes)] = df_condition[df_condition['Growth'] == growth]['Growth rate (nm/min)'].values[0]
            f['Pressure (mTorr)'][index:index+len(plumes)] = df_condition[df_condition['Growth'] == growth]['Pressure (mTorr)'].values[0]
            f['Fluence (J_per_cm2)'][index:index+len(plumes)] = df_condition[df_condition['Growth'] == growth]['Fluence (J/cm2)'].values[0]
            
            labels = np.array([df_condition[df_condition['Growth'] == growth]['Pressure (mTorr)'].values[0],
                            df_condition[df_condition['Growth'] == growth]['Fluence (J/cm2)'].values[0],
                            df_condition[df_condition['Growth'] == growth]['Growth rate (Å/pulse)'].values[0]])
            f['labels'][index:index+len(plumes)] = labels
            f['growth_name'][index:index+len(plumes)] = growth_name_dict[growth]

            index += len(plumes)


    if normalize_labels:
        # normalize the labels and create another dataset for it
        with h5py.File(target_file, 'r') as f:
            labels = np.array(f['labels'])
            
        # Create and fit the normalizer
        normalizer = EqualRangeNormalizer()
        normalizer.fit(labels)

        # Normalize the labels
        normalized_labels = normalizer.transform(labels)# normalize the labels and create another dataset for it
        with h5py.File(target_file, 'a') as f:
            f.create_dataset('normalized_labels', data=normalized_labels)