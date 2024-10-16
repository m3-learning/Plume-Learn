import h5py 
import numpy as np
import matplotlib.pyplot as plt

def remove_all_0_plume(df, metric='Area', viz=False):
    area_sum_by_metric = df.groupby('plume_index')[metric].sum()
    min_ = np.mean(area_sum_by_metric) - 3*np.std(area_sum_by_metric)
    plume_indices_to_remove = area_sum_by_metric[area_sum_by_metric < min_].index
    df_filtered = df[~df['plume_index'].isin(plume_indices_to_remove)]
    # df_filtered.reset_index(drop=True, inplace=True)
    if 'index' in df_filtered.keys():
        df_filtered.drop('index', axis=1, inplace=True)

    if viz:
        plt.figure(figsize=(5, 4))
        plt.plot(area_sum_by_metric, linewidth=3, label='Before')
        plt.plot(df_filtered.groupby('plume_index')[metric].sum(), linewidth=1, label='After')
        plt.legend()
        plt.show()
    return df_filtered


def check_fragmentation(filename):
    '''
    check the fragmentation of the hdf5 file after renaming group and dataset
    '''

    with h5py.File(filename, 'r') as f:
        total_size = 0
        allocated_size = 0
        for obj in f['PLD_Plumes'].values():
            if isinstance(obj, h5py.Dataset):
                total_size += obj.size * obj.dtype.itemsize
                allocated_size += obj.id.get_storage_size()
    
    fragmentation = (allocated_size - total_size) / allocated_size * 100
    return fragmentation

# example of renaming group and dataset
# import h5py
# # Open the file
#     with h5py.File(file, 'r+') as f:
#         print(f['PLD_Plumes'].keys())
#         # Rename the dataset
#         f['PLD_Plumes'].move('1-SrTiO3_Pre', '1-SrRuO3_Pre')
#         check_fragmentation(file)



def smooth_curve(data, window_size):
    """
    Smooths a 1D curve using a moving average, retaining the same shape as the original data.

    Parameters:
    - data: The 1D array to be smoothed.
    - window_size: The size of the moving window.

    Returns:
    - The smoothed array.
    """
    # Pad the array on both sides to handle the edges
    pad_size = window_size // 2
    padded_data = np.pad(data, pad_size, mode='edge')

    # Compute the moving average
    cumsum = np.cumsum(np.insert(padded_data, 0, 0))
    smoothed_data = (cumsum[window_size:] - cumsum[:-window_size]) / float(window_size)

    return smoothed_data


def NormalizeData(data):
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def show_images(images, labels=None, img_per_row=8, colorbar=False):

    '''
    This is a utility function used to show a series images.

    :param images: input images
    :type images: np.array

    :param labels: labels for images
    :type labels: str(, optional)

    :param img_per_row: how many images to show in one row
    :type img_per_row: int(, optional)

    :param colorbar: to determine if colobar is included 
    :type colorbar: bool(, optional)
    '''

    h = images[0].shape[1] // images[0].shape[0]*0.5 + 1
    if not labels:
        labels = range(len(images))
    fig, axes = plt.subplots(len(images)//img_per_row+1*int(len(images)%img_per_row>0), img_per_row, 
                             figsize=(16, h*len(images)//img_per_row+1))
    for i in range(len(images)):
        if len(images) <= img_per_row:
            axes[i%img_per_row].title.set_text(labels[i])
            im = axes[i%img_per_row].imshow(images[i])
            if colorbar:
                fig.colorbar(im, ax=axes[i%img_per_row])
            axes[i%img_per_row].axis('off')

        else:
            axes[i//img_per_row, i%img_per_row].title.set_text(labels[i])
            im = axes[i//img_per_row, i%img_per_row].imshow(images[i])
            if colorbar:
                fig.colorbar(im, ax=axes[i//img_per_row, i%img_per_row])
            axes[i//img_per_row, i%img_per_row].axis('off')
            
    plt.show()

def show_h5_dataset_name(ds_path, class_name=None):
    '''
    This is a utility function used to show the dataset names in a hdf5 file.

    :param ds_path: path to hdf5 file
    :type ds_path: str

    :param class_name: class name of hdf5 file
    :type class_name: str(, optional)
    '''

    with h5py.File(ds_path) as hf:
        if class_name:
            print(hf[class_name].keys())            
        else:
            print(hf.keys())
            

def load_plumes(ds_path, class_name, ds_name, process_func=None):

    '''
    This is a utility function used to load plume images from hdf5 file 
    based on the the ds_name after preprocess with process_func.

    :param ds_path: path to hdf5 file
    :type ds_path: str

    :param class_name: class name of hdf5 file
    :type class_name: str(, optional)

    :param ds_name: dataset name for plume images in hdf5 file
    :type ds_name: str

    :param process_func: preprocess function
    :type process_func: function(, optional)

    '''

    with h5py.File(ds_path) as hf:
        plumes = np.array(hf[class_name][ds_name])

    if process_func:
        plumes = process_func(plumes)

    return plumes



def load_h5_examples(ds_path, class_name, ds_name, process_func=None, show=True):

    '''
    This is a utility function used to load plume images from hdf5 file 
    based on the the ds_name after preprocess with process_func.

    :param ds_path: path to hdf5 file
    :type ds_path: str

    :param class_name: class name of hdf5 file
    :type class_name: str(, optional)

    :param ds_name: dataset name for plume images in hdf5 file
    :type ds_name: str

    :param process_func: preprocess function
    :type process_func: function(, optional)

    :param show: show the plumes images if show=True
    :type show: bool(, optional)

    '''

    with h5py.File(ds_path) as hf:
        plumes = np.array(hf[class_name][ds_name])

    if process_func:
        images = process_func(plumes)

    return plumes
