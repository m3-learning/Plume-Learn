import h5py 
import numpy as np
import matplotlib.pyplot as plt
import json

def load_json(file_path):
    with open(file_path) as f:
        data = json.load(f)
    return data


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


def remove_all_0_plume(df, index_label='plume_index', metric='Area', viz=False):
    area_sum_by_metric = df.groupby(index_label)[metric].sum()
    area_mean_by_metric = df.groupby(index_label)[metric].mean()
    df_filtered = df.copy()
    
    # # remove all 0 plume
    # plume_indices_to_remove = df_filtered[area_sum_by_metric == area_mean_by_metric].index
    # df_filtered = df_filtered[~df_filtered[index_label].isin(plume_indices_to_remove)]

    # remove outside 3 std plume
    min_ = np.mean(area_sum_by_metric) - 3*np.std(area_sum_by_metric)
    plume_indices_to_remove = area_sum_by_metric[area_sum_by_metric < min_].index
    df_filtered = df_filtered[~df_filtered[index_label].isin(plume_indices_to_remove)]
    # df_filtered.reset_index(drop=True, inplace=True)
    if 'index' in df_filtered.keys():
        df_filtered.drop('index', axis=1, inplace=True)

    if viz:
        plt.figure(figsize=(5, 4))
        plt.plot(area_sum_by_metric, linewidth=3, label='Before')
        plt.plot(df_filtered.groupby(index_label)[metric].sum(), linewidth=1, label='After')
        plt.legend()
        plt.show()
    return df_filtered



def check_fragmentation(filename, group_name='PLD_Plumes'):
    '''
    check the fragmentation of the hdf5 file after renaming group and dataset
    '''

    with h5py.File(filename, 'r') as f:
        total_size = 0
        allocated_size = 0
        for obj in f[group_name].values():
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