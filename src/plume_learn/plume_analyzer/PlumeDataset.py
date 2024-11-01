import h5py 
import numpy as np
import pandas as pd

class plume_dataset():
    def __init__(self, file_path, group_name='PLD_Plumes'):
        '''
        This is a class used to load plume images from hdf5 file 
        based on the the ds_name after preprocess with process_func.

        :param file_path: path to hdf5 file
        :type file_path: str

        :param group_name: class name of hdf5 file, default: PLD_Plumes
        :type group_name: str(, optional)

        '''
        self.file_path = file_path
        self.group_name = group_name
    
    def dataset_names(self):
        '''
        This is a utility function used to get the dataset names in a hdf5 file.

        :param ds_path: path to hdf5 file
        :type ds_path: str

        :param class_name: class name of hdf5 file
        :type class_name: str(, optional)
        '''
        with h5py.File(self.file_path) as hf:
            datasets = list(hf[self.group_name].keys())
        return datasets
                

    def load_plumes(self, dataset_name):

        '''
        This is a utility function used to load plume images from hdf5 file 
        based on the the ds_name after preprocess with process_func.

        :param dataset_name: dataset name for plume images in hdf5 file
        :type dataset_name: str
        '''
        with h5py.File(self.file_path) as hf:
            plumes = np.array(hf[self.group_name][dataset_name])
        return plumes