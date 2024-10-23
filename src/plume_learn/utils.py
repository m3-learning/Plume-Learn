import h5py 
import numpy as np
import matplotlib.pyplot as plt


def bresenham_line(point1, point2):
    """Generate the coordinates of points on a line using Bresenham's algorithm."""

    x0, y0, x1, y1 = point1[0], point1[1], point2[0], point2[1]

    points = []
    dx = abs(x1 - x0)
    dy = abs(y1 - y0)
    x, y = x0, y0
    sx = -1 if x0 > x1 else 1
    sy = -1 if y0 > y1 else 1
    if dx > dy:
        err = dx / 2.0
        while x != x1:
            points.append((x, y))
            err -= dy
            if err < 0:
                y += sy
                err += dx
            x += sx
    else:
        err = dy / 2.0
        while y != y1:
            points.append((x, y))
            err -= dx
            if err < 0:
                x += sx
                err += dy
            y += sy
    points.append((x, y))  # Include the endpoint
    return np.array(points[:-1])


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