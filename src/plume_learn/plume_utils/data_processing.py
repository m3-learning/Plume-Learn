import h5py 
import numpy as np
import matplotlib.pyplot as plt
import json

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

