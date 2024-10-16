import numpy as np
from PIL import Image
import h5py
import torch
from torch.utils.data import DataLoader, Dataset

class hdf5_dataset_image(Dataset):
    
    def __init__(self, file_path, folder=None, transform=None, data_key='data', label_key='labels'):
        self.file_path = file_path
        self.folder = folder
        self.transform = transform
        self.hf = None
        self.data_key = data_key
        self.label_key = label_key

    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            if self.folder:
                self.len = len(f[self.folder][self.label_key])
            else:
                self.len = len(f[self.label_key])
        return self.len
    
    def __getitem__(self, idx):
        if self.hf is None:
            self.hf = h5py.File(self.file_path, 'r')
        
        if self.folder:
            image = np.array(self.hf[self.folder][self.data_key][idx])
            label = np.array(self.hf[self.folder][self.label_key][idx])
        else:
            image = np.array(self.hf[self.data_key][idx])
            label = np.array(self.hf[self.label_key][idx])
        
        # Convert numpy array to PIL Image
        if image.dtype != np.uint8:
            image = (image * 255).astype(np.uint8)
        if image.ndim == 2:  # If it's a grayscale image
            image = Image.fromarray(image, mode='L')
        else:
            image = Image.fromarray(image)
        
        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label)

class hdf5_dataset_video(Dataset):
    
    def __init__(self, file_path, folder=None, transform=None, data_key='data', label_key='labels'):
        self.file_path = file_path
        self.folder = folder
        self.transform = transform
        self.hf = None
        self.data_key = data_key
        self.label_key = label_key

    def __len__(self):
        with h5py.File(self.file_path, 'r') as f:
            if self.folder:
                self.len = len(f[self.folder][self.label_key])
            else:
                self.len = len(f[self.label_key])
        return self.len
    
    def __getitem__(self, idx):
        if self.hf is None:
            self.hf = h5py.File(self.file_path, 'r')
        
        if self.folder:
            video = np.array(self.hf[self.folder][self.data_key][idx])
            label = np.array(self.hf[self.folder][self.label_key][idx])
        else:
            video = np.array(self.hf[self.data_key][idx])
            label = np.array(self.hf[self.label_key][idx])
            
        # Convert numpy array to a list of PIL Images
        if video.dtype != np.uint8:
            video = (video * 255).astype(np.uint8)
        
        video_frames = []
        for frame in video:
            if frame.ndim == 2:  # If it's a grayscale image
                frame = Image.fromarray(frame, mode='L')
            else:
                frame = Image.fromarray(frame)
            video_frames.append(frame)
        
        if self.transform:
            video_frames = [self.transform(frame) for frame in video_frames]
            video_frames = torch.stack(video_frames)
            # Remove the following line to keep the shape as [128, 250, 400]
            if video_frames.ndim == 3:
                video_frames = video_frames.unsqueeze(1)  # expand dimension for single channel video
            # video_frames = video_frames.squeeze()  # expand dimension for single channel video

        return video_frames, torch.tensor(label)
