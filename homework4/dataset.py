# dataset.py
import os
import numpy as np
import torch
from torch.utils.data import Dataset

class SDFDataset(Dataset):
    def __init__(self, data_dir):
        self.data_dir = data_dir
        pointcloud_data = np.load(os.path.join(data_dir, 'pointcloud.npz'))
        sdf_data = np.load(os.path.join(data_dir, 'sdf.npz'))
        
        self.pointcloud_points = torch.tensor(pointcloud_data['points'], dtype=torch.float32)
        self.pointcloud_normals = torch.tensor(pointcloud_data['normals'], dtype=torch.float32)
        
        self.sdf_points = torch.tensor(sdf_data['points'], dtype=torch.float32) * 2
        self.sdf_gradients = torch.tensor(sdf_data['grad'], dtype=torch.float32)
        self.sdf_values = torch.tensor(sdf_data['sdf'], dtype=torch.float32)


    def __len__(self):
        return self.sdf_points.shape[0]
    
    def __getitem__(self, idx):
        return {
            'sdf_points': self.sdf_points[idx],
            'sdf_gradients': self.sdf_gradients[idx],
            'sdf_values': self.sdf_values[idx]
        }
