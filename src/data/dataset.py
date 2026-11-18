import os
import torch
from torch.utils.data import Dataset
import rasterio
import numpy as np

class SEN12MSCRDataset(Dataset):
    """
    Parses paired SAR, Cloudy Optical, and Clear Optical GeoTIFFs.
    """
    def __init__(self, root_dir, season="ROIs1158_spring"):
        self.root_dir = os.path.join(root_dir, season)
        
        self.s1_dir = os.path.join(self.root_dir, "s1")
        self.s2_cloudy_dir = os.path.join(self.root_dir, "s2_cloudy")
        self.s2_clear_dir = os.path.join(self.root_dir, "s2_cloud_free")
        
        # We assume strict pairing: files have matching names across directories
        self.files = sorted(os.listdir(self.s1_dir))

    def _read_tiff(self, path):
        with rasterio.open(path) as src:
            # rasterio reads as (Channels, Height, Width)
            # Optical is heavily scaled, so we normalize by dividing by 10000.0 (standard for Sentinel-2)
            img = src.read().astype(np.float32) / 10000.0 
        return torch.tensor(img)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        filename = self.files[idx]
        
        sar_path = os.path.join(self.s1_dir, filename)
        cloudy_path = os.path.join(self.s2_cloudy_dir, filename)
        clear_path = os.path.join(self.s2_clear_dir, filename)
        
        sar_img = self._read_tiff(sar_path)
        opt_cloudy = self._read_tiff(cloudy_path)
        opt_clear = self._read_tiff(clear_path)
        
        # Sentinel-2 has 13 bands. We slice the core 4 (R, G, B, NIR) which are indices [1, 2, 3, 7]
        opt_cloudy = opt_cloudy[[1, 2, 3, 7], :, :]
        opt_clear = opt_clear[[1, 2, 3, 7], :, :]
        
        # Sentinel-1 has 2 bands (VV, VH)
        sar_img = sar_img[[0, 1], :, :]
        
        return sar_img, opt_cloudy, opt_clear