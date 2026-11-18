import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pywt
import numpy as np

class SARDWTFilter(nn.Module):
    def __init__(self, wavelet='haar'):
        super().__init__()
        self.wavelet = wavelet

    def forward(self, sar_tensor):
        """
        Expects sar_tensor of shape (B, C, H, W).
        Applies 2D DWT to extract structural edges and suppress speckle.
        """
        B, C, H, W = sar_tensor.shape
        device = sar_tensor.device
        
        # PyWavelets operates on numpy arrays, so we detach and convert
        sar_np = sar_tensor.detach().cpu().numpy()
        filtered_sar = np.zeros_like(sar_np)

        for b in range(B):
            for c in range(C):
                # 2D DWT: returns Approximation (LL) and Details (LH, HL, HH)
                coeffs2 = pywt.dwt2(sar_np[b, c], self.wavelet)
                LL, (LH, HL, HH) = coeffs2
                
                # Zero out the Approximation (LL) to drop low-frequency speckle/background
                # Keep the Details (LH, HL, HH) which contain the structural geometry and edges
                LL_zero = np.zeros_like(LL)
                
                # Inverse DWT to reconstruct the filtered image
                filtered_sar[b, c] = pywt.idwt2((LL_zero, (LH, HL, HH)), self.wavelet)

        return torch.tensor(filtered_sar, dtype=torch.float32, device=device)

class MockSatelliteDataset(Dataset):
    def __init__(self, num_samples=10, img_size=256):
        self.num_samples = num_samples
        self.img_size = img_size

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        # SAR: 2 channels (VV, VH)
        sar = torch.randn(2, self.img_size, self.img_size)
        # Optical Cloudy: 4 channels (R, G, B, NIR)
        opt_cloudy = torch.randn(4, self.img_size, self.img_size)
        # Optical Clear (Target): 4 channels
        opt_clear = torch.randn(4, self.img_size, self.img_size)
        
        return sar, opt_cloudy, opt_clear

# --- Verification Execution ---
if __name__ == "__main__":
    print("Initializing Data Pipeline...")
    dataset = MockSatelliteDataset(num_samples=5, img_size=256)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    dwt_filter = SARDWTFilter(wavelet='haar')
    
    for batch_idx, (sar, opt_cloudy, opt_clear) in enumerate(dataloader):
        print(f"\nBatch {batch_idx + 1}")
        print(f"Original SAR shape: {sar.shape}")
        
        # Apply DWT Filter
        filtered_sar = dwt_filter(sar)
        print(f"Filtered SAR shape: {filtered_sar.shape}")
        
        # Verify alignment for the eventual fusion bottleneck
        assert filtered_sar.shape[2:] == opt_cloudy.shape[2:], "Spatial dimensions misaligned!"
        print("Spatial dimensions mathematically aligned. Ready for Latent Compression.")
        break # Just testing the first batch