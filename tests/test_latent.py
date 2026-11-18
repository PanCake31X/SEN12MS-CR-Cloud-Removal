import sys
import os
import torch
import numpy as np
from datasets import load_dataset

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.latent_compression import SatelliteVAE

if __name__ == "__main__":
    print("Initializing Phase 2: Latent Compression with HuggingFace Streaming Data...")
    
    # Stream the dataset directly from HuggingFace (no massive local download)
    print("Connecting to HuggingFace Datasets...")
    ds = load_dataset("Hermanni/sen12mscr", split="train", streaming=True)
    
    # Grab the first sample
    sample = next(iter(ds))
    
    # Decode the binary bytes into numpy arrays as per the dataset specification
    sar = np.frombuffer(sample["sar"], dtype=np.float32).reshape(sample["sar_shape"])
    cloudy = np.frombuffer(sample["cloudy"], dtype=np.int16).reshape(sample["opt_shape"])
    
    # Convert numpy arrays to PyTorch tensors (and format to Batch, Channel, Height, Width)
    # The HuggingFace optical shape is (256, 256, 13) in HWC format, we need CHW
    sar_tensor = torch.tensor(sar).permute(2, 0, 1).unsqueeze(0).float() # SAR is already [2, 256, 256], add batch dim
    opt_cloudy_tensor = torch.tensor(cloudy).permute(2, 0, 1).unsqueeze(0).float() / 10000.0
    
    # Slice the Sentinel-2 13 bands down to the core 4 (R, G, B, NIR)
    opt_cloudy_tensor = opt_cloudy_tensor[:, [1, 2, 3, 7], :, :]
    
    print(f"\nReal Optical Input Shape:  {opt_cloudy_tensor.shape}")
    print(f"Real SAR Input Shape:      {sar_tensor.shape}")
    
    # Initialize VAE
    vae = SatelliteVAE(opt_channels=4, sar_channels=2, latent_dim=8)
    
    # Execute Encoding
    z_opt_cloudy = vae.encode_optical(opt_cloudy_tensor)
    z_sar = vae.encode_sar(sar_tensor)
    
    print(f"\nOptical Latent Shape:      {z_opt_cloudy.shape}")
    print(f"SAR Latent Shape:          {z_sar.shape}")
    
    # Execute Decoding
    reconstructed_opt = vae.decode(z_opt_cloudy)
    print(f"\nReconstructed Output Shape: {reconstructed_opt.shape}")
    
    assert z_opt_cloudy.shape[2] == 32, "Latent spatial dimension failed to compress to 32x32!"
    assert reconstructed_opt.shape == opt_cloudy_tensor.shape, "Decoder failed to reconstruct original dimensions!"
    
    print("\nReal Multi-Spectral GeoTIFFs successfully streamed and compressed into the Latent Manifold.")