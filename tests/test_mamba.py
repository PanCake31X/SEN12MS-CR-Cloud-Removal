import sys
import os
import torch

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.mamba_ss2d import CrossModalSAGF, SS2DScanner

if __name__ == "__main__":
    print("Initializing Phase 3: SS2D Mamba Backbone...")
    
    # Mock latent tensors arriving from Phase 2 (Batch, Channels, H, W)
    batch_size = 2
    latent_dim = 8
    spatial_dim = 32
    
    z_opt = torch.randn(batch_size, latent_dim, spatial_dim, spatial_dim)
    z_sar = torch.randn(batch_size, latent_dim, spatial_dim, spatial_dim)
    
    # 1. Test Cross-Modal Fusion (SAGF)
    sagf = CrossModalSAGF(latent_dim=latent_dim)
    fused_latent, mask = sagf(z_opt, z_sar)
    
    print(f"Optical Latent Shape: {z_opt.shape}")
    print(f"SAR Latent Shape:     {z_sar.shape}")
    print(f"Gating Mask Shape:    {mask.shape}")
    print(f"Fused Latent Shape:   {fused_latent.shape}")
    
    # Verify mask bounds
    assert torch.all((mask >= 0) & (mask <= 1)), "Mask values out of bounds!"
    
    # 2. Test SS2D Multi-Directional Scanning
    scanner = SS2DScanner()
    scanned_seqs = scanner(fused_latent)
    
    # Expected shape: (Batch, 4 directions, Channels, H*W)
    print(f"\nScanned Sequences Shape: {scanned_seqs.shape}")
    
    expected_len = spatial_dim * spatial_dim
    assert scanned_seqs.shape == (batch_size, 4, latent_dim, expected_len), "SS2D scan dimensions failed!"
    
    print("\nSAGF Cross-Modal Fusion and SS2D Multi-Directional Flattening verified.")