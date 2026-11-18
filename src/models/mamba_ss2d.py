import torch
import torch.nn as nn

class CrossModalSAGF(nn.Module):
    """
    Sparsity-Aware Gated Fusion (SAGF).
    Evaluates where clouds are and injects SAR structural features specifically 
    into those occluded zones, leaving clear-sky pixels unaltered.
    """
    def __init__(self, latent_dim=8):
        super().__init__()
        # Generates a spatial modulation mask from optical and SAR
        self.mask_generator = nn.Sequential(
            nn.Conv2d(latent_dim * 2, latent_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(latent_dim, 1, kernel_size=3, padding=1),
            nn.Sigmoid() # Outputs values between 0 (clear sky) and 1 (thick cloud)
        )
        self.fusion_proj = nn.Conv2d(latent_dim * 2, latent_dim, kernel_size=1)

    def forward(self, opt_latent, sar_latent):
        # Concatenate to evaluate cloud density
        combined = torch.cat([opt_latent, sar_latent], dim=1)
        
        # Generate spatial mask: 1 where clouds exist, 0 for clear sky
        gating_mask = self.mask_generator(combined)
        
        # Modulate SAR: Only pass SAR features where the mask is active
        gated_sar = sar_latent * gating_mask
        
        # Fuse modulated SAR with original Optical
        fused = self.fusion_proj(torch.cat([opt_latent, gated_sar], dim=1))
        
        return fused, gating_mask

class SS2DScanner(nn.Module):
    """
    Simulates the 2D Selective Scan (SS2D) by flattening the grid 
    into 4 continuous directional sequences to preserve spatial relationships.
    """
    def __init__(self):
        super().__init__()

    def forward(self, x):
        B, C, H, W = x.shape
        
        # 1. Forward Scan (Top-Left to Bottom-Right)
        scan1 = x.view(B, C, H * W)
        
        # 2. Reverse Scan (Bottom-Right to Top-Left)
        scan2 = torch.flip(scan1, dims=[2])
        
        # 3. Transposed Forward Scan (Top-Left downwards column-wise)
        scan3 = x.transpose(2, 3).contiguous().view(B, C, H * W)
        
        # 4. Transposed Reverse Scan
        scan4 = torch.flip(scan3, dims=[2])
        
        # Returns the 4 distinct 1D sequences ready for 1D State Space processing
        # Shape output: (Batch, 4_Directions, Channels, Sequence_Length)
        return torch.stack([scan1, scan2, scan3, scan4], dim=1)