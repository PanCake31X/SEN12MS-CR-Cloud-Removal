import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class SinusoidalTimeEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, t):
        # t shape: (B, 1, 1, 1) -> (B,)
        t = t.view(-1)
        half = self.dim // 2
        freqs = torch.exp(
            -math.log(10000) * torch.arange(half, device=t.device, dtype=torch.float32) / half
        )
        args = t[:, None] * freqs[None, :]
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)
        return emb


class BidirectionalSS2DBlock(nn.Module):
    """
    4-Way Directional Scanning Block with exact inverse un-scanning,
    long-range dilated 1D gated mixing (O(N) complexity), and FiLM time conditioning.
    """
    def __init__(self, channels, time_dim):
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels)
        self.time_mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(time_dim, channels * 2)
        )
        
        # 4 independent directional 1D depthwise-dilated mixers (long-range context)
        self.dir_convs = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(channels, channels, kernel_size=5, padding=2, groups=channels),
                nn.SiLU(),
                nn.Conv1d(channels, channels, kernel_size=5, padding=6, dilation=3, groups=channels),
            )
            for _ in range(4)
        ])
        
        # Gated projection (Mamba-style multiplicative gating)
        self.gate_proj = nn.Conv2d(channels, channels, kernel_size=1)
        self.out_proj = nn.Conv2d(channels, channels, kernel_size=1)
        
        # Feed-forward channel mixer
        self.norm2 = nn.GroupNorm(8, channels)
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, kernel_size=3, padding=1),
            nn.SiLU(),
            nn.Conv2d(channels * 2, channels, kernel_size=1)
        )

    def forward(self, x, t_emb):
        B, C, H, W = x.shape
        h = self.norm1(x)
        
        # FiLM time modulation: scale & shift
        scale, shift = self.time_mlp(t_emb).chunk(2, dim=1)
        h = h * (1.0 + scale[:, :, None, None]) + shift[:, :, None, None]
        
        gate = F.silu(self.gate_proj(h))
        
        # 4-way scan + EXACT inverse un-scan so every (h, w) aligns 1-to-1
        # 1. Row-major forward
        s1 = self.dir_convs[0](h.flatten(2)).view(B, C, H, W)
        # 2. Row-major backward
        s2 = self.dir_convs[1](h.flatten(2).flip(-1)).flip(-1).view(B, C, H, W)
        # 3. Col-major forward
        h_t = h.transpose(2, 3).contiguous()
        s3 = self.dir_convs[2](h_t.flatten(2)).view(B, C, W, H).transpose(2, 3)
        # 4. Col-major backward
        s4 = self.dir_convs[3](h_t.flatten(2).flip(-1)).flip(-1).view(B, C, W, H).transpose(2, 3)
        
        fused = (s1 + s2 + s3 + s4) * 0.25
        x = x + self.out_proj(fused * gate)
        x = x + self.ffn(self.norm2(x))
        return x


class MambaVectorFieldPredictor(nn.Module):
    """
    Production-grade Multi-Scale SS2D Velocity Predictor for Latent Flow Matching.
    """
    def __init__(self, latent_dim=8, hidden_dim=128):
        super().__init__()
        time_dim = hidden_dim * 2
        self.time_embed = nn.Sequential(
            SinusoidalTimeEmb(hidden_dim),
            nn.Linear(hidden_dim, time_dim),
            nn.SiLU(),
            nn.Linear(time_dim, time_dim)
        )
        
        # Concatenate z_t (8) + condition_sar (8) = 16 channels
        self.in_proj = nn.Conv2d(latent_dim * 2, hidden_dim, kernel_size=3, padding=1)
        
        # Encoder (32x32 -> 16x16)
        self.enc1 = BidirectionalSS2DBlock(hidden_dim, time_dim)
        self.down = nn.Conv2d(hidden_dim, hidden_dim * 2, kernel_size=3, stride=2, padding=1)
        
        # Bottleneck (16x16)
        self.mid1 = BidirectionalSS2DBlock(hidden_dim * 2, time_dim)
        self.mid2 = BidirectionalSS2DBlock(hidden_dim * 2, time_dim)
        
        # Decoder (16x16 -> 32x32)
        self.up = nn.ConvTranspose2d(hidden_dim * 2, hidden_dim, kernel_size=4, stride=2, padding=1)
        self.skip_fuse = nn.Conv2d(hidden_dim * 2, hidden_dim, kernel_size=1)
        self.dec1 = BidirectionalSS2DBlock(hidden_dim, time_dim)
        
        # Output velocity projection (zero-initialized for stable start)
        self.out_norm = nn.GroupNorm(8, hidden_dim)
        self.out_proj = nn.Conv2d(hidden_dim, latent_dim, kernel_size=3, padding=1)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

    def forward(self, z_t, t, condition_sar):
        t_emb = self.time_embed(t)
        
        x = torch.cat([z_t, condition_sar], dim=1)
        h0 = self.in_proj(x)
        
        h1 = self.enc1(h0, t_emb)          # (B, 128, 32, 32)
        h_down = self.down(h1)             # (B, 256, 16, 16)
        
        m = self.mid1(h_down, t_emb)       # (B, 256, 16, 16)
        m = self.mid2(m, t_emb)            # (B, 256, 16, 16)
        
        u = self.up(m)                     # (B, 128, 32, 32)
        u = self.skip_fuse(torch.cat([u, h1], dim=1))
        out = self.dec1(u, t_emb)          # (B, 128, 32, 32)
        
        return self.out_proj(F.silu(self.out_norm(out)))


class FlowMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, z_clear, z_cloudy, predictor, condition_sar):
        B = z_clear.shape[0]
        device = z_clear.device
        
        t = torch.rand(B, 1, 1, 1, device=device)
        z_t = (1.0 - t) * z_clear + t * z_cloudy
        true_velocity = z_cloudy - z_clear
        
        predicted_velocity = predictor(z_t, t, condition_sar)
        loss = self.mse(predicted_velocity, true_velocity)
        return loss, predicted_velocity, true_velocity