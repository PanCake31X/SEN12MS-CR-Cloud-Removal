import torch
import torch.nn as nn

class SatelliteVAE(nn.Module):
    def __init__(self, opt_channels=4, sar_channels=2, latent_dim=8):
        super().__init__()
        # Downsample factor of 8 (256x256 -> 32x32)
        # Separate encoders because SAR and Optical have entirely different physical distributions
        
        self.opt_encoder = nn.Sequential(
            nn.Conv2d(opt_channels, 32, kernel_size=4, stride=2, padding=1),  # 128x128
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),            # 64x64
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, latent_dim, kernel_size=4, stride=2, padding=1)     # 32x32
        )
        
        self.sar_encoder = nn.Sequential(
            nn.Conv2d(sar_channels, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, latent_dim, kernel_size=4, stride=2, padding=1)
        )

        # Decoder only needs to reconstruct the Optical channels (the target)
        self.opt_decoder = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(32, opt_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh() # Normalizes output to [-1, 1]
        )

    def encode_optical(self, x):
        return self.opt_encoder(x)
        
    def encode_sar(self, x):
        return self.sar_encoder(x)

    def decode(self, z):
        return self.opt_decoder(z)