import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


class ResConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.skip = nn.Conv2d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x):
        h = F.silu(self.norm1(self.conv1(x)))
        h = self.norm2(self.conv2(h))
        return F.silu(h + self.skip(x))


class RefinementUNetV3(nn.Module):
    """
    256x256 Mask-Gated Residual Refiner.
    - Backbone + mask_head warm-start cleanly from RefinementUNetV2.
    - Clear regions (1 - M) receive only a low-frequency bounded color shift,
      physically preventing the refiner from blurring clear optical detail.
    - Cloud regions (M) receive full high-frequency SAR-guided reconstruction.
    """
    def __init__(self, base_ch=48):
        super().__init__()
        in_channels = 4 + 13 + 2 + 2  # stage1 + 13-band optical + raw SAR + 5x5 SAR

        # Encoder: 256 -> 128 -> 64 -> 32
        self.enc1 = ResConvBlock(in_channels, base_ch)
        self.down1 = nn.Conv2d(base_ch, base_ch * 2, kernel_size=3, stride=2, padding=1)
        self.enc2 = ResConvBlock(base_ch * 2, base_ch * 2)
        self.down2 = nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=3, stride=2, padding=1)
        self.enc3 = ResConvBlock(base_ch * 4, base_ch * 4)
        self.down3 = nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=3, stride=2, padding=1)

        # Bottleneck: 32x32
        self.mid1 = ResConvBlock(base_ch * 4, base_ch * 4)
        self.mid2 = ResConvBlock(base_ch * 4, base_ch * 4)

        # Decoder: 32 -> 64 -> 128 -> 256
        self.up3 = nn.ConvTranspose2d(base_ch * 4, base_ch * 4, kernel_size=4, stride=2, padding=1)
        self.dec3 = ResConvBlock(base_ch * 8, base_ch * 4)
        self.up2 = nn.ConvTranspose2d(base_ch * 4, base_ch * 2, kernel_size=4, stride=2, padding=1)
        self.dec2 = ResConvBlock(base_ch * 4, base_ch * 2)
        self.up1 = nn.ConvTranspose2d(base_ch * 2, base_ch, kernel_size=4, stride=2, padding=1)
        self.dec1 = ResConvBlock(base_ch * 2, base_ch)

        # Soft cloud mask head (loaded from v2)
        self.mask_head = nn.Conv2d(base_ch, 1, kernel_size=3, padding=1)
        nn.init.zeros_(self.mask_head.weight)
        nn.init.constant_(self.mask_head.bias, 2.0)

        # Zero-initialized cloud-only high-frequency residual
        self.delta_cloud = nn.Conv2d(base_ch, 4, kernel_size=3, padding=1)
        nn.init.zeros_(self.delta_cloud.weight)
        nn.init.zeros_(self.delta_cloud.bias)

        # Zero-initialized low-frequency color/exposure correction for clear pixels
        self.color_head = nn.Conv2d(base_ch, 4, kernel_size=1)
        nn.init.zeros_(self.color_head.weight)
        nn.init.zeros_(self.color_head.bias)

    def forward(self, stage1_pred, opt_cloudy_13, sar_norm):
        sar_5x5 = F.avg_pool2d(sar_norm, kernel_size=5, stride=1, padding=2)
        x = torch.cat([stage1_pred, opt_cloudy_13, sar_norm, sar_5x5], dim=1)

        e1 = self.enc1(x)
        e2 = self.enc2(self.down1(e1))
        e3 = self.enc3(self.down2(e2))
        m = self.mid2(self.mid1(self.down3(e3)))

        d3 = self.dec3(torch.cat([self.up3(m), e3], dim=1))
        d2 = self.dec2(torch.cat([self.up2(d3), e2], dim=1))
        d1 = self.dec1(torch.cat([self.up1(d2), e1], dim=1))

        mask = torch.sigmoid(self.mask_head(d1))
        delta_cloud = self.delta_cloud(d1)

        # 17x17 low-pass guarantees clear-pixel color adjustment cannot blur fine edges
        d1_lowfreq = F.avg_pool2d(d1, kernel_size=17, stride=1, padding=8)
        delta_color = 0.15 * torch.tanh(self.color_head(d1_lowfreq))

        opt_cloudy_4 = opt_cloudy_13[:, [1, 2, 3, 7], :, :]
        clear_branch = opt_cloudy_4 + delta_color
        cloud_branch = stage1_pred + delta_cloud

        out = (1.0 - mask) * clear_branch + mask * cloud_branch
        return torch.clamp(out, -1.0, 1.0)


class ConditionalPatchDiscriminator(nn.Module):
    """
    Spectral-Norm PatchGAN Discriminator conditioned on 5x5 despeckled SAR.
    No BatchNorm -> completely stable at batch_size=1.
    Input: [opt_4 (4), sar_5x5 (2)] = 6 channels -> 30x30 patch realism logits.
    """
    def __init__(self, in_channels=6, base_ch=64):
        super().__init__()
        self.net = nn.Sequential(
            spectral_norm(nn.Conv2d(in_channels, base_ch, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_ch, base_ch * 2, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_ch * 2, base_ch * 4, kernel_size=4, stride=2, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_ch * 4, base_ch * 4, kernel_size=4, stride=1, padding=1)),
            nn.LeakyReLU(0.2, inplace=True),
            spectral_norm(nn.Conv2d(base_ch * 4, 1, kernel_size=4, stride=1, padding=1)),
        )

    def forward(self, opt_4, sar_5x5):
        return self.net(torch.cat([opt_4, sar_5x5], dim=1))