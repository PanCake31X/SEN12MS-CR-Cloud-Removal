import os
import sys
import copy
import glob
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from datasets import load_dataset
from torch.amp import autocast, GradScaler

from src.models.latent_compression import SatelliteVAE
from src.models.flow_matching import MambaVectorFieldPredictor

LATENT_SCALE = 3.0


class SobelEdgeLoss(nn.Module):
    """Computes L1 loss on horizontal and vertical spatial gradients across all 4 bands."""
    def __init__(self, channels=4):
        super().__init__()
        kx = torch.tensor([[-1., 0., 1.], [-2., 0., 2.], [-1., 0., 1.]]).view(1, 1, 3, 3)
        ky = torch.tensor([[-1., -2., -1.], [0., 0., 0.], [1., 2., 1.]]).view(1, 1, 3, 3)
        self.register_buffer("kx", kx.repeat(channels, 1, 1, 1))
        self.register_buffer("ky", ky.repeat(channels, 1, 1, 1))
        self.channels = channels

    def forward(self, pred, target):
        gx_p = F.conv2d(pred, self.kx, padding=1, groups=self.channels)
        gy_p = F.conv2d(pred, self.ky, padding=1, groups=self.channels)
        gx_t = F.conv2d(target, self.kx, padding=1, groups=self.channels)
        gy_t = F.conv2d(target, self.ky, padding=1, groups=self.channels)
        return F.l1_loss(gx_p, gx_t) + F.l1_loss(gy_p, gy_t)


def format_hf_sample(sample, device):
    sar = np.frombuffer(sample["sar"], dtype=np.float32).reshape(sample["sar_shape"])
    cloudy = np.frombuffer(sample["cloudy"], dtype=np.int16).reshape(sample["opt_shape"])
    clear = np.frombuffer(sample["target"], dtype=np.int16).reshape(sample["opt_shape"])

    sar_tensor = torch.tensor(sar).float()
    if sar_tensor.shape[-1] == 2:
        sar_tensor = sar_tensor.permute(2, 0, 1)
    sar_tensor = (torch.clamp(sar_tensor, -25.0, 0.0) / 12.5 + 1.0).unsqueeze(0).to(device)

    opt_cloudy = torch.tensor(cloudy).permute(2, 0, 1).unsqueeze(0).float().to(device) / 10000.0
    opt_clear = torch.tensor(clear).permute(2, 0, 1).unsqueeze(0).float().to(device) / 10000.0

    opt_cloudy = (torch.clamp(opt_cloudy, 0.0, 1.0) * 2.0 - 1.0)[:, [1, 2, 3, 7], :, :]
    opt_clear = (torch.clamp(opt_clear, 0.0, 1.0) * 2.0 - 1.0)[:, [1, 2, 3, 7], :, :]

    return sar_tensor, opt_cloudy, opt_clear


@torch.no_grad()
def update_ema(ema_module, live_module, decay=0.995):
    for p_ema, p_live in zip(ema_module.parameters(), live_module.parameters()):
        p_ema.data.mul_(decay).add_(p_live.data, alpha=1.0 - decay)


def build_stream(seed):
    ds = load_dataset("Hermanni/sen12mscr", split="train", streaming=True)
    return iter(ds.shuffle(buffer_size=1000, seed=seed))


def train_resume():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Stage 1 Resume (Option B: Pixel + Edge Fine-Tune) on: {device}", flush=True)

    # 1. Locate 100k checkpoint in attached Kaggle Input or Working dir
    matches = glob.glob("/kaggle/**/mamba_flow_prod_step_100000.pth", recursive=True)
    if not matches:
        raise FileNotFoundError("mamba_flow_prod_step_100000.pth not found. Ensure Version 1 output is attached in Input.")
    ckpt_path = matches[0]
    print(f"Resuming from: {ckpt_path}", flush=True)

    latent_dim = 8
    vae = SatelliteVAE(opt_channels=4, sar_channels=2, latent_dim=latent_dim).to(device)
    predictor = MambaVectorFieldPredictor(latent_dim=latent_dim, hidden_dim=128).to(device)

    ckpt = torch.load(ckpt_path, map_location=device, weights_only=True)
    vae.load_state_dict(ckpt["vae_state"])
    predictor.load_state_dict(ckpt["predictor_state"])

    # Freeze optical encoder & decoder (gradients still flow through decoder to predictor)
    for p in vae.opt_encoder.parameters():
        p.requires_grad = False
    for p in vae.opt_decoder.parameters():
        p.requires_grad = False

    # EMA copies for sharp inference
    ema_predictor = copy.deepcopy(predictor).eval()
    ema_sar_encoder = copy.deepcopy(vae.sar_encoder).eval()

    trainable_params = list(vae.sar_encoder.parameters()) + list(predictor.parameters())

    # Remaining 60,000 samples (steps 100,001 -> 160,000)
    remaining_samples = 60000
    accumulation_steps = 16
    total_opt_steps = remaining_samples // accumulation_steps

    optimizer = optim.AdamW(trainable_params, lr=1e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_opt_steps, eta_min=1e-5)
    scaler = GradScaler('cuda')

    mse_fn = nn.MSELoss()
    l1_fn = nn.L1Loss()
    edge_fn = SobelEdgeLoss(channels=4).to(device)

    seed = 1042
    print("Initializing Shuffled HuggingFace Stream (seed=1042)...", flush=True)
    iterator = build_stream(seed)

    optimizer.zero_grad()
    r_flow, r_pix, r_edge, valid_count = 0.0, 0.0, 0.0, 0
    consecutive_errors = 0

    for step in range(1, remaining_samples + 1):
        # Robust stream fetch with automatic reconnect on HF parquet drops
        sample = None
        while sample is None:
            try:
                sample = next(iterator)
                consecutive_errors = 0
            except StopIteration:
                seed += 1
                iterator = build_stream(seed)
            except Exception as e:
                consecutive_errors += 1
                print(f"[Stream Warning] Step {100000 + step}: {e} (Retry {consecutive_errors}/10)", flush=True)
                if consecutive_errors >= 10:
                    raise RuntimeError("Exceeded 10 consecutive stream failures.")
                time.sleep(5)
                seed += 1
                iterator = build_stream(seed)

        sar_t, opt_cloudy, opt_clear = format_hf_sample(sample, device)

        with autocast('cuda'):
            with torch.no_grad():
                z_cloudy = vae.encode_optical(opt_cloudy) / LATENT_SCALE
                z_clear = vae.encode_optical(opt_clear) / LATENT_SCALE
            z_sar = vae.encode_sar(sar_t)

            # Flow Matching forward
            t = torch.rand(1, 1, 1, 1, device=device)
            z_t = (1.0 - t) * z_clear + t * z_cloudy
            true_v = z_cloudy - z_clear
            pred_v = predictor(z_t, t, z_sar)

            flow_loss = mse_fn(pred_v, true_v)

            # Decoded Pixel-Space Supervision (L1 + Sobel Edge)
            z_clear_hat = z_t - t * pred_v
            pred_img = vae.decode(z_clear_hat * LATENT_SCALE)
            pix_loss = l1_fn(pred_img, opt_clear)
            edge_loss = edge_fn(pred_img, opt_clear)

            total_loss = flow_loss + 0.5 * pix_loss + 0.25 * edge_loss
            loss = total_loss / accumulation_steps

        # NaN / Inf guard
        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        scaler.scale(loss).backward()
        r_flow += flow_loss.item()
        r_pix += pix_loss.item()
        r_edge += edge_loss.item()
        valid_count += 1

        if step % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()

            update_ema(ema_predictor, predictor, decay=0.995)
            update_ema(ema_sar_encoder, vae.sar_encoder, decay=0.995)

        if step % 1000 == 0 and valid_count > 0:
            lr_now = scheduler.get_last_lr()[0]
            print(
                f"Sample {100000 + step:6d}/160000 | "
                f"Flow MSE: {r_flow/valid_count:.4f} | "
                f"Pix L1: {r_pix/valid_count:.4f} | "
                f"Edge L1: {r_edge/valid_count:.4f} | "
                f"LR: {lr_now:.6f}",
                flush=True
            )
            r_flow, r_pix, r_edge, valid_count = 0.0, 0.0, 0.0, 0

        if step % 20000 == 0:
            ema_vae_state = copy.deepcopy(vae.state_dict())
            for k, v in ema_sar_encoder.state_dict().items():
                ema_vae_state[f"sar_encoder.{k}"] = v
            ckpt_out = f"/kaggle/working/mamba_flow_stage1_step_{100000 + step}.pth"
            torch.save({
                'vae_state': ema_vae_state,
                'predictor_state': ema_predictor.state_dict(),
                'raw_predictor_state': predictor.state_dict(),
                'step': 100000 + step,
            }, ckpt_out)
            print(f"Checkpoint saved: {ckpt_out}", flush=True)

    # Save final EMA weights
    ema_vae_state = copy.deepcopy(vae.state_dict())
    for k, v in ema_sar_encoder.state_dict().items():
        ema_vae_state[f"sar_encoder.{k}"] = v

    final_path = "/kaggle/working/mamba_flow_stage1_160k_FINAL.pth"
    torch.save({
        'vae_state': ema_vae_state,
        'predictor_state': ema_predictor.state_dict(),
        'raw_predictor_state': predictor.state_dict(),
        'step': 160000,
    }, final_path)
    print(f"Stage 1 Complete. Final EMA weights saved to {final_path}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    train_resume()
