import os
import gc
import sys
import copy
import glob
import math
import time
import traceback
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision.models as tv_models
import numpy as np
import pyarrow as pa
from datasets import load_dataset
from torch.amp import autocast, GradScaler

from src.models.latent_compression import SatelliteVAE
from src.models.flow_matching import MambaVectorFieldPredictor
from src.models.inference import SARGuidedCloudRemovalPipeline
from src.models.refiner import RefinementUNetV3, ConditionalPatchDiscriminator


def get_ram_gb():
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) / (1024.0 * 1024.0)
    except Exception:
        pass
    return -1.0


class MultiLayerVGGPerceptualLoss(nn.Module):
    """Frozen multi-scale VGG16 loss (relu1_2, relu2_2, relu3_3) on 2.5x gained RGB."""
    def __init__(self):
        super().__init__()
        vgg_full = tv_models.vgg16(weights=tv_models.VGG16_Weights.DEFAULT)
        self.blocks = nn.ModuleList([
            copy.deepcopy(vgg_full.features[:4]).eval(),
            copy.deepcopy(vgg_full.features[4:9]).eval(),
            copy.deepcopy(vgg_full.features[9:16]).eval(),
        ])
        del vgg_full
        gc.collect()
        for p in self.blocks.parameters():
            p.requires_grad = False
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, pred, target):
        p = (torch.clamp((pred[:, [2, 1, 0]].float() + 1.0) * 0.5 * 2.5, 0.0, 1.0) - self.mean) / self.std
        t = (torch.clamp((target[:, [2, 1, 0]].float() + 1.0) * 0.5 * 2.5, 0.0, 1.0) - self.mean) / self.std
        loss = 0.0
        for blk in self.blocks:
            p = blk(p)
            t = blk(t)
            loss = loss + F.l1_loss(p, t)
        return loss / len(self.blocks)


def fft_amplitude_loss(pred, target):
    fp = torch.fft.rfft2(pred.float(), norm="ortho")
    ft = torch.fft.rfft2(target.float(), norm="ortho")
    return F.l1_loss(torch.log1p(fp.abs()), torch.log1p(ft.abs()))


def format_hf_sample(sample, device):
    sar = np.frombuffer(sample["sar"], dtype=np.float32).reshape(sample["sar_shape"])
    cloudy = np.frombuffer(sample["cloudy"], dtype=np.int16).reshape(sample["opt_shape"])
    clear = np.frombuffer(sample["target"], dtype=np.int16).reshape(sample["opt_shape"])

    sar_tensor = torch.from_numpy(sar.copy()).float()
    if sar_tensor.shape[-1] == 2:
        sar_tensor = sar_tensor.permute(2, 0, 1)
    sar_tensor = (torch.clamp(sar_tensor, -25.0, 0.0) / 12.5 + 1.0).unsqueeze(0).to(device, non_blocking=True)

    cloudy_13 = torch.from_numpy(cloudy.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device, non_blocking=True) / 10000.0
    clear_13 = torch.from_numpy(clear.copy()).permute(2, 0, 1).unsqueeze(0).float().to(device, non_blocking=True) / 10000.0

    opt_cloudy_13 = torch.clamp(cloudy_13, 0.0, 1.0) * 2.0 - 1.0
    opt_clear_4 = (torch.clamp(clear_13, 0.0, 1.0) * 2.0 - 1.0)[:, [1, 2, 3, 7], :, :]

    return sar_tensor, opt_cloudy_13, opt_clear_4


@torch.no_grad()
def update_ema(ema_mod, live_mod, decay=0.998):
    for p_ema, p_live in zip(ema_mod.parameters(), live_mod.parameters()):
        p_ema.data.mul_(decay).add_(p_live.data, alpha=1.0 - decay)


def build_stream(seed):
    ds = load_dataset("Hermanni/sen12mscr", split="train", streaming=True)
    return iter(ds.shuffle(buffer_size=500, seed=seed))


def train_stage2_v3():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Stage 2 v3 (Mask-Gated + SAR-PatchGAN + MultiLayerVGG/FFT) on: {device}", flush=True)

    matches = glob.glob("/kaggle/**/mamba_flow_stage2_v2_FINAL.pth", recursive=True)
    if not matches:
        raise FileNotFoundError(
            "mamba_flow_stage2_v2_FINAL.pth not found in /kaggle/input. "
            "Attach your v2 weights dataset before committing."
        )
    v2_path = matches[0]
    print(f"Warm-starting Stage 1 + Refiner backbone/mask from: {v2_path}", flush=True)

    vae = SatelliteVAE(opt_channels=4, sar_channels=2, latent_dim=8).to(device)
    predictor = MambaVectorFieldPredictor(latent_dim=8, hidden_dim=128).to(device)
    refiner = RefinementUNetV3(base_ch=48).to(device)

    ckpt = torch.load(v2_path, map_location="cpu", weights_only=True)
    vae_state_cpu = ckpt["vae_state"]
    pred_state_cpu = ckpt["predictor_state"]
    vae.load_state_dict(vae_state_cpu)
    predictor.load_state_dict(pred_state_cpu)

    missing, unexpected = refiner.load_state_dict(ckpt["refiner_state"], strict=False)
    print(f"Refiner warm-start | New heads (zero-init): {missing} | Dropped v2 heads: {unexpected}", flush=True)
    del ckpt
    gc.collect()

    stage1 = SARGuidedCloudRemovalPipeline(vae, predictor, steps=10).to(device).eval()
    for p in stage1.parameters():
        p.requires_grad = False

    ema_refiner = copy.deepcopy(refiner).eval()
    disc = ConditionalPatchDiscriminator(in_channels=6, base_ch=64).to(device)

    total_samples = 20000
    accumulation_steps = 4
    total_opt_steps = total_samples // accumulation_steps  # 5,000 updates
    warmup_steps = 250

    opt_g = optim.AdamW(refiner.parameters(), lr=1.5e-4, betas=(0.5, 0.999), weight_decay=1e-4)
    opt_d = optim.AdamW(disc.parameters(), lr=2.0e-4, betas=(0.5, 0.999), weight_decay=1e-4)

    def lr_lambda(opt_step):
        if opt_step < warmup_steps:
            return float(opt_step + 1) / float(warmup_steps)
        progress = (opt_step - warmup_steps) / max(1, total_opt_steps - warmup_steps)
        return 0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * progress))

    sched_g = optim.lr_scheduler.LambdaLR(opt_g, lr_lambda)
    sched_d = optim.lr_scheduler.LambdaLR(opt_d, lr_lambda)

    scaler_g = GradScaler('cuda')
    scaler_d = GradScaler('cuda')
    vgg_fn = MultiLayerVGGPerceptualLoss().to(device)

    seed = 4042
    iterator = build_stream(seed)
    opt_g.zero_grad()
    opt_d.zero_grad()

    r_l1, r_vgg, r_fft, r_gadv, r_dloss, valid_count = 0.0, 0.0, 0.0, 0.0, 0.0, 0
    accum_valid = 0
    consecutive_errors = 0

    for step in range(1, total_samples + 1):
        while True:
            try:
                if iterator is None:
                    iterator = build_stream(seed)
                sample = next(iterator)
                sar_t, opt_cloudy_13, opt_clear_4 = format_hf_sample(sample, device)
                del sample
                consecutive_errors = 0
                break
            except StopIteration:
                seed += 1
                iterator = None
            except Exception as e:
                consecutive_errors += 1
                print(f"[Stream Warning] Step {step}: {e} (Retry {consecutive_errors}/10)", flush=True)
                if consecutive_errors >= 10:
                    raise RuntimeError("Exceeded 10 consecutive stream failures.")
                time.sleep(5)
                seed += 1
                iterator = None

        opt_cloudy_4 = opt_cloudy_13[:, [1, 2, 3, 7], :, :]
        sar_5x5 = F.avg_pool2d(sar_t, kernel_size=5, stride=1, padding=2)

        # Ramp adversarial weight: 0 for first 1k samples (D warmup), -> 0.015 by 3k
        if step <= 1000:
            adv_w = 0.0
        elif step <= 3000:
            adv_w = 0.015 * (step - 1000) / 2000.0
        else:
            adv_w = 0.015

        # --- 1. Generator Forward & Backward ---
        for p in disc.parameters():
            p.requires_grad = False

        with autocast('cuda'):
            with torch.no_grad():
                stage1_pred = stage1(opt_cloudy_4, sar_t).clamp(-1.0, 1.0)

            refined = refiner(stage1_pred, opt_cloudy_13, sar_t)
            l1 = F.l1_loss(refined, opt_clear_4)
            vgg = vgg_fn(refined, opt_clear_4)
            fft = fft_amplitude_loss(refined, opt_clear_4)

            if adv_w > 0.0:
                g_adv = -disc(refined, sar_5x5).mean()
            else:
                g_adv = torch.zeros((), device=device)

            loss_g = (l1 + 0.20 * vgg + 2.0 * fft + adv_w * g_adv) / accumulation_steps

        if not torch.isfinite(loss_g):
            opt_g.zero_grad()
            opt_d.zero_grad()
            accum_valid = 0
            continue

        scaler_g.scale(loss_g).backward()

        # --- 2. Discriminator Forward & Backward (Hinge Loss) ---
        for p in disc.parameters():
            p.requires_grad = True

        with autocast('cuda'):
            d_real = disc(opt_clear_4, sar_5x5)
            d_fake = disc(refined.detach(), sar_5x5)
            loss_d = (F.relu(1.0 - d_real).mean() + F.relu(1.0 + d_fake).mean()) / accumulation_steps

        if not torch.isfinite(loss_d):
            opt_g.zero_grad()
            opt_d.zero_grad()
            accum_valid = 0
            continue

        scaler_d.scale(loss_d).backward()

        r_l1 += l1.item()
        r_vgg += vgg.item()
        r_fft += fft.item()
        r_gadv += g_adv.item()
        r_dloss += (loss_d.item() * accumulation_steps)
        valid_count += 1
        accum_valid += 1

        if step % accumulation_steps == 0 and accum_valid > 0:
            scaler_g.unscale_(opt_g)
            torch.nn.utils.clip_grad_norm_(refiner.parameters(), 0.5)
            scaler_g.step(opt_g)
            scaler_g.update()
            sched_g.step()
            opt_g.zero_grad()

            scaler_d.unscale_(opt_d)
            torch.nn.utils.clip_grad_norm_(disc.parameters(), 1.0)
            scaler_d.step(opt_d)
            scaler_d.update()
            sched_d.step()
            opt_d.zero_grad()

            update_ema(ema_refiner, refiner, decay=0.998)
            accum_valid = 0

        if step % 500 == 0:
            gc.collect()
            pa.default_memory_pool().release_unused()

        if step % 1000 == 0 and valid_count > 0:
            lr_now = sched_g.get_last_lr()[0]
            print(
                f"Sample {step:5d}/{total_samples} | "
                f"L1: {r_l1/valid_count:.4f} | "
                f"VGG: {r_vgg/valid_count:.4f} | "
                f"FFT: {r_fft/valid_count:.4f} | "
                f"G_adv: {r_gadv/valid_count:+.3f} | "
                f"D_hinge: {r_dloss/valid_count:.3f} | "
                f"LR: {lr_now:.6f} | RAM: {get_ram_gb():.2f} GB",
                flush=True
            )
            r_l1, r_vgg, r_fft, r_gadv, r_dloss, valid_count = 0.0, 0.0, 0.0, 0.0, 0.0, 0

        if step % 5000 == 0:
            ckpt_out = f"/kaggle/working/mamba_flow_stage2_v3_step_{step}.pth"
            torch.save({
                'refiner_state': ema_refiner.state_dict(),
                'raw_refiner_state': refiner.state_dict(),
                'disc_state': disc.state_dict(),
                'step': step,
            }, ckpt_out)
            print(f"Checkpoint saved: {ckpt_out}", flush=True)

    final_path = "/kaggle/working/mamba_flow_stage2_v3_FINAL.pth"
    torch.save({
        'vae_state': vae_state_cpu,
        'predictor_state': pred_state_cpu,
        'refiner_state': ema_refiner.state_dict(),
        'step': total_samples,
    }, final_path)
    print(f"Stage 2 v3 Complete. All-in-one weights saved to {final_path}", flush=True)
    sys.stdout.flush()
    os._exit(0)


if __name__ == "__main__":
    try:
        train_stage2_v3()
    except Exception:
        traceback.print_exc()
        sys.stdout.flush()
        sys.stderr.flush()
        os._exit(1)