import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from datasets import load_dataset
from torch.amp import autocast, GradScaler

from src.models.latent_compression import SatelliteVAE
from src.models.flow_matching import MambaVectorFieldPredictor, FlowMatchingLoss

LATENT_SCALE = 3.0

def format_hf_sample(sample, device):
    sar = np.frombuffer(sample["sar"], dtype=np.float32).reshape(sample["sar_shape"])
    cloudy = np.frombuffer(sample["cloudy"], dtype=np.int16).reshape(sample["opt_shape"])
    clear = np.frombuffer(sample["target"], dtype=np.int16).reshape(sample["opt_shape"])
    
    # Normalize Sentinel-1 SAR dB [-25, 0] -> [-1, 1]
    sar_tensor = torch.tensor(sar).float()
    if sar_tensor.shape[-1] == 2:
        sar_tensor = sar_tensor.permute(2, 0, 1)
    sar_tensor = (torch.clamp(sar_tensor, -25.0, 0.0) / 12.5 + 1.0).unsqueeze(0).to(device)
    
    # Normalize Sentinel-2 Optical [0, 10000] -> [-1, 1], core 4 bands (B, G, R, NIR)
    opt_cloudy = torch.tensor(cloudy).permute(2, 0, 1).unsqueeze(0).float().to(device) / 10000.0
    opt_clear = torch.tensor(clear).permute(2, 0, 1).unsqueeze(0).float().to(device) / 10000.0
    
    opt_cloudy = (torch.clamp(opt_cloudy, 0.0, 1.0) * 2.0 - 1.0)[:, [1, 2, 3, 7], :, :]
    opt_clear = (torch.clamp(opt_clear, 0.0, 1.0) * 2.0 - 1.0)[:, [1, 2, 3, 7], :, :]
    
    return sar_tensor, opt_cloudy, opt_clear

def pretrain_vae(vae, device, steps=5000):
    print(f"\n[Phase 1] Pre-training Optical VAE for {steps} steps...", flush=True)
    vae_params = list(vae.opt_encoder.parameters()) + list(vae.opt_decoder.parameters())
    opt = optim.AdamW(vae_params, lr=2e-4, weight_decay=1e-5)
    loss_fn = nn.L1Loss().to(device)
    scaler = GradScaler('cuda')
    
    ds = load_dataset("Hermanni/sen12mscr", split="train", streaming=True).shuffle(buffer_size=1000, seed=101)
    iterator = iter(ds)
    running = 0.0
    
    for step in range(1, steps + 1):
        try:
            sample = next(iterator)
        except StopIteration:
            iterator = iter(ds)
            sample = next(iterator)
            
        _, opt_cloudy, opt_clear = format_hf_sample(sample, device)
        opt.zero_grad()
        with autocast('cuda'):
            loss = loss_fn(vae.decode(vae.encode_optical(opt_clear)), opt_clear) + \
                   loss_fn(vae.decode(vae.encode_optical(opt_cloudy)), opt_cloudy)
        scaler.scale(loss).backward()
        scaler.step(opt)
        scaler.update()
        running += loss.item()
        
        if step % 1000 == 0:
            print(f"VAE Step {step}/{steps} | Avg L1 Recon: {running / 1000.0:.4f}", flush=True)
            running = 0.0
            
    torch.save({'vae_state': vae.state_dict()}, "/kaggle/working/pretrained_vae_5000.pth")
    print("[Phase 1] Optical VAE pre-training complete.\n", flush=True)

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Executing Production Pipeline on: {device}", flush=True)
    
    latent_dim = 8
    vae = SatelliteVAE(opt_channels=4, sar_channels=2, latent_dim=latent_dim).to(device)
    predictor = MambaVectorFieldPredictor(latent_dim=latent_dim, hidden_dim=128).to(device)
    flow_loss_fn = FlowMatchingLoss().to(device)
    
    # Phase 1: Load existing VAE if present, otherwise pre-train automatically
    vae_ckpt = "/kaggle/working/mamba_flow_checkpoint_step_5000.pth"
    if os.path.exists(vae_ckpt):
        print(f"Loading existing VAE from {vae_ckpt}...", flush=True)
        ckpt = torch.load(vae_ckpt, map_location=device, weights_only=True)
        vae.load_state_dict(ckpt["vae_state"])
    else:
        pretrain_vae(vae, device, steps=5000)
        
    # Freeze Optical Manifold
    for p in vae.opt_encoder.parameters():
        p.requires_grad = False
    for p in vae.opt_decoder.parameters():
        p.requires_grad = False
    
    # Phase 2: Production Flow Matching Training
    trainable_params = list(vae.sar_encoder.parameters()) + list(predictor.parameters())
    total_samples = 160000
    accumulation_steps = 16
    total_opt_steps = total_samples // accumulation_steps
    
    optimizer = optim.AdamW(trainable_params, lr=3e-4, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_opt_steps, eta_min=1e-5)
    scaler = GradScaler('cuda')
    
    print("[Phase 2] Initializing Shuffled HuggingFace Stream...", flush=True)
    ds = load_dataset("Hermanni/sen12mscr", split="train", streaming=True).shuffle(buffer_size=1000, seed=42)
    iterator = iter(ds)
    
    optimizer.zero_grad()
    running_loss = 0.0
    
    for step in range(1, total_samples + 1):
        try:
            sample = next(iterator)
        except StopIteration:
            ds = ds.shuffle(buffer_size=1000, seed=42 + step)
            iterator = iter(ds)
            sample = next(iterator)
            
        sar_tensor, opt_cloudy_tensor, opt_clear_tensor = format_hf_sample(sample, device)
        
        with autocast('cuda'):
            with torch.no_grad():
                z_cloudy = vae.encode_optical(opt_cloudy_tensor) / LATENT_SCALE
                z_clear = vae.encode_optical(opt_clear_tensor) / LATENT_SCALE
            z_sar = vae.encode_sar(sar_tensor)
            
            flow_loss, _, _ = flow_loss_fn(z_clear, z_cloudy, predictor, z_sar)
            loss = flow_loss / accumulation_steps
            
        scaler.scale(loss).backward()
        running_loss += flow_loss.item()
        
        if step % accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            optimizer.zero_grad()
            
        if step % 1000 == 0:
            avg_loss = running_loss / 1000.0
            lr_now = scheduler.get_last_lr()[0]
            print(f"Sample {step:6d}/{total_samples} | Avg Flow MSE: {avg_loss:.4f} | LR: {lr_now:.6f}", flush=True)
            running_loss = 0.0
            
        if step % 20000 == 0:
            ckpt_out = f"/kaggle/working/mamba_flow_prod_step_{step}.pth"
            torch.save({
                'vae_state': vae.state_dict(),
                'predictor_state': predictor.state_dict(),
                'optimizer_state': optimizer.state_dict(),
                'step': step,
            }, ckpt_out)
            print(f"Checkpoint saved: {ckpt_out}", flush=True)
            
    final_path = "/kaggle/working/mamba_flow_prod_FINAL.pth"
    torch.save({
        'vae_state': vae.state_dict(),
        'predictor_state': predictor.state_dict(),
    }, final_path)
    print(f"Training complete. Final weights saved to {final_path}", flush=True)
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    train()
