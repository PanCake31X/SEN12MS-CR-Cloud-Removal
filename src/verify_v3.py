import os
import sys
import torch
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
    
    sar_tensor = torch.tensor(sar).float()
    if sar_tensor.shape[-1] == 2:
        sar_tensor = sar_tensor.permute(2, 0, 1)
    sar_tensor = (torch.clamp(sar_tensor, -25.0, 0.0) / 12.5 + 1.0).unsqueeze(0).to(device)
    
    opt_cloudy = torch.tensor(cloudy).permute(2, 0, 1).unsqueeze(0).float().to(device) / 10000.0
    opt_clear = torch.tensor(clear).permute(2, 0, 1).unsqueeze(0).float().to(device) / 10000.0
    
    opt_cloudy = (torch.clamp(opt_cloudy, 0.0, 1.0) * 2.0 - 1.0)[:, [1, 2, 3, 7], :, :]
    opt_clear = (torch.clamp(opt_clear, 0.0, 1.0) * 2.0 - 1.0)[:, [1, 2, 3, 7], :, :]
    
    return sar_tensor, opt_cloudy, opt_clear

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    vae = SatelliteVAE(opt_channels=4, sar_channels=2, latent_dim=8).to(device)
    predictor = MambaVectorFieldPredictor(latent_dim=8, hidden_dim=128).to(device)
    flow_loss_fn = FlowMatchingLoss().to(device)
    
    ckpt = torch.load("/kaggle/working/mamba_flow_checkpoint_step_5000.pth", map_location=device, weights_only=True)
    vae.load_state_dict(ckpt["vae_state"])
    for p in vae.opt_encoder.parameters():
        p.requires_grad = False
    for p in vae.opt_decoder.parameters():
        p.requires_grad = False
        
    trainable = list(vae.sar_encoder.parameters()) + list(predictor.parameters())
    optimizer = optim.AdamW(trainable, lr=3e-4, weight_decay=1e-4)
    scaler = GradScaler('cuda')
    
    ds = load_dataset("Hermanni/sen12mscr", split="train", streaming=True).shuffle(buffer_size=1000, seed=42)
    iterator = iter(ds)
    
    total_samples = 5000
    accum = 4
    optimizer.zero_grad()
    running = 0.0
    
    for step in range(1, total_samples + 1):
        sample = next(iterator)
        sar_t, opt_cloudy, opt_clear = format_hf_sample(sample, device)
        
        with autocast('cuda'):
            with torch.no_grad():
                z_cloudy = vae.encode_optical(opt_cloudy) / LATENT_SCALE
                z_clear = vae.encode_optical(opt_clear) / LATENT_SCALE
            z_sar = vae.encode_sar(sar_t)
            flow_loss, _, _ = flow_loss_fn(z_clear, z_cloudy, predictor, z_sar)
            loss = flow_loss / accum
            
        scaler.scale(loss).backward()
        running += flow_loss.item()
        
        if step % accum == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()
            
        if step % 1000 == 0:
            print(f"Step {step}/{total_samples} | Avg Flow MSE: {running / 1000.0:.4f}", flush=True)
            running = 0.0
            
    out_path = "/kaggle/working/mamba_flow_v3_verify.pth"
    torch.save({'vae_state': vae.state_dict(), 'predictor_state': predictor.state_dict()}, out_path)
    print(f"Saved verification checkpoint: {out_path}", flush=True)
    sys.stdout.flush()
    os._exit(0)

if __name__ == "__main__":
    main()
