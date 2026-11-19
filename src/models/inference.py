import torch
import torch.nn as nn

LATENT_SCALE = 3.0

class MultiStepODESolver(nn.Module):
    """
    Executes N-step Euler integration from t=1 (cloudy) to t=0 (clear)
    in unit-variance latent space.
    """
    def __init__(self, steps=10):
        super().__init__()
        self.steps = steps

    def forward(self, z_cloudy, condition_sar, predictor):
        B = z_cloudy.shape[0]
        device = z_cloudy.device
        
        z = z_cloudy.clone()
        dt = 1.0 / self.steps
        
        for s in range(self.steps):
            t_curr = torch.full((B, 1, 1, 1), 1.0 - s * dt, device=device)
            v = predictor(z, t_curr, condition_sar)
            z = z - v * dt
            
        return z

class SARGuidedCloudRemovalPipeline(nn.Module):
    def __init__(self, vae, predictor, steps=10):
        super().__init__()
        self.vae = vae
        self.predictor = predictor
        self.solver = MultiStepODESolver(steps=steps)

    def forward(self, opt_cloudy, sar_norm):
        z_cloudy = self.vae.encode_optical(opt_cloudy) / LATENT_SCALE
        z_sar = self.vae.encode_sar(sar_norm)
        
        z_clear_pred = self.solver(z_cloudy, z_sar, self.predictor)
        opt_clear_pred = self.vae.decode(z_clear_pred * LATENT_SCALE)
        return opt_clear_pred