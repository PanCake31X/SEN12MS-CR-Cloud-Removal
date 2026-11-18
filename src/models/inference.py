import torch
import torch.nn as nn

class OneStepODESolver(nn.Module):
    """
    Executes a 1-step Euler integration to transport the cloudy observation 
    directly to the clear ground truth using the predicted vector field.
    """
    def __init__(self):
        super().__init__()

    def forward(self, z_cloudy, condition_sar, predictor):
        B = z_cloudy.shape[0]
        device = z_cloudy.device
        
        # In our Flow Formulation: z_t = (1 - t)*z_clear + t*z_cloudy
        # Therefore, at t=1, z_t is entirely the cloudy image.
        t_start = torch.ones(B, 1, 1, 1, device=device)
        
        # Predict the velocity at the cloudy state
        predicted_velocity = predictor(z_cloudy, t_start, condition_sar)
        
        # 1-Step Euler Integration:
        # z_clear = z_cloudy - velocity * delta_t (where delta_t is 1.0)
        z_clear_pred = z_cloudy - predicted_velocity
        
        return z_clear_pred

class SARGuidedCloudRemovalPipeline(nn.Module):
    """
    The complete production pipeline wrapping Phase 1 through 5.
    """
    def __init__(self, vae, predictor):
        super().__init__()
        self.vae = vae
        self.predictor = predictor
        self.solver = OneStepODESolver()

    def forward(self, opt_cloudy, sar_filtered):
        # 1. Compress into latent manifold
        z_cloudy = self.vae.encode_optical(opt_cloudy)
        z_sar = self.vae.encode_sar(sar_filtered)
        
        # 2. Solve the ODE trajectory to hallucinate/restore clear features
        z_clear_pred = self.solver(z_cloudy, z_sar, self.predictor)
        
        # 3. Decode back to high-resolution optical pixel space
        opt_clear_pred = self.vae.decode(z_clear_pred)
        
        return opt_clear_pred