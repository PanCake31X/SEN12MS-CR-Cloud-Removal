import torch
import torch.nn as nn
from .mamba_ss2d import SS2DScanner

class MambaVectorFieldPredictor(nn.Module):
    """
    Predicts the vector field (velocity) to transport z_t to the clear latent.
    Simulates the Mamba block processing and un-flattening.
    """
    def __init__(self, latent_dim=8):
        super().__init__()
        self.scanner = SS2DScanner()
        
        # A lightweight 1D convolution to simulate the Mamba SSM processing
        # on the 4 directional sequences.
        self.simulated_ssm = nn.Conv1d(latent_dim, latent_dim, kernel_size=3, padding=1)
        
        # Reconstructs the 2D spatial grid from the fused sequences
        self.unflatten = nn.Sequential(
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(latent_dim, latent_dim, kernel_size=3, padding=1)
        )

    def forward(self, z_t, t, condition_sar):
        B, C, H, W = z_t.shape
        
        # In a full model, t (timestep) and condition_sar would modulate the features here
        # For this skeleton, we concatenate the SAR condition to z_t
        combined = z_t + condition_sar 
        
        # 1. Flatten via SS2D (Shape: B, 4, C, L)
        scanned_seqs = self.scanner(combined)
        
        # 2. Process each of the 4 directions independently (simulating Mamba S6 blocks)
        processed_seqs = []
        for i in range(4):
            # Extract direction i (Shape: B, C, L)
            seq = scanned_seqs[:, i, :, :]
            processed = self.simulated_ssm(seq)
            processed_seqs.append(processed)
            
        # 3. Average the 4 directional features to fuse global context
        fused_1d = sum(processed_seqs) / 4.0
        
        # 4. Un-flatten back to 2D spatial grid
        fused_2d = fused_1d.view(B, C, H, W)
        
        # 5. Output the predicted velocity vector field
        predicted_velocity = self.unflatten(fused_2d)
        return predicted_velocity

class FlowMatchingLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, z_clear, z_cloudy, predictor, condition_sar):
        """
        Calculates the Optimal Transport Flow Matching loss.
        z_clear: The target ground truth (z_0)
        z_cloudy: The source observation (z_1)
        """
        B = z_clear.shape[0]
        device = z_clear.device
        
        # 1. Sample a random timestep t between 0 and 1
        # Shape (B, 1, 1, 1) for broadcasting
        t = torch.rand(B, 1, 1, 1, device=device)
        
        # 2. Optimal Transport Interpolation (straight line between clear and cloudy)
        z_t = (1 - t) * z_clear + t * z_cloudy
        
        # 3. Calculate the True Velocity (the straight-line target)
        true_velocity = z_cloudy - z_clear
        
        # 4. Predict the Velocity using the Mamba Backbone
        predicted_velocity = predictor(z_t, t, condition_sar)
        
        # 5. Regression Loss
        loss = self.mse(predicted_velocity, true_velocity)
        return loss, predicted_velocity, true_velocity