import sys
import os
import torch

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.flow_matching import MambaVectorFieldPredictor, FlowMatchingLoss

if __name__ == "__main__":
    print("Initializing Phase 4: Flow Matching & Generative Training...")
    
    # Mock latent tensors (Batch, Channels, H, W)
    batch_size = 2
    latent_dim = 8
    spatial_dim = 32
    
    z_clear = torch.randn(batch_size, latent_dim, spatial_dim, spatial_dim)
    z_cloudy = torch.randn(batch_size, latent_dim, spatial_dim, spatial_dim)
    condition_sar = torch.randn(batch_size, latent_dim, spatial_dim, spatial_dim)
    
    # Initialize Predictor and Loss Module
    predictor = MambaVectorFieldPredictor(latent_dim=latent_dim)
    loss_fn = FlowMatchingLoss()
    
    # Execute Flow Matching Forward Pass
    loss, pred_v, true_v = loss_fn(z_clear, z_cloudy, predictor, condition_sar)
    
    print(f"\nPredicted Velocity Shape: {pred_v.shape}")
    print(f"True Velocity Shape:      {true_v.shape}")
    print(f"Calculated Flow Loss (MSE): {loss.item():.4f}")
    
    # Assertions
    assert pred_v.shape == z_clear.shape, "Vector field spatial dimensions misaligned!"
    assert loss.dim() == 0, "Loss is not a scalar!"
    
    print("\nOptimal Transport Flow Matching loss calculated successfully.")
    print("The model is learning to regress straight vector fields.")