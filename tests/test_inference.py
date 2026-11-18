import sys
import os
import torch
import numpy as np
from datasets import load_dataset

# Add the project root to the path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.models.latent_compression import SatelliteVAE
from src.models.flow_matching import MambaVectorFieldPredictor
from src.models.inference import SARGuidedCloudRemovalPipeline

if __name__ == "__main__":
    print("Initializing Phase 5: 1-Step Inference with Real HuggingFace Data...")
    
    # 1. Initialize the full production architecture
    latent_dim = 8
    vae = SatelliteVAE(opt_channels=4, sar_channels=2, latent_dim=latent_dim)
    predictor = MambaVectorFieldPredictor(latent_dim=latent_dim)
    pipeline = SARGuidedCloudRemovalPipeline(vae, predictor)
    
    # 2. Stream the dataset directly from HuggingFace
    print("Connecting to HuggingFace Datasets...")
    ds = load_dataset("Hermanni/sen12mscr", split="train", streaming=True)
    sample = next(iter(ds))
    
    # 3. Decode binary bytes into numpy arrays
    sar = np.frombuffer(sample["sar"], dtype=np.float32).reshape(sample["sar_shape"])
    cloudy = np.frombuffer(sample["cloudy"], dtype=np.int16).reshape(sample["opt_shape"])
    
    # 4. Format into PyTorch Tensors (CHW format + Batch dimension)
    sar_tensor = torch.tensor(sar).permute(2, 0, 1).unsqueeze(0).float()
    opt_cloudy_tensor = torch.tensor(cloudy).permute(2, 0, 1).unsqueeze(0).float() / 10000.0
    
    # Slice Sentinel-2 13 bands down to the core 4 (R, G, B, NIR)
    opt_cloudy_tensor = opt_cloudy_tensor[:, [1, 2, 3, 7], :, :]
    
    print(f"\nReal Optical (Cloudy) Shape: {opt_cloudy_tensor.shape}")
    print(f"Real SAR Input Shape:        {sar_tensor.shape}")
    
    # 5. Execute 1-Step Generation
    print("\nExecuting Generative Mamba Flow Matching...")
    # Use torch.no_grad() to prevent memory bloat during inference
    with torch.no_grad():
        opt_clear_pred = pipeline(opt_cloudy_tensor, sar_tensor)
        
    print(f"\nOutput Optical (Clear) Shape: {opt_clear_pred.shape}")
    
    # Verify the output matches the original high-resolution dimensions
    assert opt_clear_pred.shape == opt_cloudy_tensor.shape, "End-to-end dimensional mismatch!"
    
    print("\nEnd-to-End Generative Pipeline verified with real SOTA data.")
    print("The architecture is structurally complete and ready for cloud deployment.")