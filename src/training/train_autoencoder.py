import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import os
import sys

# --- CONFIGURATION ---
DATA_FILENAME = 'airsim_dataset.npz'
EMBEDDING_SIZE = 128
EPOCHS = 75 # <--- MODIFIED TO 75 EPOCHS
BATCH_SIZE = 64
LR = 1e-3

# --- 1. DEFINE THE AUTOENCODER MODEL ---
class Autoencoder(nn.Module):
    def __init__(self, embedding_size=EMBEDDING_SIZE):
        super(Autoencoder, self).__init__()
        
        # Encoder: 64x64x1 -> 128-feature vector
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 16, 4, 2, 1), # 1x64x64 -> 16x32x32
            nn.ReLU(),
            nn.Conv2d(16, 32, 4, 2, 1), # 32x32x32 -> 32x16x16
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, 2, 1), # 64x16x16 -> 64x8x8
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, embedding_size), # 4096 -> 128
            nn.Tanh() # Activation for stable embedding range [-1, 1]
        )
        
        # Decoder: 128-feature vector -> 64x64x1 (Reconstruction Check)
        self.decoder = nn.Sequential(
            nn.Linear(embedding_size, 64 * 8 * 8),
            nn.Unflatten(1, (64, 8, 8)),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # 64x8x8 -> 32x16x16
            nn.ReLU(),
            nn.ConvTranspose2d(32, 16, 4, 2, 1), # 32x16x16 -> 16x32x32
            nn.ReLU(),
            nn.ConvTranspose2d(16, 1, 4, 2, 1), # 16x32x32 -> 1x64x64
            nn.Sigmoid() # Output pixels in [0, 1] range
        )

    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded, encoded

def run_autoencoder_training():
    # Load Data
    data = np.load(DATA_FILENAME)
    raw_obs = data['obs'] 

    # Preprocessing: Convert to PyTorch format (N, C, H, W) and normalize [0, 1]
    obs_tensor = torch.tensor(raw_obs, dtype=torch.float32).permute(0, 3, 1, 2) / 255.0
    
    # Resize tensor to 64x64 using interpolate
    obs_tensor = nn.functional.interpolate(obs_tensor, size=(64, 64), mode='area')
    
    dataset = TensorDataset(obs_tensor, obs_tensor)
    dataloader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # Initialize Model and Training
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = Autoencoder().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=LR)
    
    print(f"\n[INFO] Starting Autoencoder Training on {device}")
    print(f"[INFO] Dataset Size: {len(raw_obs)} samples, {EPOCHS} epochs.")
    
    # Training Loop
    for epoch in range(EPOCHS):
        total_loss = 0
        for data in dataloader:
            inputs, _ = data
            inputs = inputs.to(device)
            
            # Forward pass
            outputs, embeddings = model(inputs)
            loss = criterion(outputs, inputs)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        avg_loss = total_loss / len(dataloader)
        print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")

    # Save the ENCODER part of the model only
    encoder_path = 'src/training/autoencoder_encoder.pth'
    torch.save(model.encoder.state_dict(), encoder_path)
    print(f"\n[INFO] Autoencoder training complete. Encoder saved to {encoder_path}")

if __name__ == "__main__":
    run_autoencoder_training()
