import os
import sys
import torch
import torch.nn as nn
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from stable_baselines3.common.utils import set_random_seed
import airsim

# --- CONFIGURATION (MLP MAX SPEED) ---
LOG_DIR = "./airsim_mlp_logs"
MODEL_SAVE_PATH = "airsim_ppo_mlp_model"
TOTAL_TIMESTEPS = 50000 # Retaining 50,000 steps for quick test
N_STEPS = 2048 # <--- MODIFIED to support 1024 batch size
BATCH_SIZE = 1024 # <--- MODIFIED to 1024
EMBEDDING_SIZE = 128
ENCODER_WEIGHTS_PATH = 'src/training/autoencoder_encoder.pth' 

# --- 1. DEFINE THE FEATURE EXTRACTOR ---
class MLPPassThroughFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = EMBEDDING_SIZE):
        super(MLPPassThroughFeatureExtractor, self).__init__(observation_space, features_dim)
        
        n_input_channels = observation_space.shape[0] # Should be 128
        
        self.pass_through = nn.Sequential(
            nn.Linear(n_input_channels, features_dim),
        )
        self._features_dim = features_dim

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations


policy_kwargs = {
    "features_extractor_class": MLPPassThroughFeatureExtractor,
    "features_extractor_kwargs": dict(features_dim=EMBEDDING_SIZE),
    "net_arch": dict(pi=[64, 64], vf=[64, 64]) 
}
# --- END OF CUSTOM FEATURE EXTRACTOR ---


# --- Environment Import ---
from airsim_mlp_env import AirSimMLPEnv 


def run_training():
    print("Setting up MLP State-Embedding Training (HIGH SPEED)...")

    # --- 1. Environment Setup ---
    env_instance = AirSimMLPEnv()
    env = DummyVecEnv([lambda: env_instance]) 

    # --- 2. Callback for Saving ---
    checkpoint_callback = CheckpointCallback(
      save_freq=50000, save_path=LOG_DIR, name_prefix=MODEL_SAVE_PATH
    )

    # --- 3. Model Setup (MLP Policy) ---
    model = PPO(
        "MlpPolicy", 
        env, 
        policy_kwargs=policy_kwargs, 
        learning_rate=3e-5,
        n_steps=N_STEPS,
        batch_size=BATCH_SIZE,
        gamma=0.99, 
        verbose=1, 
        tensorboard_log=LOG_DIR, 
        device="cuda"
    )

    # --- 4. Start Training ---
    print(f"Starting training for {TOTAL_TIMESTEPS} timesteps...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=checkpoint_callback)

    # --- 5. Final Save ---
    model.save(MODEL_SAVE_PATH + "_final.zip")
    print(f"Training finished. Model saved to {MODEL_SAVE_PATH}_final.zip")
    env_instance.close()

if __name__ == "__main__":
    set_random_seed(42)
    run_training()
