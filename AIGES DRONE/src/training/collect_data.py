import numpy as np
import os
import sys
import airsim
import random

# Add the envs folder to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'envs')))
from airsim_env import AirSimDroneEnv

DATA_STEPS = 10000  # Number of data points to collect
DATA_FILENAME = 'airsim_dataset.npz'

def collect_data():
    print(f"Starting data collection for {DATA_STEPS} steps...")
    
    # Initialize environment
    env = AirSimDroneEnv()
    
    # Arrays to store data
    observations = []
    actions = []
    
    obs, info = env.reset()
    
    for i in range(DATA_STEPS):
        # Generate random action (pure exploration)
        action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        
        observations.append(obs.copy())
        actions.append(action.copy())
        
        if terminated or truncated:
            obs, info = env.reset()
        
        if (i + 1) % 1000 == 0:
            print(f"Collected {i + 1}/{DATA_STEPS} frames.")

    env.close()

    # Convert lists to numpy arrays
    observations = np.array(observations)
    actions = np.array(actions)
    
    # Save the dataset
    np.savez_compressed(DATA_FILENAME, obs=observations, act=actions)
    print(f"\nSuccessfully saved {observations.shape[0]} samples to {DATA_FILENAME}")

if __name__ == "__main__":
    collect_data()
