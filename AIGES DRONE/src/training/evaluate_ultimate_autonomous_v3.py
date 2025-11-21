#!/usr/bin/env python3
import os
import sys
import time
import math
import torch
import gymnasium as gym
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
import airsim
import cv2

# --- PRO CONFIGURATION ---
WINDOWS_HOST_IP = "172.24.176.1"
MODEL_PATH = "airsim_mlp_logs/airsim_ppo_mlp_model_50000_steps.zip"

# Flight Physics Constants
SAFE_DISTANCE = 12.0         # Start evasive maneuvers
EMERGENCY_DISTANCE = 4.0     # Immediate panic stop
TARGET_ALTITUDE = -10.0      # Locked altitude
MAX_VX = 3.0                 # Cruise speed
MAX_VY = 2.0                 # Strafe speed
MAX_VZ = 1.0                 # Climb speed
YAW_RATE_MAX = 30.0          # Rotation speed

# PID Controller Gains
ALT_KP = 1.5
ALT_KI = 0.001
ALT_KD = 0.8

# Observation
EMBEDDING_SIZE = 128

# --- 1. ENVIRONMENT CLASS ---
class AirSimMLPEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(self):
        super(AirSimMLPEnv, self).__init__()
        print(f"Connecting to AirSim at {WINDOWS_HOST_IP}...")
        self.client = airsim.MultirotorClient(ip=WINDOWS_HOST_IP)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        self.observation_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(EMBEDDING_SIZE,), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def _get_feature_vector(self):
        return np.zeros(EMBEDDING_SIZE, dtype=np.float32)

    def step(self, action):
        return self._get_feature_vector(), 0.0, False, False, {}

    def reset(self, seed=None, options=None):
        return self._get_feature_vector(), {}

    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)

# --- 2. FEATURE EXTRACTOR ---
class MLPPassThroughFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = EMBEDDING_SIZE):
        super().__init__(observation_space, features_dim)
        self.pass_through = torch.nn.Sequential(torch.nn.Linear(EMBEDDING_SIZE, features_dim))
        self._features_dim = features_dim
    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return observations

policy_kwargs = {
    "features_extractor_class": MLPPassThroughFeatureExtractor,
    "features_extractor_kwargs": dict(features_dim=EMBEDDING_SIZE),
    "net_arch": dict(pi=[64, 64], vf=[64, 64]) 
}

# --- 3. PID CONTROLLER ---
class PIDController:
    def __init__(self, kp, ki, kd, min_val, max_val):
        self.kp = kp; self.ki = ki; self.kd = kd
        self.min_val = min_val; self.max_val = max_val
        self.integral = 0.0; self.prev_error = 0.0
        self.last_time = time.time()

    def compute(self, target, current):
        now = time.time()
        dt = now - self.last_time if (now - self.last_time) > 0 else 0.1
        error = target - current
        self.integral += error * dt
        derivative = (error - self.prev_error) / dt
        output = (self.kp * error) + (self.ki * self.integral) + (self.kd * derivative)
        self.prev_error = error; self.last_time = now
        return max(self.min_val, min(self.max_val, output))

# --- 4. INTELLIGENT VISION SYSTEM ---
def get_safe_direction(client):
    """Reads depth camera to find obstacles and best escape route."""
    try:
        responses = client.simGetImages([airsim.ImageRequest("0", airsim.ImageType.DepthPlanar, True, False)])
    except: return 100.0, (0.0, 0.0), "NONE"

    if not responses: return 100.0, (0.0, 0.0), "NONE"

    img = responses[0]
    raw = np.array(img.image_data_float, dtype=np.float32)
    if raw.size == 0: return 100.0, (0.0, 0.0), "NONE"

    # Reshape
    w = img.width
    h = img.height
    try:
        depth_img = raw.reshape((h, w))
    except:
        depth_img = raw.reshape((w, h))
    
    # Filter valid range
    depth_img = np.clip(depth_img, 0.0, 50.0)
    valid_mask = (depth_img > 0.5) 
    
    if not np.any(valid_mask): return 100.0, (0.0, 0.0), "NONE"

    min_dist = float(np.min(depth_img[valid_mask]))

    # --- FIND OPEN SPACE (Fixed Variable Names) ---
    h_mid = h // 2
    w_mid = w // 2
    scale = 2.5 
    
    # Use consistent variable names (h_mid, w_mid)
    score_up = np.mean(depth_img[0:h_mid, :]) if np.any(depth_img[0:h_mid, :]) else 0
    score_down = np.mean(depth_img[h_mid:h, :]) if np.any(depth_img[h_mid:h, :]) else 0
    score_left = np.mean(depth_img[:, 0:w_mid]) if np.any(depth_img[:, 0:w_mid]) else 0
    score_right = np.mean(depth_img[:, w_mid:w]) if np.any(depth_img[:, w_mid:w]) else 0
    
    scores = {"UP": score_up, "DOWN": score_down, "LEFT": score_left, "RIGHT": score_right}
    best = max(scores, key=scores.get)
    
    if best == "UP":    avoidance = (0.0, -scale)
    elif best == "DOWN":  avoidance = (0.0, scale)
    elif best == "LEFT":  avoidance = (-scale, 0.0)
    elif best == "RIGHT": avoidance = (scale, 0.0)
    else: avoidance = (0.0, 0.0)

    return min_dist, avoidance, best

# --- 5. MAIN AUTONOMOUS LOOP ---
def evaluate():
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model not found at {MODEL_PATH}")
        return

    print(f"Loading trained model...")
    env_instance = AirSimMLPEnv()
    env = DummyVecEnv([lambda: env_instance])
    client = env_instance.client

    model = PPO.load(MODEL_PATH, env=env, custom_objects=None, device="auto")
    alt_pid = PIDController(ALT_KP, ALT_KI, ALT_KD, -MAX_VZ, MAX_VZ)

    print("\n--- 100% AUTONOMOUS MODE ENGAGED ---")
    
    # Takeoff Sequence
    client.takeoffAsync().join()
    client.moveToZAsync(TARGET_ALTITUDE, 3).join()
    obs = env.reset()
    if isinstance(obs, tuple): obs = obs[0]

    try:
        while True:
            # 1. Get AI "Intent"
            action_raw, _ = model.predict(obs, deterministic=True)
            action = np.asarray(action_raw).flatten()
            
            # 2. Get Vision Data
            min_dist, avoidance_vel, direction = get_safe_direction(client)
            avoid_vy, avoid_vz = avoidance_vel
            
            # 3. Altitude Control
            z_curr = client.getMultirotorState().kinematics_estimated.position.z_val
            pid_vz = alt_pid.compute(TARGET_ALTITUDE, z_curr)

            # 4. Fusion Control Logic
            
            if min_dist < EMERGENCY_DISTANCE:
                print(f"🚨 EMERGENCY: {min_dist:.1f}m! Backing up & Evading {direction}!")
                vx = -2.0       
                vy = float(escape_vy) if 'escape_vy' in locals() else avoid_vy * 1.5
                vz = float(avoid_vz) + pid_vz
                yaw_rate = 0.0 
                
            elif min_dist < SAFE_DISTANCE:
                print(f"⚠️ CAUTION: {min_dist:.1f}m. Steering {direction}.")
                vx = 1.0
                vy = float(avoid_vy)
                vz = pid_vz + float(avoid_vz)
                yaw_rate = 0.0
                
            else:
                vx = MAX_VX
                ai_turn = float(action[2]) 
                vy = float(action[0]) * 1.5 
                vz = pid_vz
                yaw_rate = ai_turn * YAW_RATE_MAX

            # 5. Send Command (Cast everything to float for AirSim compatibility)
            client.moveByVelocityAsync(
                float(vx), float(vy), float(vz), 0.1, 
                airsim.DrivetrainType.ForwardOnly, 
                airsim.YawMode(is_rate=True, yaw_or_rate=float(yaw_rate))
            ).join()

            # 6. Step Env
            obs, rewards, dones, infos = env.step(action_raw)
            
            if isinstance(dones, (list, np.ndarray)) and dones[0]:
                obs = env.reset()
                alt_pid.reset()
                
    except KeyboardInterrupt:
        print("\nLanding...")
        client.landAsync().join()
        client.armDisarm(False)
        client.enableApiControl(False)

if __name__ == "__main__":
    evaluate()
