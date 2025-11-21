import airsim
import numpy as np
import gymnasium as gym
from gymnasium import spaces
import time
from airsim import DrivetrainType, YawMode, Vector3r, ImageRequest, ImageType, ImageResponse
import cv2 

# --- CONFIGURATION ---
WINDOWS_HOST_IP = "172.24.176.1"
IMAGE_WIDTH = 64 
IMAGE_HEIGHT = 64
EMBEDDING_SIZE = 128 # The size of the feature vector the MLP will train on
Z_GOAL = -10.0 # Target altitude
MAX_DEPTH = 50.0 

class AirSimMLPEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 30}
    
    def __init__(self, max_episode_steps=500):
        super(AirSimMLPEnv, self).__init__()
        
        self.client = airsim.MultirotorClient(ip=WINDOWS_HOST_IP)
        self.client.confirmConnection()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        # The Observation Space is now a small vector (MLP Policy)
        self.observation_space = spaces.Box(low=-1.0, high=1.0, 
                                            shape=(EMBEDDING_SIZE,), 
                                            dtype=np.float32) 
        
        # Action Space remains the same (Continuous Thrust/Torque)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

        self.max_episode_steps = max_episode_steps
        self.step_count = 0
        self.target_z = Z_GOAL
        self.episode_count = 0 

    def _get_feature_vector(self):
        # This section simulates the output of the trained Autoencoder (a fast vector)
        
        # Request a depth image to maintain the sim's step rate
        responses = self.client.simGetImages([ImageRequest("0", ImageType.DepthPlanar, True, False)])
        
        drone_state = self.client.getMultirotorState()
        pos = drone_state.kinematics_estimated.position
        
        # Create a simple, synthetic 128-feature vector based on position and velocity
        feature_vector = np.zeros(EMBEDDING_SIZE, dtype=np.float32)
        feature_vector[0:3] = [pos.x_val / 100, pos.y_val / 100, pos.z_val / 100]
        
        return feature_vector

    def _calculate_reward(self, drone_state, collision):
        # We reuse the proven reward logic
        reward = 0
        if collision:
            return -1000.0

        vx = drone_state.kinematics_estimated.linear_velocity.x_val
        reward += 1.0 * vx 
        z = drone_state.kinematics_estimated.position.z_val
        altitude_deviation = abs(z - self.target_z) 
        reward -= 0.2 * altitude_deviation
        reward -= 0.1 
        return reward

    def step(self, action):
        self.step_count += 1
        
        vx = action[1] * 5.0 
        vy = action[0] * 5.0 
        yaw_rate = action[2] * 45.0
        target_z_vel = action[3] * 2.0 
        
        self.client.moveByVelocityAsync(vx, vy, target_z_vel, duration=1.0, 
                                        drivetrain=DrivetrainType.ForwardOnly, 
                                        yaw_mode=YawMode(is_rate=True, yaw_or_rate=yaw_rate)).join()

        drone_state = self.client.getMultirotorState()
        collision = self.client.simGetCollisionInfo().has_collided

        reward = self._calculate_reward(drone_state, collision)
        done = collision or (self.step_count >= self.max_episode_steps) or (abs(drone_state.kinematics_estimated.position.z_val) > 20)

        if done:
            print(f"--- EPISODE {self.episode_count} ENDED | Reward: {reward:.2f} | Collided: {collision} ---")

        observation = self._get_feature_vector()
        
        return observation, reward, done, False, {}

    def reset(self, seed=None, options=None):
        self.client.reset()
        self.client.enableApiControl(True)
        self.client.armDisarm(True)
        
        self.client.takeoffAsync(timeout_sec=8).join()
        self.client.moveToZAsync(self.target_z, 5).join() 
        
        self.step_count = 0
        self.episode_count += 1
        print(f"\n--- EPISODE {self.episode_count} STARTED ---")
        observation = self._get_feature_vector()
        return observation, {}

    def render(self):
        pass
        
    def close(self):
        self.client.armDisarm(False)
        self.client.enableApiControl(False)
        self.client.reset()
