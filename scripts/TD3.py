import gymnasium as gym
import bluesky_gym
from stable_baselines3 import TD3

bluesky_gym.register_envs()

timesteps = 1e5  # reduced for faster testing on my CPU
env_name = 'SectorCREnv-v0' 

# TD3 Training
print("Training TD3 on SectorCR")
env = gym.make(env_name, render_mode=None)
model = TD3(
    "MultiInputPolicy", 
    env, 
    verbose=1, 
    learning_rate=3e-4, 
    tensorboard_log="./tensorboard_logs/"
)

model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="TD3_SectorCR")
model.save(f"td3_{env_name.lower()}")
env.close()

