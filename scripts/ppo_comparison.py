import gymnasium as gym
import bluesky_gym
from stable_baselines3 import PPO
from sb3_contrib import RecurrentPPO

bluesky_gym.register_envs()
timesteps = 6e5 # may need to reduce for faster testing
env_name = 'DescentEnv-v0' # can change to other environments

## standard ppo
print("### STANDARD PPO ###")
env = gym.make(env_name)
model = PPO.load("standard_ppo_descent")  # ("MultiInputPolicy", env, verbose=1, learning_rate=3e-4, tensorboard_log="./tensorboard_logs/")
model.set_env(env)
model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="PPO_2")
model.save(f"standard_ppo_{env_name.lower()}")
env.close()

## recurrent ppo w 2 layers
print("### RECURRENT PPO WITH 2 LAYERS ###")
env = gym.make(env_name)

policy_kwargs = dict(
    lstm_hidden_size=256,  
    n_lstm_layers=2, 
    enable_critic_lstm=True,  
)

model = RecurrentPPO("MultiInputLstmPolicy", env, verbose=1, learning_rate=5e-4, tensorboard_log="./tensorboard_logs/", policy_kwargs=policy_kwargs, n_epochs=17, n_steps=2048, ent_coef=0.02, batch_size=512)
# for loading a pre-trained model, uncomment below and comment above
# model = RecurrentPPO.load("recurrent_ppo_2layer_descent")
model.set_env(env)
model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="RecurrentPPO_1")
model.save(f"recurrent_ppo_2layer_{env_name.lower()}")
env.close() 

## recurrent ppo w 4 layers
print("### RECURRENT PPO WITH 4 LAYERS")
env = gym.make(env_name)
policy_kwargs = dict(
    lstm_hidden_size=256,  
    n_lstm_layers=4, 
    enable_critic_lstm=True,  
)

model = RecurrentPPO("MultiInputLstmPolicy", env, verbose=1, learning_rate=5e-4, tensorboard_log="./tensorboard_logs/", policy_kwargs=policy_kwargs, n_epochs=17, n_steps=2048, ent_coef=0.02, batch_size=512)
# for loading a pre-trained model, uncomment below and comment above
# model = RecurrentPPO.load("recurrent_ppo_4layer_descent")
model.set_env(env)
model.learn(total_timesteps=timesteps, reset_num_timesteps=False, tb_log_name="RecurrentPPO_2")
model.save(f"recurrent_ppo_4layer_{env_name.lower()}")
env.close()
