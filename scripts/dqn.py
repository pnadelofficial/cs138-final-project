import math
import time
import random
import matplotlib
import matplotlib.pyplot as plt
from collections import namedtuple, deque
from itertools import count
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import gymnasium as gym
from gymnasium.wrappers import TimeLimit
import bluesky_gym

from utils import *

###################
# Hyperparameters #
###################
RUN_ID = 1
BATCH_SIZE = 256
GAMMA = 0.99
EPS_START = 0.9
EPS_END = 0.01
EPS_DECAY = 5000
TAU = 0.005
LR = 1e-4
HIDDEN_SIZE = 256
N_BINS_PER_DIM = 15
NUM_EPISODES = 600

##################
# Initialization #
##################
device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

bluesky_gym.register_envs()
env = gym.make('SectorCREnv-v0', render_mode='human')

original_action_space = env.action_space
action_dim = original_action_space.shape[0]
n_discrete_actions = N_BINS_PER_DIM ** action_dim

state, info = env.reset()
state = state_dict_to_tensor(state)
n_observations = state.shape[-1]

policy_net = DQN(n_observations, N_BINS_PER_DIM, N_BINS_PER_DIM ).to(device)
target_net = DQN(n_observations, N_BINS_PER_DIM, N_BINS_PER_DIM).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
memory = ReplayMemory(10000)

############
# Training #
############
steps_done = 0
episode_durations = []
episode_rewards = []
print("Beginning training")
start = time.time.now()

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    total_reward = 0
    state, info = env.reset()
    state = state_dict_to_tensor(state)
    for t in count():
        discrete_action = select_action(state)
        heading_action = discrete_action[0].squeeze(0)
        speed_action = discrete_action[1].squeeze(0)
        continuous_action = discrete_to_continuous(heading_action, speed_action)
        observation, reward, terminated, truncated, _ = env.step(continuous_action)
        total_reward += reward
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = state_dict_to_tensor(observation)

        # Store the transition in memory
        memory.push(state, (heading_action, speed_action), next_state, reward) # storing the DISCRETE action not the continuous for training

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            episode_durations.append(t + 1)
            episode_rewards.append(reward)
            break
end = time.time.now()
dur = end - curr
print(f'Training complete with {dur/60} minutes.')

##########
# Saving #
##########
base_path = f"./{RUN_ID}"
os.makedirs(base_path, exists_ok=True)

reward_avg = np.convolve(np.array([r.item() for r in episode_rewards]), np.ones(20)/20, mode='valid')
plt.title("Average Reward")
plt.xlabel("Episode")
plt.ylabel("20-Episode Running Average")
plt.plot(reward_avg)
plt.savefig(f"{base_path}/average_reward.png")

plt.clf()
plt.title("Durations")
plt.xlabel('Episode')
plt.ylabel('Duration')
durations_avg = np.convolve(np.array([r.item() for r in episode_rewards]), np.ones(20)/20, mode='valid')
plt.plot(episode_durations)
plt.plot(durations_avg)
plt.savefig(f"{base_path}/average_duration.png")

torch.save(policy_net.state_dict(), f"{base_path}/policy_net")