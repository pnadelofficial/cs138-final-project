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

########################
# Helpers not in utils #
########################
# need to change this too to support speed and heading
def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPS_END + (EPS_START - EPS_END) * \
        math.exp(-1. * steps_done / EPS_DECAY)
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            # t.max(1) will return the largest column value of each row.
            # second column on max result is index of where max element was
            # found, so we pick action with the larger expected reward.
            # return policy_net(state).max(1).indices.view(1, 1)
            heading, speed = policy_net(state)
            heading_action = heading.max(1).indices.view(1, 1)
            speed_action = speed.max(1).indices.view(1, 1)
            return heading_action, speed_action
    else:
        # Explore: sample random DISCRETE actions
        heading_action = torch.tensor(
            [[random.randrange(n_bins_per_dim)]],
            device=device,
            dtype=torch.long
        )
        speed_action = torch.tensor(
            [[random.randrange(n_bins_per_dim)]],
            device=device,
            dtype=torch.long
        )

    return heading_action, speed_action

def optimize_model():
    if len(memory) < BATCH_SIZE:
        return
    transitions = memory.sample(BATCH_SIZE)
    # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
    # detailed explanation). This converts batch-array of Transitions
    # to Transition of batch-arrays.
    batch = Transition(*zip(*transitions))

    # Compute a mask of non-final states and concatenate the batch elements
    # (a final state would've been the one after which simulation ended)
    non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                          batch.next_state)), device=device, dtype=torch.bool)
    non_final_next_states = torch.cat([s for s in batch.next_state
                                                if s is not None])
    state_batch = torch.cat(batch.state)
    reward_batch = torch.cat(batch.reward)

    # need to update this as well to have separate losses for heading and speed actions
    heading_action_batch = torch.cat([a[0] for a in batch.action])
    heading_action_batch = heading_action_batch.unsqueeze(1)
    speed_action_batch = torch.cat([a[1] for a in batch.action])
    speed_action_batch = speed_action_batch.unsqueeze(1)

    # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
    # columns of actions taken. These are the actions which would've been taken
    # for each batch state according to policy_net
    # get q values from the policy
    heading, speed = policy_net(state_batch)
    # get q values for action taken
    state_heading_values = heading.gather(1, heading_action_batch)
    state_speed_values = speed.gather(1, speed_action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    # init next state values
    next_heading_values = torch.zeros(BATCH_SIZE, device=device)
    next_speed_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        heading_q_next, speed_q_next = target_net(non_final_next_states)
        next_heading_values[non_final_mask] = heading_q_next.max(1).values
        next_speed_values[non_final_mask] = speed_q_next.max(1).values

    # Compute the expected Q values
    expected_heading_values = (next_heading_values * GAMMA) + reward_batch
    expected_speed_values = (next_speed_values * GAMMA) + reward_batch

    # Compute Huber loss FOR BOTH and combine
    criterion = nn.SmoothL1Loss()
    heading_loss = criterion(state_heading_values, expected_heading_values.unsqueeze(1))
    speed_loss = criterion(state_speed_values, expected_speed_values.unsqueeze(1))
    loss = heading_loss + speed_loss

    # Optimize the model
    optimizer.zero_grad()
    loss.backward()
    # In-place gradient clipping
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()

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
start = time.time()

for i_episode in range(NUM_EPISODES):
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
end = time.time()
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