import math
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

device = torch.device(
    "cuda" if torch.cuda.is_available() else
    "mps" if torch.backends.mps.is_available() else
    "cpu"
)

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, n_observations, n_heading, n_speed, hidden_size=128):
        super().__init__()
        self.layer1 = nn.Linear(n_observations, hidden_size)
        self.layer2 = nn.Linear(hidden_size, hidden_size)

        self.heading_layer = nn.Linear(hidden_size, n_heading)
        self.speed_layer = nn.Linear(hidden_size, n_speed)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        heading = self.heading_layer(x)
        speed = self.speed_layer(x)
        return heading, speed

def discrete_to_continuous(heading_action, speed_action):
    # basically we are locating where the discrete action is
    # in the continuum of continuous actions using binning
    # imagine a histogram, except were doing it backwards
    heading_low = original_action_space.low[0]
    heading_high = original_action_space.high[0]
    speed_low = original_action_space.low[1]
    speed_high = original_action_space.high[1]

    heading_idx = heading_action.item()
    speed_idx = speed_action.item()

    heading = heading_low + (heading_high - heading_low) * (heading_idx / (n_bins_per_dim - 1))
    speed = speed_low + (speed_high - speed_low) * (speed_idx / (n_bins_per_dim - 1))

    return np.array([heading, speed])

# for some reason bluesky emit's its state a dict
# we need to turn it into a tensor
def state_dict_to_tensor(state_dict):
    values = []
    for key in sorted(state_dict.keys()):
        val = state_dict[key]
        if isinstance(val, np.ndarray):
            values.append(val.flatten())
        else:
            values.append(np.array([val]))
    flat_state = np.concatenate(values)
    return torch.tensor(flat_state, dtype=torch.float32, device=device).unsqueeze(0)

