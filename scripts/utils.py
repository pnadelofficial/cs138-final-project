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