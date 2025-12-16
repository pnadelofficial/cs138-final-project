# inspired by https://sb3-contrib.readthedocs.io/en/master/modules/ppo_recurrent.html
# decided to reimplement this algorithm with thorough commenting
# using the same framework with stable_baselines for ease

from copy import deepcopy
from typing import Any, ClassVar, TypeVar

import numpy as np
import torch

from gymnasium import spaces

from stable_baselines3.common.buffers import RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.on_policy_algorithm import OnPolicyAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.type_aliases import GymEnv, MaybeCallback, Schedule
from stable_baselines3.common.utils import FloatSchedule, explained_variance, obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from sb3_contrib.common.recurrent.buffers import RecurrentDictRolloutBuffer, RecurrentRolloutBuffer
from sb3_contrib.common.recurrent.policies import RecurrentActorCriticPolicy
from sb3_contrib.common.recurrent.type_aliases import RNNStates
from sb3_contrib.ppo_recurrent.policies import CnnLstmPolicy, MlpLstmPolicy, MultiInputLstmPolicy

class RecurrentPPO(OnPolicyAlgorithm):
    def __init__(policy, env, learning_rate, n_steps, batch_size, n_epochs, gamma, gae_lambda, clip_range, clip_range_vf, normalize_advantage, ent_coef, max_grad_norm, use_sde, sde_sample_freq, target_kl, stats_window_size, tensorboard_log, policy_kwargs, verbose, seed, device):
        