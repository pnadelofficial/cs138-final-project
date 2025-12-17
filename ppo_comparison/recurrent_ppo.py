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

    # mapping from strings to policy class
    policy_aliases: ClassVar[dict[str, type[BasePolicy]]] = {
        "MlpLstmPolicy": MlpLstmPolicy,
        "CnnLstmPolicy": CnnLstmPolicy,
        "MultiInputLstmPolicy": MultiInputLstmPolicy,
    }

    def __init__(
        self,
        policy, 
        env, 
        learning_rate=3e-4, 
        n_steps=128,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        normalize_advantage=True,
        ent_coef=0.0,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=False,
        sde_sample_freq=-1,
        stats_window_size=100,
        tensorboard_log="./runs/",
        policy_kwargs=None,
        verbose=0,
        seed=None,
        device="auto",
        _init_setup_model=True):
        """
        Initializes the RecurrentPPO algorithm. 
        """

        # hyperparameters/kwargs for base on policy algorithm 
        supported_action_spaces = (
            spaces.Box,
            spaces.Discrete,
            spaces.MultiDiscrete,
            spaces.MultiBinary,
        ),
        super().__init__(
            policy, env, learning_rate=learning_rate, n_steps=n_steps, gamma=gamma, gae_lambda=gae_lambda, ent_coef=ent_coef, vf_coef=vf_coef, max_grad_norm=max_grad_norm, use_sde=use_sde, sde_sample_freq=sde_sample_freq,stats_window_size=stats_window_size, tensorboard_log=tensorboard_log, policy_kwargs=policy_kwargs, verbose=verbose, seed=seed, device=device, _init_setup_model=False, supported_action_spaces=supported_action_spaces
        )

        # recurrent ppo specific hyperparameters/kwargs
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.clip_range = clip_range
        self.normalize_advantage = normalize_advantage

        self._last_lstm_states = None

        # checking if the model has been initialized
        if _init_setup_model:
            self._setup_model()
        
    def _setup_model(self) -> None:
        """
        Sets up the model by:
            (1) finding and creating the correct buffer (data structure for holding past states)
            (2) initializing the actor and critic agents and putting them on device
            (3) initializing the last lstm states with zeros
            (4) initializing value clipping
        """
        self._setup_lr_schedule() # comes from the super class - checks for a learning rate scheduler, otherwise defaults to linear
        self.set_random_seed(self.seed) # comes from the super class - sets random seed for everything

        # selects the correct buffer for storing past states for the LSTM
        if isinstance(self.observation_space, spaces.Dict):
            buffer_cls = RecurrentDictRolloutBuffer # this is really critical as all of the BlueSky envs use spaces.Dict to store their state-action data
        else:
            buffer_cls = RecurrentRolloutBuffer # this will go unused but i added it for completeness

        # from the super class - creating policy network
        self.policy = self.policy_class(
            self.observation_space,
            self.action_space,
            self.lr_schedule,
            use_sde=self.use_sde,
            **self.policy_kwargs,
        )
        self.policy = self.policy.to(self.device) # and moves it on to the device

        # assuming that the LSTM architecture is being used for both the actor and critic
        lstm = self.policy.lstm_actor

        # creating holder for the last lstm states
        self._last_lstm_states = RNNStates(
            (
                torch.zeros((lstm.num_layers, self.n_envs, lstm.hidden_size), device=self.device), # the last lstm state is of shape the number of layers (we vary this from 2 to 4) x the number of envs (will be 1, but needed to shape consistences) x the hidden size of the network (usually 256)
                torch.zeros((lstm.num_layers, self.n_envs, lstm.hidden_size), device=self.device)
            ),
            (
                torch.zeros((lstm.num_layers, self.n_envs, lstm.hidden_size), device=self.device),
                torch.zeros((lstm.num_layers, self.n_envs, lstm.hidden_size), device=self.device)
            )
        )

        # initialize the buffer
        self.rollout_buffer = buffer_cls(
            self.n_steps,
            self.observation_space,
            self.action_space,
            (self.n_steps, lstm.num_layers, self.n_envs, lstm.hidden_size), # size of the buffer
            self.device,
            gamma=self.gamma,
            gae_lambda=self.gae_lambda,
            n_envs=self.n_envs
        )

        # initialize the schedules for clipping policy values
        self.clip_range = FloatSchedule(self.clip_range) # float schedule varies the clipping value through the number of epochs
        
    def collect_rollouts(self, env, callback, rollout_buffer, n_rollout_steps) -> bool:
        """
        Will collect states, action and observations from the poilcy and fill ad RolloutBuffer. 
        """
        self.policy.set_training_mode(False) # switch off of training mode as all of this will happen AFTER a training step
        n_steps = 0
        rollout_buffer.reset()

        callback.on_rollout_start()
        lstm_states = deepcopy(self._last_lstm_states) # avoiding inplace changes

        # while the number of steps is less than the max number of steps for the rollout we are going to 
        ## (1) convert observation tensors into observation dicts so that we can pass them to the env
        ## (2) clip the action to the action space
        ## (3) pass the new action into the env
        ## (4) observe a new observation and a reward
        ## (5) wrap up environment if it is in a terminal state
        ## (6) add data to the rollout buffer
        while n_steps < n_rollout_steps:  
            # (1) converting to dicts
            with torch.no_grad():
                obs_tensor = obs_as_tensor(self._last_obs, self.device)
                episode_starts = torch.tensor(self._last_episode_starts, dtype=th.float32, device=self.device)
                actions, values, log_probs, lstm_states = self.policy(obs_tensor, lstm_states, episode_starts) # using the policy to get new observations
            actions = actions.cpu().numpy() # cast as numpy so that we can clip it and then pass it into the env

            # (2) clipping
            clipped_actions = actions
            if isinstance(self.action_space, spaces.Box):
                clipped_actions = np.clip(actions, self.action_space.low, self.action_space.high)
            
            # (3) passing new action to the env/ (4) getting new observation
            new_obs, rewards, dones, infos = env.step(clipped_actions)

            ## managing logging and the callback
            self.num_timesteps += env.num_envs
            callback.update_locals(locals())
            if not callback.on_step():
                return False    
            self._update_info_buffer(infos, dones)
            n_steps += 1

            # (5) clean up if we reached the terminal state
            for idx, done_ in enumerate(dones):
                if (
                    done_ 
                    and infos[idx].get("terminal_observation") is not None # if we are in the terminal state
                    and infos[idx].get("TimeLimit.truncated", False) # if we have bee truncated by the time limit (200 steps in BlueSky Gym)
                ):
                    terminal_obs = self.policy.obs_to_tensor(infos[idx]["terminal_observation"])[0] # last observation
                    with torch.no_grad():
                        terminal_lstm_state = (
                            lstm_states.vf[0][:, idx : idx + 1, :].contiguous(),
                            lstm_states.vf[1][:, idx : idx + 1, :].contiguous(),
                        )
                        episode_starts = th.tensor([False], dtype=th.float32, device=self.device)
                        terminal_value = self.policy.predict_values(terminal_obs, terminal_lstm_state, episode_starts)[0]
                    rewards[idx] += self.gamma * terminal_value
            
            # (6) add data to buffer
            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                lstm_states=self._last_lstm_states
            )

            self._last_obs = new_obs
            self._last_episode_starts = dones
            self._last_lstm_states = lstm_states
        
        # outside of the while loop, we then need to compute a value from the policy of what the last time step is
        with th.no_grad():
            episode_starts = th.tensor(dones, dtype=th.float32, device=self.device)
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device), lstm_states.vf, episode_starts)

        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)
        callback.on_rollout_end()

        return True

    def train(self) -> None:
        """
        The training algorithm that will optimize the LSTMs based on the rollout buffer. This method supercedes a train method in the super class. This is how the learn method functions.
        """
        self.policy.set_training_mode(True) # put the model in train mode
        self._update_learning_rate(self.policy.optimizer) # need to update the learning rate if we are using a scheduler
        clip_range = self.clip_range(self._current_progress_remaining) # determine what clip range we are in

        # init values for training
        entropy_losses = []
        pg_losses = []
        value_losses = []
        clip_fractions = []
        continue_training = True

        for epoch in range(self.n_epoch):
            approx_kl_divs = []
            for rollout_data in self.rollout_buffer.get(self.batch_size): # getting a batch size worth of rollout data
                actions = rollout_data.actions # actions for each
                mask = rollout_data.mask > 1e-8 # converting float to bool by thresholding at nearly 0

                # pass the observations, lstm_states and episode_starts to the policy
                values, log_prob, entropy = self.policy.evaluate_actions(
                    rollout_data.observations,
                    actions,
                    rollout_data.lstm_states,
                    rollout_data.episode_starts,
                )

                values = values.flatten()
                advantages = rollout_data.advantages
                advantages = (advantages - advantages[mask].mean()) / (advantages[mask].std() + 1e-8) # normalizing the advantages so they are in the same range

                # ratio between old and new policy, should be one at the first iteration
                ratio = torch.exp(log_prob - rollout_data.old_log_prob)

                # clip loss from each model
                policy_1_loss = advantages * ratio
                policy_2_loss = advantages * torch.clamp(ratio, 1 - clip_range, 1 + clip_range)
                policy_loss = -torch.mean(torch.min(policy_loss_1, policy_loss_2)[mask])

                # log to lists
                pg_losses.append(policy_loss.item())
                clip_fraction = torch.mean((torch.abs(ratio - 1) > clip_range).float()[mask]).item()
                clip_fractions.append(clip_fraction)

                # clipping between the old and new value then calculating the loss
                values_pred = rollout_data.old_values + torch.clamp(values - rollout_data.old_values, 1, -1)
                value_loss = torch.mean(((rollout_data.returns - values_pred) ** 2)[mask]) # calculating the loss for the values
                value_losses.append(value_loss.item()) # logging

                # calculate entropy loss
                if entroy is None:
                    # approximating entropy if there is none
                    entropy_loss = -torch.mean(-log_prob[mask])
                else:
                    entropy_loss = -torch.mean(entropy[mask])
                entropy_losses.append(entropy_loss.item())

                loss = policy_loss + self.ent_coef * entropy_loss + self.vf_coef * value_loss # final loss function

                # Optimization step
                self.policy.optimizer.zero_grad()
                loss.backward()
                # Clip grad norm
                th.nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
                self.policy.optimizer.step()

            if not continue_training:
                break

        self._n_updates += self.n_epochs
        explained_var = explained_variance(self.rollout_buffer.values.flatten(), self.rollout_buffer.returns.flatten())

        # further logging/tensorboard lgging
        self.logger.record("train/entropy_loss", np.mean(entropy_losses))
        self.logger.record("train/policy_gradient_loss", np.mean(pg_losses))
        self.logger.record("train/value_loss", np.mean(value_losses))
        self.logger.record("train/approx_kl", np.mean(approx_kl_divs))
        self.logger.record("train/clip_fraction", np.mean(clip_fractions))
        self.logger.record("train/loss", loss.item())
        self.logger.record("train/explained_variance", explained_var)
        if hasattr(self.policy, "log_std"):
            self.logger.record("train/std", th.exp(self.policy.log_std).mean().item())

        self.logger.record("train/n_updates", self._n_updates, exclude="tensorboard")
        self.logger.record("train/clip_range", clip_range)

    # learn method that called the super class
    def learn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 1,
        tb_log_name: str = "RecurrentPPO",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ):
        return super().learn(
            total_timesteps=total_timesteps,
            callback=callback,
            log_interval=log_interval,
            tb_log_name=tb_log_name,
            reset_num_timesteps=reset_num_timesteps,
            progress_bar=progress_bar,
        )

    def _excluded_save_params(self) -> list[str]:
        return super()._excluded_save_params() + ["_last_lstm_states"] 