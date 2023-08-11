# Copyright 2023 OmniSafeAI Team. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================


from __future__ import annotations

import torch
import torch
from safepo.utils.util import get_shape_from_act_space, get_shape_from_obs_space


class VectorizedOnPolicyBuffer:
    """
    A buffer for storing vectorized on-policy data for reinforcement learning.

    Args:
        obs_space (gymnasium.Space): The observation space.
        act_space (gymnasium.Space): The action space.
        size (int): The maximum size of the buffer.
        gamma (float, optional): The discount factor for rewards. Defaults to 0.99.
        lam (float, optional): The lambda parameter for GAE computation. Defaults to 0.95.
        lam_c (float, optional): The lambda parameter for cost GAE computation. Defaults to 0.95.
        standardized_adv_r (bool, optional): Whether to standardize advantage rewards. Defaults to True.
        standardized_adv_c (bool, optional): Whether to standardize advantage costs. Defaults to True.
        device (torch.device, optional): The device to store tensors on. Defaults to "cpu".
        num_envs (int, optional): The number of parallel environments. Defaults to 1.
    """
    def __init__(
        self,
        obs_space,
        act_space,
        size: int,
        gamma: float=0.99,
        lam: float=0.95,
        lam_c: float=0.95,
        standardized_adv_r: bool = True,
        standardized_adv_c: bool = True,
        device: torch.device = "cpu",
        num_envs: int = 1,
    ) -> None:
        self.buffers: list[dict[str, torch.tensor]] = [
            {
                "obs": torch.zeros(
                    (size, *obs_space.shape), dtype=torch.float32, device=device
                ),
                "act": torch.zeros(
                    (size, *act_space.shape), dtype=torch.float32, device=device
                ),
                "reward": torch.zeros(size, dtype=torch.float32, device=device),
                "cost": torch.zeros(size, dtype=torch.float32, device=device),
                "done": torch.zeros(size, dtype=torch.float32, device=device),
                "value_r": torch.zeros(size, dtype=torch.float32, device=device),
                "value_c": torch.zeros(size, dtype=torch.float32, device=device),
                "adv_r": torch.zeros(size, dtype=torch.float32, device=device),
                "adv_c": torch.zeros(size, dtype=torch.float32, device=device),
                "target_value_r": torch.zeros(size, dtype=torch.float32, device=device),
                "target_value_c": torch.zeros(size, dtype=torch.float32, device=device),
                "log_prob": torch.zeros(size, dtype=torch.float32, device=device),
            }
            for _ in range(num_envs)
        ]
        self._gamma = gamma
        self._lam = lam
        self._lam_c = lam_c
        self._standardized_adv_r = standardized_adv_r
        self._standardized_adv_c = standardized_adv_c
        self.ptr_list = [0] * num_envs
        self.path_start_idx_list = [0] * num_envs
        self._device = device
        self.num_envs = num_envs

    def store(self, **data: torch.Tensor) -> None:
        """
        Store vectorized data into the buffer.

        Args:
            **data: Keyword arguments specifying data tensors to be stored.
        """
        for i, buffer in enumerate(self.buffers):
            assert self.ptr_list[i] < buffer["obs"].shape[0], "Buffer overflow"
            for key, value in data.items():
                buffer[key][self.ptr_list[i]] = value[i]
            self.ptr_list[i] += 1

    def finish_path(
        self,
        last_value_r: torch.Tensor | None = None,
        last_value_c: torch.Tensor | None = None,
        idx: int = 0,
    ) -> None:
        """
        Finalize the trajectory path and compute advantages and value targets.

        Args:
            last_value_r (torch.Tensor, optional): The last value estimate for rewards. Defaults to None.
            last_value_c (torch.Tensor, optional): The last value estimate for costs. Defaults to None.
            idx (int, optional): Index of the environment. Defaults to 0.
        """
        if last_value_r is None:
            last_value_r = torch.zeros(1, device=self._device)
        if last_value_c is None:
            last_value_c = torch.zeros(1, device=self._device)
        path_slice = slice(self.path_start_idx_list[idx], self.ptr_list[idx])
        last_value_r = last_value_r.to(self._device)
        last_value_c = last_value_c.to(self._device)
        rewards = torch.cat([self.buffers[idx]["reward"][path_slice], last_value_r])
        costs = torch.cat([self.buffers[idx]["cost"][path_slice], last_value_c])
        values_r = torch.cat([self.buffers[idx]["value_r"][path_slice], last_value_r])
        values_c = torch.cat([self.buffers[idx]["value_c"][path_slice], last_value_c])

        adv_r, target_value_r = calculate_adv_and_value_targets(
            values_r,
            rewards,
            lam=self._lam,
            gamma=self._gamma,
        )
        adv_c, target_value_c = calculate_adv_and_value_targets(
            values_c,
            costs,
            lam=self._lam_c,
            gamma=self._gamma,
        )
        self.buffers[idx]["adv_r"][path_slice] = adv_r
        self.buffers[idx]["adv_c"][path_slice] = adv_c
        self.buffers[idx]["target_value_r"][path_slice] = target_value_r
        self.buffers[idx]["target_value_c"][path_slice] = target_value_c

        self.path_start_idx_list[idx] = self.ptr_list[idx]

    def get(self) -> dict[str, torch.Tensor]:
        """
        Retrieve collected data from the buffer.

        Returns:
            dict[str, torch.Tensor]: A dictionary containing collected data tensors.
        """
        data_pre = {k: [v] for k, v in self.buffers[0].items()}
        for buffer in self.buffers[1:]:
            for k, v in buffer.items():
                data_pre[k].append(v)
        data = {k: torch.cat(v, dim=0) for k, v in data_pre.items()}
        adv_mean = data["adv_r"].mean()
        adv_std = data["adv_r"].std()
        cadv_mean = data["adv_c"].mean()
        if self._standardized_adv_r:
            data["adv_r"] = (data["adv_r"] - adv_mean) / (adv_std + 1e-8)
        if self._standardized_adv_c:
            data["adv_c"] = data["adv_c"] - cadv_mean
        self.ptr_list = [0] * self.num_envs
        self.path_start_idx_list = [0] * self.num_envs

        return data


def discount_cumsum(vector_x: torch.Tensor, discount: float) -> torch.Tensor:
    """
    Compute the discounted cumulative sum of a tensor along its first dimension.

    This function computes the discounted cumulative sum of the input tensor `vector_x` along
    its first dimension. The discount factor `discount` is applied to compute the weighted sum
    of future values. The resulting tensor has the same shape as the input tensor.

    Args:
        vector_x (torch.Tensor): Input tensor with shape `(length, ...)`.
        discount (float): Discount factor for future values.

    Returns:
        torch.Tensor: Tensor containing the discounted cumulative sum of `vector_x`.
    """
    length = vector_x.shape[0]
    vector_x = vector_x.type(torch.float64)
    cumsum = vector_x[-1]
    for idx in reversed(range(length - 1)):
        cumsum = vector_x[idx] + discount * cumsum
        vector_x[idx] = cumsum
    return vector_x


def calculate_adv_and_value_targets(
    values: torch.Tensor,
    rewards: torch.Tensor,
    lam: float,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    # GAE formula: A_t = \sum_{k=0}^{n-1} (lam*gamma)^k delta_{t+k}
    deltas = rewards[:-1] + gamma * values[1:] - values[:-1]
    adv = discount_cumsum(deltas, gamma * lam)
    target_value = adv + values[:-1]
    return adv, target_value

def _flatten(T, N, x):
    return x.reshape(T * N, *x.shape[2:]) # del

def _cast(x):
    return x.transpose(1,0,2).reshape(-1, *x.shape[2:]) # del

class SeparatedReplayBuffer(object):
    """Buffer for storing and managing data collected during training.

    Args:
        config (dict): Configuration parameters for the replay buffer.
        obs_space: Observation space of the environment.
        share_obs_space: Shared observation space of the environment (if applicable).
        act_space: Action space of the environment.
    """
    
    def __init__(self, config, obs_space, share_obs_space, act_space):
        self.episode_length = config["episode_length"]
        self.n_rollout_threads = config["n_rollout_threads"]
        self.rnn_hidden_size = config["hidden_size"]
        self.recurrent_N = config["recurrent_N"]
        self.gamma = config["gamma"]
        self.gae_lambda = config["gae_lambda"]
        self._use_gae = config["use_gae"]
        self._use_popart = config["use_popart"]
        self._use_valuenorm = config["use_valuenorm"]
        self._use_proper_time_limits = config["use_proper_time_limits"]
        self.device = config.get("device", "cpu")
        self.algo = config["algorithm_name"]

        obs_shape = get_shape_from_obs_space(obs_space)
        share_obs_shape = get_shape_from_obs_space(share_obs_space)

        if type(obs_shape[-1]) == list:
            obs_shape = obs_shape[:1] # del

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1] # del

        self.aver_episode_costs = torch.zeros(self.episode_length + 1, self.n_rollout_threads, *obs_shape, device=self.device)

        self.share_obs = torch.zeros(self.episode_length + 1, self.n_rollout_threads, *share_obs_shape, device=self.device)
        self.obs = torch.zeros(self.episode_length + 1, self.n_rollout_threads, *obs_shape, device=self.device)

        self.rnn_states = torch.zeros(self.episode_length + 1, self.n_rollout_threads, self.recurrent_N, self.rnn_hidden_size, device=self.device)
        self.rnn_states_critic = torch.zeros_like(self.rnn_states)

        self.rnn_states = torch.zeros(self.episode_length + 1, self.n_rollout_threads, self.recurrent_N, self.rnn_hidden_size, device=self.device)
        self.rnn_states_critic = torch.zeros_like(self.rnn_states)
        self.rnn_states_cost = torch.zeros_like(self.rnn_states)

        self.value_preds = torch.zeros(self.episode_length + 1, self.n_rollout_threads, 1, device=self.device)
        self.returns = torch.zeros(self.episode_length + 1, self.n_rollout_threads, 1, device=self.device)
        
        if act_space.__class__.__name__ == 'Discrete':
            self.available_actions = torch.ones(self.episode_length + 1, self.n_rollout_threads, act_space.n, device=self.device)
        else:
            self.available_actions = None

        act_shape = get_shape_from_act_space(act_space)

        self.actions = torch.zeros(self.episode_length, self.n_rollout_threads, act_shape, device=self.device)
        self.action_log_probs = torch.zeros(self.episode_length, self.n_rollout_threads, act_shape, device=self.device)
        self.rewards = torch.zeros(self.episode_length, self.n_rollout_threads, 1, device=self.device)
        
        self.costs = torch.zeros_like(self.rewards)
        self.cost_preds = torch.zeros_like(self.value_preds)
        self.cost_returns = torch.zeros_like(self.returns)

        self.masks = torch.ones(self.episode_length + 1, self.n_rollout_threads, 1, device=self.device)
        self.bad_masks = torch.ones_like(self.masks)
        self.active_masks = torch.ones_like(self.masks)

        self.factor = torch.ones(self.episode_length, self.n_rollout_threads, 1, device=self.device)

        self.step = 0

    def update_factor(self, factor):
        self.factor.copy_(factor)

    def return_aver_insert(self, aver_episode_costs):
        # self.aver_episode_costs = aver_episode_costs.copy()
        self.aver_episode_costs = aver_episode_costs.clone()

    def insert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
               value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None, costs=None,
               cost_preds=None, rnn_states_cost=None, done_episodes_costs_aver=None, aver_episode_costs = 0):
        """
        Inserts data from a single time step into the replay buffer.

        Args:
            share_obs: Shared observations for the time step.
            obs: Observations for the time step.
            rnn_states: RNN states for the main network.
            rnn_states_critic: RNN states for the critic network.
            actions: Actions taken at the time step.
            action_log_probs: Log probabilities of the actions.
            value_preds: Value predictions at the time step.
            rewards: Rewards received at the time step.
            masks: Masks indicating whether the episode is done.
            bad_masks: Masks indicating bad episodes (optional).
            active_masks: Masks indicating active episodes (optional).
            available_actions: Available actions for discrete action spaces (optional).
            costs: Costs associated with the time step (optional).
            cost_preds: Cost predictions at the time step (optional).
            rnn_states_cost: RNN states for cost prediction (optional).
            done_episodes_costs_aver: Average costs of done episodes (optional).
            aver_episode_costs: Average episode costs (optional).

        Note:
            This method inserts data for a single time step into the replay buffer and updates the internal step counter.
        """
        self.share_obs[self.step + 1].copy_(share_obs)
        self.obs[self.step + 1].copy_(obs)
        self.rnn_states[self.step + 1].copy_(rnn_states)
        self.rnn_states_critic[self.step + 1].copy_(rnn_states_critic)
        self.actions[self.step].copy_(actions)
        self.action_log_probs[self.step].copy_(action_log_probs)
        self.value_preds[self.step].copy_(value_preds)
        self.rewards[self.step].copy_(rewards)
        self.masks[self.step + 1].copy_(masks)
        if bad_masks is not None:
            self.bad_masks[self.step + 1].copy_(bad_masks)
        if active_masks is not None:
            self.active_masks[self.step + 1].copy_(active_masks)
        if available_actions is not None:
            self.available_actions[self.step + 1].copy_(available_actions)
        if costs is not None:
            self.costs[self.step].copy_(costs)
        if cost_preds is not None:
            self.cost_preds[self.step].copy_(cost_preds)
        if rnn_states_cost is not None:
            self.rnn_states_cost[self.step + 1].copy_(rnn_states_cost)

        self.step = (self.step + 1) % self.episode_length
    
    def after_update(self):
        self.share_obs[0].copy_(self.share_obs[-1])
        self.obs[0].copy_(self.obs[-1])
        self.rnn_states[0].copy_(self.rnn_states[-1])
        self.rnn_states_critic[0].copy_(self.rnn_states_critic[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])
        self.active_masks[0].copy_(self.active_masks[-1])
        if self.available_actions is not None:
            self.available_actions[0].copy_(self.available_actions[-1])

    def chooseafter_update(self): # del
        self.rnn_states[0].copy_(self.rnn_states[-1])
        self.rnn_states_critic[0].copy_(self.rnn_states_critic[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self, next_value, value_normalizer=None):
        """
        Computes the discounted cumulative returns for each time step.

        Args:
            next_value: Estimated value of the next time step.
            value_normalizer: Normalizer for value predictions (optional).

        Note:
            This method calculates the discounted cumulative returns (GAE or regular) for each time step,
            taking into account various buffer settings and optional value normalization.

        Returns:
            None
        """
        self.value_preds[-1] = next_value
        gae = 0
        for step in reversed(range(self.rewards.shape[0])):
            delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])

    def compute_cost_returns(self, next_cost, value_normalizer=None):
        self.cost_preds[-1] = next_cost
        gae = 0
        for step in reversed(range(self.costs.shape[0])):
            delta = self.costs[step] + self.gamma * value_normalizer.denormalize(self.cost_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.cost_preds[step])
            gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
            self.cost_returns[step] = gae + value_normalizer.denormalize(self.cost_preds[step])

    def feed_forward_generator(self, advantages, num_mini_batch=None, mini_batch_size=None, cost_adv=None):        
        episode_length, n_rollout_threads = self.rewards.shape[0:2]

        batch_size = n_rollout_threads * episode_length

        if mini_batch_size is None:
            assert batch_size >= num_mini_batch, (
                "PPO requires the number of processes ({}) "
                "* number of steps ({}) = {} "
                "to be greater than or equal to the number of PPO mini batches ({})."
                "".format(n_rollout_threads, episode_length, n_rollout_threads * episode_length,
                          num_mini_batch))
            mini_batch_size = batch_size // num_mini_batch

        rand = torch.randperm(batch_size)
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        share_obs = self.share_obs[:-1].reshape(-1, *self.share_obs.shape[2:])
        obs = self.obs[:-1].reshape(-1, *self.obs.shape[2:])
        rnn_states = self.rnn_states[:-1].reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].reshape(-1, *self.rnn_states_critic.shape[2:])
        rnn_states_cost = self.rnn_states_cost[:-1].reshape(-1, *self.rnn_states_cost.shape[2:])

        actions = self.actions.reshape(-1, self.actions.shape[-1])
        if self.available_actions is not None:
            available_actions = self.available_actions[:-1].reshape(-1, self.available_actions.shape[-1])
        value_preds = self.value_preds[:-1].reshape(-1, 1)
        returns = self.returns[:-1].reshape(-1, 1)
        cost_preds = self.cost_preds[:-1].reshape(-1, 1)
        cost_returns = self.cost_returns[:-1].reshape(-1, 1)
        masks = self.masks[:-1].reshape(-1, 1)
        active_masks = self.active_masks[:-1].reshape(-1, 1)
        action_log_probs = self.action_log_probs.reshape(-1, self.action_log_probs.shape[-1])
        aver_episode_costs = self.aver_episode_costs

        if self.factor is not None:
            factor = self.factor.reshape(-1, self.factor.shape[-1])
        advantages = advantages.reshape(-1, 1)
        if cost_adv is not None:
            cost_adv = cost_adv.reshape(-1, 1)

        for indices in sampler:
            share_obs_batch = share_obs[indices]
            obs_batch = obs[indices]
            rnn_states_batch = rnn_states[indices]
            rnn_states_critic_batch = rnn_states_critic[indices]
            rnn_states_cost_batch = rnn_states_cost[indices]

            actions_batch = actions[indices]
            if self.available_actions is not None:
                available_actions_batch = available_actions[indices]
            else:
                available_actions_batch = None
            value_preds_batch = value_preds[indices]
            return_batch = returns[indices]
            cost_preds_batch = cost_preds[indices]
            cost_return_batch = cost_returns[indices]
            masks_batch = masks[indices]
            active_masks_batch = active_masks[indices]
            old_action_log_probs_batch = action_log_probs[indices]
            if advantages is None:
                adv_targ = None
            else:
                adv_targ = advantages[indices]
            if cost_adv is None:
                cost_adv_targ = None
            else:
                cost_adv_targ = cost_adv[indices]
            if self.factor is None:
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch
            else:
                if self.algo == "mappolag":
                    factor_batch = factor[indices]
                    yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch, cost_preds_batch, cost_return_batch, rnn_states_cost_batch, cost_adv_targ, aver_episode_costs
                elif self.algo == "macpo":
                    factor_batch = factor[indices]
                    yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch, cost_preds_batch, cost_return_batch, rnn_states_cost_batch, cost_adv_targ, aver_episode_costs
                else:
                    factor_batch = factor[indices]
                    yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch

