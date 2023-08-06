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
    def __init__(  # pylint: disable=too-many-arguments
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
        """Store vectorized data into vectorized buffer."""
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
    return x.reshape(T * N, *x.shape[2:])

def _cast(x):
    return x.transpose(1,0,2).reshape(-1, *x.shape[2:])

class SeparatedReplayBuffer(object):
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
            obs_shape = obs_shape[:1]

        if type(share_obs_shape[-1]) == list:
            share_obs_shape = share_obs_shape[:1]

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
        # if train_episode_costs_aver is not None:
        #     self.train_episode_costs_aver[self.step + 1] = train_episode_costs_aver.copy()

        self.step = (self.step + 1) % self.episode_length

    def chooseinsert(self, share_obs, obs, rnn_states, rnn_states_critic, actions, action_log_probs,
                     value_preds, rewards, masks, bad_masks=None, active_masks=None, available_actions=None):
        self.share_obs[self.step] = share_obs.copy_()
        self.obs[self.step] = obs.copy_()
        self.rnn_states[self.step + 1] = rnn_states.copy_()
        self.rnn_states_critic[self.step + 1] = rnn_states_critic.copy_()
        self.actions[self.step] = actions.copy_()
        self.action_log_probs[self.step] = action_log_probs.copy_()
        self.value_preds[self.step] = value_preds.copy_()
        self.rewards[self.step] = rewards.copy_()
        self.masks[self.step + 1] = masks.copy_()
        if bad_masks is not None:
            self.bad_masks[self.step + 1] = bad_masks.copy_()
        if active_masks is not None:
            self.active_masks[self.step] = active_masks.copy_()
        if available_actions is not None:
            self.available_actions[self.step] = available_actions.copy_()

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

    def chooseafter_update(self):
        self.rnn_states[0].copy_(self.rnn_states[-1])
        self.rnn_states_critic[0].copy_(self.rnn_states_critic[-1])
        self.masks[0].copy_(self.masks[-1])
        self.bad_masks[0].copy_(self.bad_masks[-1])

    def compute_returns(self, next_value, value_normalizer=None):
        """
        use proper time limits, the difference of use or not is whether use bad_mask
        """
        if self._use_proper_time_limits:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                            + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(self.value_preds[step])
                    else:
                        self.returns[step] = (self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]) * self.bad_masks[step + 1] \
                            + (1 - self.bad_masks[step + 1]) * self.value_preds[step]
        else:
            if self._use_gae:
                self.value_preds[-1] = next_value
                gae = 0
                for step in reversed(range(self.rewards.shape[0])):
                    if self._use_popart or self._use_valuenorm:

                        delta = self.rewards[step] + self.gamma * value_normalizer.denormalize(self.value_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.value_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + value_normalizer.denormalize(self.value_preds[step])
                    else:
                        delta = self.rewards[step] + self.gamma * self.value_preds[step + 1] * self.masks[step + 1] - self.value_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.returns[step] = gae + self.value_preds[step]
            else:
                self.returns[-1] = next_value
                for step in reversed(range(self.rewards.shape[0])):
                    self.returns[step] = self.returns[step + 1] * self.gamma * self.masks[step + 1] + self.rewards[step]

    def compute_cost_returns(self, next_cost, value_normalizer=None):

        if self._use_proper_time_limits:
            if self._use_gae:
                self.cost_preds[-1] = next_cost
                gae = 0
                for step in reversed(range(self.costs.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.costs[step] + self.gamma * value_normalizer.denormalize(self.cost_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.cost_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.cost_returns[step] = gae + value_normalizer.denormalize(self.cost_preds[step])
                    else:
                        delta = self.costs[step] + self.gamma * self.cost_preds[step + 1] * self.masks[step + 1] - self.cost_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        gae = gae * self.bad_masks[step + 1]
                        self.cost_returns[step] = gae + self.cost_preds[step]
            else:
                self.cost_returns[-1] = next_cost
                for step in reversed(range(self.costs.shape[0])):
                    if self._use_popart:
                        self.cost_returns[step] = (self.cost_returns[step + 1] * self.gamma * self.masks[step + 1] + self.costs[step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * value_normalizer.denormalize(self.cost_preds[step])
                    else:
                        self.cost_returns[step] = (self.cost_returns[step + 1] * self.gamma * self.masks[step + 1] + self.costs[step]) * self.bad_masks[step + 1] \
                                             + (1 - self.bad_masks[step + 1]) * self.cost_preds[step]
        else:
            if self._use_gae:
                self.cost_preds[-1] = next_cost
                gae = 0
                for step in reversed(range(self.costs.shape[0])):
                    if self._use_popart or self._use_valuenorm:
                        delta = self.costs[step] + self.gamma * value_normalizer.denormalize(self.cost_preds[step + 1]) * self.masks[step + 1] - value_normalizer.denormalize(self.cost_preds[step])
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.cost_returns[step] = gae + value_normalizer.denormalize(self.cost_preds[step])
                    else:
                        delta = self.costs[step] + self.gamma * self.cost_preds[step + 1] * self.masks[step + 1] - self.cost_preds[step]
                        gae = delta + self.gamma * self.gae_lambda * self.masks[step + 1] * gae
                        self.cost_returns[step] = gae + self.cost_preds[step]
            else:
                self.cost_returns[-1] = next_cost
                for step in reversed(range(self.costs.shape[0])):
                    self.cost_returns[step] = self.cost_returns[step + 1] * self.gamma * self.masks[step + 1] + self.costs[step]

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
                if self.algo == "macppo":
                    factor_batch = factor[indices]
                    yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch, cost_preds_batch, cost_return_batch, rnn_states_cost_batch, cost_adv_targ
                elif self.algo == "mappolag":
                    factor_batch = factor[indices]
                    yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch, cost_preds_batch, cost_return_batch, rnn_states_cost_batch, cost_adv_targ, aver_episode_costs
                elif self.algo == "macpo":
                    factor_batch = factor[indices]
                    yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch, cost_preds_batch, cost_return_batch, rnn_states_cost_batch, cost_adv_targ, aver_episode_costs
                else:
                    factor_batch = factor[indices]
                    yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch

    def naive_recurrent_generator(self, advantages, num_mini_batch):
        n_rollout_threads = self.rewards.shape[1]
        assert n_rollout_threads >= num_mini_batch, (
            "PPO requires the number of processes ({}) "
            "to be greater than or equal to the number of "
            "PPO mini batches ({}).".format(n_rollout_threads, num_mini_batch))
        num_envs_per_batch = n_rollout_threads // num_mini_batch
        perm = torch.randperm(n_rollout_threads)
        for start_ind in range(0, n_rollout_threads, num_envs_per_batch):
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            factor_batch = []
            for offset in range(num_envs_per_batch):
                ind = perm[start_ind + offset]
                share_obs_batch.append(self.share_obs[:-1, ind])
                obs_batch.append(self.obs[:-1, ind])
                rnn_states_batch.append(self.rnn_states[0:1, ind])
                rnn_states_critic_batch.append(self.rnn_states_critic[0:1, ind])
                actions_batch.append(self.actions[:, ind])
                if self.available_actions is not None:
                    available_actions_batch.append(self.available_actions[:-1, ind])
                value_preds_batch.append(self.value_preds[:-1, ind])
                return_batch.append(self.returns[:-1, ind])
                masks_batch.append(self.masks[:-1, ind])
                active_masks_batch.append(self.active_masks[:-1, ind])
                old_action_log_probs_batch.append(self.action_log_probs[:, ind])
                adv_targ.append(advantages[:, ind])
                if self.factor is not None:
                    factor_batch.append(self.factor[:,ind])

            # [N[T, dim]]
            T, N = self.episode_length, num_envs_per_batch
            # These are all from_numpys of size (T, N, -1)
            share_obs_batch = torch.stack(share_obs_batch, 1)
            obs_batch = torch.stack(obs_batch, 1)
            actions_batch = torch.stack(actions_batch, 1)
            if self.available_actions is not None:
                available_actions_batch = torch.stack(available_actions_batch, 1)
            if self.factor is not None:
                factor_batch=torch.stack(factor_batch,1)
            value_preds_batch = torch.stack(value_preds_batch, 1)
            return_batch = torch.stack(return_batch, 1)
            masks_batch = torch.stack(masks_batch, 1)
            active_masks_batch = torch.stack(active_masks_batch, 1)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch, 1)
            adv_targ = torch.stack(adv_targ, 1)

            # States is just a (N, -1) from_numpy [N[1,dim]]
            rnn_states_batch = torch.stack(rnn_states_batch, 1).reshape(N, *self.rnn_states.shape[2:])
            rnn_states_critic_batch = torch.stack(rnn_states_critic_batch, 1).reshape(N, *self.rnn_states_critic.shape[2:])

            # Flatten the (T, N, ...) from_numpys to (T * N, ...)
            share_obs_batch = _flatten(T, N, share_obs_batch)
            obs_batch = _flatten(T, N, obs_batch)
            actions_batch = _flatten(T, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(T, N, available_actions_batch)
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch=_flatten(T,N,factor_batch)
            value_preds_batch = _flatten(T, N, value_preds_batch)
            return_batch = _flatten(T, N, return_batch)
            masks_batch = _flatten(T, N, masks_batch)
            active_masks_batch = _flatten(T, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(T, N, old_action_log_probs_batch)
            adv_targ = _flatten(T, N, adv_targ)
            if self.factor is not None:
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch
            else:
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch

    def recurrent_generator(self, advantages, num_mini_batch, data_chunk_length):
        episode_length, n_rollout_threads = self.rewards.shape[0:2]
        batch_size = n_rollout_threads * episode_length
        data_chunks = batch_size // data_chunk_length  # [C=r*T/L]
        mini_batch_size = data_chunks // num_mini_batch

        assert episode_length * n_rollout_threads >= data_chunk_length, (
            "PPO requires the number of processes ({}) * episode length ({}) "
            "to be greater than or equal to the number of "
            "data chunk length ({}).".format(n_rollout_threads, episode_length, data_chunk_length))
        assert data_chunks >= 2, ("need larger batch size")

        rand = torch.randperm(data_chunks)
        sampler = [rand[i*mini_batch_size:(i+1)*mini_batch_size] for i in range(num_mini_batch)]

        if len(self.share_obs.shape) > 3:
            share_obs = self.share_obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.share_obs.shape[2:])
            obs = self.obs[:-1].transpose(1, 0, 2, 3, 4).reshape(-1, *self.obs.shape[2:])
        else:
            share_obs = _cast(self.share_obs[:-1])
            obs = _cast(self.obs[:-1])

        actions = _cast(self.actions)
        action_log_probs = _cast(self.action_log_probs)
        advantages = _cast(advantages)
        value_preds = _cast(self.value_preds[:-1])
        returns = _cast(self.returns[:-1])
        masks = _cast(self.masks[:-1])
        active_masks = _cast(self.active_masks[:-1])
        if self.factor is not None:
            factor = _cast(self.factor)
        # rnn_states = _cast(self.rnn_states[:-1])
        # rnn_states_critic = _cast(self.rnn_states_critic[:-1])
        rnn_states = self.rnn_states[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states.shape[2:])
        rnn_states_critic = self.rnn_states_critic[:-1].transpose(1, 0, 2, 3).reshape(-1, *self.rnn_states_critic.shape[2:])

        if self.available_actions is not None:
            available_actions = _cast(self.available_actions[:-1])

        for indices in sampler:
            share_obs_batch = []
            obs_batch = []
            rnn_states_batch = []
            rnn_states_critic_batch = []
            actions_batch = []
            available_actions_batch = []
            value_preds_batch = []
            return_batch = []
            masks_batch = []
            active_masks_batch = []
            old_action_log_probs_batch = []
            adv_targ = []
            factor_batch = []
            for index in indices:
                ind = index * data_chunk_length
                # size [T+1 N M Dim]-->[T N Dim]-->[N T Dim]-->[T*N,Dim]-->[L,Dim]
                share_obs_batch.append(share_obs[ind:ind+data_chunk_length])
                obs_batch.append(obs[ind:ind+data_chunk_length])
                actions_batch.append(actions[ind:ind+data_chunk_length])
                if self.available_actions is not None:
                    available_actions_batch.append(available_actions[ind:ind+data_chunk_length])
                value_preds_batch.append(value_preds[ind:ind+data_chunk_length])
                return_batch.append(returns[ind:ind+data_chunk_length])
                masks_batch.append(masks[ind:ind+data_chunk_length])
                active_masks_batch.append(active_masks[ind:ind+data_chunk_length])
                old_action_log_probs_batch.append(action_log_probs[ind:ind+data_chunk_length])
                adv_targ.append(advantages[ind:ind+data_chunk_length])
                # size [T+1 N Dim]-->[T N Dim]-->[T*N,Dim]-->[1,Dim]
                rnn_states_batch.append(rnn_states[ind])
                rnn_states_critic_batch.append(rnn_states_critic[ind])
                if self.factor is not None:
                    factor_batch.append(factor[ind:ind+data_chunk_length])
            L, N = data_chunk_length, mini_batch_size

            # These are all from_numpys of size (N, L, Dim)
            share_obs_batch = torch.stack(share_obs_batch)
            obs_batch = torch.stack(obs_batch)

            actions_batch = torch.stack(actions_batch)
            if self.available_actions is not None:
                available_actions_batch = torch.stack(available_actions_batch)
            if self.factor is not None:
                factor_batch = torch.stack(factor_batch)
            value_preds_batch = torch.stack(value_preds_batch)
            return_batch = torch.stack(return_batch)
            masks_batch = torch.stack(masks_batch)
            active_masks_batch = torch.stack(active_masks_batch)
            old_action_log_probs_batch = torch.stack(old_action_log_probs_batch)
            adv_targ = torch.stack(adv_targ)

            # States is just a (N, -1) from_numpy
            rnn_states_batch = torch.stack(rnn_states_batch).reshape(N, *self.rnn_states.shape[2:])
            rnn_states_critic_batch = torch.stack(rnn_states_critic_batch).reshape(N, *self.rnn_states_critic.shape[2:])

            share_obs_batch = _flatten(L, N, share_obs_batch)
            obs_batch = _flatten(L, N, obs_batch)
            actions_batch = _flatten(L, N, actions_batch)
            if self.available_actions is not None:
                available_actions_batch = _flatten(L, N, available_actions_batch)
            else:
                available_actions_batch = None
            if self.factor is not None:
                factor_batch = _flatten(L, N, factor_batch)
            value_preds_batch = _flatten(L, N, value_preds_batch)
            return_batch = _flatten(L, N, return_batch)
            masks_batch = _flatten(L, N, masks_batch)
            active_masks_batch = _flatten(L, N, active_masks_batch)
            old_action_log_probs_batch = _flatten(L, N, old_action_log_probs_batch)
            adv_targ = _flatten(L, N, adv_targ)
            if self.factor is not None:
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch, factor_batch
            else:
                yield share_obs_batch, obs_batch, rnn_states_batch, rnn_states_critic_batch, actions_batch, value_preds_batch, return_batch, masks_batch, active_masks_batch, old_action_log_probs_batch, adv_targ, available_actions_batch
