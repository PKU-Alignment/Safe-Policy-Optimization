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


class VectorizedOnPolicyBuffer:
    def __init__(  # pylint: disable=too-many-arguments
        self,
        obs_space,
        act_space,
        size: int,
        gamma: float,
        lam: float,
        lam_c: float,
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
