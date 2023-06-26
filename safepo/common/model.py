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

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.distributions import Normal


def build_mlp_network(sizes):
    layers = list()
    for j in range(len(sizes) - 1):
        act = nn.Tanh if j < len(sizes) - 2 else nn.Identity
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        nn.init.kaiming_uniform_(affine_layer.weight, a=np.sqrt(5))
        layers += [affine_layer, act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """Actor network."""

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.mean = build_mlp_network([obs_dim, 64, 64, act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim), requires_grad=True)

    def forward(self, obs: torch.Tensor):
        mean = self.mean(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)


class VCritic(nn.Module):
    """Critic network."""

    def __init__(self, obs_dim):
        super().__init__()
        self.critic = build_mlp_network([obs_dim, 64, 64, 1])

    def forward(self, obs):
        return torch.squeeze(self.critic(obs), -1)


class ActorVCritic(nn.Module):
    """Actor critic policy."""

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.reward_critic = VCritic(obs_dim)
        self.cost_critic = VCritic(obs_dim)
        self.actor = Actor(obs_dim, act_dim)

    def get_value(self, obs):
        return self.critic(obs)

    def step(self, obs, deterministic=False):
        dist = self.actor(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        value_r = self.reward_critic(obs)
        value_c = self.cost_critic(obs)
        return action, log_prob, value_r, value_c
