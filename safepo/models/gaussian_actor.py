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
import numpy as np
import torch
import torch.nn as nn
from torch.distributions.normal import Normal

from safepo.models.actor import Actor
from safepo.models.model_utils import build_mlp_network


class MLPGaussianActor(Actor):
    def __init__(
            self,
            obs_dim,
            act_dim,
            hidden_sizes,
            activation,
            weight_initialization,
            shared=None):
        super().__init__(obs_dim, act_dim, weight_initialization)
        log_std = np.log(0.5) * np.ones(self.act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std),
                                          requires_grad=False)

        if shared is not None:  # use shared layers
            action_head = nn.Linear(hidden_sizes[-1], act_dim)
            self.net = nn.Sequential(shared, action_head, nn.Identity())
        else:
            layers = [self.obs_dim] + list(hidden_sizes) + [self.act_dim]
            self.net = build_mlp_network(
                layers,
                activation=activation,
                weight_initialization=weight_initialization
            )

    def dist(self, obs):
        mu = self.net(obs)
        return Normal(mu, self.std)

    def detach_dist(self, obs):
        mu = self.net(obs).detach()
        return Normal(mu, self.std.detach())

    def focops_return(self, obs, act):
        mean = self.net(obs)
        std = torch.exp(self.log_std)
        std = std.expand_as(mean)
        normal = Normal(mean, std)
        return normal.log_prob(act).sum(-1, keepdim=True), mean, std


    def log_prob_from_dist(self, pi, act) -> torch.Tensor:
        # Last axis sum needed for Torch Normal distribution
        return pi.log_prob(act).sum(axis=-1)

    def sample(self, obs):
        pi = self.dist(obs)
        a = pi.sample()
        logp_a = self.log_prob_from_dist(pi, a)

        return a, logp_a

    def set_log_std(self, frac):
        """ To support annealing exploration noise.
            frac is annealing from 1. to 0 over course of training"""
        assert 0 <= frac <= 1
        new_stddev = 0.499 * frac + 0.01  # annealing from 0.5 to 0.01
        log_std = np.log(new_stddev) * np.ones(self.act_dim, dtype=np.float32)
        self.log_std = torch.nn.Parameter(torch.as_tensor(log_std),
                                          requires_grad=False)

    @property
    def std(self):
        """ Standard deviation of distribution."""
        return torch.exp(self.log_std)

    def predict(self, obs):
        """ Predict action based on observation without exploration noise.
            Use this method for evaluation purposes. """
        action = self.net(obs)
        log_p = torch.ones_like(action)  # avoid type conflicts at evaluation

        return action, log_p
