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
from torch.distributions.categorical import Categorical
from safepo.models.Actor import Actor
from safepo.models.model_utils import build_mlp_network

class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation,
                 weight_initialization, shared=None):
        super().__init__(obs_dim, act_dim, weight_initialization, shared=shared)
        if shared is not None:
            raise NotImplementedError
        self.net = build_mlp_network(
            [obs_dim] + list(hidden_sizes) + [act_dim],
            activation=activation,
            weight_initialization=weight_initialization
        )

    def dist(self, obs):

        logits = self.net(obs)
        return Categorical(logits=logits)

    def log_prob_from_dist(self, pi, act):

        return pi.log_prob(act)

    def sample(self, obs):

        dist = self.dist(obs)
        a = dist.sample()
        logp_a = self.log_prob_from_dist(dist, a)

        return a, logp_a
