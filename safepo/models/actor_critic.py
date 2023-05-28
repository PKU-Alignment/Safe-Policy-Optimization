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
from gymnasium.spaces import Box, Discrete

from safepo.common.online_mean_std import OnlineMeanStd
from safepo.models.critic import Critic
from safepo.models.MLP_Categorical_Actor import MLPCategoricalActor
from safepo.models.gaussian_actor import MLPGaussianActor
from safepo.models.model_utils import build_mlp_network


class ActorCritic(nn.Module):
    def __init__(self,
                 observation_space,
                 action_space,
                 use_standardized_obs,
                 use_scaled_rewards,
                 use_shared_weights,
                 policy_config,
                 weight_initialization='kaiming_uniform'
    ):
        super().__init__()
        self.obs_shape = observation_space.shape
        self.obs_oms = OnlineMeanStd(shape=self.obs_shape) \
            if use_standardized_obs else None

        # policy builder depends on action space
        act_dim = action_space.shape[0]
        obs_dim = observation_space.shape[0]
        layer_units = [obs_dim] + list(policy_config['actor']['hidden_sizes'])
        activation_function = policy_config['actor']['activation']
        if use_shared_weights:
            shared = build_mlp_network(
                layer_units,
                activation=activation_function,
                weight_initialization=weight_initialization,
                output_activation=activation_function
            )
        else:
            shared = None

        self.pi = MLPGaussianActor(obs_dim=obs_dim,
                           act_dim=act_dim,
                           shared=shared,
                           weight_initialization=weight_initialization,
                           **ac_kwargs['pi'])
        self.v = Critic(obs_dim,
                           shared=shared,
                           **ac_kwargs['val'])

        self.ret_oms = OnlineMeanStd(shape=(1,)) if use_scaled_rewards else None

    def forward(self,
                obs: torch.Tensor
                ) -> tuple:
        return self.step(obs)

    def step(self,
             obs: torch.Tensor
             ):
        """
            If training, this includes exploration noise!
            Expects that obs is not pre-processed.

            Returns:
                action, value, log_prob(action)
            Note:
                Training mode can be activated with ac.train()
                Evaluation mode is activated by ac.eval()
        """
        with torch.no_grad():
            if self.obs_oms:
                # Note: Update RMS in Algorithm.running_statistics() method
                # self.obs_oms.update(obs) if self.training else None
                obs = self.obs_oms(obs)
            v = self.v(obs)
            if self.training:
                a, logp_a = self.pi.sample(obs)
            else:
                a, logp_a = self.pi.predict(obs)

        return a.numpy(), v.numpy(), logp_a.numpy()

    def act(self,
            obs: torch.Tensor
            ) -> np.ndarray:
        return self.step(obs)[0]

    def update(self, frac):
        """update internals of actors

            1) Updates exploration parameters
            + for Gaussian actors update log_std

        frac: progress of epochs, i.e. current epoch / total epochs
                e.g. 10 / 100 = 0.1

        """
        if hasattr(self.pi, 'set_log_std'):
            self.pi.set_log_std(1 - frac)
