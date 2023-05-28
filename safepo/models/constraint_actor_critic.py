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
import torch

from safepo.models.actor_critic import ActorCritic
from safepo.models.critic import Critic


class ConstraintActorCritic(ActorCritic):
    def __init__(
        self,
        policy_config,
        observation_space,
        action_space,
        use_standardized_obs,
        use_scaled_rewards,
        use_shared_weights,
        weight_initialization,
    ):
        super().__init__(
            policy_config=policy_config,
            observation_space=observation_space,
            action_space=action_space,
            use_standardized_obs=use_standardized_obs,
            use_scaled_rewards=use_scaled_rewards,
            use_shared_weights=use_shared_weights,
            weight_initialization=weight_initialization,
        )
        self.cost_critic = Critic(
            obs_dim=self.obs_shape[0], shared=None, **policy_config["critic"]
        )

    def step(self, obs: torch.Tensor) -> tuple:
        """Produce action, value, log_prob(action).
        If training, this includes exploration noise!

        Note:
            Training mode can be activated with ac.train()
            Evaluation mode is activated by ac.eval()
        """
        with torch.no_grad():
            if self.obs_oms:
                # Note: do the updates at the end of batch!
                # self.obs_oms.update(obs) if self.training else None
                obs = self.obs_oms(obs)
            v = self.critic(obs)
            c = self.cost_critic(obs)

            if self.training:
                a, logp_a = self.actor.sample(obs)
            else:
                a, logp_a = self.actor.predict(obs)

        return a.numpy(), v.numpy(), c.numpy(), logp_a.numpy()
