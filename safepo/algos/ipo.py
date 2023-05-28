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
import torch.nn.functional as F

from safepo.algos.policy_gradient import PG


class IPO(PG):
    def __init__(
            self,
            algo: str = 'ipo',
            cost_limit: float = 25.,
            clip: float = 0.2,
            kappa: float = 0.01,
            penalty_max: float = 1.0,
            use_standardized_reward=True,
            use_standardized_cost=True,
            use_standardized_obs=False,
            use_cost_value_function=True,
            use_kl_early_stopping=True,
            **kwargs
    ):
        PG.__init__(
            self,
            algo=algo,
            use_cost_value_function=use_cost_value_function,
            use_kl_early_stopping=use_kl_early_stopping,
            use_standardized_reward=use_standardized_reward,
            use_standardized_cost=use_standardized_cost,
            use_standardized_obs=use_standardized_obs,
            **kwargs
        )
        self.clip = clip
        self.cost_limit = cost_limit
        self.kappa = kappa
        self.penalty_max = penalty_max

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('Penalty')

    def compute_loss_pi(self, data: dict, **kwargs) -> tuple:
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        ratio_clip = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)

        surr_adv = (torch.min(ratio * data['adv'], ratio_clip * data['adv'])).mean()
        surr_cadv = (torch.max(ratio * data['cost_adv'], ratio_clip * data['cost_adv'])).mean()

        ep_costs = self.logger.get_stats('EpCosts')[0]
        c = self.cost_limit - ep_costs

        penalty = self.kappa / (c + 1e-8)
        if penalty < 0 or penalty > self.penalty_max:
            penalty = self.penalty_max

        self.logger.store(Penalty=penalty)

        loss_pi = -surr_adv + penalty * surr_cadv
        loss_pi = loss_pi.mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2
                     / dist.stddev ** 2).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio_clip.mean().item())
        return loss_pi, pi_info
