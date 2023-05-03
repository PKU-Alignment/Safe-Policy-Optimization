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
from safepo.algos.policy_gradient import PG
from safepo.algos.lagrangian_base import Lagrangian
import safepo.common.mpi_tools as mpi_tools

class PPO_Lag(PG,Lagrangian):
    '''

    '''
    def __init__(
            self,
            algo='ppo-lag',
            cost_limit=25.,
            clip=0.2,
            lagrangian_multiplier_init=0.001,
            lambda_lr=0.035,
            lambda_optimizer='Adam',
            use_standardized_reward=True,
            use_standardized_cost=True,
            use_standardized_obs=False,
            use_reward_scaling=False,
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
            use_reward_scaling=use_reward_scaling,
            **kwargs
        )

        Lagrangian.__init__(
            self,
            cost_limit=cost_limit,
            lagrangian_multiplier_init=lagrangian_multiplier_init,
            lambda_lr=lambda_lr,
            lambda_optimizer=lambda_optimizer
        )

        self.clip = clip

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('LagrangeMultiplier',
                                self.lagrangian_multiplier.item())


    def compute_loss_pi(self, data: dict):
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])
        ratio_clip = torch.clamp(ratio, 1-self.clip, 1+self.clip)
        loss_pi = -(torch.min(ratio * data['adv'], ratio_clip * data['adv'])).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # ensure that lagrange multiplier is positive
        penalty = self.lambda_range_projection(self.lagrangian_multiplier).item()
        loss_pi += penalty * ((ratio * data['cost_adv']).mean())
        loss_pi /= (1 + penalty)

        # Useful extra info
        approx_kl = .5 * (data['log_p'] - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def update(self):
        raw_data = self.buf.get()
        # pre-process data
        data = self.pre_process_data(raw_data)
        # Note that logger already uses MPI statistics across all processes..
        ep_costs = self.logger.get_stats('EpCosts')[0]
        # First update Lagrange multiplier parameter
        self.update_lagrange_multiplier(ep_costs)
        # now update policy and value network
        self.update_policy_net(data=data)
        self.update_value_net(data=data)
        self.update_cost_net(data=data)
        # Update running statistics, e.g. observation standardization
        # Note: observations from are raw outputs from environment
        self.update_running_statistics(raw_data)
