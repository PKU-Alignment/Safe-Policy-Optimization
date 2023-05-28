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
from collections import deque, namedtuple

import torch

from safepo.algos.lagrangian_base import Lagrangian
from safepo.algos.policy_gradient import PG


class CPPOPid(PG):
    def __init__(
            self,
            algo: str = 'cppo-pid',
            cost_limit: float = 25.,
            clip: float = 0.2,
            lagrangian_multiplier_init: float = 0.001,
            pid_Kp=0.01,
            pid_Ki=0.01,
            pid_Kd=0.01,
            pid_d_delay=10,
            pid_delta_p_ema_alpha=0.95,  # 0 for hard update, 1 for no update
            pid_delta_d_ema_alpha=0.95,
            sum_norm=True,  # L = (J_r - lam * J_c) / (1 + lam); lam <= 0
            diff_norm=False,  # L = (1 - lam) * J_r - lam * J_c; 0 <= lam <= 1
            penalty_max=100,  # only used if sum_norm=diff_norm=False
            use_lagrangian_penalty=True,
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
        # self.update_params_from_local(locals())

        self.clip = clip
        self.cost_limit = cost_limit

        # pid
        self.pid_Kp = pid_Kp
        self.pid_Ki = pid_Ki
        self.pid_Kd = pid_Kd
        self.pid_d_delay = pid_d_delay
        self.pid_delta_p_ema_alpha = pid_delta_p_ema_alpha
        self.pid_delta_d_ema_alpha = pid_delta_d_ema_alpha

        self.sum_norm = sum_norm
        self.diff_norm = diff_norm
        self.penalty_max = penalty_max

        self.pid_i = self.cost_penalty = lagrangian_multiplier_init
        self.cost_ds = deque(maxlen=self.pid_d_delay)
        self.cost_ds.append(0)
        self._delta_p = 0
        self._cost_d = 0

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('LagrangeMultiplier', self.cost_penalty)
        self.logger.log_tabular('pid_Kp', self.pid_Kp)
        self.logger.log_tabular('pid_Ki', self.pid_Ki)
        self.logger.log_tabular('pid_Kd', self.pid_Kd)

    def compute_loss_pi(self, data: dict, **kwargs):
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])
        ratio_clip = torch.clamp(ratio, 1 - self.clip, 1 + self.clip)

        surr_adv = (torch.min(ratio * data['adv'], ratio_clip * data['adv'])).mean()
        surr_cadv = (torch.max(ratio * data['cost_adv'], ratio_clip * data['cost_adv'])).mean()

        loss_pi = - surr_adv
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        # ensure that lagrange multiplier is positive
        penalty = self.cost_penalty
        # loss_pi += penalty * ((ratio * data['cost_adv']).mean())
        loss_pi += penalty * surr_cadv
        loss_pi /= (1 + penalty)

        # Useful extra info
        approx_kl = .5 * (data['log_p'] - _log_p).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def pid_update(self, ep_cost_avg):
        delta = float(ep_cost_avg - self.cost_limit)  # ep_cost_avg: tensor
        self.pid_i = max(0., self.pid_i + delta * self.pid_Ki)
        if self.diff_norm:
            self.pid_i = max(0., min(1., self.pid_i))
        a_p = self.pid_delta_p_ema_alpha
        self._delta_p *= a_p
        self._delta_p += (1 - a_p) * delta
        a_d = self.pid_delta_d_ema_alpha
        self._cost_d *= a_d
        self._cost_d += (1 - a_d) * float(ep_cost_avg)
        pid_d = max(0., self._cost_d - self.cost_ds[0])
        pid_o = (self.pid_Kp * self._delta_p + self.pid_i +
                 self.pid_Kd * pid_d)
        self.cost_penalty = max(0., pid_o)
        if self.diff_norm:
            self.cost_penalty = min(1., self.cost_penalty)
        if not (self.diff_norm or self.sum_norm):
            self.cost_penalty = min(self.cost_penalty, self.penalty_max)
        self.cost_ds.append(self._cost_d)

    def update(self):
        raw_data = self.buf.get()
        # pre-process data
        data = self.pre_process_data(raw_data)
        # Note that logger already uses MPI statistics across all processes..
        ep_costs = self.logger.get_stats('EpCosts')[0]
        # First update Lagrange multiplier parameter
        self.pid_update(ep_costs)
        # now update policy and value network
        self.update_policy_net(data=data)
        self.update_value_net(data=data)
        self.update_cost_net(data=data)
        # Update running statistics, e.g. observation standardization
        # Note: observations from are raw outputs from environment
        self.update_running_statistics(raw_data)
