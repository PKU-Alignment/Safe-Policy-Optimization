import torch
from safepo.algos.policy_graident import PG
# from safepo.common.core import ConstrainedPolicyGradientAlgorithm
from safepo.algos.lagrangian_base import Lagrangian
import safepo.common.mpi_tools as mpi_tools

class PPO_Lagrangian(PG,Lagrangian):
    def __init__(
            self,
            alg: str = 'ppo_lagrangian',
            cost_limit: float = 25.,
            clip: float = 0.2,
            lagrangian_multiplier_init: float = 0.001,
            lambda_lr: float = 0.035,
            lambda_optimizer: str = 'Adam',
            use_lagrangian_penalty: bool = True,
            use_standardized_advantages: bool = True,
            **kwargs
    ):
        PG.__init__(
            self,
            alg=alg,
            cost_limit=cost_limit,
            lagrangian_multiplier_init=lagrangian_multiplier_init,
            lambda_lr=lambda_lr,
            lambda_optimizer=lambda_optimizer,
            use_cost_value_function=True,
            use_kl_early_stopping=True,
            use_lagrangian_penalty=use_lagrangian_penalty,
            use_standardized_advantages=use_standardized_advantages,
            **kwargs
        )

        Lagrangian.__init__(
            self,
            cost_limit=cost_limit,
            use_lagrangian_penalty=use_lagrangian_penalty,
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
        loss_pi = -(torch.min(ratio, ratio_clip) * data['adv']).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        if self.use_lagrangian_penalty:
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



# import torch
# from safepo.algos.ppo import PPO
# from safepo.common.core import ConstrainedPolicyGradientAlgorithm
# import safepo.common.mpi_tools as mpi_tools # for debug

# class PPO_Lagrangian(PPO,ConstrainedPolicyGradientAlgorithm):
#     def __init__(
#             self,
#             alg: str = 'ppo_lagrangian',
#             cost_limit: float = 25.,
#             lagrangian_multiplier_init: float = 0.001,
#             lambda_lr: float = 0.05,
#             lambda_optimizer: str = 'SGD',
#             use_lagrangian_penalty: bool = True,
#             use_standardized_advantages: bool = True,
#             use_reward_scaling:bool = False,
#             **kwargs
#     ):
#         PPO.__init__(
#             self,
#             alg=alg,
#             cost_limit=cost_limit,
#             lagrangian_multiplier_init=lagrangian_multiplier_init,
#             lambda_lr=lambda_lr,
#             lambda_optimizer=lambda_optimizer,
#             use_cost_value_function=True,
#             use_kl_early_stopping=False,
#             use_lagrangian_penalty=use_lagrangian_penalty,
#             use_standardized_advantages=use_standardized_advantages,
#             use_reward_scaling=use_reward_scaling,
#             **kwargs
#         )

#         ConstrainedPolicyGradientAlgorithm.__init__(
#             self,
#             cost_limit=cost_limit,
#             use_lagrangian_penalty=use_lagrangian_penalty,
#             lagrangian_multiplier_init=lagrangian_multiplier_init,
#             lambda_lr=lambda_lr,
#             lambda_optimizer=lambda_optimizer
#         )

#     def algorithm_specific_logs(self):
#         super().algorithm_specific_logs()
#         self.logger.log_tabular('LagrangeMultiplier', self.lagrangian_multiplier.item())

#     def compute_loss_pi(self, data: dict):
#         # Policy loss
#         dist, _log_p = self.ac.pi(data['obs'], data['act'])
#         ratio = torch.exp(_log_p - data['log_p'])
#         # ratio_clip = torch.clamp(ratio, 1-self.clip, 1+self.clip)
#         # loss_pi = -(torch.min(ratio, ratio_clip) * data['adv']).mean()

#         loss_pi = - (torch.min(
#             ratio * data['adv'], torch.clamp(ratio, 1-self.clip, 1+self.clip) * data['adv']
#         )).mean()

#         loss_c = (torch.min(
#             ratio * data['cost_adv'], torch.clamp(ratio, 1-self.clip, 1+self.clip) * data['cost_adv']
#         )).mean()
#         # loss_c = (ratio * data['cost_adv']).mean()

#         loss_pi -= self.entropy_coef * dist.entropy().mean()

#         if self.use_lagrangian_penalty:
#             # ensure that lagrange multiplier is positive
#             # need to detach() important !!! 
#             penalty = self.lambda_range_projection(self.lagrangian_multiplier).detach()
#             print('penalty:', penalty) if mpi_tools.proc_id() == 0 else None
            
#             loss_pi += penalty * loss_c
#             loss_pi /= (1 + penalty)
            
#         # Useful extra info
#         approx_kl = (0.5 * (dist.mean - data['act']) ** 2
#                      / dist.stddev ** 2).mean().item()
#         ent = dist.entropy().mean().item()
#         pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

#         return loss_pi, pi_info

#     def update(self):
#         raw_data = self.buf.get()
#         # pre-process data
#         data = self.pre_process_data(raw_data)
#         # sub-sampling accelerates calculations
#         self.fvp_obs = data['obs'][::4]
#         # Note that logger already uses MPI statistics across all processes..
#         ep_costs = self.logger.get_stats('EpCosts')[0]
#         # First update Lagrange multiplier parameter
#         self.update_lagrange_multiplier(ep_costs)
#         # now update policy and value network
#         self.update_policy_net(data=data)
#         self.update_value_net(data=data)
#         self.update_cost_net(data=data)
#         # Update running statistics, e.g. observation standardization
#         # Note: observations from are raw outputs from environment
#         self.update_running_statistics(raw_data)