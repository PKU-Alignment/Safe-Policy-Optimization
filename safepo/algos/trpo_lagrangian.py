import torch
from safepo.algos.trpo import TRPO
from safepo.algos.lagrangian_base import Lagrangian


class TRPO_Lagrangian(TRPO,Lagrangian):
    def __init__(
            self,
            alg: str = 'trpo_lagrangian',
            cost_limit: float = 25.0,
            lagrangian_multiplier_init: float = 0.001,
            lambda_lr: float = 0.05,
            lambda_optimizer: str = 'SGD',
            use_lagrangian_penalty: bool = True,
            **kwargs
    ):
        TRPO.__init__(
            self,
            alg=alg,
            cost_limit=cost_limit,
            lagrangian_multiplier_init=lagrangian_multiplier_init,
            lambda_lr=lambda_lr,
            lambda_optimizer=lambda_optimizer,
            use_cost_value_function=True,
            use_kl_early_stopping=False,
            use_lagrangian_penalty=use_lagrangian_penalty,
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

    def algorithm_specific_logs(self):
        super().algorithm_specific_logs()
        self.logger.log_tabular('LagrangeMultiplier',
                                self.lagrangian_multiplier.item())

    def compute_loss_pi(self, data: dict) -> tuple:
        # Policy loss
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        loss_pi = -(ratio * data['adv']).mean()
        loss_pi -= self.entropy_coef * dist.entropy().mean()

        if self.use_lagrangian_penalty:
            # ensure that lagrange multiplier is positive
            penalty = torch.clamp_min(self.lagrangian_multiplier,0.0)
            loss_pi += penalty * (ratio * data['cost_adv']).mean()
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
        # sub-sampling accelerates calculations
        self.fvp_obs = data['obs'][::4]
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