import torch
import torch.nn.functional as F
from safepo.algos.policy_graident import PG
class P3O(PG):
    def __init__(
            self,
            algo: str = 'p3o',
            cost_limit: float = 25.,
            clip: float = 0.2,
            kappa: float = 20.0,
            use_standardized_advantages: bool = True,
            **kwargs
    ):
        super().__init__(
            algo=algo,
            use_standardized_advantages=use_standardized_advantages,
            use_kl_early_stopping=True,
            use_cost_value_function=True,
            **kwargs
        )
        self.clip = clip
        self.cost_limit = cost_limit
        self.kappa = kappa

    def compute_loss_pi(self, data: dict) -> tuple:
        dist, _log_p = self.ac.pi(data['obs'], data['act'])
        ratio = torch.exp(_log_p - data['log_p'])

        ratio_clip = torch.clamp(ratio, 1-self.clip, 1+self.clip)

        ratio = torch.min(ratio_clip, ratio)
        surr_adv = (ratio * data['adv']).mean()
        surr_cadv = (ratio * data['cost_adv']).mean()
        ep_costs = self.logger.get_stats('EpCosts')[0]
        c = ep_costs - self.cost_limit
        loss_pi = -surr_adv + self.kappa * F.relu(surr_cadv + c)
        loss_pi = loss_pi.mean()

        # Useful extra info
        approx_kl = (0.5 * (dist.mean - data['act']) ** 2
                     / dist.stddev ** 2).mean().item()
        ent = dist.entropy().mean().item()
        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio_clip.mean().item())

        return loss_pi, pi_info