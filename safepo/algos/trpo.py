import torch
from safepo.algos.natural_pg import NPG
import safepo.common.mpi_tools as mpi_tools
from safepo.common.utils import get_flat_params_from, set_param_values_to_model

class TRPO(NPG):
    def __init__(
            self,
            algo: str = 'trpo',
            **kwargs
    ):
        super().__init__(algo=algo, **kwargs)

    def search_step_size(self,
                         step_dir,
                         g_flat,
                         p_dist,
                         data,
                         total_steps: int = 15,
                         decay: float = 0.8
                        ):
        """ 
            TRPO performs line-search until constraint satisfaction.
        """
        step_frac = 1.0
        _theta_old = get_flat_params_from(self.ac.pi.net)
        expected_improve = g_flat.dot(step_dir)

        # while not within_trust_region:
        for j in range(total_steps):
            new_theta = _theta_old + step_frac * step_dir
            set_param_values_to_model(self.ac.pi.net, new_theta)
            acceptance_step = j + 1

            with torch.no_grad():
                loss_pi, pi_info = self.compute_loss_pi(data=data)
                # determine KL div between new and old policy
                q_dist = self.ac.pi.dist(data['obs'])
                torch_kl = torch.distributions.kl.kl_divergence(
                    p_dist, q_dist).mean().item()
            loss_improve = self.loss_pi_before - loss_pi.item()
            # average processes....
            torch_kl = mpi_tools.mpi_avg(torch_kl)
            loss_improve = mpi_tools.mpi_avg(loss_improve)

            self.logger.log("Expected Improvement: %.3f Actual: %.3f" % (
                expected_improve, loss_improve))
            if not torch.isfinite(loss_pi):
                self.logger.log('WARNING: loss_pi not finite')
            elif loss_improve < 0:
                self.logger.log('INFO: did not improve improve <0')
            elif torch_kl > self.target_kl * 1.5:
                self.logger.log('INFO: violated KL constraint.')
            else:
                # step only if surrogate is improved and when within trust reg.
                self.logger.log(f'Accept step at i={acceptance_step}')
                break
            step_frac *= decay
        else:
            self.logger.log('INFO: no suitable step found...')
            step_dir = torch.zeros_like(step_dir)
            acceptance_step = 0

        set_param_values_to_model(self.ac.pi.net, _theta_old)

        return step_frac * step_dir, acceptance_step