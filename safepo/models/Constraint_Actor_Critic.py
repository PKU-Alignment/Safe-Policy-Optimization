import torch
from safepo.models.Critic import Critic
from safepo.models.Actor_Critic import ActorCritic

class ConstraintActorCritic(ActorCritic):
    def __init__(
            self,
            **kwargs
    ):
        super().__init__(**kwargs)
        self.c = Critic(
            obs_dim=self.obs_shape[0],
            shared=None,
            **self.ac_kwargs['val'])

    def step(self,
             obs: torch.Tensor
             ) -> tuple:
        """ Produce action, value, log_prob(action).
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
            v = self.v(obs)
            c = self.c(obs)

            if self.training:
                a, logp_a = self.pi.sample(obs)
            else:
                a, logp_a = self.pi.predict(obs)

        return a.numpy(), v.numpy(), c.numpy(), logp_a.numpy()