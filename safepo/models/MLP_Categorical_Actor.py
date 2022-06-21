from torch.distributions.categorical import Categorical
from safepo.models.Actor import Actor
from safepo.models.model_utils import build_mlp_network

class MLPCategoricalActor(Actor):

    def __init__(self, obs_dim, act_dim, hidden_sizes, activation,
                 weight_initialization, shared=None):
        super().__init__(obs_dim, act_dim, weight_initialization, shared=shared)
        if shared is not None:
            raise NotImplementedError
        self.net = build_mlp_network(
            [obs_dim] + list(hidden_sizes) + [act_dim],
            activation=activation,
            weight_initialization=weight_initialization
        )

    def dist(self, obs):
        
        logits = self.net(obs)
        return Categorical(logits=logits)

    def log_prob_from_dist(self, pi, act):

        return pi.log_prob(act)

    def sample(self, obs):

        dist = self.dist(obs)
        a = dist.sample()
        logp_a = self.log_prob_from_dist(dist, a)

        return a, logp_a