import torch
import torch.nn as nn
from safepo.models.model_utils import build_mlp_network

class Critic(nn.Module):

    def __init__(self, obs_dim, hidden_sizes, activation, shared=None):
        super().__init__()
        if shared is None:
            self.net = build_mlp_network([obs_dim] + list(hidden_sizes) + [1],
                                           activation=activation)
        else:  # use shared layers
            value_head = nn.Linear(hidden_sizes[-1], 1)
            self.net = nn.Sequential(shared, value_head, nn.Identity())

    def forward(self, obs):
        return torch.squeeze(self.net(obs),-1)
