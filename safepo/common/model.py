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


from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from torch.distributions import Normal
from safepo.utils.act import ACTLayer
from safepo.utils.mlp import MLPBase
from safepo.utils.util import check, init
from safepo.utils.util import get_shape_from_obs_space


def build_mlp_network(sizes):
    """
    Build a multi-layer perceptron (MLP) neural network.

    This function constructs an MLP network with the specified layer sizes and activation functions.

    Args:
        sizes (list of int): List of integers representing the sizes of each layer in the network.

    Returns:
        nn.Sequential: An instance of PyTorch's Sequential module representing the constructed MLP.
    """
    layers = list()
    for j in range(len(sizes) - 1):
        act = nn.Tanh if j < len(sizes) - 2 else nn.Identity
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        nn.init.kaiming_uniform_(affine_layer.weight, a=np.sqrt(5))
        layers += [affine_layer, act()]
    return nn.Sequential(*layers)


class Actor(nn.Module):
    """
    Actor network for policy-based reinforcement learning.

    This class represents an actor network that outputs a distribution over actions given observations.

    Args:
        obs_dim (int): Dimensionality of the observation space.
        act_dim (int): Dimensionality of the action space.

    Attributes:
        mean (nn.Sequential): MLP network representing the mean of the action distribution.
        log_std (nn.Parameter): Learnable parameter representing the log standard deviation of the action distribution.

    Example:
        obs_dim = 10
        act_dim = 2
        actor = Actor(obs_dim, act_dim)
        observation = torch.randn(1, obs_dim)
        action_distribution = actor(observation)
    """

    def __init__(self, obs_dim: int, act_dim: int):
        super().__init__()
        self.mean = build_mlp_network([obs_dim, 256, 256, act_dim])
        self.log_std = nn.Parameter(torch.zeros(act_dim), requires_grad=True)

    def forward(self, obs: torch.Tensor):
        mean = self.mean(obs)
        std = torch.exp(self.log_std)
        return Normal(mean, std)


class VCritic(nn.Module):
    """
    Critic network for value-based reinforcement learning.

    This class represents a critic network that estimates the value function for input observations.

    Args:
        obs_dim (int): Dimensionality of the observation space.

    Attributes:
        critic (nn.Sequential): MLP network representing the critic function.

    Example:
        obs_dim = 10
        critic = VCritic(obs_dim)
        observation = torch.randn(1, obs_dim)
        value_estimate = critic(observation)
    """

    def __init__(self, obs_dim):
        super().__init__()
        self.critic = build_mlp_network([obs_dim, 256, 256, 1])

    def forward(self, obs):
        return torch.squeeze(self.critic(obs), -1)


class ActorVCritic(nn.Module):
    """
    Actor-critic policy for reinforcement learning.

    This class represents an actor-critic policy that includes an actor network, two critic networks for reward
    and cost estimation, and provides methods for taking policy steps and estimating values.

    Args:
        obs_dim (int): Dimensionality of the observation space.
        act_dim (int): Dimensionality of the action space.

    Example:
        obs_dim = 10
        act_dim = 2
        actor_critic = ActorVCritic(obs_dim, act_dim)
        observation = torch.randn(1, obs_dim)
        action, log_prob, reward_value, cost_value = actor_critic.step(observation)
        value_estimate = actor_critic.get_value(observation)
    """

    def __init__(self, obs_dim, act_dim):
        super().__init__()
        self.reward_critic = VCritic(obs_dim)
        self.cost_critic = VCritic(obs_dim)
        self.actor = Actor(obs_dim, act_dim)

    def get_value(self, obs):
        """
        Estimate the value of observations using the critic network.

        Args:
            obs (torch.Tensor): Input observation tensor.

        Returns:
            torch.Tensor: Estimated value for the input observation.
        """
        return self.critic(obs)

    def step(self, obs, deterministic=False):
        """
        Take a policy step based on observations.

        Args:
            obs (torch.Tensor): Input observation tensor.
            deterministic (bool): Flag indicating whether to take a deterministic action.

        Returns:
            tuple: Tuple containing action tensor, log probabilities of the action, reward value estimate,
                   and cost value estimate.
        """

        dist = self.actor(obs)
        if deterministic:
            action = dist.mean
        else:
            action = dist.rsample()
        log_prob = dist.log_prob(action).sum(axis=-1)
        value_r = self.reward_critic(obs)
        value_c = self.cost_critic(obs)
        return action, log_prob, value_r, value_c

class MultiAgentActor(nn.Module):
    """
    Multi-agent actor network for reinforcement learning.

    This class represents a multi-agent actor network that takes observations as input and produces actions and
    action probabilities as outputs. It includes options for using recurrent layers and policy active masks.

    Args:
        config (dict): Configuration parameters for the actor network.
        obs_space: Observation space of the environment.
        action_space: Action space of the environment.
        device (torch.device): Device to run the network on (default is "cpu").

    Attributes:
        hidden_size (int): Size of the hidden layers.
        config (dict): Configuration parameters for the actor network.
        _gain (float): Gain factor for action scaling.
        _use_orthogonal (bool): Flag indicating whether to use orthogonal initialization.
        _use_policy_active_masks (bool): Flag indicating whether to use policy active masks.
        _use_naive_recurrent_policy (bool): Flag indicating whether to use naive recurrent policy.
        _use_recurrent_policy (bool): Flag indicating whether to use recurrent policy.
        _recurrent_N (int): Number of recurrent layers.
        tpdv (dict): Dictionary with data type and device for tensor conversion.
        
    Example:
        config = {"hidden_size": 256, "gain": 0.1, ...}
        obs_space = gym.spaces.Box(low=0, high=1, shape=(4,))
        action_space = gym.spaces.Discrete(2)
        actor = MultiAgentActor(config, obs_space, action_space)
        observation = torch.randn(1, 4)
        rnn_states = torch.zeros(1, 256)
        masks = torch.ones(1, 1)
        actions, action_log_probs, new_rnn_states = actor(observation, rnn_states, masks)
        action = torch.tensor([0])
        action_log_probs, dist_entropy = actor.evaluate_actions(observation, rnn_states, action, masks)
    """

    def __init__(self, config, obs_space, action_space, device=torch.device("cpu")):
        super(MultiAgentActor, self).__init__()
        self.hidden_size = config["hidden_size"]
        self.config=config
        self._gain = config["gain"]
        self._use_orthogonal = config["use_orthogonal"]
        self._use_policy_active_masks = config["use_policy_active_masks"]
        self._use_naive_recurrent_policy = config["use_naive_recurrent_policy"]
        self._use_recurrent_policy = config["use_recurrent_policy"]
        self._recurrent_N = config["recurrent_N"]
        self.tpdv = dict(dtype=torch.float32, device=device)

        obs_shape = get_shape_from_obs_space(obs_space)
        base =  MLPBase
        self.base = base(self.config, obs_shape)
        self.act = ACTLayer(action_space, self.hidden_size, self._use_orthogonal, self._gain, self.config)

        self.to(device)

    def forward(self, obs, rnn_states, masks, available_actions=None, deterministic=False):
        """
        Perform a forward pass through the network to generate actions and log probabilities.

        Args:
            obs (torch.Tensor): Input observation tensor.
            rnn_states (torch.Tensor): Recurrent states tensor.
            masks (torch.Tensor): Mask tensor.
            available_actions (torch.Tensor, optional): Available actions tensor (default: None).
            deterministic (bool, optional): Flag indicating whether to take deterministic actions (default: False).

        Returns:
            tuple: Tuple containing action tensor, log probability tensor, and new recurrent states tensor.
        """

        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        actor_features = self.base(obs)
        actions, action_log_probs = self.act(actor_features, available_actions, deterministic)

        return actions, action_log_probs, rnn_states

    def evaluate_actions(self, obs, rnn_states, action, masks, available_actions=None, active_masks=None):
        """
        Evaluate the actions based on the network's policy.

        Args:
            obs (torch.Tensor): Input observation tensor.
            rnn_states (torch.Tensor): Recurrent states tensor.
            action (torch.Tensor): Action tensor.
            masks (torch.Tensor): Mask tensor.
            available_actions (torch.Tensor, optional): Available actions tensor (default: None).
            active_masks (torch.Tensor, optional): Active masks tensor (default: None).

        Returns:
            tuple: Tuple containing action log probabilities tensor, distribution entropy tensor,
                   action mean tensor, action standard deviation tensor, and other optional tensors.
        """
        obs = check(obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        action = check(action).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)
        if available_actions is not None:
            available_actions = check(available_actions).to(**self.tpdv)

        if active_masks is not None:
            active_masks = check(active_masks).to(**self.tpdv)

        actor_features = self.base(obs)

        if self.config["algorithm_name"]== "macpo":
            action_log_probs, dist_entropy, action_mu, action_std, _ = self.act.evaluate_actions_trpo(actor_features,
                                                                                                   action,
                                                                                                   available_actions,
                                                                                                   active_masks=
                                                                                                   active_masks if self._use_policy_active_masks
                                                                                                   else None)
            return action_log_probs, dist_entropy, action_mu, action_std

        else:
            action_log_probs, dist_entropy = self.act.evaluate_actions(actor_features,
                                                                    action, available_actions,
                                                                    active_masks=
                                                                    active_masks if self._use_policy_active_masks
                                                                    else None)

            return action_log_probs, dist_entropy


class MultiAgentCritic(nn.Module):
    """
    Multi-agent critic network.

    This class represents a multi-agent critic network used in reinforcement learning algorithms.
    It consists of a base network (CNN or MLP), recurrent layers (if applicable), and a value output layer.

    Args:
        config (dict): Configuration dictionary.
        cent_obs_space (gym.spaces.Space): Centralized observation space.
        device (torch.device): Device to use for computations (default: cuda:0).

    Attributes:
        hidden_size (int): Size of the hidden layer.
        _use_orthogonal (bool): Flag indicating whether to use orthogonal initialization.
        _use_naive_recurrent_policy (bool): Flag indicating whether to use naive recurrent policy.
        _use_recurrent_policy (bool): Flag indicating whether to use recurrent policy.
        _recurrent_N (int): Number of recurrent layers.
        tpdv (dict): Dictionary for tensor properties.
    """
    
    def __init__(self, config, cent_obs_space, device=torch.device("cuda:0")):
        super(MultiAgentCritic, self).__init__()
        self.hidden_size = config["hidden_size"]
        self._use_orthogonal = config["use_orthogonal"]
        self._use_naive_recurrent_policy = config["use_naive_recurrent_policy"]
        self._use_recurrent_policy = config["use_recurrent_policy"]
        self._recurrent_N = config["recurrent_N"]
        self.tpdv = dict(dtype=torch.float32, device=device)
        init_method = [nn.init.xavier_uniform_, nn.init.orthogonal_][self._use_orthogonal]

        cent_obs_shape = get_shape_from_obs_space(cent_obs_space)
        base =  MLPBase
        self.base = base(config, cent_obs_shape)

        def init_(m):
            return init(m, init_method, lambda x: nn.init.constant_(x, 0), gain=0)

        self.v_out = init_(nn.Linear(self.hidden_size, 1))

        self.to(device)

    def forward(self, cent_obs, rnn_states, masks):
        """
        Perform a forward pass through the network to compute value estimates.

        Args:
            cent_obs (torch.Tensor): Centralized observation tensor.
            rnn_states (torch.Tensor): Recurrent states tensor.
            masks (torch.Tensor): Mask tensor.

        Returns:
            tuple: Tuple containing value estimates tensor and new recurrent states tensor.
        """

        cent_obs = check(cent_obs).to(**self.tpdv)
        rnn_states = check(rnn_states).to(**self.tpdv)
        masks = check(masks).to(**self.tpdv)

        critic_features = self.base(cent_obs)
        values = self.v_out(critic_features)

        return values, rnn_states
    