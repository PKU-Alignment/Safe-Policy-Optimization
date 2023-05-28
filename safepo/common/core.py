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
import numpy as np
import scipy.signal
import torch
import torch.nn as nn
import torch.optim as optim

registered_actors = dict()  # global dict that holds pointers to functions


def get_optimizer(opt: str, module: torch.nn.Module, lr: float):
    """
    Returns an initialized optimizer from PyTorch.
    """
    assert hasattr(optim, opt), f"Optimizer={opt} not found in torch."
    optimizer = getattr(optim, opt)

    return optimizer(module.parameters(), lr=lr)


def initialize_layer(init_function: str, layer: torch.nn.Module):
    if init_function == "kaiming_uniform":  # this the default!
        nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
    elif init_function == "xavier_normal":
        nn.init.xavier_normal_(layer.weight)
    # glorot is also known as xavier uniform
    elif init_function == "glorot" or init_function == "xavier_uniform":
        nn.init.xavier_uniform_(layer.weight)
    elif init_function == "orthogonal":  # matches values from baselines repo.
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
    else:
        raise NotImplementedError


def register_actor(actor_name):
    """
    register actor into global dict
    """

    def wrapper(func):
        registered_actors[actor_name] = func
        return func

    return wrapper


def get_registered_actor_fn(actor_type: str, distribution_type: str):
    assert distribution_type == "categorical" or distribution_type == "gaussian"
    actor_fn = actor_type + "_" + distribution_type
    msg = f"Did not find: {actor_fn} in registered actors."
    assert actor_fn in registered_actors, msg
    return registered_actors[actor_fn]


def combined_shape(length: int, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)


def count_vars(module):
    return sum([np.prod(p.shape) for p in module.parameters()])


def discount_cumsum(x, discount):
    """
    magic from rllab for computing discounted cumulative sums of vectors.
    input:
        vector x,
        [x0,
         x1,
         x2]
    output:
        [x0 + discount * x1 + discount^2 * x2,
         x1 + discount * x2,
         x2]
    """
    return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
