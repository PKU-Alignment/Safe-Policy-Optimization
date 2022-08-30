import torch
import torch.nn as nn
import numpy as np
def initialize_layer(
        init_function: str,
        layer: torch.nn.Module
):
    if init_function == 'kaiming_uniform':  # this the default!
        nn.init.kaiming_uniform_(layer.weight, a=np.sqrt(5))
    elif init_function == 'xavier_normal':
        nn.init.xavier_normal_(layer.weight)
    # glorot is also known as xavier uniform
    elif init_function == 'glorot' or init_function == 'xavier_uniform':
        nn.init.xavier_uniform_(layer.weight)
    elif init_function == 'orthogonal':  # matches values from baselines repo.
        nn.init.orthogonal_(layer.weight, gain=np.sqrt(2))
    else:
        raise NotImplementedError

def convert_str_to_torch_functional(activation):
    if isinstance(activation, str):  # convert string to torch functional
        activations = {
            'identity': nn.Identity,
            'relu': nn.ReLU,
            'sigmoid': nn.Sigmoid,
            'softplus': nn.Softplus,
            'tanh': nn.Tanh
        }
        assert activation in activations
        activation = activations[activation]
    assert issubclass(activation, torch.nn.Module)
    return activation

def build_mlp_network(
        sizes,
        activation,
        output_activation='identity',
        weight_initialization='kaiming_uniform'
):
    activation = convert_str_to_torch_functional(activation)
    output_activation = convert_str_to_torch_functional(output_activation)
    layers = list()
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        affine_layer = nn.Linear(sizes[j], sizes[j + 1])
        initialize_layer(weight_initialization, affine_layer)
        layers += [affine_layer, act()]
    return nn.Sequential(*layers)