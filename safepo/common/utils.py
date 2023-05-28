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
import os

import random
import numpy as np
import torch
import yaml

def get_defaults_kwargs_yaml(algo, env_id):
    path = os.path.abspath(__file__).split('/')[:-2]
    cfg_path = os.path.join('/', *path,'configs',"{}.yaml".format(algo))
    with open(cfg_path, "r") as f:
        try:
            kwargs = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format(algo, exc)
    kwargs_name = env_id if env_id in kwargs.keys() else 'defaults'
    return kwargs[kwargs_name]

def save_eval_kwargs(log_dir, eval_kwargs):
    """To save eval kwargs."""
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, 'eval_kwargs.yaml')
    with open(path, "w") as f:
        yaml.dump(eval_kwargs, f)

def get_flat_params_from(model):
    """Get all model parameters as a single vector."""
    flat_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            d = param.data
            d = d.view(-1)  # flatten tensor
            flat_params.append(d)
    assert flat_params is not [], 'No gradients were found in model parameters.'

    return torch.cat(flat_params)

def get_flat_gradients_from(model):
    """Get all model gradients as a single vector."""
    grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            g = param.grad
            grads.append(g.view(-1))  # flatten tensor and append
    assert grads is not [], 'No gradients were found in model parameters.'

    return torch.cat(grads)

def conjugate_gradients(Avp, b, nsteps, residual_tol=1e-10, eps=1e-6):
    """Conjugate gradient algorithm.
        see https://en.wikipedia.org/wiki/Conjugate_gradient_method
    """
    x = torch.zeros_like(b)
    r = b - Avp(x)
    p = r.clone()
    rdotr = torch.dot(r, r)

    fmtstr = "%10i %10.3g %10.3g"
    titlestr = "%10s %10s %10s"
    verbose = False

    for i in range(nsteps):
        if verbose: print(fmtstr % (i, rdotr, np.linalg.norm(x)))
        z = Avp(p)
        alpha = rdotr / (torch.dot(p, z) + eps)
        x += alpha * p
        r -= alpha * z
        new_rdotr = torch.dot(r, r)
        if torch.sqrt(new_rdotr) < residual_tol:
            break
        mu = new_rdotr / (rdotr + eps)
        p = r + mu * p
        rdotr = new_rdotr

    return x

def set_param_values_to_model(model, vals):
    assert isinstance(vals, torch.Tensor)
    i = 0
    for name, param in model.named_parameters():
        if param.requires_grad:  # param has grad and, hence, must be set
            orig_size = param.size()
            size = np.prod(list(param.size()))
            new_values = vals[i:i + size]
            # set new param values
            new_values = new_values.view(orig_size)
            param.data = new_values
            i += size  # increment array position
    assert i == len(vals), f'Lengths do not match: {i} vs. {len(vals)}'


def seed_everything(seed: int) -> None:
    """Set global random seed for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
