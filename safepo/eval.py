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
import argparse
import warnings
from safepo.algos import REGISTRY
import os
import yaml
import torch
import numpy as np
import gym
from gym.utils.save_video import save_video
# from gym.wrappers import Monitor
from safepo.common.logger import setup_logger_kwargs

try:
    import safety_gym
except ImportError:
    warnings.warn('safety_gym package not found.')

try:
    import bullet_safety_gym
except ImportError:
    warnings.warn('Bullet-Safety-Gym package not found.')

def get_defaults_kwargs_yaml(path):
    path = os.path.join('/', *path.split('/')[:-1]) if path[0] == '/' else os.path.join(*path.split('/')[:-1])
    path = os.path.join(path, 'eval_kwargs.yaml')
    with open(path, "r") as f:
        try:
            kwargs = yaml.load(f, Loader=yaml.FullLoader)
        except yaml.YAMLError as exc:
            assert False, "{}.yaml error: {}".format('eval_kwargs', exc)
    return kwargs

if __name__ == '__main__':
    """
        It is used to visulize your model after trainning,
        *./mp4 files will be saved in your model directory.

        Note: In this file, We use the latest interface from gym==0.26.1,
        It does not supported by current Safepo. Currently,
        It is a testing feature in our project, and it ***only
        supports Safety-gym environments***, We will soon support
        gym environmrnts.
    """

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument('--path', type=str, required=True,
                        help='The path to save models which you want to visulize.')
    parser.add_argument('--model-name', type=str,
                        help='The single model name which you want to visulize.')
    parser.add_argument('--seed', default=0, type=int,
                        help='The seed for testing.')
    parser.add_argument('--all', action='store_true',
                        help='The signal which specify wheather you want to visulize all models in the path folder.')
    parser.add_argument('--eval_ep', type=int, default=10,
                        help='The number of visulized episodes for a single model.')
    parser.add_argument('--width', type=int, default=512,
                        help='The movies width.')
    parser.add_argument('--height', type=int, default=512,
                        help='The movies height.')

    args, unparsed_args = parser.parse_known_args()
    default_log_dir = os.path.join(args.path, "./evals")

    assert args.model_name is not None or args.all, "Please specify the way you want to eval models."

    path = args.path
    kwargs = get_defaults_kwargs_yaml(path)
    if kwargs['seed'] != args.seed:
        print("Warning: The seed for evaluation is not same with seed for trainning.")
    kwargs.update({'seed': args.seed})

    seed = kwargs['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)

    algo = kwargs['algo']
    kwargs.pop('algo')
    kwargs.update({
        'logger_kwargs': setup_logger_kwargs(base_dir='', exp_name='', seed=0),
        'enable_eval': True,
    })

    model = REGISTRY[algo](kwargs['env_id'],**kwargs)

    render_kwargs = {
        'mode': 'rgb_array',
        'width': args.width,
        'height': args.height,
    }

    if args.all:
        for item in os.scandir(path):
            if item.is_file() and item.name.split('.')[-1] == 'pt':

                model.ac.pi.load_state_dict(torch.load(os.path.join(path, item.name), map_location=torch.device('cpu')))
                env = gym.make(kwargs['env_id'])
                env.seed(seed=seed)
                out_dir = os.path.join(default_log_dir, item.name.split('.')[0])

                for i in range(args.eval_ep):
                    render_list = []
                    o = env.reset()
                    render_list.append(env.render(**render_kwargs))
                    d = False

                    while not d:
                        a, v, cv, logp = model.ac.step(
                        torch.as_tensor(o, dtype=torch.float32))
                        o, r, d, env_info = env.step(a)

                        render_list.append(env.render(**render_kwargs))

                    save_video(
                        frames=render_list,
                        video_folder=out_dir,
                        name_prefix=f'eval_{i}',
                        fps=30,
                    )

                env.close()
    else:

        model.ac.pi.load_state_dict(torch.load(os.path.join(path, args.model_name + '.pt'), map_location=torch.device('cpu')))
        env = gym.make(kwargs['env_id'])
        out_dir = os.path.join(default_log_dir, args.model_name)
        env.seed(seed=seed)

        for i in range(args.eval_ep):
            render_list = []
            o = env.reset()
            render_list.append(env.render(**render_kwargs))
            d = False

            while not d:
                a, v, cv, logp = model.ac.step(
                torch.as_tensor(o, dtype=torch.float32))
                o, r, d, env_info = env.step(a)

                render_list.append(env.render(**render_kwargs))

            save_video(
                frames=render_list,
                video_folder=out_dir,
                name_prefix=f'eval_{i}',
                fps=30,
            )

        env.close()
