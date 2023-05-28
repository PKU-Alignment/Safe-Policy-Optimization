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
from copy import deepcopy

import torch

from safepo.algorithms import REGISTRY
from safepo.common import multi_processing_utils
from safepo.common.experiment_analysis import EnvironmentEvaluator
from safepo.common.logger import setup_logger_kwargs
from safepo.common.utils import get_defaults_kwargs_yaml, save_eval_kwargs


class Runner(object):

    def __init__(self,
                 algo: str,
                 env_id: str,
                 log_dir: str,
                 seed: int,
                 unparsed_args: list = ()):
        '''Initial Parameters.'''

        self.algo = algo
        self.env_id = env_id
        self.configs = get_defaults_kwargs_yaml(algo=algo, env_id=env_id)
        # update algorithm kwargs with unparsed arguments from command line
        keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
        values = [eval(v) for v in unparsed_args[1::2]]
        unparsed_dict = {k: v for k, v in zip(keys, values)}
        self.configs.update(**unparsed_dict)
        # e.g. Safexp-PointGoal1-v0-ppo
        self.exp_name = '-'.join([self.env_id, self.algo, "seed-" + str(seed)])
        self.logger_kwargs = setup_logger_kwargs(base_dir=log_dir,
                                                 exp_name=self.exp_name,
                                                 seed=seed)
        self.configs.update({
            'algo': algo,
            'env_id': env_id,
            'seed': seed
        })

        self.configs.update({'logger_kwargs': self.logger_kwargs})

        # save_eval_kwargs(self.kwargs['log_dir'], self.kwargs)

        # assigned by class methods
        # self.model = None
        # self.env = None
        # self.scheduler = None

    def _evaluate_model(self):
        evaluator = EnvironmentEvaluator(log_dir=self.logger_kwargs['log_dir'])
        evaluator.eval(env=self.env, ac=self.model, num_evaluations=128)
        # Close opened files to avoid number of open files overflow
        evaluator.close()

    def _load_torch_safe(self):
        self.model = torch.load("./model.pt")

    # don't use, need to fix
    def _fill_scheduler(self, target_fn) -> None:
        """Create tasks for multi-process execution. This method is called when
        model.compile(individual_processes=True) is enabled.
        """
        ts = list()
        for task_number in range(1, self.num_runs + 1):
            kwargs = self.kwargs.copy()
            _seed = task_number + self.init_seed
            logger_kwargs = setup_logger_kwargs(base_dir=self.log_dir,
                                                exp_name=self.exp_name,
                                                seed=_seed,
                                                use_tensor_board=True,
                                                verbose=False)
            kwargs.update(logger_kwargs=logger_kwargs,
                          seed=_seed,
                          algo=self.algo,
                          env_id=self.env_id)
            t = multi_processing_utils.Task(
                id=_seed,
                target_function=target_fn,
                kwargs=kwargs)
            ts.append(t)
        self.scheduler.fill(tasks=ts)
        # adjust number of cores if num_runs < num_cores
        self.scheduler.num_cores = min(self.num_runs, self.scheduler.num_cores)

    @classmethod
    def _run_mp_training(self, cls, **kwargs):

        algo = kwargs.pop('algo')
        env_id = kwargs.pop('env_id')
        logger_kwargs = kwargs.pop('logger_kwargs')
        algo = REGISTRY[self.algo](env_id=env_id,**kwargs)
        evaluator = EnvironmentEvaluator(log_dir=logger_kwargs['log_dir'])
        ac, env = algo.learn()
        evaluator.eval(env=env, ac=ac, num_evaluations=128)
        evaluator.close()

    def _eval_once(self, actor_critic, env, render) -> tuple:
        done = False
        self.env.render() if render else None
        x = self.env.reset()
        ret = 0.
        costs = 0.
        episode_length = 0
        while not done:
            self.env.render() if render else None
            obs = torch.as_tensor(x, dtype=torch.float32)
            action, value, info = actor_critic(obs)
            x, r, done, info = env.step(action)
            costs += info.get('cost', 0)
            ret += r
            episode_length += 1
        return ret, episode_length, costs

    def eval(self, **kwargs) -> None:

        if self.multiple_individual_processes:
            # Note that Multi-process models are evaluated in _run_mp_training method.
            pass
        else:
            # Set in evaluation mode before evaluation, which is different with *torch.no_grad()*
            # More details in https://stackoverflow.com/questions/60018578/what-does-model-eval-do-in-pytorch
            self.model.eval()
            self._evaluate_model()
            self.model.train()  # switch back to train mode

    def train(self):
        """Train the agent."""
        agent = REGISTRY[self.algo](configs=self.configs)
        self.model, self.env = agent.learn()

    def play(self) -> None:
        """ Visualize model after training."""
        #assert self.trained, 'Call model.train() before model.play()'
        # self.eval(episodes=5, render=True)

        env_id = self.env_id
        epochs = self.kwargs.pop('epochs')
        defaults = deepcopy(self.kwargs)
        defaults.update(epochs=epochs)
        defaults.update(logger_kwargs=self.logger_kwargs)
        algo = REGISTRY[self.algo](env_id=env_id,**defaults)
        return algo


    def summary(self):
        """ print nice outputs to console."""
        raise NotImplementedError
