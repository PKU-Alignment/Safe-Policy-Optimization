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

from safepo.algorithms import REGISTRY
from safepo.common.logger import setup_logger_kwargs
from safepo.common.utils import get_defaults_kwargs_yaml


class Runner(object):
    def __init__(
        self, algo: str, env_id: str, log_dir: str, seed: int, unparsed_args: list = ()
    ):
        """Initial Parameters."""
        self.algo = algo
        self.env_id = env_id
        self.configs = get_defaults_kwargs_yaml(algo=algo, env_id=env_id)
        self.configs.update({"algo": algo, "env_id": env_id, "seed": seed})
        # update algorithm kwargs with unparsed arguments from command line
        keys = [k[2:] for k in unparsed_args[0::2]]  # remove -- from argument
        values = [eval(v) for v in unparsed_args[1::2]]
        unparsed_dict = {k: v for k, v in zip(keys, values)}
        self.configs.update(**unparsed_dict)
        # e.g. Safexp-PointGoal1-v0-ppo
        self.exp_name = "-".join([self.env_id, self.algo, "seed-" + str(seed)])
        self.logger_kwargs = setup_logger_kwargs(
            base_dir=log_dir, exp_name=self.exp_name, seed=seed
        )

        self.configs.update({"logger_kwargs": self.logger_kwargs})

    def train(self):
        """Train the agent."""
        agent = REGISTRY[self.algo](configs=self.configs)
        self.agent, self.env = agent.learn()

    # def _evaluate_model(self):
    #     evaluator = EnvironmentEvaluator(log_dir=self.logger_kwargs['log_dir'])
    #     evaluator.eval(env=self.env, ac=self.model, num_evaluations=128)
    #     # Close opened files to avoid number of open files overflow
    #     evaluator.close()

    # def _eval_once(self, actor_critic, env, render) -> tuple:
    #     done = False
    #     self.env.render() if render else None
    #     x = self.env.reset()
    #     ret = 0.
    #     costs = 0.
    #     episode_length = 0
    #     while not done:
    #         self.env.render() if render else None
    #         obs = torch.as_tensor(x, dtype=torch.float32)
    #         action, value, info = actor_critic(obs)
    #         x, r, done, info = env.step(action)
    #         costs += info.get('cost', 0)
    #         ret += r
    #         episode_length += 1
    #     return ret, episode_length, costs

    # def eval(self, **kwargs) -> None:

    #     self.model.eval()
    #     self._evaluate_model()
    #     self.model.train()  # switch back to train mode

    # def play(self) -> None:
    #     """ Visualize model after training."""
    #     env_id = self.env_id
    #     epochs = self.kwargs.pop('epochs')
    #     defaults = deepcopy(self.kwargs)
    #     defaults.update(epochs=epochs)
    #     defaults.update(logger_kwargs=self.logger_kwargs)
    #     algo = REGISTRY[self.algo](env_id=env_id,**defaults)
    #     return algo
