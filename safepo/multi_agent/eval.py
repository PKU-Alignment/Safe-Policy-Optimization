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


from marl_utils.safety_gymnasium_config import (get_args, load_cfg,
                               set_np_formatting)
from marl_utils.process_sg_marl import process_MultiAgentRL
from marl_utils.process_sarl import *

import torch
import numpy as np
import os
from safepo.multi_agent.marl_utils.env_wrappers import  ShareDummyVecEnv, ShareEnv


def make_train_env(env_cfg):
    def get_env_fn(rank):
        def init_env():
            env=ShareEnv(
                scenario=env_cfg['scenario'],
                agent_conf=env_cfg['agent_conf'],
                agent_obsk=env_cfg['agent_obsk'],
                render_mode="rgb_array",
            )
            env.reset(seed=env_cfg['seed'] + rank * 1000)
            return env

        return init_env
    return ShareDummyVecEnv([get_env_fn(0)])


def make_eval_env(env_cfg):
    def get_env_fn(rank):
        def init_env():
            env=ShareEnv(
                scenario=env_cfg['scenario'],
                agent_conf=env_cfg['agent_conf'],
                agent_obsk=env_cfg['agent_obsk'],
                render_mode="rgb_array",
            )
            env.reset(seed=env_cfg['seed'] + rank * 1000)
            return env

        return init_env
    return ShareDummyVecEnv([get_env_fn(0)])

def train():

    if args.algo in ["mappo", "happo", "ippo", "macpo", "mappolag"]:
        args.task_type = "MultiAgent"
        torch.set_num_threads(4)

        env = make_train_env(cfg_train)
        eval_env = make_eval_env(cfg_train)

        runner = process_MultiAgentRL(args, env, eval_env, config=cfg_train, model_dir=args.model_dir)
        episode_rewards, episode_costs = [], []
        for _ in range(3):
            rew, cost=runner.eval(100000)
            episode_rewards.append(rew)
            episode_costs.append(cost)
        return episode_rewards, episode_costs
    else:
        print("Unrecognized algorithm!\nAlgorithm should be one of: [happo, hatrpo, mappo,ippo,maddpg,sac,td3,trpo,ppo,ddpg]")


if __name__ == '__main__':
    set_np_formatting()
    base_dir = '/home/jiayi/zjy_dev/Safe-Policy-Optimization/safepo/multi_agent/runs/benchmark_velocity_5.0'
    save_dir = base_dir.replace('runs', 'results')
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    for env_name in os.listdir(base_dir):
        env_dir = os.path.join(base_dir, env_name)
        with open(f'{save_dir}/eval_result.txt','a') as f:
            f.write('\n'+'\\textsc{' + env_name + '} ')
        for algo_name in os.listdir(env_dir):
            algo_dir = os.path.join(env_dir, algo_name)
            reward_list = []
            cost_list = []
            for seed in os.listdir(algo_dir):
                seed_path = os.path.join(algo_dir, seed)
                if os.path.isdir(seed_path) and 'model' in seed_path:
                    print("Algorithm: ", algo_name, "Env: ", env_name, "Seed: ", seed)
                    args = get_args(
                        algo=algo_name,
                        task=env_name,
                        model_dir=seed_path,
                        seed=np.random.randint(0, 1000000),
                    )
                    cfg, cfg_train, logdir = load_cfg(args)
                    curr_reward_list, curr_cost_list = train()
                    reward_list=reward_list+curr_reward_list
                    cost_list=cost_list+curr_cost_list
            rew_mean = np.mean(reward_list)
            rew_std = np.std(reward_list)
            cost_mean = np.mean(cost_list)
            cost_std = np.std(cost_list)
            with open(f'{save_dir}/eval_result.txt','a') as f:
                f.write(f'& {rew_mean:.2f} $\pm$ {rew_std:.2f} & {cost_mean:.2f} $\pm$ {cost_std:.2f} ')
                