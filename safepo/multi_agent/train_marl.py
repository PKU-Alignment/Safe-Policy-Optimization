from marl_utils.config import (get_args, load_cfg,
                               set_np_formatting, set_seed)
from marl_utils.process_marl import process_MultiAgentRL
from marl_utils.process_sarl import *

import torch
from safety_gymnasium.tasks.masafe.mujoco_multi import MujocoMulti
from safety_gymnasium.tasks.masafe.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env_args = {"scenario": all_args.scenario,
                        "agent_conf": all_args.agent_conf,
                        "agent_obsk": all_args.agent_obsk,
                        "episode_limit": 1000}
            env = MujocoMulti(env_args=env_args)
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    if all_args.n_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            env_args = {"scenario": all_args.scenario,
                        "agent_conf": all_args.agent_conf,
                        "agent_obsk": all_args.agent_obsk,
                        "episode_limit": 1000}
            env = MujocoMulti(env_args=env_args)
            env.seed(all_args.seed * 50000 + rank * 10000)
            return env

        return init_env

    if all_args.n_eval_rollout_threads == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(all_args.n_eval_rollout_threads)])


def train():
    print("Algorithm: ", args.algo)
    # Agent: 4x3
    # agent_index = [[[0, 1, 2],[ 3, 4, 5]],
    #                [[0, 1, 2],[ 3, 4, 5]]]
    # Agent: 2x6

    if args.algo in ["mappo", "happo", "ippo", "macpo", "mappolag"]:
        # maddpg exists a bug now
        args.task_type = "MultiAgent"
        torch.set_num_threads(4)

        env = make_train_env(args)
        eval_env = make_eval_env(args)

        runner = process_MultiAgentRL(args, env, eval_env, config=cfg_train, model_dir=args.model_dir)

        if args.model_dir != "":
            runner.eval(100000)
        else:
            print("Start Training")
            runner.run()

    else:
        print("Unrecognized algorithm!\nAlgorithm should be one of: [happo, hatrpo, mappo,ippo,maddpg,sac,td3,trpo,ppo,ddpg]")


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    train()