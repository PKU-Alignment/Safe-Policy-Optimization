import os
import sys

import torch
from safepo.multi_agent.marl_utils.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv, ShareEnv

from safepo.multi_agent.marl_utils.safety_gymnasium_config import (get_args, load_cfg,
                               set_np_formatting, set_seed)
from safepo.multi_agent.marl_utils.process_sg_marl import process_MultiAgentRL

def make_train_env(env_cfg):
    def get_env_fn(rank):
        def init_env():
            env=ShareEnv(
                scenario=env_cfg['scenario'],
                agent_conf=env_cfg['agent_conf'],
                agent_obsk=env_cfg['agent_obsk'],
            )
            env.reset(seed=env_cfg['seed'] + rank * 1000)
            return env

        return init_env

    if env_cfg['n_rollout_threads'] == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(env_cfg['n_rollout_threads'])])


def make_eval_env(env_cfg):
    def get_env_fn(rank):
        def init_env():
            env=ShareEnv(
                scenario=env_cfg['scenario'],
                agent_conf=env_cfg['agent_conf'],
                agent_obsk=env_cfg['agent_obsk'],
            )
            env.reset(seed=env_cfg['seed']*50000 + rank * 1000)
            return env

        return init_env

    if env_cfg['n_rollout_threads'] == 1:
        return ShareDummyVecEnv([get_env_fn(0)])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(env_cfg['n_rollout_threads'])])


def train():
    print("Algorithm: ", args.algo)

    if args.algo in ["mappo", "happo", "ippo", "macpo", "mappolag"]:
        # maddpg exists a bug now
        args.task_type = "MultiAgent"
        torch.set_num_threads(4)

        env = make_train_env(cfg_train)
        eval_env = make_eval_env(cfg_train)

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
    if args.write_terminal:
        train()
    else:
        terminal_log_name = "terminal.log"
        error_log_name = "error.log"
        terminal_log_name = f"seed{args.seed}_{terminal_log_name}"
        error_log_name = f"seed{args.seed}_{error_log_name}"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(cfg_train['log_dir']):
            os.makedirs(cfg_train['log_dir'], exist_ok=True)
        with open(
            os.path.join(
                f"{cfg_train['log_dir']}",
                terminal_log_name,
            ),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(
                    f"{cfg_train['log_dir']}",
                    error_log_name,
                ),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                train()