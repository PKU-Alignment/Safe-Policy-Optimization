import sys
import os
from safepo.multi_agent.marl_utils.config import (get_args, load_cfg, parse_sim_params,
                               set_np_formatting, set_seed)
from safepo.multi_agent.marl_utils.parse_task import parse_task
from safepo.multi_agent.marl_utils.process_marl import process_MultiAgentRL


def train():
    print("Algorithm: ", args.algo)
    agent_index = [[[0, 1, 2, 3, 4, 5]],
                   [[0, 1, 2, 3, 4, 5]]]

    if args.algo in ["mappo", "happo", "macpo", "mappolag"]:
        # maddpg exists a bug now
        args.task_type = "MultiAgent"
        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

        runner = process_MultiAgentRL(args, env, env, config=cfg_train, model_dir=args.model_dir)

        if args.model_dir != "":
            runner.eval(100000)
        else:
            runner.run()

    else:
        print("Unrecognized algorithm!\nAlgorithm should be one of: [happo, mappo, macpo, mappolag]")


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)

    if args.write_terminal:
        sim_params = parse_sim_params(args, cfg, cfg_train)
        set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
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
                sim_params = parse_sim_params(args, cfg, cfg_train)
                set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
                train()
