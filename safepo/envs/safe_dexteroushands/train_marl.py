from marl_utils.config import set_np_formatting, set_seed, get_args, parse_sim_params, load_cfg
from marl_utils.parse_task import parse_task
from marl_utils.process_sarl import *
from marl_utils.process_marl import process_MultiAgentRL


def train():
    print("Algorithm: ", args.algo)
    # Agent: 4x3
    # agent_index = [[[0, 1, 2],[ 3, 4, 5]],
    #                [[0, 1, 2],[ 3, 4, 5]]]
    # Agent: 2x6
    agent_index = [[[0, 1, 2, 3, 4, 5]],
                   [[0, 1, 2, 3, 4, 5]]]

    if args.algo in ["mappo", "happo", "ippo", "macpo", "mappolag"]:
        # maddpg exists a bug now
        args.task_type = "MultiAgent"

        task, env = parse_task(args, cfg, cfg_train, sim_params, agent_index)

        runner = process_MultiAgentRL(args,env=env, config=cfg_train, model_dir=args.model_dir)

        if args.model_dir != "":
            runner.eval(100000)
        else:
            runner.run()

    


    else:
        print("Unrecognized algorithm!\nAlgorithm should be one of: [happo, hatrpo, mappo,ippo,maddpg,sac,td3,trpo,ppo,ddpg]")


if __name__ == '__main__':
    set_np_formatting()
    args = get_args()
    cfg, cfg_train, logdir = load_cfg(args)
    sim_params = parse_sim_params(args, cfg, cfg_train)
    set_seed(cfg_train.get("seed", -1), cfg_train.get("torch_deterministic", False))
    train()
