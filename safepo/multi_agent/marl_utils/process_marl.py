# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.



def process_MultiAgentRL(args,env, eval_env, config, model_dir=""):

    config["n_rollout_threads"] = env.num_envs
    config["n_eval_rollout_threads"] = env.num_envs

    if args.algo in ["mappo", "happo", "ippo"]:
        # on policy marl
        from safepo.multi_agent.marl_algorithms.algorithms.runner import \
            Runner
        marl = Runner(
            vec_env=env,
            vec_eval_env=eval_env,
            config=config,
            model_dir=model_dir
            )
    if args.algo in ["macpo", "mappolag"]:
        # safe rl
        from safepo.multi_agent.marl_algorithms.algorithms.runner_macpo import \
            Runner
        marl = Runner(
            vec_env=env,
            vec_eval_env=eval_env,
            config=config,
            model_dir=model_dir
            )

    return marl
