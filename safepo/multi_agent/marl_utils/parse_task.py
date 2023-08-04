# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import json
from safepo.envs.safe_dexteroushands.tasks.shadow_hand_bottle_cap import ShadowHandBottleCap
from safepo.envs.safe_dexteroushands.tasks.shadow_hand_catch_abreast import ShadowHandCatchAbreast
from safepo.envs.safe_dexteroushands.tasks.shadow_hand_catch_over2underarm import ShadowHandCatchOver2Underarm
from safepo.envs.safe_dexteroushands.tasks.shadow_hand_catch_underarm import ShadowHandCatchUnderarm
from safepo.envs.safe_dexteroushands.tasks.shadow_hand_door_close_inward import ShadowHandDoorCloseInward
from safepo.envs.safe_dexteroushands.tasks.shadow_hand_door_close_outward import ShadowHandDoorCloseOutward
from safepo.envs.safe_dexteroushands.tasks.shadow_hand_door_open_inward import ShadowHandDoorOpenInward
from safepo.envs.safe_dexteroushands.tasks.shadow_hand_door_open_outward import ShadowHandDoorOpenOutward
from safepo.envs.safe_dexteroushands.tasks.shadow_hand_lift_underarm import ShadowHandLiftUnderarm
from safepo.envs.safe_dexteroushands.tasks.shadow_hand_over import ShadowHandOver

from safepo.envs.safe_dexteroushands.tasks.base.multi_vec_task import \
    MultiVecTaskPython

def parse_task(args, cfg, cfg_train, sim_params, agent_index):

    # create native task and pass custom config
    device_id = args.device_id
    rl_device = args.device

    cfg["seed"] = cfg_train.get("seed", -1)
    cfg_task = cfg["env"]
    cfg_task["seed"] = cfg["seed"]
    task = eval(args.task)(
        cfg=cfg,
        sim_params=sim_params,
        physics_engine=args.physics_engine,
        device_type=args.device,
        device_id=device_id,
        headless=args.headless,
        agent_index=agent_index,
        is_multi_agent=True)
    env = MultiVecTaskPython(task, rl_device)

    return env
