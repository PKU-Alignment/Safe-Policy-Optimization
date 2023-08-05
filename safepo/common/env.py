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

from __future__ import annotations

from typing import Callable
import safety_gymnasium
from safety_gymnasium.wrappers import SafeAutoResetWrapper, SafeNormalizeObservation, SafeRescaleAction, SafeUnsqueeze
from safety_gymnasium.vector.async_vector_env import SafetyAsyncVectorEnv
from safepo.multi_agent.marl_utils.env_wrappers import ShareSubprocVecEnv, ShareDummyVecEnv, ShareEnv
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
    

def make_env(num_envs: int, env_id: str, seed: int|None = None):
    # create and wrap the environment
    if num_envs > 1:
        def create_env() -> Callable:
            """Creates an environment that can enable or disable the environment checker."""
            env = safety_gymnasium.make(env_id)
            env = SafeRescaleAction(env, -1.0, 1.0)
            env = SafeNormalizeObservation(env)
            return env
        env_fns = [create_env for _ in range(num_envs)]
        env = SafetyAsyncVectorEnv(env_fns)
        env.reset(seed=seed)
        obs_space = env.single_observation_space
        act_space = env.single_action_space
    else:
        env = safety_gymnasium.make(env_id)
        env.reset(seed=seed)
        obs_space = env.observation_space
        act_space = env.action_space
        env = SafeAutoResetWrapper(env)
        env = SafeRescaleAction(env, -1.0, 1.0)
        env = SafeNormalizeObservation(env)
        env = SafeUnsqueeze(env)
    
    return env, obs_space, act_space

def make_ma_mujoco_env(args, cfg_train):
    def get_env_fn(rank):
        def init_env():
            env=ShareEnv(
                scenario=args.scenario,
                agent_conf=args.agent_conf,
            )
            env.reset(seed=args.seed + rank * 1000)
            return env

        return init_env

    if cfg_train['n_rollout_threads']== 1:
        return ShareDummyVecEnv([get_env_fn(0)], cfg_train['device'])
    else:
        return ShareSubprocVecEnv([get_env_fn(i) for i in range(cfg_train['n_rollout_threads'])])

def make_ma_shadow_hand_env(args, cfg, cfg_train, sim_params, agent_index):

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
