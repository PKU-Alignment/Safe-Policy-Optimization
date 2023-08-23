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
import random
from distutils.util import strtobool

import numpy as np
import copy
import torch
import time
import yaml
import argparse


multi_agent_velocity_map = {
    'Safety2x4AntVelocity-v0': {
        'agent_conf': '2x4',
        'scenario': 'Ant',
    },
    'Safety4x2AntVelocity-v0': {
        'agent_conf': '4x2',
        'scenario': 'Ant',
    },
    'Safety2x3HalfCheetahVelocity-v0': {
        'agent_conf': '2x3',
        'scenario': 'HalfCheetah',
    },
    'Safety6x1HalfCheetahVelocity-v0': {
        'agent_conf': '6x1',
        'scenario': 'HalfCheetah',
    },
    'Safety3x1HopperVelocity-v0': {
        'agent_conf': '3x1',
        'scenario': 'Hopper',
    },
    'Safety2x3Walker2dVelocity-v0': {
        'agent_conf': '2x3',
        'scenario': 'Walker2d',
    },
    'Safety2x1SwimmerVelocity-v0': {
        'agent_conf': '2x1',
        'scenario': 'Swimmer',
    },
    'Safety9|8HumanoidVelocity-v0': {
        'agent_conf': '9|8',
        'scenario': 'Humanoid',
    },
}

multi_agent_goal_tasks = [
    "SafetyPointMultiGoal0-v0",
    "SafetyPointMultiGoal1-v0",
    "SafetyPointMultiGoal2-v0",
    "SafetyAntMultiGoal0-v0",
    "SafetyAntMultiGoal1-v0",
    "SafetyAntMultiGoal2-v0",
]

isaac_gym_map = {
    "ShadowHandOver_Safe_finger": "shadow_hand_over_safe_finger",
    "ShadowHandOver_Safe_joint": "shadow_hand_over_safe_joint",
    "ShadowHandCatchOver2Underarm_Safe_finger": "shadow_hand_catch_over_2_underarm_safe_finger",
    "ShadowHandCatchOver2Underarm_Safe_joint": "shadow_hand_catch_over_2_underarm_safe_joint",
    "FreightFrankaCloseDrawer": "freight_franka_close_drawer",
    "FreightFrankaPickAndPlace": "freight_franka_pick_and_place",
}

def set_np_formatting():
    np.set_printoptions(edgeitems=30, infstr='inf',
                        linewidth=4000, nanstr='nan', precision=2,
                        suppress=False, threshold=10000, formatter=None)

def warn_task_name():
    raise Exception(
        "Unrecognized task!")

def set_seed(seed, torch_deterministic=False):
    
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.autograd.set_detect_anomaly(True)
    
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

    return seed

def parse_sim_params(args, cfg, cfg_train):
    # initialize sim
    try:
        from isaacgym import gymapi, gymutil
    except ImportError:
        raise Exception("Please install isaacgym to run Isaac Gym tasks!")
    sim_params = gymapi.SimParams()
    sim_params.dt = 1./60.
    sim_params.num_client_threads = args.slices

    if args.physics_engine == gymapi.SIM_FLEX:
        if args.device != "cpu":
            print("WARNING: Using Flex with GPU instead of PHYSX!")
        sim_params.flex.shape_collision_margin = 0.01
        sim_params.flex.num_outer_iterations = 4
        sim_params.flex.num_inner_iterations = 10
    elif args.physics_engine == gymapi.SIM_PHYSX:
        sim_params.physx.solver_type = 1
        sim_params.physx.num_position_iterations = 4
        sim_params.physx.num_velocity_iterations = 0
        sim_params.physx.num_threads = 4
        sim_params.physx.use_gpu = args.use_gpu
        sim_params.physx.num_subscenes = args.subscenes
        sim_params.physx.max_gpu_contact_pairs = 8 * 1024 * 1024

    sim_params.use_gpu_pipeline = args.use_gpu_pipeline
    sim_params.physx.use_gpu = args.use_gpu

    # if sim options are provided in cfg, parse them and update/override above:
    if "sim" in cfg:
        gymutil.parse_sim_config(cfg["sim"], sim_params)

    # Override num_threads if passed on the command line
    if args.physics_engine == gymapi.SIM_PHYSX and args.num_threads > 0:
        sim_params.physx.num_threads = args.num_threads

    return sim_params

def single_agent_args():
    custom_parameters = [
        {"name": "--seed", "type": int, "default":0, "help": "Random seed"},
        {"name": "--use-eval", "type": lambda x: bool(strtobool(x)), "default": False, "help": "Use evaluation environment for testing"},
        {"name": "--task", "type": str, "default": "SafetyPointGoal1-v0", "help": "The task to run"},
        {"name": "--num-envs", "type": int, "default": 10, "help": "The number of parallel game environments"},
        {"name": "--experiment", "type": str, "default": "single_agent_exp", "help": "Experiment name"},
        {"name": "--log-dir", "type": str, "default": "../runs", "help": "directory to save agent logs"},
        {"name": "--device", "type": str, "default": "cpu", "help": "The device to run the model on"},
        {"name": "--device-id", "type": int, "default": 0, "help": "The device id to run the model on"},
        {"name": "--write-terminal", "type": lambda x: bool(strtobool(x)), "default": True, "help": "Toggles terminal logging"},
        {"name": "--headless", "type": lambda x: bool(strtobool(x)), "default": False, "help": "Toggles headless mode"},
        {"name": "--total-steps", "type": int, "default": 10000000, "help": "Total timesteps of the experiments"},
        {"name": "--steps-per-epoch", "type": int, "default": 20000, "help": "The number of steps to run in each environment per policy rollout"},
        {"name": "--randomize", "type": bool, "default": False, "help": "Wheather to randomize the environments' initial states"},
        {"name": "--cost-limit", "type": float, "default": 25.0, "help": "cost_lim"},
        {"name": "--lagrangian-multiplier-init", "type": float, "default": 0.001, "help": "initial value of lagrangian multiplier"},
        {"name": "--lagrangian-multiplier-lr", "type": float, "default": 0.035, "help": "learning rate of lagrangian multiplier"},
    ]
    # Create argument parser
    parser = argparse.ArgumentParser(description="RL Policy")
    issac_parameters = copy.deepcopy(custom_parameters)
    for param in custom_parameters:
        param_name = param.pop("name")
        parser.add_argument(param_name, **param)

    # Parse arguments

    args = parser.parse_args()
    cfg_env={}
    base_path = os.path.dirname(os.path.abspath(__file__)).replace("utils", "multi_agent")
    if args.task in isaac_gym_map.keys():
        try:
            from isaacgym import gymutil
        except ImportError:
            raise Exception("Please install isaacgym to run Isaac Gym tasks!")
        args = gymutil.parse_arguments(description="RL Policy", custom_parameters=issac_parameters)
        args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'
        cfg_env_path = "marl_cfg/{}.yaml".format(isaac_gym_map[args.task])
        with open(os.path.join(base_path, cfg_env_path), 'r') as f:
            cfg_env = yaml.load(f, Loader=yaml.SafeLoader)
            cfg_env["name"] = args.task
            if "task" in cfg_env:
                if "randomize" not in cfg_env["task"]:
                    cfg_env["task"]["randomize"] = args.randomize
                else:
                    cfg_env["task"]["randomize"] = False
    return args, cfg_env


def multi_agent_args(algo):

    # Define custom parameters
    custom_parameters = [
        {"name": "--use-eval", "type": lambda x: bool(strtobool(x)), "default": False, "help": "Use evaluation environment for testing"},
        {"name": "--task", "type": str, "default": "SafetyAntMultiGoal1-v0", "help": "The task to run"},
        {"name": "--agent-conf", "type": str, "default": "2x4", "help": "The agent configuration"},
        {"name": "--scenario", "type": str, "default": "Ant", "help": "The scenario"},
        {"name": "--experiment", "type": str, "default": "Base", "help": "Experiment name"},
        {"name": "--seed", "type": int, "default":0, "help": "Random seed"},
        {"name": "--model-dir", "type": str, "default": "", "help": "Choose a model dir"},
        {"name": "--safety-bound", "type": float, "default": 25.0, "help": "cost_lim"},
        {"name": "--device", "type": str, "default": "cpu", "help": "The device to run the model on"},
        {"name": "--device-id", "type": int, "default": 0, "help": "The device id to run the model on"},
        {"name": "--write-terminal", "type": lambda x: bool(strtobool(x)), "default": True, "help": "Toggles terminal logging"},
        {"name": "--headless", "type": lambda x: bool(strtobool(x)), "default": False, "help": "Toggles headless mode"},
        {"name": "--total-steps", "type": int, "default": None, "help": "Total timesteps of the experiments"},
        {"name": "--num-envs", "type": int, "default": None, "help": "The number of parallel game environments"},
        {"name": "--randomize", "type": bool, "default": False, "help": "Wheather to randomize the environments' initial states"},
    ]
    # Create argument parser
    parser = argparse.ArgumentParser(description="RL Policy")
    issac_parameters = copy.deepcopy(custom_parameters)
    for param in custom_parameters:
        param_name = param.pop("name")
        parser.add_argument(param_name, **param)

    # Parse arguments

    args = parser.parse_args()

    if args.task in isaac_gym_map.keys():
        try:
            from isaacgym import gymutil
        except ImportError:
            raise Exception("Please install isaacgym to run Isaac Gym tasks!")
        args = gymutil.parse_arguments(description="RL Policy", custom_parameters=issac_parameters)
        args.device = args.sim_device_type if args.use_gpu_pipeline else 'cpu'
    cfg_train_path = "marl_cfg/{}/config.yaml".format(algo)
    base_path = os.path.dirname(os.path.abspath(__file__)).replace("utils", "multi_agent")
    with open(os.path.join(base_path, cfg_train_path), 'r') as f:
        cfg_train = yaml.load(f, Loader=yaml.SafeLoader)
        if args.task in multi_agent_velocity_map.keys():
            cfg_train.update(cfg_train.get("mamujoco"))
            args.agent_conf = multi_agent_velocity_map[args.task]["agent_conf"]
            args.scenario = multi_agent_velocity_map[args.task]["scenario"]
        elif args.task in multi_agent_goal_tasks:
            cfg_train.update(cfg_train.get("mamujoco"))

    cfg_train["use_eval"] = args.use_eval
    cfg_train["safety_bound"]=args.safety_bound
    cfg_train["algorithm_name"]=algo
    cfg_train["device"] = args.device + ":" + str(args.device_id)

    cfg_train["env_name"] = args.task

    if args.total_steps:
        cfg_train["num_env_steps"] = args.total_steps
    if args.num_envs:
        cfg_train["n_rollout_threads"] = args.num_envs
        cfg_train["n_eval_rollout_threads"] = args.num_envs
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    cfg_train['log_dir']="../runs/"+args.experiment+'/'+args.task+'/'+algo+'/'+relpath
    cfg_env={}
    if args.task in isaac_gym_map.keys():
        cfg_env_path = "marl_cfg/{}.yaml".format(isaac_gym_map[args.task])
        with open(os.path.join(base_path, cfg_env_path), 'r') as f:
            cfg_env = yaml.load(f, Loader=yaml.SafeLoader)
            cfg_env["name"] = args.task
            if "task" in cfg_env:
                if "randomize" not in cfg_env["task"]:
                    cfg_env["task"]["randomize"] = args.randomize
                else:
                    cfg_env["task"]["randomize"] = False
            else:
                cfg_env["task"] = {"randomize": False}
    elif args.task in multi_agent_velocity_map.keys() or args.task in multi_agent_goal_tasks:
        pass
    else:
        warn_task_name()

    return args, cfg_env, cfg_train

