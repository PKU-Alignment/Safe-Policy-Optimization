# SafePO-Baselines

<img src="assets/main.jpg" width="1000" border="1"/>

Safepo baselines provides many safe-based algorithms implementations, including Single-agent RL and Multi-agent RL. Codes are implemented based on the latest version of pytorch and include mainstream CMDP-based algorithms in the field of safe reinforcement learning. We provide a unified framework and interface for these algorithms, which helps people to fairly compare the performance of different algorithms under the same learning mode and parameters. All algorithms are implemented based on cpu multi-threading and minibatch, which greatly improves the training speed of the algorithm.

## Overview of Algorithms
Here we provide a table for Safe RL algorithms that the benchmark concludes.
|Algorithm| Proceedings&Cites | Paper Links |Code URL | Official Code Repo | Official Code Framework | Official Code Last Update | Official Github Stars |
|:-------------:|:------------:|:-------------:|:-------------:|:---------------------------:|:------------:|---------------|---------------|
|PPO Lagrangian | &cross; | [Openai](https://cdn.openai.com/safexp-short.pdf) |[code](https://github.com/PKU-MARL/Safe-Policy-Optimization/blob/main/safepo/algos/ppo_lagrangian.py)| [Official repo](https://github.com/openai/safety-starter-agents) | Tensorflow 1 | ![GitHub last commit](https://img.shields.io/github/last-commit/openai/safety-starter-agents?label=last%20update) | [![GitHub stars](https://img.shields.io/github/stars/openai/safety-starter-agents)](https://github.com/openai/safety-starter-agents/stargazers) |
|TRPO Lagrangian | &cross; | [Openai](https://cdn.openai.com/safexp-short.pdf) |[code](https://github.com/PKU-MARL/Safe-Policy-Optimization/blob/main/safepo/algos/trpo_lagrangian.py)| [Official repo](https://github.com/openai/safety-starter-agents) | Tensorflow 1 | ![GitHub last commit](https://img.shields.io/github/last-commit/openai/safety-starter-agents?label=last%20update) | [![GitHub stars](https://img.shields.io/github/stars/openai/safety-starter-agents)](https://github.com/openai/safety-starter-agents/stargazers) |
|FOCOPS | Neurips 2020 (Cite: 27) | [arxiv](https://arxiv.org/pdf/2002.06506.pdf) |[code](https://github.com/PKU-MARL/Safe-Policy-Optimization/blob/main/safepo/algos/focops.py)| [Official repo](https://github.com/ymzhang01/focops) | pytorch | ![GitHub last commit](https://img.shields.io/github/last-commit/ymzhang01/focops?label=last%20update) | [![GitHub stars](https://img.shields.io/github/stars/ymzhang01/focops)](https://github.com/ymzhang01/focops/stargazers) |
|CPO | ICML 2017(Cite: 663) | [arxiv](https://arxiv.org/abs/1705.10528) |[code](https://github.com/PKU-MARL/Safe-Policy-Optimization/blob/main/safepo/algos/cpo.py)| &cross; | &cross; | &cross; | &cross; |
|PCPO | ICLR 2020(Cite: 67) | [arxiv](https://arxiv.org/pdf/2010.03152.pdf) |[code](https://github.com/PKU-MARL/Safe-Policy-Optimization/blob/main/safepo/algos/pcpo.py)| [Official repo](https://sites.google.com/view/iclr2020-pcpo) | theano | &cross; | &cross; |
|P3O | IJCAI 2022(Cite: 0) | [arxiv](https://arxiv.org/pdf/2205.11814.pdf) |[code](https://github.com/PKU-MARL/Safe-Policy-Optimization/blob/main/safepo/algos/p3o.py)| &cross; | &cross; | &cross; | &cross; |
|MACPO | Preprint(Cite: 4) | [arxiv](https://arxiv.org/pdf/2110.02793.pdf) |[code](https://github.com/PKU-MARL/Safe-Policy-Optimization/blob/main/safepo/algos/marl/safe-marl-baselines/algorithms/algorithms/macpo_trainer.py)| [Official repo](https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation) | pytorch | ![GitHub last commit](https://img.shields.io/github/last-commit/chauncygu/Multi-Agent-Constrained-Policy-Optimisation?label=last%20update) | [![GitHub stars](https://img.shields.io/github/stars/chauncygu/Safe-Multi-Agent-Isaac-Gym)](https://github.com/chauncygu/Safe-Multi-Agent-Isaac-Gym/stargazers) |
|MAPPO_Lagrangian | Preprint(Cite: 4) | [arxiv](https://arxiv.org/pdf/2110.02793.pdf) |[code]()| [Official repo](https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation) | pytorch | ![GitHub last commit](https://img.shields.io/github/last-commit/chauncygu/Multi-Agent-Constrained-Policy-Optimisation?label=last%20update) | [![GitHub stars](https://img.shields.io/github/stars/chauncygu/Safe-Multi-Agent-Isaac-Gym)](https://github.com/chauncygu/Safe-Multi-Agent-Isaac-Gym/stargazers) |
|HATRPO | ICLR 2022 (Cite: 10) | [arxiv](https://arxiv.org/pdf/2109.11251.pdf) |[code](https://github.com/PKU-MARL/Safe-Policy-Optimization/blob/main/safepo/algos/marl/safe-marl-baselines/algorithms/algorithms/hatrpo_trainer.py)| [Official repo](https://github.com/cyanrain7/TRPO-in-MARL) | pytorch | ![GitHub last commit](https://img.shields.io/github/last-commit/cyanrain7/TRPO-in-MARL?label=last%20update) | [![GitHub stars](https://img.shields.io/github/stars/cyanrain7/TRPO-in-MARL)](https://github.com/cyanrain7/TRPO-in-MARL/stargazers) |
|HAPPO (Purely reward optimisation) | ICLR 2022 (Cite: 10) | [arxiv](https://arxiv.org/pdf/2109.11251.pdf) |[code](https://github.com/PKU-MARL/Safe-Policy-Optimization/blob/main/safepo/algos/marl/safe-marl-baselines/algorithms/algorithms/happo_trainer.py)| [Official repo](https://github.com/cyanrain7/TRPO-in-MARL) | pytorch | ![GitHub last commit](https://img.shields.io/github/last-commit/cyanrain7/TRPO-in-MARL?label=last%20update) | [![GitHub stars](https://img.shields.io/github/stars/cyanrain7/TRPO-in-MARL)](https://github.com/cyanrain7/TRPO-in-MARL/stargazers) |
|MAPPO (Purely reward optimisation) | Preprint(Cite: 98) | [arxiv](https://arxiv.org/pdf/2103.01955.pdf) |[code](https://github.com/PKU-MARL/Safe-Policy-Optimization/blob/main/safepo/algos/marl/safe-marl-baselines/algorithms/algorithms/mappo_trainer.py)| [Official repo](https://github.com/marlbenchmark/on-policy) | pytorch | ![GitHub last commit](https://img.shields.io/github/last-commit/marlbenchmark/on-policy?label=last%20update) | [![GitHub stars](https://img.shields.io/github/stars/marlbenchmark/on-policy)](https://github.com/marlbenchmark/on-policy/stargazers) |
|IPPO (Purely reward optimisation) | Preprint(Cite: 28) | [arixiv](https://arxiv.org/pdf/2011.09533.pdf) |[code](https://github.com/PKU-MARL/Safe-Policy-Optimization/blob/main/safepo/algos/marl/safe-marl-baselines/algorithms/algorithms/ippo_trainer.py)| &cross; | &cross; | &cross; | &cross; |


## Supported Environment 

## Installation

### Sacred
Sacred is a tool to configure, organize, log and reproduce computational experiments. It is designed to introduce only minimal overhead, while encouraging modularity and configurability of experiments. You can install it from [Sacred doc](https://sacred.readthedocs.io/en/stable/).

### Mujoco
MuJoCo stands for Multi-Joint dynamics with Contact. It is a general purpose physics engine that aims to facilitate research and development in robotics, biomechanics, graphics and animation, machine learning, and other areas which demand fast and accurate simulation of articulated structures interacting with their environment. You can install from [Mujoco github](https://github.com/deepmind/mujoco).

### Safety Gym
Safety Gym, a suite of environments and tools for measuring progress towards reinforcement learning agents that respect safety constraints while training. You can install from [Safety Gym github](https://github.com/openai/safety-gym).

### Bullet Safety Gym
"Bullet-Safety-Gym" is a free and open-source framework to benchmark and assess safety specifications in Reinforcement Learning (RL) problems.[Bullet Safety Gym github](https://github.com/SvenGronauer/Bullet-Safety-Gym).

### Conda-Environment

```
conda create -n Single python=3.8
pip install torch==1.8.0+cu111 torchvision==0.9.0+cu111 torchaudio==0.8.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -e .
```
### Machine Configuration
We test all algorithms and experiments in CPU: **AMD Ryzen Threadripper PRO 3975WX 32-Cores** and **GPU: NVIDIA GeForce RTX 3090, Driver Version: 495.44**.

## Getting Started
### Single Agent
#### Train
All algorithm codes are in file:Parallel_algorithm, for example, if you want to run ppo_lagrangian in safety_gym:Safexp-PointGoal1-v0, with cpu cores:4, seed:0,

```
python train.py --env_id Safexp-PointGoal1-v0 --algo ppo_lagrangian --cores 4
```
#### Configures
|  Argprase   | default  | info|
|  ----       | ----  | ----|
| --algo       | required | the name of algorithm exec |
| --cores | int| the number of cpu physical cores you use|
| --seed | int| the seed you use|
| --check_freq       | int: 25 | check the snyc parameter |
| --entropy_coef | float:0.01| the parameter of entropy|
| --gamma| float:0.99 | the value of dicount|
| --lam | float: 0.95 | the value of GAE lambda |
| --lam_c| float: 0.95| the value of GAE cost lambda |
| --max_ep_len | int: 1000| unless environment have the default value else, we take 1000 as default value|
| --max_grad_norm| float: 0.5| the clip of parameters|
| --num_mini_batches| int: 16| used for value network tranining|
| --optimizer| Adam | the optimizer of Policy other : SGD, other class in torch.optim|
| --pi_lr | float: 3e-4| the learning rate of policy|
| --steps_per_epoch| int: 32000| the number of interactor steps|
| --target_kl | float: 0.01| the value of trust region|
| --train_pi_iterations| int: 80| the number of policy learn iterations|
| --train_v_iterations| int: 40| the number of value network and cost value network iterations|
| --use_cost_value_function| bool: False| use cost_value_function or not|
|--use_entropy|bool:False| use entropy or not|
|--use_reward_penalty| bool:False| use reward_penalty or not|

E.g. if we want use trpo_lagrangian in environment: with 10 cores and seed:0, we can run the following command:
```
python train.py --algo trpo_lagrangian --env_id Safexp-PointGoal1-v0 --cores 10 --seed 0
```
### Mult-agent
#### About this repository

This repository provides a safe MARL baseline benchmark for safe MARL research on challenging tasks of safety DexterousHands (which is developed for MARL, named as Safe MAIG, for details, see [Safe MAIG](https://github.com/chauncygu/Safe-Multi-Agent-Isaac-Gym)), in which the [MACPO](https://arxiv.org/pdf/2110.02793.pdf), [MAPPO-lagrangian](https://arxiv.org/pdf/2110.02793.pdf), [MAPPO](https://arxiv.org/abs/2103.01955), [HAPPO](https://arxiv.org/abs/2109.11251), [IPPO](https://arxiv.org/abs/2011.09533) are all implemented to investigate the safety and reward performance.






#### Installation

Details regarding installation of IsaacGym can be found [here](https://developer.nvidia.com/isaac-gym). We currently support the `Preview Release 3` version of IsaacGym.

#### Pre-requisites

The code has been tested on Ubuntu 18.04 with Python 3.7. The minimum recommended NVIDIA driver
version for Linux is `470` (dictated by support of IsaacGym).

It uses [Anaconda](https://www.anaconda.com/) to create virtual environments.
To install Anaconda, follow instructions [here](https://docs.anaconda.com/anaconda/install/linux/).

Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples` 
directory, like `joint_monkey.py`. Follow troubleshooting steps described in the Isaac Gym Preview 2 
install instructions if you have any trouble running the samples.

#### install this repo
Once Isaac Gym is installed and samples work within your current python environment, install marl package ```algos/marl``` with:

```bash
pip install -e .
```

#### Running the benchmarks

To train your first policy, run this line in ```algos/marl```:

```bash
python train.py --task=ShadowHandOver --algo=macpo
```

#### Select an algorithm

To select an algorithm, pass `--algo=ppo/mappo/happo/hatrpo` in ```algos/marl``` 
as an argument:

```bash
python train.py --task=ShadowHandOver --algo=macpo
```

At present, we only support these four algorithms.
<!-- ### Loading trained models // Checkpoints

Checkpoints are saved in the folder `models/` 

To load a trained checkpoint and only perform inference (no training), pass `--test` 
as an argument:

```bash
python train.py --task=ShadowHandOver --checkpoint=models/shadow_hand_over/ShadowHandOver.pth --test
``` -->

<!--## <span id="task">Tasks</span>-->

#### Select tasks

Source code for tasks can be found in `dexteroushandenvs/tasks`. 

Until now we only suppose the following environments:

| Environments | ShadowHandOver | ShadowHandCatchUnderarm | ShadowHandTwoCatchUnderarm | ShadowHandCatchAbreast | ShadowHandOver2Underarm |
|  :----:  | :----:  | :----:  | :----:  | :----:  | :----:  |
| Description | These environments involve two fixed-position hands. The hand which starts with the object must find a way to hand it over to the second hand. | These environments again have two hands, however now they have some additional degrees of freedom that allows them to translate/rotate their centre of masses within some constrained region. | These environments involve coordination between the two hands so as to throw the two objects between hands (i.e. swapping them). | This environment is similar to ShadowHandCatchUnderarm, the difference is that the two hands are changed from relative to side-by-side posture. | This environment is is made up of half ShadowHandCatchUnderarm and half ShadowHandCatchOverarm, the object needs to be thrown from the vertical hand to the palm-up hand |
| Actions Type | Continuous | Continuous | Continuous | Continuous | Continuous |
| Total Action Num | 40    | 52    | 52    | 52    | 52    |
| Action Values     | [-1, 1]    | [-1, 1]    | [-1, 1]    | [-1, 1]    | [-1, 1]    |
| Action Index and Description     | [detail](#action1)    | [detail](#action2)   | [detail](#action3)    | [detail](#action4)    | [detail](#action5)    |
| Observation Shape     | (num_envs, 2, 211)    | (num_envs, 2, 217)    | (num_envs, 2, 217)    | (num_envs, 2, 217)    | (num_envs, 2, 217)    |
| Observation Values     | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    |
| Observation Index and Description     | [detail](#obs1)    | [detail](#obs2)   | [detail](#obs3)    | [detail](#obs4)    | [detail](#obs4)    |
| State Shape     | (num_envs, 2, 398)    | (num_envs, 2, 422)    | (num_envs, 2, 422)    | (num_envs, 2, 422)    | (num_envs, 2, 422)    | 
| State Values     | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    | [-5, 5]    |
| Rewards     | Rewards is the pose distance between object and goal. You can check out the details [here](#r1)| Rewards is the pose distance between object and goal. You can check out the details [here](#r2)    | Rewards is the pose distance between object and goal. You can check out the details [here](#r3)    | Rewards is the pose distance between two object and  two goal, this means that both objects have to be thrown in order to be swapped over. You can check out the details [here](#r4)    | Rewards is the pose distance between object and goal. You can check out the details [here](#r2)    |
| Demo     | <img src="assets/image_folder/0v1.gif" align="middle" width="550" border="1"/>    | <img src="assets/image_folder/hand_catch_underarm.gif" align="middle" width="140" border="1"/>    | <img src="assets/image_folder/two_catch.gif" align="middle" width="130" border="1"/>    | <img src="assets/image_folder/1v1.gif" align="middle" width="130" border="1"/>    | <img src="assets/image_folder/2.gif" align="middle" width="130" border="1"/>    |






## Demo
If you want to see some demo with our benchmark, you can check it [Demo](https://sites.google.com/view/safepo-benchmark)


## The Team
The Baseline is a project contributed by MARL team at Peking University, please contact yaodong.yang@pku.edu.cn if you are interested to collaborate.
We also thank the list of contributors from the following open source repositories: 
[Spinning Up](https://spinningup.openai.com/en/latest/), [Bullet-Safety-Gym](https://github.com/SvenGronauer/Bullet-Safety-Gym/tree/master/bullet_safety_gym/envs), [SvenG](https://github.com/SvenGronauer/RL-Safety-Algorithms), [Safety Gym](https://github.com/openai/safety-gym).
