<div align="center">
  <img src="assets/logo.png" width="75%"/>
</div>

<div align="center">

[![Organization](https://img.shields.io/badge/Organization-PKU--Alignment-blue)](https://github.com/PKU-Alignment)
[![License](https://img.shields.io/github/license/PKU-Alignment/Safe-Policy-Optimization?label=license)](#license)
[![codecov](https://codecov.io/gh/PKU-Alignment/Safe-Policy-Optimization/graph/badge.svg?token=KF0UM0UNXW)](https://codecov.io/gh/PKU-Alignment/Safe-Policy-Optimization)
[![Documentation Status](https://readthedocs.org/projects/safe-policy-optimization/badge/?version=latest)](https://safe-policy-optimization.readthedocs.io/en/latest/?badge=latest)

</div>

**What's New**: 

- We have release documentation for [Safe Policy Optimization](https://safe-policy-optimization.readthedocs.io).
- The **benchmark results** of SafePO can be viewed at [Benchmark](https://safe-policy-optimization.readthedocs.io/en/latest/algorithms/general.html).

**Safe Policy Optimization (SafePO)**  is a comprehensive algorithm benchmark for Safe Reinforcement Learning (Safe RL). It provides RL research community with a unified platform for processing and evaluating algorithms in various safe reinforcement learning environments. In order to better help the community study this problem, SafePO is developed with the following key features:

<div align=center>
    <img src="assets/arch.png" width="800" border="1"/>
</div>

**Correctness.** For a benchmark, it is critical to ensure its correctness and reliability.
%To achieve this goal, we examine the implementation of SafePO carefully.
Firstly, each algorithm is implemented strictly according to the original paper (e.g., ensuring consistency with the gradient flow of the original paper, etc). Secondly, for algorithms with a commonly acknowledged open-source code base, we compare our implementation with those line by line, in order to double-check the correctness. Finally, we compare SafePO with existing benchmarks (e.g., [Safety-Starter-Agents](https://github.com/openai/safety-starter-agents) and [RL-Safety-Algorithms](https://github.com/SvenGronauer/RL-Safety-Algorithms)) outperforms other existing implementations.

**Extensibility.** SafePO enjoys high extensibility thanks to its architecture. New algorithms can be integrated to SafePO by inheriting from base algorithms and only implementing their unique features. For example, we integrate PPO by inheriting from policy gradient and only adding the clip ratio variable and rewriting the function that computes the loss of policy. In a similar way, algorithms can be easily added to SafePO.

**Logging and Visualization.** Another important functionality of SafePO is logging and visualization. Supporting both TensorBoard and WandB, we offer code for the visualizations of more than 40 parameters and intermediate computation results, for the purpose of inspecting the training process. Common parameters and metrics such as KL-divergence, SPS (step per second), and variance of cost are visualized universally. During training, users are able to inspect the changes of every parameter, collect the log file, and obtain saved checkpoint models. The complete and comprehensive visualization allows easier observation, model selection, and comparison.

**Documentation.** In addition to its code implementation, SafePO comes with an [extensive documentation](https://safe-policy-optimization.readthedocs.io). We include detailed guidance on installation and propose solutions to common issues. Moreover, we provide instructions on simple usage and advanced customization of SafePO. Official information concerning maintenance, ethical and responsible use are stated clearly for reference.


- [Overview of Algorithms](#overview-of-algorithms)
- [Supported Environments](#supported-environments)
- [Safety-Gymnasium](#safety-gymnasium)
- [Safe-Dexterous-Hands](#safe-dexterous-hands)
  - [Prerequisites](#prerequisites)
  - [Selected Tasks](#selected-tasks)
- [What's More](#whats-more)
- [Pre-requisites](#pre-requisites)
- [Conda-Environment](#conda-environment)
- [Getting Started](#getting-started)
  - [Single-Agent](#single-agent)
  - [Multi-Agent](#multi-agent)
    - [Safety DexterousHands](#safety-dexteroushands)
    - [Safety-Gymnasium Multi-agent Velocity](#safety-gymnasium-multi-agent-velocity)
    - [Multi-Agent Benchmark](#multi-agent-benchmark)
- [Machine Configuration](#machine-configuration)
- [Ethical and Responsible Use](#ethical-and-responsible-use)
- [PKU-Alignment Team](#pku-alignment-team)

## Overview of Algorithms

Here we provide a table of Safe RL algorithms that the benchmark includes.


**note: Four more classic RL algorithms are also included in the benchmark, namely PG, NaturalPG, TRPO, and PPO.**

|                                 Algorithm                                  |    Proceedings&Cites    |                                 Official Code Repo                                  |                                                         Official Code Last Update                                                          |                                                                      Official Github Stars                                                                      |
| :------------------------------------------------------------------------: | :---------------------: | :---------------------------------------------------------------------------------: | :----------------------------------------------------------------------------------------------------------------------------------------: | :-------------------------------------------------------------------------------------------------------------------------------------------------------------: |
|             [PPO-Lag](https://cdn.openai.com/safexp-short.pdf)             |           :x:           |          [Tensorflow 1 ](https://github.com/openai/safety-starter-agents)           |             ![GitHub last commit](https://img.shields.io/github/last-commit/openai/safety-starter-agents?label=last%20update)              |         [![GitHub stars](https://img.shields.io/github/stars/openai/safety-starter-agents)](https://github.com/openai/safety-starter-agents/stargazers)         |
|            [TRPO-Lag](https://cdn.openai.com/safexp-short.pdf)             |           :x:           |           [Tensorflow 1](https://github.com/openai/safety-starter-agents)           |             ![GitHub last commit](https://img.shields.io/github/last-commit/openai/safety-starter-agents?label=last%20update)              |         [![GitHub stars](https://img.shields.io/github/stars/openai/safety-starter-agents)](https://github.com/openai/safety-starter-agents/stargazers)         |
|               [CUP](https://arxiv.org/pdf/2209.07089.pdf)               | Neurips 2022 (Cite: 6) |                   [Pytorch](https://github.com/zmsn-2077/CUP-safe-rl)                    |                   ![GitHub last commit](https://img.shields.io/github/last-commit/zmsn-2077/CUP-safe-rl?label=last%20update)                    |                     [![GitHub stars](https://img.shields.io/github/stars/zmsn-2077/CUP-safe-rl)](https://github.com/zmsn-2077/CUP-safe-rl/stargazers)                     |
|               [FOCOPS](https://arxiv.org/pdf/2002.06506.pdf)               | Neurips 2020 (Cite: 27) |                   [Pytorch](https://github.com/ymzhang01/focops)                    |                   ![GitHub last commit](https://img.shields.io/github/last-commit/ymzhang01/focops?label=last%20update)                    |                     [![GitHub stars](https://img.shields.io/github/stars/ymzhang01/focops)](https://github.com/ymzhang01/focops/stargazers)                     |
|                  [CPO](https://arxiv.org/abs/1705.10528)                   |  ICML 2017(Cite: 663)   |                                         :x:                                         |                                                                    :x:                                                                     |                                                                               :x:                                                                               |
|                [PCPO](https://arxiv.org/pdf/2010.03152.pdf)                |   ICLR 2020(Cite: 67)   |                [Theano](https://sites.google.com/view/iclr2020-pcpo)                |                                                                    :x:                                                                     |                                                                               :x:                                                                               |
|                [RCPO](https://arxiv.org/pdf/1805.11074.pdf)                |  ICLR 2019 (Cite: 238)  |                                         :x:                                         |                                                                    :x:                                                                     |                                                                               :x:                                                                               |
|              [CPPO-PID](https://arxiv.org/pdf/2007.03964.pdf)              | Neurips 2020(Cite: 71)  |     [Pytorch](https://github.com/astooke/rlpyt/tree/master/rlpyt/projects/safe)     |                     ![GitHub last commit](https://img.shields.io/github/last-commit/astooke/rlpyt?label=last%20update)                     |                        [![GitHub stars](https://img.shields.io/github/stars/astooke/rlpyt)](https://github.com/astooke/rlpyt/stargazers)                        |
|               [MACPO](https://arxiv.org/pdf/2110.02793.pdf)                |    Preprint(Cite: 4)    | [Pytorch](https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation) | ![GitHub last commit](https://img.shields.io/github/last-commit/chauncygu/Multi-Agent-Constrained-Policy-Optimisation?label=last%20update) | [![GitHub stars](https://img.shields.io/github/stars/chauncygu/Safe-Multi-Agent-Isaac-Gym)](https://github.com/chauncygu/Safe-Multi-Agent-Isaac-Gym/stargazers) |
|             [MAPPO-Lag](https://arxiv.org/pdf/2110.02793.pdf)              |    Preprint(Cite: 4)    | [Pytorch](https://github.com/chauncygu/Multi-Agent-Constrained-Policy-Optimisation) | ![GitHub last commit](https://img.shields.io/github/last-commit/chauncygu/Multi-Agent-Constrained-Policy-Optimisation?label=last%20update) | [![GitHub stars](https://img.shields.io/github/stars/chauncygu/Safe-Multi-Agent-Isaac-Gym)](https://github.com/chauncygu/Safe-Multi-Agent-Isaac-Gym/stargazers) |
| [HAPPO (Purely reward optimisation)](https://arxiv.org/pdf/2109.11251.pdf) |  ICLR 2022 (Cite: 10)   |                [Pytorch](https://github.com/cyanrain7/TRPO-in-MARL)                 |                ![GitHub last commit](https://img.shields.io/github/last-commit/cyanrain7/TRPO-in-MARL?label=last%20update)                 |               [![GitHub stars](https://img.shields.io/github/stars/cyanrain7/TRPO-in-MARL)](https://github.com/cyanrain7/TRPO-in-MARL/stargazers)               |
| [MAPPO (Purely reward optimisation)](https://arxiv.org/pdf/2103.01955.pdf) |   Preprint(Cite: 98)    |                [Pytorch](https://github.com/marlbenchmark/on-policy)                |                ![GitHub last commit](https://img.shields.io/github/last-commit/marlbenchmark/on-policy?label=last%20update)                |              [![GitHub stars](https://img.shields.io/github/stars/marlbenchmark/on-policy)](https://github.com/marlbenchmark/on-policy/stargazers)              |

## Supported Environments: Safety-Gymnasium

Here is a list of all the environments support for now; some are being tested in our baselines, and we will gradually release them in later updates. For more details, please refer to [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium).

### Gymnasium-based Environments

<table border="1">
  <thead>
    <tr>
      <th>Category</th>
      <th>Task</th>
      <th>Agent</th>
      <th>Example</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td rowspan="4">Safe Navigation</td>
      <td>Goal[012]</td>
      <td rowspan="4">Point, Car, Doggo, Racecar, Ant</td>
      <td rowspan="4">SafetyPointGoal1-v0</td>
    </tr>
    <tr>
      <td>Button[012]</td>
    </tr>
    <tr>
      <td>Push[012]</td>
    </tr>
    <tr>
      <td>Circle[012]</td>
    </tr>
    <tr>
      <td>Velocity</td>
      <td>Velocity</td>
      <td>HalfCheetah, Hopper, Swimmer, Walker2d, Ant, Humanoid</td>
      <td>SafetyAntVelocity-v1</td>
    </tr>
  </tbody>
</table>

**note**: Safe velocity tasks support both single-agent and multi-agent algorithms, while safe navigation tasks only support single-agent algorithms currently.


### Isaac Gym-based Environments

Ensure that Isaac Gym works on your system by running one of the examples from the `python/examples` 
directory, like `joint_monkey.py`. Please follow troubleshooting steps described in the Isaac Gym Preview Release 3/4
install instructions if you have any trouble running the samples.


### Selected Tasks

| Base Environments            | Description                                                                                                                                                           | Demo                                                        |
| ---------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ----------------------------------------------------------- |
| ShadowHandOver_Safe_finger               | These environments involve two fixed-position hands. The hand which starts with the object must find a way to hand it over to the second hand.                        | <img src="assets/hand/0v1.gif" align="middle" width="250"/> |
| ShadowHandCatchOver2Underarm_Safe_finger | This environment is made up of half ShadowHandCatchUnderarm and half ShadowHandCatchOverarm, the object needs to be thrown from the vertical hand to the palm-up hand | <img src="assets/hand/2.gif" align="middle" width="250"/>   |

**We implement some different constraints to the base environments, expanding the setting to both single-agent and multi-agent.**

<img src="assets/hand.png" align="middle" width="1000"/>


## Pre-requisites

To use SafePO-Baselines, you need to install environments. Please refer to [Mujoco](https://mujoco.org/), [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) for more details on installation. Details regarding the installation of IsaacGym can be found [here](https://developer.nvidia.com/isaac-gym). We currently support the `Preview Release 3` version of IsaacGym.

## Conda-Environment

```bash
conda create -n safe python=3.8
conda activate safe
# because the cuda version, we recommend you install pytorch manual.
pip install -e .
```

> For detailed instructions, please refer to [Installation.md](Installation.md).

## Getting Started

### One line benchmark

To verify the performance of SafePO, you can run the following one line:

```bash
conda create -n safepo python=3.8
conda activate safepo
make benchmark
``````

### Single-Agent

each algorithm file is the entrance. Running `ALGO.py` with arguments about algorithms and environments does the training. For example, to run PPO-Lag in SafetyPointGoal1-v0 with seed 0, you can use the following command:

```bash
cd safepo/single_agent
python ppo_lag.py --task SafetyPointGoal1-v0 --seed 0
```

To run a benchmark parallelly, for example, you can use the following command to run `PPO-Lag`, `TRPO-Lag` in `SafetyAntVelocity-v1`, `SafetyHalfCheetahVelocity-v1`: 

```bash
cd safepo/single_agent
python benchmark.py --tasks SafetyAntVelocity-v1 SafetyHalfCheetahVelocity-v1 --algo ppo_lag trpo_lag --workers 2
```

The command above will run two processes in parallel, each process will run one algorithm in one environment. The results will be saved in `./runs/`.

### Multi-Agent

We also provide a safe MARL algorithm benchmark for safe MARL research on the challenging tasks of Safety DexterousHands and Safety-Gymnasium multi-agent velocity tasks. HAPPO, MACPO, MAPPO-Lag and MAPPO have already been implemented.


#### Multi-Agent Training

To train a multi-agent algorithm:

```bash
cd safepo/multi_agent
python macpo.py --task Safety2x4AntVelocity-v0 --experiment benchmark
```

You can also train on isaac-gym based environment:

```bash
cd safepo/multi_agent
python macpo.py --task ShadowHandOver_Safe_joint --experiment benchmark
```

**As Isaac Gym is not holding in PyPI, you should install it manually, then clone [Safety-Gymnasium](https://github.com/PKU-Alignment/safety-gymnasium) instead of installing from PyPI.**

### Plot the result

After running the experiment, you can use the following command to plot the results:

```bash
cd safepo
python plot.py --logdir ./runs/benchmark
```

## Machine Configuration

We test all algorithms and experiments on **CPU: AMD Ryzen Threadripper PRO 3975WX 32-Cores** and **GPU: NVIDIA GeForce RTX 3090, Driver Version: 495.44**.

## Ethical and Responsible Use

SafePO aims to benefit safe RL community research, and is released under the [Apache-2.0 license](https://github.com/PKU-Alignment/Safe-Policy-Optimization/blob/main/LICENSE). Illegal usage or any violation of the license is not allowed.

## PKU-Alignment Team

The Baseline is a project contributed by PKU-Alignment at Peking University. We also thank the list of contributors of the following open source repositories:
[Spinning Up](https://spinningup.openai.com/en/latest/), [Bullet-Safety-Gym](https://github.com/SvenGronauer/Bullet-Safety-Gym/tree/master/bullet_safety_gym/envs), [Safety-Gym](https://github.com/openai/safety-gym).
