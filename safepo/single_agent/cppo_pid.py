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

import argparse
import os
import random
import sys
import time
from collections import deque
from distutils.util import strtobool

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import ConstantLR, LinearLR
from torch.utils.data import DataLoader, TensorDataset

from safepo.common.buffer import VectorizedOnPolicyBuffer
from safepo.common.env import make_env
from safepo.common.lagrange import PIDLagrangian as Lagrange
from safepo.common.logger import EpochLogger
from safepo.common.model import ActorVCritic


def parse_args():
    # training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="the device (cpu or cuda) to run the code",
    )
    parser.add_argument(
        "--torch-threads", type=int, default=4, help="number of threads for torch"
    )
    parser.add_argument(
        "--num-envs",
        type=int,
        default=10,
        help="the number of parallel game environments",
    )
    parser.add_argument(
        "--total-steps",
        type=int,
        default=10000000,
        help="total timesteps of the experiments",
    )
    parser.add_argument(
        "--env-id",
        type=str,
        default="SafetyPointGoal1-v0",
        help="the id of the environment",
    )
    parser.add_argument(
        "--use-eval",
        type=lambda x: bool(strtobool(x)),
        default=False,
        nargs="?",
        const=False,
        help="toggles evaluation",
    )
    parser.add_argument(
        "--eval-episodes",
        type=int,
        default=3,
        help="the number of episodes for final evaluation",
    )
    # general algorithm parameters
    parser.add_argument(
        "--steps-per-epoch",
        type=int,
        default=20000,
        help="the number of steps to run in each environment per policy rollout",
    )
    parser.add_argument(
        "--update-iters",
        type=int,
        default=40,
        help="the max iteration to update the policy",
    )
    parser.add_argument(
        "--batch-size", type=int, default=64, help="the number of mini-batches"
    )
    parser.add_argument(
        "--entropy_coef", type=float, default=0.0, help="coefficient of the entropy"
    )
    parser.add_argument(
        "--target-kl",
        type=float,
        default=0.02,
        help="the target KL divergence threshold",
    )
    parser.add_argument(
        "--max-grad-norm",
        type=float,
        default=40.0,
        help="the maximum norm for the gradient clipping",
    )
    parser.add_argument(
        "--critic-norm-coef",
        type=float,
        default=0.001,
        help="the critic norm coefficient",
    )
    parser.add_argument(
        "--gamma", type=float, default=0.99, help="the discount factor gamma"
    )
    parser.add_argument(
        "--lam",
        type=float,
        default=0.95,
        help="the lambda for the reward general advantage estimation",
    )
    parser.add_argument(
        "--lam-c",
        type=float,
        default=0.95,
        help="the lambda for the cost general advantage estimation",
    )
    parser.add_argument(
        "--standardized-adv-r",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="toggles reward advantages standardization",
    )
    parser.add_argument(
        "--standardized-adv-c",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="toggles cost advantages standardization",
    )
    parser.add_argument(
        "--actor-lr",
        type=float,
        default=3e-4,
        help="the learning rate of the actor network",
    )
    parser.add_argument(
        "--critic-lr",
        type=float,
        default=3e-4,
        help="the learning rate of the critic network",
    )
    parser.add_argument(
        "--linear-lr-decay",
        type=lambda x: bool(strtobool(x)),
        default=True,
        nargs="?",
        const=True,
        help="toggles learning rate annealing for policy and value networks",
    )
    # logger parameters
    parser.add_argument(
        "--log-dir",
        type=str,
        default="../runs",
        help="directory to save agent logs",
    )
    parser.add_argument(
        "--write-terminal",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="toggles terminal logging",
    )
    parser.add_argument(
        "--use-tensorboard",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="toggles tensorboard logging",
    )
    # algorithm specific parameters
    parser.add_argument(
        "--clip", type=float, default=0.2, help="the surrogate clipping coefficient"
    )
    parser.add_argument(
        "--cost-limit",
        type=float,
        default=25.0,
        help="the cost limit for the safety constraint",
    )
    parser.add_argument(
        "--lagrangian-multiplier-init",
        type=float,
        default=0.001,
        help="the initial value of the lagrangian multiplier",
    )
    parser.add_argument(
        "--pid-kp", type=float, default=0.1, help="the Kp of PID controller"
    )
    parser.add_argument(
        "--pid-ki", type=float, default=0.01, help="the Ki of PID controller"
    )
    parser.add_argument(
        "--pid-kd", type=float, default=0.01, help="the Kd of PID controller"
    )
    parser.add_argument(
        "--pid-d-delay", type=int, default=10, help="the delay of derivative term"
    )
    parser.add_argument(
        "--pid-delta-p-ema-alpha",
        type=float,
        default=0.95,
        help="the exponential moving average alpha of the proportional term of the PID controller.",
    )
    parser.add_argument(
        "--pid-delta-d-ema-alpha",
        type=float,
        default=0.95,
        help="the exponential moving average alpha of the derivative term of the PID controller.",
    )
    parser.add_argument(
        "--sum-norm",
        type=lambda x: bool(strtobool(x)),
        default=True,
        help="toggles to normalize the sum of the cost",
    )
    parser.add_argument(
        "--diff-norm",
        type=lambda x: bool(strtobool(x)),
        default=False,
        help="toggles to normalize the derivate of the cost",
    )
    parser.add_argument(
        "--penalty-max", type=float, default=100.0, help="the max penalty coefficient"
    )

    args = parser.parse_args()
    return args


def main(args):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(args.torch_threads)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

    # set training steps
    local_steps_per_epoch = args.steps_per_epoch // args.num_envs
    epochs = args.total_steps // args.steps_per_epoch

    env, obs_space, act_space = make_env(
        num_envs=args.num_envs, env_id=args.env_id, seed=args.seed
    )
    eval_env, _, _ = make_env(num_envs=1, env_id=args.env_id, seed=None)

    # create the actor-critic module
    policy = ActorVCritic(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
    ).to(device)
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=args.actor_lr)
    if args.linear_lr_decay:
        actor_scheduler = LinearLR(
            actor_optimizer,
            start_factor=1.0,
            end_factor=0.0,
            total_iters=epochs,
            verbose=True,
        )
    else:
        actor_scheduler = ConstantLR(actor_optimizer)
    reward_critic_optimizer = torch.optim.Adam(
        policy.reward_critic.parameters(), lr=args.critic_lr
    )
    cost_critic_optimizer = torch.optim.Adam(
        policy.cost_critic.parameters(), lr=args.critic_lr
    )

    # create the vectorized on-policy buffer
    buffer = VectorizedOnPolicyBuffer(
        obs_space=obs_space,
        act_space=act_space,
        size=args.steps_per_epoch,
        gamma=0.99,
        lam=0.95,
        lam_c=0.95,
        standardized_adv_r=True,
        standardized_adv_c=True,
        device=device,
        num_envs=args.num_envs,
    )

    # setup lagrangian multiplier
    lagrange = Lagrange(
        pid_kd=args.pid_kd,
        pid_kp=args.pid_kp,
        pid_ki=args.pid_ki,
        pid_d_delay=args.pid_d_delay,
        pid_delta_p_ema_alpha=args.pid_delta_p_ema_alpha,
        pid_delta_d_ema_alpha=args.pid_delta_d_ema_alpha,
        sum_norm=args.sum_norm,
        diff_norm=args.diff_norm,
        penalty_max=args.penalty_max,
        cost_limit=args.cost_limit,
        lagrangian_multiplier_init=args.lagrangian_multiplier_init,
    )

    # set up the logger
    dict_args = vars(args)
    logger = EpochLogger(
        log_dir=args.log_dir,
        seed=str(args.seed),
        use_tensorboard=args.use_tensorboard,
    )
    rew_deque = deque(maxlen=50)
    cost_deque = deque(maxlen=50)
    len_deque = deque(maxlen=50)
    eval_rew_deque = deque(maxlen=50)
    eval_cost_deque = deque(maxlen=50)
    eval_len_deque = deque(maxlen=50)
    logger.save_config(dict_args)
    logger.setup_torch_saver(policy.actor)
    logger.log("Start with training.")

    time.time()

    # training loop
    for epoch in range(epochs):
        rollout_start_time = time.time()
        obs, _ = env.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32, device=device)
        ep_ret, ep_cost, ep_len = (
            np.zeros(args.num_envs),
            np.zeros(args.num_envs),
            np.zeros(args.num_envs),
        )

        # collect samples until we have enough to update
        for steps in range(local_steps_per_epoch):
            with torch.no_grad():
                act, log_prob, value_r, value_c = policy.step(obs, deterministic=False)
            next_obs, reward, cost, terminated, truncated, info = env.step(
                act.detach().squeeze().cpu().numpy()
            )
            ep_ret += reward
            ep_cost += cost
            ep_len += 1
            next_obs, reward, cost, terminated, truncated = (
                torch.as_tensor(x, dtype=torch.float32, device=device)
                for x in (next_obs, reward, cost, terminated, truncated)
            )
            if "final_observation" in info:
                info["final_observation"] = np.array(
                    [
                        array if array is not None else np.zeros(obs.shape[-1])
                        for array in info["final_observation"]
                    ],
                )
                info["final_observation"] = torch.as_tensor(
                    info["final_observation"],
                    dtype=torch.float32,
                    device=device,
                )
            buffer.store(
                obs=obs,
                act=act,
                reward=reward,
                cost=cost,
                value_r=value_r,
                value_c=value_c,
                log_prob=log_prob,
            )

            obs = next_obs
            epoch_end = steps >= local_steps_per_epoch - 1
            for idx, (done, time_out) in enumerate(zip(terminated, truncated)):
                if epoch_end or done or time_out:
                    last_value_r = torch.zeros(1, device=device)
                    last_value_c = torch.zeros(1, device=device)
                    if not done:
                        if epoch_end:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    obs[idx], deterministic=False
                                )
                        if time_out:
                            with torch.no_grad():
                                _, _, last_value_r, last_value_c = policy.step(
                                    info["final_observation"][idx], deterministic=False
                                )
                        last_value_r = last_value_r.unsqueeze(0)
                        last_value_c = last_value_c.unsqueeze(0)
                    if done or time_out:
                        rew_deque.append(ep_ret[idx])
                        cost_deque.append(ep_cost[idx])
                        len_deque.append(ep_len[idx])
                        logger.store(
                            **{
                                "Metrics/EpRet": np.mean(rew_deque),
                                "Metrics/EpCost": np.mean(cost_deque),
                                "Metrics/EpLen": np.mean(len_deque),
                            }
                        )
                        ep_ret[idx] = 0.0
                        ep_cost[idx] = 0.0
                        ep_len[idx] = 0.0

                    buffer.finish_path(
                        last_value_r=last_value_r, last_value_c=last_value_c, idx=idx
                    )
        rollout_end_time = time.time()

        eval_start_time = time.time()

        eval_episodes = 1 if epoch < epochs - 1 else 10
        if args.use_eval:
            for _ in range(eval_episodes):
                eval_done = False
                eval_obs, _ = eval_env.reset()
                eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=device)
                eval_rew, eval_cost, eval_len = 0.0, 0.0, 0.0
                while not eval_done:
                    with torch.no_grad():
                        act, log_prob, value_r, value_c = policy.step(
                            eval_obs, deterministic=True
                        )
                    next_obs, reward, cost, terminated, truncated, info = env.step(
                        act.detach().squeeze().cpu().numpy()
                    )
                    next_obs = torch.as_tensor(
                        next_obs, dtype=torch.float32, device=device
                    )
                    eval_rew += reward
                    eval_cost += cost
                    eval_len += 1
                    eval_done = terminated[0] or truncated[0]
                    eval_obs = next_obs
                eval_rew_deque.append(eval_rew)
                eval_cost_deque.append(eval_cost)
                eval_len_deque.append(eval_len)
            logger.store(
                **{
                    "Metrics/EvalEpRet": np.mean(eval_rew),
                    "Metrics/EvalEpCost": np.mean(eval_cost),
                    "Metrics/EvalEpLen": np.mean(eval_len),
                }
            )

        eval_end_time = time.time()

        # update lagrange multiplier
        ep_costs = logger.get_stats("Metrics/EpCost")
        lagrange.update_lagrange_multiplier(ep_costs)

        # update policy
        data = buffer.get()
        old_distribution = policy.actor(data["obs"])

        # comnpute advantage
        advantage = data["adv_r"] - lagrange.lagrangian_multiplier * data["adv_c"]
        advantage /= lagrange.lagrangian_multiplier + 1

        dataloader = DataLoader(
            dataset=TensorDataset(
                data["obs"],
                data["act"],
                data["log_prob"],
                data["target_value_r"],
                data["target_value_c"],
                advantage,
            ),
            batch_size=128,
            shuffle=True,
        )
        update_counts = 0
        final_kl = torch.ones_like(old_distribution.loc)
        for i in track(range(args.update_iters), description="Updating..."):
            for (
                obs_b,
                act_b,
                log_prob_b,
                target_value_r_b,
                target_value_c_b,
                adv_b,
            ) in dataloader:
                reward_critic_optimizer.zero_grad()
                loss_r = nn.functional.mse_loss(
                    policy.reward_critic(obs_b), target_value_r_b
                )
                for param in policy.reward_critic.parameters():
                    loss_r += param.pow(2).sum() * 0.001
                loss_r.backward()
                clip_grad_norm_(
                    policy.reward_critic.parameters(),
                    40.0,
                )
                reward_critic_optimizer.step()

                cost_critic_optimizer.zero_grad()
                loss_c = nn.functional.mse_loss(
                    policy.cost_critic(obs_b), target_value_c_b
                )
                for param in policy.cost_critic.parameters():
                    loss_c += param.pow(2).sum() * 0.001
                loss_c.backward()
                clip_grad_norm_(
                    policy.cost_critic.parameters(),
                    40.0,
                )
                cost_critic_optimizer.step()

                distribution = policy.actor(obs_b)
                log_prob = distribution.log_prob(act_b).sum(dim=-1)
                ratio = torch.exp(log_prob - log_prob_b)
                ratio_cliped = torch.clamp(
                    ratio,
                    1 - args.clip,
                    1 + args.clip,
                )
                loss_pi = -torch.min(ratio * adv_b, ratio_cliped * adv_b).mean()
                actor_optimizer.zero_grad()
                loss_pi.backward()
                clip_grad_norm_(policy.actor.parameters(), 40.0)
                actor_optimizer.step()

                logger.store(
                    **{
                        "Loss/Loss_reward_critic": loss_r.mean().item(),
                        "Loss/Loss_cost_critic": loss_c.mean().item(),
                        "Loss/Loss_actor": loss_pi.mean().item(),
                    }
                )
            new_distribution = policy.actor(data["obs"])
            kl = (
                torch.distributions.kl.kl_divergence(old_distribution, new_distribution)
                .sum(-1, keepdim=True)
                .mean()
                .item()
            )
            final_kl = kl
            update_counts += 1
            if kl > args.target_kl:
                logger.log(f"Early stopping at iter {i + 1} due to reaching max kl")
                break
        update_end_time = time.time()
        actor_scheduler.step()

        # log data
        logger.log_tabular("Metrics/EpRet")
        logger.log_tabular("Metrics/EpCost")
        logger.log_tabular("Metrics/EpLen")
        if args.use_eval:
            logger.log_tabular("Metrics/EvalEpRet")
            logger.log_tabular("Metrics/EvalEpCost")
            logger.log_tabular("Metrics/EvalEpLen")
        logger.log_tabular("Train/Epoch", epoch + 1)
        logger.log_tabular("Train/TotalSteps", (epoch + 1) * args.steps_per_epoch)
        logger.log_tabular("Train/StopIter", update_counts)
        logger.log_tabular("Train/KL", final_kl)
        logger.log_tabular("Train/LagragianMultiplier", lagrange.lagrangian_multiplier)
        logger.log_tabular("Train/LR", actor_scheduler.get_last_lr()[0])
        logger.log_tabular("Loss/Loss_reward_critic")
        logger.log_tabular("Loss/Loss_cost_critic")
        logger.log_tabular("Loss/Loss_actor")
        logger.log_tabular("Time/Rollout", rollout_end_time - rollout_start_time)
        if args.use_eval:
            logger.log_tabular("Time/Eval", eval_end_time - eval_start_time)
        logger.log_tabular("Time/Update", update_end_time - eval_end_time)
        logger.log_tabular("Time/Total", update_end_time - rollout_start_time)
        logger.log_tabular("Value/RewardAdv", data["adv_r"].mean().item())
        logger.log_tabular("Value/CostAdv", data["adv_c"].mean().item())

        logger.dump_tabular()
        if epoch % 100 == 0:
            logger.torch_save(itr=epoch)
    logger.close()


if __name__ == "__main__":
    args = parse_args()
    relpath = time.strftime("%Y-%m-%d-%H-%M-%S")
    subfolder = "-".join(["seed", str(args.seed).zfill(3)])
    relpath = "-".join([subfolder, relpath])
    algo = os.path.basename(__file__).split(".")[0]
    args.log_dir = os.path.join(args.log_dir, args.env_id, algo, relpath)
    if not args.write_terminal:
        terminal_log_name = "terminal.log"
        error_log_name = "error.log"
        terminal_log_name = f"seed{args.seed}_{terminal_log_name}"
        error_log_name = f"seed{args.seed}_{error_log_name}"
        sys.stdout = sys.__stdout__
        sys.stderr = sys.__stderr__
        if not os.path.exists(args.log_dir):
            os.makedirs(args.log_dir, exist_ok=True)
        with open(
            os.path.join(
                f"{args.log_dir}",
                terminal_log_name,
            ),
            "w",
            encoding="utf-8",
        ) as f_out:
            sys.stdout = f_out
            with open(
                os.path.join(
                    f"{args.log_dir}",
                    error_log_name,
                ),
                "w",
                encoding="utf-8",
            ) as f_error:
                sys.stderr = f_error
                main(args)
    else:
        main(args)
