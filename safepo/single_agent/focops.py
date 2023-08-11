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
from torch.distributions import Normal
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.optim.lr_scheduler import LinearLR
from torch.utils.data import DataLoader, TensorDataset

from safepo.common.buffer import VectorizedOnPolicyBuffer
from safepo.common.env import make_sa_mujoco_env
from safepo.common.lagrange import Lagrange
from safepo.common.logger import EpochLogger
from safepo.common.model import ActorVCritic



def parse_args():
    # training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")
    parser.add_argument("--focops-eta", type=float, default=0.02, help="the eta of the focops")
    parser.add_argument("--focops-lam", type=float, default=1.5, help="the lambda of the focops")

    args = parser.parse_args()
    return args

def single_agent_args():
    # training parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0, help="seed of the experiment")
    parser.add_argument("--device", type=str, default="cpu", help="the device (cpu or cuda) to run the code")
    parser.add_argument("--num-envs", type=int, default=10, help="the number of parallel game environments")
    parser.add_argument("--total-steps", type=int, default=10000000, help="total timesteps of the experiments",)
    parser.add_argument("--env-id", type=str, default="SafetyPointGoal1-v0", help="the id of the environment",)
    parser.add_argument("--use-eval", type=lambda x: bool(strtobool(x)), default=False, help="toggles evaluation",)
    # general algorithm parameters
    parser.add_argument("--steps-per-epoch", type=int, default=20000, help="the number of steps to run in each environment per policy rollout",)
    parser.add_argument("--critic-lr", type=float, default=1e-3, help="the learning rate of the critic network")
    # logger parameters
    parser.add_argument("--log-dir", type=str, default="../runs", help="directory to save agent logs")
    parser.add_argument("--write-terminal", type=lambda x: bool(strtobool(x)), default=True, help="toggles terminal logging")
    parser.add_argument("--use-tensorboard", type=lambda x: bool(strtobool(x)), default=False, help="toggles tensorboard logging")
    # algorithm specific parameters
    parser.add_argument("--cost-limit", type=float, default=25.0, help="the cost limit for the safety constraint")
    parser.add_argument("--lagrangian-multiplier-init", type=float, default=0.001, help="the initial value of the lagrangian multiplier")
    parser.add_argument("--lagrangian-multiplier-lr", type=float, default=0.035, help="the learning rate of the lagrangian multiplier")
    parser.add_argument("--focops-eta", type=float, default=0.02, help="the eta of the focops")
    parser.add_argument("--focops-lam", type=float, default=1.5, help="the lambda of the focops")
    args = parser.parse_args()
    return args

def main(args):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device(args.device)

    # set training steps
    local_steps_per_epoch = args.steps_per_epoch // args.num_envs
    epochs = args.total_steps // args.steps_per_epoch

    env, obs_space, act_space = make_sa_mujoco_env(
        num_envs=args.num_envs, env_id=args.env_id, seed=args.seed
    )
    eval_env, _, _ = make_sa_mujoco_env(num_envs=1, env_id=args.env_id, seed=None)

    # create the actor-critic module
    policy = ActorVCritic(
        obs_dim=obs_space.shape[0],
        act_dim=act_space.shape[0],
    ).to(device)
    actor_optimizer = torch.optim.Adam(policy.actor.parameters(), lr=3e-4)
    actor_scheduler = LinearLR(
        actor_optimizer,
        start_factor=1.0,
        end_factor=0.0,
        total_iters=epochs,
        verbose=True,
    )
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
        size=local_steps_per_epoch,
        device=device,
        num_envs=args.num_envs,
    )

    # setup lagrangian multiplier
    lagrange = Lagrange(
        cost_limit=args.cost_limit,
        lagrangian_multiplier_init=args.lagrangian_multiplier_init,
        lagrangian_multiplier_lr=args.lagrangian_multiplier_lr,
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
        with torch.no_grad():
            old_distribution = policy.actor(data["obs"])
            old_mean = old_distribution.mean
            old_std = old_distribution.stddev

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
                old_mean,
                old_std,
            ),
            batch_size=64,
            shuffle=True,
        )
        update_counts = 0
        final_kl = torch.ones_like(old_distribution.loc)
        for i in track(range(40), description="Updating..."):
            for (
                obs_b,
                act_b,
                log_prob_b,
                target_value_r_b,
                target_value_c_b,
                adv_b,
                old_mean_b,
                old_std_b,
            ) in dataloader:
                reward_critic_optimizer.zero_grad()
                loss_r = nn.functional.mse_loss(
                    policy.reward_critic(obs_b), target_value_r_b
                )
                for param in policy.reward_critic.parameters():
                    loss_r += param.pow(2).sum() * 0.001
                loss_r.backward()
                clip_grad_norm_(policy.reward_critic.parameters(), 40.0)
                reward_critic_optimizer.step()

                cost_critic_optimizer.zero_grad()
                loss_c = nn.functional.mse_loss(policy.cost_critic(obs_b), target_value_c_b)
                for param in policy.cost_critic.parameters():
                    loss_c += param.pow(2).sum() * 0.001
                loss_c.backward()
                clip_grad_norm_(policy.cost_critic.parameters(), 40.0)
                cost_critic_optimizer.step()

                old_distribution_b = Normal(loc=old_mean_b, scale=old_std_b)

                distribution = policy.actor(obs_b)
                log_prob = distribution.log_prob(act_b).sum(dim=-1)
                ratio = torch.exp(log_prob - log_prob_b)
                temp_kl = torch.distributions.kl_divergence(
                    distribution, old_distribution_b
                ).sum(-1, keepdim=True)

                loss_pi = (temp_kl - (1 / 1.5) * ratio * adv_b) * (
                    temp_kl.detach() <= 0.02
                ).type(torch.float32)

                loss_pi = loss_pi.mean()

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
            if kl > 0.02:
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
        if (epoch+1) % 100 == 0 or epoch == 0:
            logger.torch_save(itr=epoch)
            logger.save_state(
                state_dict={
                    "Normalizer": env.obs_rms,
                },
                itr = epoch
            )
    logger.close()


if __name__ == "__main__":
    args = single_agent_args()
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
