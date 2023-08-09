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
from typing import Callable

import numpy as np
import torch
import torch.nn as nn
import torch.optim
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from safepo.common.buffer import VectorizedOnPolicyBuffer
from safepo.common.env import make_sa_mujoco_env
from safepo.common.lagrange import Lagrange
from safepo.common.logger import EpochLogger
from safepo.common.model import ActorVCritic
from safepo.utils.config import single_agent_args


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

    args = parser.parse_args()
    return args


def get_flat_params_from(model: torch.nn.Module) -> torch.Tensor:
    flat_params = []
    for _, param in model.named_parameters():
        if param.requires_grad:
            data = param.data
            data = data.view(-1)  # flatten tensor
            flat_params.append(data)
    assert flat_params, "No gradients were found in model parameters."
    return torch.cat(flat_params)


def conjugate_gradients(
    fisher_product: Callable[[torch.Tensor], torch.Tensor],
    policy: ActorVCritic,
    fvp_obs: torch.Tensor,
    vector_b: torch.Tensor,
    num_steps: int = 10,
    residual_tol: float = 1e-10,
    eps: float = 1e-6,
) -> torch.Tensor:
    vector_x = torch.zeros_like(vector_b)
    vector_r = vector_b - fisher_product(vector_x, policy, fvp_obs)
    vector_p = vector_r.clone()
    rdotr = torch.dot(vector_r, vector_r)

    for _ in range(num_steps):
        vector_z = fisher_product(vector_p, policy, fvp_obs)
        alpha = rdotr / (torch.dot(vector_p, vector_z) + eps)
        vector_x += alpha * vector_p
        vector_r -= alpha * vector_z
        new_rdotr = torch.dot(vector_r, vector_r)
        if torch.sqrt(new_rdotr) < residual_tol:
            break
        vector_mu = new_rdotr / (rdotr + eps)
        vector_p = vector_r + vector_mu * vector_p
        rdotr = new_rdotr
    return vector_x


def set_param_values_to_model(model: torch.nn.Module, vals: torch.Tensor) -> None:
    assert isinstance(vals, torch.Tensor)
    i: int = 0
    for _, param in model.named_parameters():
        if param.requires_grad:  # param has grad and, hence, must be set
            orig_size = param.size()
            size = np.prod(list(param.size()))
            new_values = vals[i : int(i + size)]
            # set new param values
            new_values = new_values.view(orig_size)
            param.data = new_values
            i += int(size)  # increment array position
    assert i == len(vals), f"Lengths do not match: {i} vs. {len(vals)}"


def get_flat_gradients_from(model: torch.nn.Module) -> torch.Tensor:
    grads = []
    for _, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            grad = param.grad
            grads.append(grad.view(-1))  # flatten tensor and append
    assert grads, "No gradients were found in model parameters."
    return torch.cat(grads)


def fvp(
    params: torch.Tensor,
    policy: ActorVCritic,
    fvp_obs: torch.Tensor,
) -> torch.Tensor:
    policy.actor.zero_grad()
    current_distribution = policy.actor(fvp_obs)
    with torch.no_grad():
        old_distribution = policy.actor(fvp_obs)
    kl = torch.distributions.kl.kl_divergence(
        old_distribution, current_distribution
    ).mean()

    grads = torch.autograd.grad(kl, tuple(policy.actor.parameters()), create_graph=True)
    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])

    kl_p = (flat_grad_kl * params).sum()
    grads = torch.autograd.grad(
        kl_p,
        tuple(policy.actor.parameters()),
        retain_graph=False,
    )

    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads])

    return flat_grad_grad_kl + params * 0.1


def main(args):
    # set the random seed, device and number of threads
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True
    torch.set_num_threads(4)
    device = torch.device(
        "cuda" if torch.cuda.is_available() and args.device == "cuda" else "cpu"
    )

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
        fvp_obs = data["obs"][:: 1]
        theta_old = get_flat_params_from(policy.actor)
        policy.actor.zero_grad()

        # comnpute advantage
        advantage = data["adv_r"] - lagrange.lagrangian_multiplier * data["adv_c"]
        advantage /= lagrange.lagrangian_multiplier + 1

        # compute loss_pi
        distribution = policy.actor(data["obs"])
        log_prob = distribution.log_prob(data["act"]).sum(dim=-1)
        ratio = torch.exp(log_prob - data["log_prob"])
        loss_pi = -(ratio * advantage).mean()

        loss_pi.backward()

        grads = -get_flat_gradients_from(policy.actor)
        x = conjugate_gradients(fvp, policy, fvp_obs, grads, 15)
        assert torch.isfinite(x).all(), "x is not finite"
        xHx = torch.dot(x, fvp(x, policy, fvp_obs))
        assert xHx.item() >= 0, "xHx is negative"
        alpha = torch.sqrt(2 * 0.01 / (xHx + 1e-8))
        step_direction = x * alpha
        assert torch.isfinite(step_direction).all(), "step_direction is not finite"

        theta_new = theta_old + step_direction
        set_param_values_to_model(policy.actor, theta_new)
        with torch.no_grad():
            new_distribution = policy.actor(data["obs"])
            final_kl = (
                torch.distributions.kl.kl_divergence(distribution, new_distribution)
                .mean()
                .item()
            )

        logger.store(
            **{
                "Misc/Alpha": alpha.item(),
                "Misc/FinalStepNorm": torch.norm(step_direction).mean().item(),
                "Misc/xHx": xHx.item(),
                "Misc/gradient_norm": torch.norm(grads).mean().item(),
                "Misc/H_inv_g": x.norm().item(),
                "Loss/Loss_actor": loss_pi.mean().item(),
                "Train/KL": final_kl,
            },
        )

        dataloader = DataLoader(
            dataset=TensorDataset(
                data["obs"],
                data["target_value_r"],
                data["target_value_c"],
            ),
            batch_size=128,
            shuffle=True,
        )
        for _ in track(range(10), description="Updating..."):
            for (
                obs_b,
                target_value_r_b,
                target_value_c_b,
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

                logger.store(
                    **{
                        "Loss/Loss_reward_critic": loss_r.mean().item(),
                        "Loss/Loss_cost_critic": loss_c.mean().item(),
                    }
                )
        update_end_time = time.time()

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
        logger.log_tabular("Train/KL")
        logger.log_tabular("Train/LagragianMultiplier", lagrange.lagrangian_multiplier)
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
        logger.log_tabular("Misc/Alpha")
        logger.log_tabular("Misc/FinalStepNorm")
        logger.log_tabular("Misc/xHx")
        logger.log_tabular("Misc/gradient_norm")
        logger.log_tabular("Misc/H_inv_g")

        logger.dump_tabular()
        if epoch % 100 == 0:
            logger.torch_save(itr=epoch)
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
