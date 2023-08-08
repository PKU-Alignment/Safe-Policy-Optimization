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

import os
import random
import sys
import time
from collections import deque
from typing import Callable

from distutils.util import strtobool
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim
from rich.progress import track
from torch.nn.utils.clip_grad import clip_grad_norm_
from torch.utils.data import DataLoader, TensorDataset

from safepo.common.buffer import VectorizedOnPolicyBuffer
from safepo.common.env import make_sa_mujoco_env
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
                        act, log_prob, value_r, value_c = policy.step(eval_obs, deterministic=True)
                    next_obs, reward, cost, terminated, truncated, info = env.step(
                        act.detach().squeeze().cpu().numpy()
                    )
                    next_obs = torch.as_tensor(next_obs, dtype=torch.float32, device=device)
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

        # update policy
        data = buffer.get()
        fvp_obs = data["obs"][:: 1]
        theta_old = get_flat_params_from(policy.actor)
        policy.actor.zero_grad()
        # compute loss_pi
        temp_distribution = policy.actor(data["obs"])
        log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
        ratio = torch.exp(log_prob - data["log_prob"])
        loss_pi_r = -(ratio * data["adv_r"]).mean()
        loss_reward_before = loss_pi_r.item()
        old_distribution = policy.actor(data["obs"])

        loss_pi_r.backward()

        grads = -get_flat_gradients_from(policy.actor)
        x = conjugate_gradients(fvp, policy, fvp_obs, grads, 15)
        assert torch.isfinite(x).all(), "x is not finite"
        xHx = torch.dot(x, fvp(x, policy, fvp_obs))
        assert xHx.item() >= 0, "xHx is negative"
        alpha = torch.sqrt(2 * 0.01 / (xHx + 1e-8))

        policy.actor.zero_grad()
        temp_distribution = policy.actor(data["obs"])
        log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
        ratio = torch.exp(log_prob - data["log_prob"])
        loss_pi_c = (ratio * data["adv_c"]).mean()
        loss_cost_before = loss_pi_c.item()

        loss_pi_c.backward()

        b_grads = get_flat_gradients_from(policy.actor)
        ep_costs = logger.get_stats("Metrics/EpCost") - args.cost_limit

        p = conjugate_gradients(fvp, policy, fvp_obs, b_grads, 15)
        q = xHx
        r = grads.dot(p)
        s = b_grads.dot(p)

        if b_grads.dot(b_grads) <= 1e-6 and ep_costs < 0:
            A = torch.zeros(1)
            B = torch.zeros(1)
            optim_case = 4
        else:
            assert torch.isfinite(r).all(), "r is not finite"
            assert torch.isfinite(s).all(), "s is not finite"

            A = q - r**2 / (s + 1e-8)
            B = 2 * 0.01 - ep_costs**2 / (s + 1e-8)

            if ep_costs < 0 and B < 0:
                optim_case = 3
            elif ep_costs < 0 <= B:
                optim_case = 2
            elif ep_costs >= 0 and B >= 0:
                optim_case = 1
                logger.log("Alert! Attempting feasible recovery!", "yellow")
            else:
                optim_case = 0
                logger.log("Alert! Attempting infeasible recovery!", "red")

        if optim_case in (3, 4):
            alpha = torch.sqrt(2 * 0.01 / (xHx + 1e-8))
            nu_star = torch.zeros(1)
            lambda_star = 1 / (alpha + 1e-8)
            step_direction = alpha * x

        elif optim_case in (1, 2):

            def project(
                data: torch.Tensor, low: torch.Tensor, high: torch.Tensor
            ) -> torch.Tensor:
                """Project data to [low, high] interval."""
                return torch.clamp(data, low, high)

            lambda_a = torch.sqrt(A / B)
            lambda_b = torch.sqrt(q / (2 * 0.01))
            r_num = r.item()
            eps_cost = ep_costs + 1e-8
            if ep_costs < 0:
                lambda_a_star = project(
                    lambda_a, torch.as_tensor(0.0), r_num / eps_cost
                )
                lambda_b_star = project(
                    lambda_b, r_num / eps_cost, torch.as_tensor(torch.inf)
                )
            else:
                lambda_a_star = project(
                    lambda_a, r_num / eps_cost, torch.as_tensor(torch.inf)
                )
                lambda_b_star = project(
                    lambda_b, torch.as_tensor(0.0), r_num / eps_cost
                )

            def f_a(lam: torch.Tensor) -> torch.Tensor:
                return -0.5 * (A / (lam + 1e-8) + B * lam) - r * ep_costs / (s + 1e-8)

            def f_b(lam: torch.Tensor) -> torch.Tensor:
                return -0.5 * (q / (lam + 1e-8) + 2 * 0.01 * lam)

            lambda_star = (
                lambda_a_star
                if f_a(lambda_a_star) >= f_b(lambda_b_star)
                else lambda_b_star
            )

            nu_star = torch.clamp(lambda_star * ep_costs - r, min=0) / (s + 1e-8)

            step_direction = 1.0 / (lambda_star + 1e-8) * (x - nu_star * p)

        else:
            lambda_star = torch.zeros(1)
            nu_star = torch.sqrt(2 * 0.01 / (s + 1e-8))
            step_direction = -nu_star * p

        step_frac = 1.0
        theta_old = get_flat_params_from(policy.actor)
        expected_reward_improve = grads.dot(step_direction)

        kl = torch.zeros(1)
        for step in range(15):
            new_theta = theta_old + step_frac * step_direction
            set_param_values_to_model(policy.actor, new_theta)
            acceptance_step = step + 1

            with torch.no_grad():
                try:
                    temp_distribution = policy.actor(data["obs"])
                    log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
                    ratio = torch.exp(log_prob - data["log_prob"])
                    loss_reward = -(ratio * data["adv_r"]).mean()
                except ValueError:
                    step_frac *= 0.8
                    continue
                temp_distribution = policy.actor(data["obs"])
                log_prob = temp_distribution.log_prob(data["act"]).sum(dim=-1)
                ratio = torch.exp(log_prob - data["log_prob"])
                loss_cost = (ratio * data["adv_c"]).mean()
                current_distribution = policy.actor(data["obs"])
                kl = torch.distributions.kl.kl_divergence(
                    old_distribution, current_distribution
                ).mean()
            loss_reward_improve = loss_reward_before - loss_reward.item()
            loss_cost_diff = loss_cost.item() - loss_cost_before

            logger.log(
                f"Expected Improvement: {expected_reward_improve} Actual: {loss_reward_improve}",
            )
            if not torch.isfinite(loss_reward) and not torch.isfinite(loss_cost):
                logger.log("WARNING: loss_pi not finite")
            if not torch.isfinite(kl):
                logger.log("WARNING: KL not finite")
                continue
            if loss_reward_improve < 0 if optim_case > 1 else False:
                logger.log("INFO: did not improve improve <0")
            elif loss_cost_diff > max(-ep_costs, 0):
                logger.log(f"INFO: no improve {loss_cost_diff} > {max(-ep_costs, 0)}")
            elif kl > 0.01:
                logger.log(f"INFO: violated KL constraint {kl} at step {step + 1}.")
            else:
                logger.log(f"Accept step at i={step + 1}")
                break
            step_frac *= 0.8
        else:
            logger.log("INFO: no suitable step found...")
            step_direction = torch.zeros_like(step_direction)
            acceptance_step = 0

        theta_new = theta_old + step_frac * step_direction
        set_param_values_to_model(policy.actor, theta_new)

        logger.store(
            **{
                "Misc/Alpha": alpha.item(),
                "Misc/FinalStepNorm": torch.norm(step_direction).mean().item(),
                "Misc/xHx": xHx.item(),
                "Misc/gradient_norm": torch.norm(grads).mean().item(),
                "Misc/H_inv_g": x.norm().item(),
                "Misc/AcceptanceStep": acceptance_step,
                "Loss/Loss_actor": (loss_pi_r + loss_pi_c).mean().item(),
                "Train/KL": kl,
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
                loss_r = nn.functional.mse_loss(policy.reward_critic(obs_b), target_value_r_b)
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
        logger.log_tabular("Misc/AcceptanceStep")

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
