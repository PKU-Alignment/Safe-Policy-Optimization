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
import time
from copy import deepcopy

from tqdm import tqdm

import numpy as np
import safety_gymnasium
import torch
import torch.optim

from safepo.algorithms.base import PolicyGradient
from safepo.common.core import get_optimizer
from safepo.common.buffer import Buffer
from safepo.common.logger import EpochLogger
from safepo.common.utils import seed_everything
from safepo.models.constraint_actor_critic import ConstraintActorCritic


class PG(PolicyGradient):
    def __init__(self, configs):
        """Policy Gradient."""
        # create Environment
        self.env_id = configs["env_id"]
        self.configs = configs
        self.env = safety_gymnasium.make(self.env_id)

        # set up logger
        self.logger = EpochLogger(**self.configs["logger_kwargs"])
        self.logger.save_config(self.configs)

        # set seed
        seed_everything(self.configs["seed"])

        # setup policy module
        self.policy = ConstraintActorCritic(
            policy_config=self.configs["policy"],
            observation_space=self.env.observation_space,
            action_space=self.env.action_space,
            use_standardized_obs=configs["use_standardized_obs"],
            use_scaled_rewards=configs["use_reward_scaling"],
            use_shared_weights=configs["use_shared_weights"],
            weight_initialization=configs["weight_initialization"],
        )

        # set up buffer
        self.buf = Buffer(
            policy=self.policy,
            obs_dim=self.env.observation_space.shape,
            act_dim=self.env.action_space.shape,
            size=configs["steps_per_epoch"],
            gamma=configs["gamma"],
            lam=configs["lam"],
            advantage_type=configs["advantage_type"],
            use_scaled_rewards=configs["use_reward_scaling"],
            standardize_env_obs=configs["use_standardized_obs"],
            use_standardized_reward=configs["use_standardized_reward"],
            use_standardized_cost=configs["use_standardized_cost"],
            lam_c=configs["lam_c"],
            use_reward_penalty=configs["use_reward_penalty"],
        )

        # set up optimizers for policy module
        self.pi_optimizer = get_optimizer(
            configs["optimizer"], module=self.policy.actor, lr=configs["pi_lr"]
        )
        self.vf_optimizer = get_optimizer(
            "Adam", module=self.policy.critic, lr=configs["vf_lr"]
        )
        if configs["use_cost_value_function"]:
            self.cf_optimizer = get_optimizer(
                "Adam", module=self.policy.cost_critic, lr=configs["vf_lr"]
            )

        # set up scheduler for policy learning rate decay
        self.scheduler = self._init_learning_rate_scheduler()

        # Set up model saving
        self.logger.setup_torch_saver(self.policy.actor)
        self.logger.torch_save()

        # Setup statistics
        self.start_time = time.time()
        self.epoch_time = time.time()
        self.loss_pi_before = 0.0
        self.loss_v_before = 0.0
        self.loss_c_before = 0.0
        self.logger.log("Start with training.")

    def _init_learning_rate_scheduler(self):
        scheduler = None
        if self.configs["use_linear_lr_decay"]:
            # Linear anneal
            def lm(epoch):
                return 1 - epoch / self.epochs

            scheduler = torch.optim.lr_scheduler.LambdaLR(
                optimizer=self.pi_optimizer, lr_lambda=lm
            )
        return scheduler

    def algorithm_specific_logs(self):
        """Use this method to collect log information.
        e.g. log lagrangian for lagrangian-base , log q, r, s, c for cpo, etc
        """
        pass

    def compute_loss_pi(self, data: dict):
        """Computing pi/actor loss.

        Returns:
            torch.Tensor
        """
        # Policy loss
        dist, _log_p = self.policy.actor(data["obs"], data["act"])
        ratio = torch.exp(_log_p - data["log_p"])

        # Compute loss via ratio and advantage
        loss_pi = -(ratio * data["adv"]).mean()
        loss_pi -= self.configs["entropy_coef"] * dist.entropy().mean()

        # Useful extra info
        approx_kl = (
            (0.5 * (dist.mean - data["act"]) ** 2 / dist.stddev**2).mean().item()
        )

        # Compute policy's entropy
        ent = dist.entropy().mean().item()

        pi_info = dict(kl=approx_kl, ent=ent, ratio=ratio.mean().item())

        return loss_pi, pi_info

    def compute_loss_v(self, obs, ret):
        """Computing value loss.

        Returns:
            torch.Tensor
        """
        return ((self.policy.critic(obs) - ret) ** 2).mean()

    def compute_loss_c(self, obs, ret):
        """Computing cost loss.

        Returns:
            torch.Tensor
        """
        return ((self.policy.cost_critic(obs) - ret) ** 2).mean()

    def learn(self):
        """
        This is main function for algorithm update, divided into the following steps:
            (1). self.rollout: collect interactive data from environment
            (2). self.udpate: perform actor/critic updates
            (3). log epoch/update information for visualization and terminal log print.

        Returns:
            model and environment
        """
        # Main loop: collect experience in env and update/log each epoch
        for epoch in range(1, self.configs["epochs"] + 1):
            self.epoch_time = time.time()

            # Update internals of AC
            if self.configs["use_exploration_noise_anneal"]:
                self.policy.update(frac=epoch / self.configs["epochs"])
            # collect data and store
            self.roll_out()
            # if self.algo == "focops":
            #     ep_costs = self.logger.get_stats('EpCosts')[0]
            #     self.update_lagrange_multiplier(ep_costs)

            # Update: actor, critic, running statistics
            self.update()

            # Log and store information
            self.log(epoch)

            # save model
            if epoch % 100 == 0:
                self.logger.torch_save(itr=epoch)

        # close opened files to avoid number of open files overflow
        self.logger.close()
        return self.policy, self.env

    def log(self, epoch: int):
        # Log info about epoch
        total_env_steps = epoch * self.configs["steps_per_epoch"]
        fps = self.configs["steps_per_epoch"] / (time.time() - self.epoch_time)

        # Step the actor learning rate scheduler if provided
        if self.scheduler and self.configs["use_linear_lr_decay"]:
            current_lr = self.scheduler.get_last_lr()[0]
            self.scheduler.step()
        else:
            current_lr = self.configs["pi_lr"]

        self.logger.log_tabular("Epoch", epoch + 1)
        self.logger.log_tabular("EpRet", min_and_max=True, std=True)
        self.logger.log_tabular("EpCosts", min_and_max=True, std=True)
        self.logger.log_tabular("EpLen", min_and_max=True)
        self.logger.log_tabular("Values/V", min_and_max=True)
        self.logger.log_tabular("Values/Adv", min_and_max=True)
        if self.configs["use_cost_value_function"]:
            self.logger.log_tabular("Values/C", min_and_max=True)
        self.logger.log_tabular("Loss/Pi", std=False)
        self.logger.log_tabular("Loss/Value")
        self.logger.log_tabular("Loss/DeltaPi")
        self.logger.log_tabular("Loss/DeltaValue")
        if self.configs["use_cost_value_function"]:
            self.logger.log_tabular("Loss/Cost")
            self.logger.log_tabular("Loss/DeltaCost")
        self.logger.log_tabular("Entropy")
        self.logger.log_tabular("KL")
        self.logger.log_tabular("Misc/StopIter")
        self.logger.log_tabular("Misc/Seed", self.configs["seed"])
        self.logger.log_tabular("PolicyRatio")
        self.logger.log_tabular("LR", current_lr)
        if self.configs["use_reward_scaling"]:
            reward_scale_mean = self.policy.ret_oms.mean.item()
            reward_scale_stddev = self.policy.ret_oms.std.item()
            self.logger.log_tabular("Misc/RewScaleMean", reward_scale_mean)
            self.logger.log_tabular("Misc/RewScaleStddev", reward_scale_stddev)
        if self.configs["use_exploration_noise_anneal"]:
            noise_std = np.exp(self.policy.actor.log_std[0].item())
            self.logger.log_tabular("Misc/ExplorationNoiseStd", noise_std)
        # Some child classes may add information to logs
        self.algorithm_specific_logs()
        self.logger.log_tabular("TotalEnvSteps", total_env_steps)
        self.logger.log_tabular("Time", int(time.time() - self.start_time))
        self.logger.log_tabular("FPS", int(fps))

        self.logger.dump_tabular()

    def pre_process_data(self, raw_data: dict):
        """
        Pre-process data, e.g. standardize observations, rescale rewards if
            enabled by arguments.

        Parameters
        ----------
        raw_data
            dictionary holding information obtain from environment interactions

        Returns
        -------
        dict
            holding pre-processed data, i.e. observations and rewards
        """
        data = deepcopy(raw_data)
        # Note: use_reward_scaling is currently applied in Buffer...
        # If self.use_reward_scaling:
        #     rew = self.ac.ret_oms(data['rew'], subtract_mean=False, clip=True)
        #     data['rew'] = rew

        if self.configs["use_standardized_obs"]:
            obs = data["obs"]
            data["obs"] = self.policy.obs_oms(obs, clip=False)
        return data

    def roll_out(self):
        """collect data and store to experience buffer."""
        print(self.configs["seed"])

        obs, _ = self.env.reset(seed=self.configs["seed"])
        ep_ret, ep_costs, ep_len = 0.0, 0.0, 0
        if self.configs["use_reward_penalty"]:
            # consider reward penalty parameter in reward calculation: r' = r - c
            assert hasattr(self, "lagrangian_multiplier")
            assert hasattr(self, "lambda_range_projection")
            penalty_param = self.lambda_range_projection(self.lagrangian_multiplier)
        else:
            penalty_param = 0

        # c_gamma_step = 0
        for t in tqdm(range(self.configs["steps_per_epoch"])):
            # print("current_steps: ", t)

            action, value, cost_value, logp = self.policy.step(
                torch.as_tensor(obs, dtype=torch.float32)
            )
            next_obs, reward, cost, terminated, truncated, info = self.env.step(action)

            ep_ret += reward
            if self.configs["use_discount_cost_update_lag"]:
                ep_costs += (self.gamma**ep_len) * cost
            else:
                ep_costs += cost
            ep_len += 1

            # save to buffer
            self.buf.store(obs=obs, act=action, rew=reward, val=value, logp=logp, cost=cost, cost_val=cost_value)

            # store values for statistic purpose
            if self.configs["use_cost_value_function"]:
                self.logger.store(**{"Values/V": value, "Values/C": cost_value})
            else:
                self.logger.store(**{"Values/V": value})

            obs = next_obs

            timeout = ep_len == self.configs["max_ep_len"]
            terminal = terminated or timeout
            epoch_ended = t == self.configs["steps_per_epoch"]

            if terminal or epoch_ended:
                if timeout or epoch_ended:
                    _, value, cost_value, _ = self.policy(
                        torch.as_tensor(obs, dtype=torch.float32)
                    )
                else:
                    value, cost_value = 0.0, 0.0

                # compute GAE in buffer
                self.buf.finish_path(
                    value, cost_value, penalty_param=float(penalty_param)
                )

                # Only save EpRet / EpLen if trajectory finished
                if terminal:
                    self.logger.store(EpRet=ep_ret, EpLen=ep_len, EpCosts=ep_costs)
                obs, info = self.env.reset()
                ep_ret, ep_costs, ep_len = 0.0, 0.0, 0

    def update(self):
        """
        Update actor, critic, running statistics
        """
        raw_data = self.buf.get()
        # Pre-process data: standardize observations, advantage estimation, etc.
        data = self.pre_process_data(raw_data)
        self.update_value_net(data=data)
        if self.configs["use_cost_value_function"]:
            self.update_cost_net(data=data)
        self.update_policy_net(data=data)

    def update_policy_net(self, data) -> None:
        # Get prob. distribution before updates: used to measure KL distance
        with torch.no_grad():
            self.p_dist = self.policy.actor.detach_dist(data["obs"])

        # Get loss and info values before update
        pi_l_old, pi_info_old = self.compute_loss_pi(data)
        self.loss_pi_before = pi_l_old.item()

        # Train policy with multiple steps of gradient descent
        for i in range(self.configs["train_pi_iterations"]):
            self.pi_optimizer.zero_grad()
            loss_pi, pi_info = self.compute_loss_pi(data=data)
            loss_pi.backward()
            # Apply L2 norm
            if self.configs["use_max_grad_norm"]:
                torch.nn.utils.clip_grad_norm_(
                    self.policy.actor.parameters(), self.configs["max_grad_norm"]
                )
            self.pi_optimizer.step()

            q_dist = self.policy.actor.dist(data["obs"])
            torch_kl = (
                torch.distributions.kl.kl_divergence(self.p_dist, q_dist).mean().item()
            )

            if self.configs["use_kl_early_stopping"]:
                # Average KL for consistent early stopping across processes
                if torch_kl > 2.0:
                    self.logger.log(f"Reached ES criterion after {i+1} steps.")
                    break

        # Track when policy iteration is stopped; Log changes from update
        self.logger.store(
            **{
                "Loss/Pi": self.loss_pi_before,
                "Loss/DeltaPi": loss_pi.item() - self.loss_pi_before,
                "Misc/StopIter": i + 1,
                "Values/Adv": data["adv"].numpy(),
                "Entropy": pi_info["ent"],
                "KL": torch_kl,
                "PolicyRatio": pi_info["ratio"],
            }
        )

    def update_value_net(self, data: dict) -> None:
        # Divide whole local epoch data into mini_batches which is mbs size
        mbs = self.configs["steps_per_epoch"] // self.configs["num_mini_batches"]
        assert mbs >= 16, f"Batch size {mbs}<16"

        loss_v = self.compute_loss_v(data["obs"], data["target_v"])
        self.loss_v_before = loss_v.item()

        indices = np.arange(self.configs["steps_per_epoch"])
        val_losses = []
        for _ in range(self.configs["train_v_iterations"]):
            # Shuffle for mini-batch updates
            np.random.shuffle(indices)
            # 0 to mini_batch_size with batch_train_size step
            for start in range(0, self.configs["steps_per_epoch"], mbs):
                end = start + mbs  # iterate mini batch times
                mb_indices = indices[start:end]
                self.vf_optimizer.zero_grad()
                loss_v = self.compute_loss_v(
                    obs=data["obs"][mb_indices], ret=data["target_v"][mb_indices]
                )
                loss_v.backward()
                val_losses.append(loss_v.item())
                self.vf_optimizer.step()

        self.logger.store(
            **{
                "Loss/DeltaValue": np.mean(val_losses) - self.loss_v_before,
                "Loss/Value": self.loss_v_before,
            }
        )

    def update_cost_net(self, data: dict) -> None:
        """Update cost value function"""
        self.loss_c_before = self.compute_loss_c(data["obs"], data["target_c"]).item()

        # Divide whole local epoch data into mini_batches which is mbs size
        mbs = self.configs["steps_per_epoch"] // self.configs["num_mini_batches"]
        assert mbs >= 16, f"Batch size {mbs}<16"

        indices = np.arange(self.configs["local_steps_per_epoch"])
        losses = []

        # Train cost value network
        for _ in range(self.configs["train_v_iterations"]):
            # Shuffle for mini-batch updates
            np.random.shuffle(indices)
            # 0 to mini_batch_size with batch_train_size step
            for start in range(0, self.configs["local_steps_per_epoch"], mbs):
                # Iterate mini batch times
                end = start + mbs
                mb_indices = indices[start:end]

                self.cf_optimizer.zero_grad()
                loss_c = self.compute_loss_c(
                    obs=data["obs"][mb_indices], ret=data["target_c"][mb_indices]
                )
                loss_c.backward()
                losses.append(loss_c.item())
                self.cf_optimizer.step()

        self.logger.store(
            **{
                "Loss/DeltaCost": np.mean(losses) - self.loss_c_before,
                "Loss/Cost": self.loss_c_before,
            }
        )
