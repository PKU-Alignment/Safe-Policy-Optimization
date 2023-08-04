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
import time
import csv
import json
import os.path as osp

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from safepo.multi_agent.marl_algorithms.algorithms.utils.separated_buffer import \
    SeparatedReplayBuffer
from safepo.common.logger import convert_json


def _t2n(x):
    return x.detach().cpu().numpy()

class Runner:

    def __init__(self,
                 vec_env,
                 vec_eval_env,
                 config,
                 model_dir=""
                 ):
        self.envs = vec_env
        self.eval_envs = vec_eval_env
        # parameters
        self.env_name = config["env_name"]
        self.algorithm_name = config["algorithm_name"]
        self.experiment_name = config["experiment_name"]
        self.use_centralized_V = config["use_centralized_V"]
        self.use_obs_instead_of_state = config["use_obs_instead_of_state"]
        self.num_env_steps = config["num_env_steps"]
        self.episode_length = config["episode_length"]
        self.n_rollout_threads = config["n_rollout_threads"]
        self.n_eval_rollout_threads = config["n_eval_rollout_threads"]
        self.use_linear_lr_decay = config["use_linear_lr_decay"]
        self.hidden_size = config["hidden_size"]
        self.use_render = config["use_render"]
        self.recurrent_N = config["recurrent_N"]
        self.use_single_network = config["use_single_network"]
        # interval
        self.save_interval = config["save_interval"]
        self.use_eval = config["use_eval"]
        self.eval_interval = config["eval_interval"]
        self.eval_episodes = config["eval_episodes"]
        self.single_eval_episodes = config["single_eval_episodes"]
        self.log_interval = config["log_interval"]

        self.seed = config["seed"]
        self.model_dir = model_dir

        self.num_agents = self.envs.num_agents

        self.device = config["device"]

        torch.autograd.set_detect_anomaly(True)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark = True

        self.run_dir = config["run_dir"]
        self.log_dir = str(config["log_dir"]+'/logs_seed{}'.format(self.seed))
        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)
        self.writter = SummaryWriter(self.log_dir)
        self.save_dir = str(config["log_dir"]+'/models_seed{}'.format(self.seed))
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.output_file = open(  # noqa: SIM115 # pylint: disable=consider-using-with
            os.path.join(self.log_dir, "progress.csv"),
            encoding="utf-8",
            mode="w",
        )
        self.csv_writer = csv.writer(self.output_file)
        self.csv_writer.writerow(['Train/Steps', 'Train/Episode Reward', 'Train/Episode Cost', \
                                  'Eval/Episode Reward', 'Eval/Episode Cost'])
        
        # save config
        config_json = convert_json(config)
        config_json["exp_name"] = self.experiment_name
        output = json.dumps(
            config_json, separators=(",", ":\t"), indent=4, sort_keys=True
        )
        with open(osp.join(self.log_dir, "config.json"), "w") as out:
            out.write(output)

        if self.algorithm_name == "happo":
            from safepo.multi_agent.marl_algorithms.algorithms.happo_policy import \
                HAPPO_Policy as Policy
            from safepo.multi_agent.marl_algorithms.algorithms.happo_trainer import \
                HAPPO as TrainAlgo
        if self.algorithm_name == "hatrpo":
            from safepo.multi_agent.marl_algorithms.algorithms.mappolag_policy import \
                MAPPO_L_Policy as Policy
            from safepo.multi_agent.marl_algorithms.algorithms.mappolag_trainer import \
                R_MAPPO_Lagr as TrainAlgo
        if self.algorithm_name == "mappo":
            from safepo.multi_agent.marl_algorithms.algorithms.mappo_policy import \
                MAPPO_Policy as Policy
            from safepo.multi_agent.marl_algorithms.algorithms.mappo_trainer import \
                MAPPO as TrainAlgo
        if self.algorithm_name == "macpo":
            from safepo.multi_agent.marl_algorithms.algorithms.macpo_policy import \
                MACPO_Policy as Policy
            from safepo.multi_agent.marl_algorithms.algorithms.macpo_trainer import \
                MACPO as TrainAlgo
        if self.algorithm_name == "ippo":
            from safepo.multi_agent.marl_algorithms.algorithms.ippo_policy import \
                IPPO_Policy as Policy
            from safepo.multi_agent.marl_algorithms.algorithms.ippo_trainer import \
                IPPO as TrainAlgo

        self.policy = []
        for agent_id in range(self.num_agents):
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            # policy network
            po = Policy(config,
                        self.envs.observation_space[agent_id],
                        share_observation_space,
                        self.envs.action_space[agent_id],
                        device = self.device)
            self.policy.append(po)

        if self.model_dir != "":
            self.restore()

        self.trainer = []
        self.buffer = []
        for agent_id in range(self.num_agents):
            # algorithm
            tr = TrainAlgo(config, self.policy[agent_id], device = self.device)
            # buffer
            share_observation_space = self.envs.share_observation_space[agent_id] if self.use_centralized_V else self.envs.observation_space[agent_id]
            bu = SeparatedReplayBuffer(config,
                                       self.envs.observation_space[agent_id],
                                       share_observation_space,
                                       self.envs.action_space[agent_id])
            self.buffer.append(bu)
            self.trainer.append(tr)

    def run(self):
        self.warmup()

        start = time.time()
        episodes = int(self.num_env_steps) // self.episode_length // self.n_rollout_threads

        train_episode_rewards = torch.zeros(1, self.n_rollout_threads, device=self.device)
        train_episode_costs = torch.zeros(1, self.n_rollout_threads, device=self.device)

        for episode in range(episodes):
            if self.use_linear_lr_decay:
                self.trainer.policy.lr_decay(episode, episodes)

            done_episodes_rewards = []
            done_episodes_costs = []

            for step in range(self.episode_length):
                # Sample actions
                values, actions, action_log_probs, rnn_states, rnn_states_critic, env_actions = self.collect(step)

                # Obser reward and next obs
                obs, share_obs, rewards, costs, dones, infos, _ = self.envs.step(env_actions)
                obs = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
                share_obs = torch.as_tensor(share_obs, dtype=torch.float32, device=self.device)
                rewards = torch.as_tensor(rewards, dtype=torch.float32, device=self.device)
                costs = torch.as_tensor(costs, dtype=torch.float32, device=self.device)
                dones = torch.as_tensor(dones, dtype=torch.float32, device=self.device)

                dones_env = torch.all(dones, dim=1)
                reward_env = torch.mean(rewards, dim=1).flatten()
                cost_env = torch.mean(costs, dim=1).flatten()

                train_episode_rewards += reward_env
                train_episode_costs += cost_env

                for t in range(self.n_rollout_threads):
                    if dones_env[t]:
                        done_episodes_rewards.append(train_episode_rewards[:, t].clone())
                        train_episode_rewards[:, t] = 0
                        done_episodes_costs.append(train_episode_costs[:, t].clone())
                        train_episode_costs[:, t] = 0

                data = obs, share_obs, rewards, dones, infos, \
                       values, actions, action_log_probs, \
                       rnn_states, rnn_states_critic

                # insert data into buffer
                self.insert(data)

            # compute return and update network
            self.compute()
            train_infos = self.train()

            # post process
            total_num_steps = (episode + 1) * self.episode_length * self.n_rollout_threads
            # save model
            if (episode % self.save_interval == 0 or episode == episodes - 1):
                self.save()

            # log information
            if episode % self.log_interval == 0:
                end = time.time()
                print("\nAlgo {} Exp {} updates {}/{} episodes, total num timesteps {}/{}, FPS {}.\n"
                      .format(self.algorithm_name,
                              self.experiment_name,
                              episode,
                              episodes,
                              total_num_steps,
                              self.num_env_steps,
                              int(total_num_steps / (end - start))))

                self.log_train(train_infos, total_num_steps)

            # eval
            eval_rewards='Not Recorded'
            eval_costs='Not Recorded'
            if episode % self.eval_interval == 0 and self.use_eval:
                eval_rewards, eval_costs = self.eval(total_num_steps)

            if len(done_episodes_rewards) != 0:
                aver_episode_rewards = torch.stack(done_episodes_rewards).mean()
                aver_episode_costs = torch.stack(done_episodes_costs).mean()
                print("some episodes done, average rewards: {}, average costs: {}".format(aver_episode_rewards, aver_episode_costs))
                self.writter.add_scalar("train_episode_rewards", aver_episode_rewards,
                                            total_num_steps)
                self.writter.add_scalar("train_episode_costs", aver_episode_costs,
                                            total_num_steps)
                self.csv_writer.writerow([total_num_steps, aver_episode_rewards.item(), aver_episode_costs.item(), eval_rewards, eval_costs])
                self.output_file.flush()

    def warmup(self):
        # reset env
        obs, share_obs, _ = self.envs.reset()
        obs = torch.as_tensor(obs, dtype=torch.float32).to(self.device)
        share_obs = torch.as_tensor(share_obs, dtype=torch.float32).to(self.device)
        # replay buffer
        if not self.use_centralized_V:
            share_obs = obs

        for agent_id in range(self.num_agents):
            self.buffer[agent_id].share_obs[0] = share_obs[:, agent_id].clone()
            self.buffer[agent_id].obs[0] = obs[:, agent_id].clone()

    @torch.no_grad()
    def collect(self, step):
        value_collector = []
        action_collector = []
        action_log_prob_collector = []
        rnn_state_collector = []
        rnn_state_critic_collector = []
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            value, action, action_log_prob, rnn_state, rnn_state_critic \
                = self.trainer[agent_id].policy.get_actions(self.buffer[agent_id].share_obs[step],
                                                            self.buffer[agent_id].obs[step],
                                                            self.buffer[agent_id].rnn_states[step],
                                                            self.buffer[agent_id].rnn_states_critic[step],
                                                            self.buffer[agent_id].masks[step])
            value_collector.append(value.detach())
            action_collector.append(action.detach())

            action_log_prob_collector.append(action_log_prob.detach())
            rnn_state_collector.append(rnn_state.detach())
            rnn_state_critic_collector.append(rnn_state_critic.detach())
        # TODO: padding for humanoid
        if self.env_name == "Safety9|8HumanoidVelocity-v0":
            zeros = torch.zeros(action_collector[-1].shape[0], 1)
            action_collector[-1]=torch.cat((action_collector[-1], zeros), dim=1)
        values = torch.transpose(torch.stack(value_collector), 1, 0)
        actions = torch.transpose(torch.stack(action_collector), 1, 0)
        # action_log_probs = torch.transpose(torch.stack(action_log_prob_collector), 1, 0)
        rnn_states = torch.transpose(torch.stack(rnn_state_collector), 1, 0)
        rnn_states_critic = torch.transpose(torch.stack(rnn_state_critic_collector), 1, 0)

        return values, action_collector, action_log_prob_collector, rnn_states, rnn_states_critic, actions.detach().numpy()

    def insert(self, data):
        obs, share_obs, rewards, dones, infos, \
        values, actions, action_log_probs, rnn_states, rnn_states_critic = data

        dones_env = torch.all(dones, axis=1)

        rnn_states[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size, device=self.device)
        rnn_states_critic[dones_env == True] = torch.zeros(
            (dones_env == True).sum(), self.num_agents, *self.buffer[0].rnn_states_critic.shape[2:], device=self.device)

        masks = torch.ones(self.n_rollout_threads, self.num_agents, 1, device=self.device)
        masks[dones_env == True] = torch.zeros((dones_env == True).sum(), self.num_agents, 1, device=self.device)

        active_masks = torch.ones(self.n_rollout_threads, self.num_agents, 1, device=self.device)
        active_masks[dones == True] = torch.zeros((dones == True).sum(), 1, device=self.device)
        active_masks[dones_env == True] = torch.ones((dones_env == True).sum(), self.num_agents, 1, device=self.device)


        if not self.use_centralized_V:
            share_obs = obs
        # TODO: padding for humanoid
        if self.env_name == "Safety9|8HumanoidVelocity-v0":
            actions[1]=actions[1][:, :8]
        for agent_id in range(self.num_agents):
            self.buffer[agent_id].insert(share_obs[:, agent_id], obs[:, agent_id], rnn_states[:, agent_id],
                                         rnn_states_critic[:, agent_id], actions[agent_id],
                                         action_log_probs[agent_id],
                                         values[:, agent_id], rewards[:, agent_id], masks[:, agent_id], None,
                                         active_masks[:, agent_id], None)

    def train(self):
        train_infos = []
        # random update order

        action_dim = 1
        factor = torch.ones(self.episode_length, self.n_rollout_threads, action_dim, device=self.device)

        for agent_id in torch.randperm(self.num_agents):
            action_dim=self.buffer[agent_id].actions.shape[-1]

            self.trainer[agent_id].prep_training()
            self.buffer[agent_id].update_factor(factor)
            available_actions = None if self.buffer[agent_id].available_actions is None \
                else self.buffer[agent_id].available_actions[:-1].reshape(-1, *self.buffer[agent_id].available_actions.shape[2:])

            if self.algorithm_name == "hatrpo":
                old_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            else:
                old_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            train_info = self.trainer[agent_id].train(self.buffer[agent_id])

            if self.algorithm_name == "hatrpo":
                new_actions_logprob, _, _, _, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))
            else:
                new_actions_logprob, _ =self.trainer[agent_id].policy.actor.evaluate_actions(self.buffer[agent_id].obs[:-1].reshape(-1, *self.buffer[agent_id].obs.shape[2:]),
                                                            self.buffer[agent_id].rnn_states[0:1].reshape(-1, *self.buffer[agent_id].rnn_states.shape[2:]),
                                                            self.buffer[agent_id].actions.reshape(-1, *self.buffer[agent_id].actions.shape[2:]),
                                                            self.buffer[agent_id].masks[:-1].reshape(-1, *self.buffer[agent_id].masks.shape[2:]),
                                                            available_actions,
                                                            self.buffer[agent_id].active_masks[:-1].reshape(-1, *self.buffer[agent_id].active_masks.shape[2:]))

            action_prod = torch.prod(torch.exp(new_actions_logprob.detach()-old_actions_logprob.detach()).reshape(self.episode_length,self.n_rollout_threads,action_dim), dim=-1, keepdim=True)
            factor = factor*action_prod.detach()
            train_infos.append(train_info)
            self.buffer[agent_id].after_update()

        return train_infos

    def save(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model = self.trainer[agent_id].policy.model
                torch.save(policy_model.state_dict(), str(self.save_dir) + "/model_agent" + str(agent_id) + ".pt")
            else:
                policy_actor = self.trainer[agent_id].policy.actor
                torch.save(policy_actor.state_dict(), str(self.save_dir) + "/actor_agent" + str(agent_id) + ".pt")
                policy_critic = self.trainer[agent_id].policy.critic
                torch.save(policy_critic.state_dict(), str(self.save_dir) + "/critic_agent" + str(agent_id) + ".pt")

    def restore(self):
        for agent_id in range(self.num_agents):
            if self.use_single_network:
                policy_model_state_dict = torch.load(str(self.model_dir) + '/model_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].model.load_state_dict(policy_model_state_dict)
            else:
                policy_actor_state_dict = torch.load(str(self.model_dir) + '/actor_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].actor.load_state_dict(policy_actor_state_dict)
                policy_critic_state_dict = torch.load(str(self.model_dir) + '/critic_agent' + str(agent_id) + '.pt')
                self.policy[agent_id].critic.load_state_dict(policy_critic_state_dict)

    def log_train(self, train_infos, total_num_steps):
        for agent_id in range(self.num_agents):
            for k, v in train_infos[agent_id].items():
                agent_k = "agent%i/" % agent_id + k
                self.writter.add_scalar(agent_k, v, total_num_steps)

    def log_env(self, env_infos, total_num_steps):
        for k, v in env_infos.items():
            if len(v) > 0:
                self.writter.add_scalar(k, np.mean(v), total_num_steps)


    @torch.no_grad()
    def eval(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        eval_episode_costs = []
        one_episode_rewards = torch.zeros(1, self.n_rollout_threads, device=self.device)
        one_episode_costs = torch.zeros(1, self.n_rollout_threads, device=self.device)

        eval_obs, _, _ = self.eval_envs.reset()
        eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=self.device)

        eval_rnn_states = torch.zeros(self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size,
                                   device=self.device)
        eval_masks = torch.ones(self.n_eval_rollout_threads, self.num_agents, 1, device=self.device)

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_actions, temp_rnn_state = \
                    self.trainer[agent_id].policy.act(eval_obs[:, agent_id],
                                                      eval_rnn_states[:, agent_id],
                                                      eval_masks[:, agent_id],
                                                      deterministic=True)
                eval_rnn_states[:, agent_id] = temp_rnn_state
                eval_actions_collector.append(eval_actions)
            # TODO: padding for humanoid
            if self.env_name == "Safety9|8HumanoidVelocity-v0":
                zeros = torch.zeros(eval_actions_collector[-1].shape[0], 1)
                eval_actions_collector[-1]=torch.cat((eval_actions_collector[-1], zeros), dim=1)
            eval_actions = torch.transpose(torch.stack(eval_actions_collector), 1, 0).detach().numpy()

            # Obser reward and next obs
            eval_obs, _, eval_rewards, eval_costs, eval_dones, eval_infos, _ = self.eval_envs.step(
                eval_actions)
            eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=self.device)
            eval_rewards = torch.as_tensor(eval_rewards, dtype=torch.float32, device=self.device)
            eval_costs = torch.as_tensor(eval_costs, dtype=torch.float32, device=self.device)
            eval_dones = torch.as_tensor(eval_dones, dtype=torch.float32, device=self.device)

            reward_env = torch.mean(eval_rewards, dim=1).flatten()
            cost_env = torch.mean(eval_costs, dim=1).flatten()

            one_episode_rewards += reward_env
            one_episode_costs += cost_env

            eval_dones_env = torch.all(eval_dones, dim=1)

            eval_rnn_states[eval_dones_env == True] = torch.zeros(
                (eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size, device=self.device)

            eval_masks = torch.ones(self.n_eval_rollout_threads, self.num_agents, 1, device=self.device)
            eval_masks[eval_dones_env == True] = torch.zeros((eval_dones_env == True).sum(), self.num_agents, 1,
                                                          device=self.device)

            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(one_episode_rewards[:, eval_i].mean().item())
                    one_episode_rewards[:, eval_i] = 0
                    eval_episode_costs.append(one_episode_costs[:, eval_i].mean().item())
                    one_episode_costs[:, eval_i] = 0

            if eval_episode >= self.single_eval_episodes:
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards,
                                  'eval_average_episode_costs': eval_episode_costs}
                self.log_env(eval_env_infos, total_num_steps)
                print("eval_average_episode_rewards is {}.".format(np.mean(eval_episode_rewards)),
                      "eval_average_episode_costs is {}.".format(np.mean((eval_episode_costs))))
                return np.mean(eval_episode_rewards), np.mean(eval_episode_costs)

    @torch.no_grad()
    def render(self, total_num_steps):
        eval_episode = 0
        eval_episode_rewards = []
        eval_episode_costs = []
        frames = []
        frames.append(self.eval_envs.render())
        save_video_path=os.path.join(self.save_dir, 'video')
        one_episode_rewards = torch.zeros(1, self.n_rollout_threads, device=self.device)
        one_episode_costs = torch.zeros(1, self.n_rollout_threads, device=self.device)

        eval_obs, _, _ = self.eval_envs.reset()
        eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=self.device)

        eval_rnn_states = torch.zeros(self.n_eval_rollout_threads, self.num_agents, self.recurrent_N, self.hidden_size,
                                   device=self.device)
        eval_masks = torch.ones(self.n_eval_rollout_threads, self.num_agents, 1, device=self.device)

        while True:
            eval_actions_collector = []
            for agent_id in range(self.num_agents):
                self.trainer[agent_id].prep_rollout()
                eval_actions, temp_rnn_state = \
                    self.trainer[agent_id].policy.act(eval_obs[:, agent_id],
                                                      eval_rnn_states[:, agent_id],
                                                      eval_masks[:, agent_id],
                                                      deterministic=True)
                eval_rnn_states[:, agent_id] = temp_rnn_state
                eval_actions_collector.append(eval_actions)
            # TODO: padding for humanoid
            if self.env_name == "Safety9|8HumanoidVelocity-v0":
                zeros = torch.zeros(eval_actions_collector[-1].shape[0], 1)
                eval_actions_collector[-1]=torch.cat((eval_actions_collector[-1], zeros), dim=1)
            eval_actions = torch.transpose(torch.stack(eval_actions_collector), 1, 0).detach().numpy()

            # Obser reward and next obs
            eval_obs, _, eval_rewards, eval_costs, eval_dones, eval_infos, _ = self.eval_envs.step(
                eval_actions)
            frames.append(self.eval_envs.render())
            eval_obs = torch.as_tensor(eval_obs, dtype=torch.float32, device=self.device)
            eval_rewards = torch.as_tensor(eval_rewards, dtype=torch.float32, device=self.device)
            eval_costs = torch.as_tensor(eval_costs, dtype=torch.float32, device=self.device)
            eval_dones = torch.as_tensor(eval_dones, dtype=torch.float32, device=self.device)

            reward_env = torch.mean(eval_rewards, dim=1).flatten()
            cost_env = torch.mean(eval_costs, dim=1).flatten()

            one_episode_rewards += reward_env
            one_episode_costs += cost_env

            eval_dones_env = torch.all(eval_dones, dim=1)

            eval_rnn_states[eval_dones_env == True] = torch.zeros(
                (eval_dones_env == True).sum(), self.num_agents, self.recurrent_N, self.hidden_size, device=self.device)

            eval_masks = torch.ones(self.n_eval_rollout_threads, self.num_agents, 1, device=self.device)
            eval_masks[eval_dones_env == True] = torch.zeros((eval_dones_env == True).sum(), self.num_agents, 1,
                                                          device=self.device)

            if eval_dones_env.any(): 
                save_video(
                    frames,
                    save_video_path,
                    fps=30,
                    episode_trigger=lambda x: True,
                    episode_index=eval_episode,
                    name_prefix='eval',
                )
                frames=[]
                frames.append(self.eval_envs.render())
            for eval_i in range(self.n_eval_rollout_threads):
                if eval_dones_env[eval_i]:
                    eval_episode += 1
                    eval_episode_rewards.append(one_episode_rewards[:, eval_i].mean().item())
                    one_episode_rewards[:, eval_i] = 0
                    eval_episode_costs.append(one_episode_costs[:, eval_i].mean().item())
                    one_episode_costs[:, eval_i] = 0

            if eval_episode >= self.single_eval_episodes:
                eval_env_infos = {'eval_average_episode_rewards': eval_episode_rewards,
                                  'eval_average_episode_costs': eval_episode_costs}
                self.log_env(eval_env_infos, total_num_steps)
                print("eval_average_episode_rewards is {}.".format(np.mean(eval_episode_rewards)),
                      "eval_average_episode_costs is {}.".format(np.mean((eval_episode_costs))))
                return np.mean(eval_episode_rewards), np.mean(eval_episode_costs)

    @torch.no_grad()
    def compute(self):
        for agent_id in range(self.num_agents):
            self.trainer[agent_id].prep_rollout()
            next_value = self.trainer[agent_id].policy.get_values(self.buffer[agent_id].share_obs[-1],
                                                                self.buffer[agent_id].rnn_states_critic[-1],
                                                                self.buffer[agent_id].masks[-1])
            next_value = next_value.detach()
            self.buffer[agent_id].compute_returns(next_value, self.trainer[agent_id].value_normalizer)
