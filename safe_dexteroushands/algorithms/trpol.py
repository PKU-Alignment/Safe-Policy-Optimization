from datetime import datetime
import os
import time

from gym.spaces import Space

import numpy as np
import statistics
from collections import deque
from torch.nn.functional import softplus
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.autograd import grad
from .storage import RolloutStorage
from torch.distributions import Normal
EPS = 1e-8

def gaussian_kl(mean1, std1, mean2, std2):
    """
    Calculate KL-divergence between two Gaussian distributions N(mu1, sigma1) and N(mu2, sigma2)
    """
    normal1 = Normal(mean1, std1)
    normal2 = Normal(mean2, std2)
    return torch.distributions.kl.kl_divergence(normal1,normal2).sum(-1, keepdim=True)


def set_params(model, new_params):
    """
    Set the parameters of parameterized_fun to new_params
    Parameters
    ----------
    parameterized_fun : torch.nn.Sequential
        the function approximator to be updated
    new_params : torch.FloatTensor
        a flattened version of the parameters to be set
    """

    index = 0
    for params in model.parameters():
        params_length = len(params.view(-1))
        new_param = new_params[index: index + params_length]
        new_param = new_param.view(params.size())
        params.data.copy_(new_param)
        index += params_length


def flatten(vecs):
    """
    Return an unrolled, concatenated copy of vecs
    Parameters
    ----------
    vecs : Tensor or list
        a list of Pytorch Tensor objects
    Returns
    -------
    flattened : torch.FloatTensor
        the flattened version of vecs
    """

    flattened = torch.cat([v.view(-1) for v in vecs])

    return flattened


def flat_grad(functional_output, inputs, retain_graph=False, create_graph=False):
    """
    Return a flattened view of the gradients of functional_output w.r.t. inputs
    Parameters
    ----------
    functional_output : torch.FloatTensor
        The output of the function for which the gradient is to be calculated
    inputs : torch.FloatTensor (with requires_grad=True)
        the variables w.r.t. which the gradient will be computed
    retain_graph : bool
        whether to keep the computational graph in memory after computing the
        gradient (not required if create_graph is True)
    create_graph : bool
        whether to create a computational graph of the gradient computation
        itself
    Return
    ------
    flat_grads : torch.FloatTensor
        a flattened view of the gradients of functional_output w.r.t. inputs
    """

    if create_graph:
        retain_graph = True

    grads = grad(functional_output, inputs, retain_graph=retain_graph, create_graph=create_graph)
    flat_grads = flatten(grads)

    return flat_grads


def get_flat_params(parameterized_fun):
    """
    Get a flattened view of the parameters of a function approximator
    Parameters
    ----------
    parameterized_fun : torch.nn.Sequential
        the function approximator for which the parameters are to be returned
    Returns
    -------
    flat_params : torch.FloatTensor
        a flattened view of the parameters of parameterized_fun
    """
    parameters = parameterized_fun.parameters()
    flat_params = flatten([param.view(-1) for param in parameters])

    return flat_params


class TRPOL:

    def __init__(self,
                 vec_env,
                 logger,
                 actor_class,
                 critic_class,
                 cost_critic_class,
                 num_transitions_per_env,
                 num_learning_epochs,
                 num_mini_batches,
                 cost_lim,
                 clip_param=0.2,
                 gamma=0.998,
                 lam=0.95,
                 init_noise_std=1.0,
                 value_loss_coef=1.0,
                 entropy_coef=0.0,
                 learning_rate=1e-3,
                 max_grad_norm=0.5,
                 use_clipped_value_loss=True,
                 schedule="fixed",
                 desired_kl=None,
                 model_cfg=None,
                 device='cpu',
                 sampler='sequential',
                 log_dir='run',
                 is_testing=False,
                 print_log=True,
                 apply_reset=False,
                 asymmetric=False
                 ):

        if not isinstance(vec_env.observation_space, Space):
            raise TypeError("vec_env.observation_space must be a gym Space")
        if not isinstance(vec_env.state_space, Space):
            raise TypeError("vec_env.state_space must be a gym Space")
        if not isinstance(vec_env.action_space, Space):
            raise TypeError("vec_env.action_space must be a gym Space")
        self.observation_space = vec_env.observation_space
        self.action_space = vec_env.action_space
        self.state_space = vec_env.state_space
        self.cost_lim = cost_lim
        self.device = device
        self.asymmetric = asymmetric
        self.max_kl = 0.02
        self.desired_kl = desired_kl
        self.schedule = schedule
        self.step_size = learning_rate
        self.logger = logger

        # PPO components
        self.vec_env = vec_env
        self.actor = actor_class(self.observation_space.shape, self.state_space.shape, self.action_space.shape,init_noise_std, model_cfg, asymmetric=asymmetric)
        self.actor.to(self.device)
        
        self.critic = critic_class(self.observation_space.shape, self.state_space.shape, model_cfg, asymmetric=asymmetric)
        self.critic.to(self.device)

        self.cost_critic = cost_critic_class(self.observation_space.shape, self.state_space.shape, model_cfg, asymmetric=asymmetric)
        self.cost_critic.to(self.device)
        self.storage = RolloutStorage(self.vec_env.num_envs, num_transitions_per_env, self.observation_space.shape,self.state_space.shape, self.action_space.shape, self.device, sampler)
        self.optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)
        self.cost_critic_optimizer = optim.Adam(self.cost_critic.parameters(), lr=learning_rate)

        # PPO parameters
        self.clip_param = clip_param
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.num_transitions_per_env = num_transitions_per_env
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.gamma = gamma
        self.lam = lam
        self.max_grad_norm = max_grad_norm
        self.use_clipped_value_loss = use_clipped_value_loss
        self.damping = 0.1

        self.penalty_param = torch.tensor(0.0001,requires_grad=True).float()
        self.penalty = softplus(self.penalty_param)
        self.penalty_optimizer = optim.Adam([self.penalty_param], lr=0.05)

        # Log
        self.log_dir = log_dir
        self.print_log = print_log
        self.writer = SummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.tot_timesteps = 0
        self.tot_time = 0
        self.is_testing = is_testing
        self.current_learning_iteration = 0

        self.apply_reset = apply_reset

    def test(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.actor.eval()

    def load(self, path):
        self.actor.load_state_dict(torch.load(path))
        self.current_learning_iteration = int(path.split("_")[-1].split(".")[0])
        self.actor.train()

    def save(self, path):
        torch.save(self.actor.state_dict(), path)

    def run(self, num_learning_iterations, log_interval=1):
        current_obs = self.vec_env.reset()
        current_states = self.vec_env.get_state()

        if self.is_testing:
            while True:
                with torch.no_grad():
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                    # Compute the action
                    actions = self.actor_critic.act_inference(current_obs)
                    # Step the vec_environment
                    next_obs, rews, dones, infos = self.vec_env.step(actions)
                    current_obs.copy_(next_obs)
        else:
            rewbuffer = deque(maxlen=100)
            rewbuffer.append(0)
            costbuffer = deque(maxlen=100)
            costbuffer.append(0)
            lenbuffer = deque(maxlen=100)
            lenbuffer.append(0)
            cur_reward_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_cost_sum = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)
            cur_episode_length = torch.zeros(self.vec_env.num_envs, dtype=torch.float, device=self.device)

            reward_sum = []
            cost_sum = []
            episode_length = []
            

            for it in range(self.current_learning_iteration, num_learning_iterations):
                start = time.time()
                ep_infos = []
                ep_cost = []
                # Rollout
                for _ in range(self.num_transitions_per_env):
                    if self.apply_reset:
                        current_obs = self.vec_env.reset()
                        current_states = self.vec_env.get_state()
                    # Compute the action
                    actions, actions_log_prob, mu, std = self.actor.act(current_obs, current_states)
                    values = self.critic.act(current_obs, current_states)
                    cost_values = self.cost_critic.act(current_obs, current_states)

                    # Step the vec_environment
                    next_obs, rews, costs, dones, infos = self.vec_env.step(actions)
                    next_states = self.vec_env.get_state()
                    # Record the transition
                    self.storage.add_transitions(current_obs, current_states, actions, rews, costs, dones, values, cost_values, actions_log_prob, mu, std)
                    current_obs.copy_(next_obs)
                    current_states.copy_(next_states)
                    # Book keeping
                    ep_infos.append(infos)

                    if self.print_log:
                        cur_reward_sum[:] += rews
                        #JM :
                        cur_cost_sum[:] += costs
                        cur_episode_length[:] += 1

                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        reward_sum.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        # JM
                        cost_sum.extend(cur_cost_sum[new_ids][:, 0].cpu().numpy().tolist())
                        ep_cost.extend(cur_cost_sum[new_ids][:, 0].cpu().numpy().tolist())
                        episode_length.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_cost_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0

                avg_cost = 0
                if len(ep_cost) == 0:
                    pass
                else:
                    avg_cost = np.mean(ep_cost) - self.cost_lim
                    print("avg_cost", avg_cost)
                    loss_penalty = -self.penalty_param * torch.tensor(avg_cost)
                    self.penalty_optimizer.zero_grad()
                    loss_penalty.backward()
                    self.penalty_optimizer.step()
                    print("penalty_param",self.penalty_param)

                if self.print_log:
                    # reward_sum = [x[0] for x in reward_sum]
                    # episode_length = [x[0] for x in episode_length]
                    rewbuffer.extend(reward_sum)
                    costbuffer.extend(cost_sum)
                    lenbuffer.extend(episode_length)

                # _, _, , _, _ = self.actor_critic.act(current_obs, current_states)
                last_values = self.critic.act(current_obs, current_states)
                last_cost_values = self.cost_critic.act(current_obs, current_states)
                stop = time.time()
                collection_time = stop - start

                mean_trajectory_length, mean_reward, mean_cost = self.storage.get_statistics()
                print("mean_cost", mean_cost)
                # Learning step
                start = stop
                self.storage.compute_returns(last_values, self.gamma, self.lam)
                self.storage.compute_costs(last_cost_values, self.gamma, self.lam)
                mean_value_loss, mean_surrogate_loss = self.update(avg_cost)
                self.storage.clear()
                stop = time.time()
                learn_time = stop - start

                self.writer.add_scalar('Train/mean_reward', statistics.mean(rewbuffer),it)
                self.writer.add_scalar('Train/mean_cost', statistics.mean(costbuffer),it)
                self.logger.store(Reward=statistics.mean(rewbuffer))
                self.logger.store(Cost=statistics.mean(costbuffer))
                self.logger.store(Epoch=it)
                self.logger.log_tabular("Epoch", average_only=True)
                self.logger.log_tabular('Reward', average_only=True)
                self.logger.log_tabular('Cost', average_only=True)
                self.logger.dump_tabular()

    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_transitions_per_env * self.vec_env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                self.writer.add_scalar('Episode/' + key, value, locs['it'])
                ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n"""
        mean_std = self.actor.log_std.exp().mean()

        self.writer.add_scalar('Loss/value_function', locs['mean_value_loss'], locs['it'])
        self.writer.add_scalar('Loss/surrogate', locs['mean_surrogate_loss'], locs['it'])
        self.writer.add_scalar('Policy/mean_noise_std', mean_std.item(), locs['it'])
        if len(locs['rewbuffer']) > 0:
            self.writer.add_scalar('Train/mean_reward', statistics.mean(locs['rewbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_reward/time', statistics.mean(locs['rewbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)
        
        if len(locs['costbuffer']) > 0:
            self.writer.add_scalar('Train/mean_cost', statistics.mean(locs['costbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_episode_length', statistics.mean(locs['lenbuffer']), locs['it'])
            self.writer.add_scalar('Train/mean_cost/time', statistics.mean(locs['costbuffer']), self.tot_time)
            self.writer.add_scalar('Train/mean_episode_length/time', statistics.mean(locs['lenbuffer']), self.tot_time)

        self.writer.add_scalar('Train/penalty', self.penalty_param, self.tot_time)

        self.writer.add_scalar('Train2/mean_reward/step', locs['mean_reward'], locs['it'])
        self.writer.add_scalar('Train2/mean_cost/step', locs['mean_cost'], locs['it'])
        self.writer.add_scalar('Train2/mean_episode_length/episode', locs['mean_trajectory_length'], locs['it'])

        fps = int(self.num_transitions_per_env * self.vec_env.num_envs / (locs['collection_time'] + locs['learn_time']))

        str = f" \033[1m Learning iteration {locs['it']}/{locs['num_learning_iterations']} \033[0m "

        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                              'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward:':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean cost:':>{pad}} {statistics.mean(locs['costbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean cost/step:':>{pad}} {locs['mean_cost']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Value function loss:':>{pad}} {locs['mean_value_loss']:.4f}\n"""
                          f"""{'Surrogate loss:':>{pad}} {locs['mean_surrogate_loss']:.4f}\n"""
                          f"""{'Mean action noise std:':>{pad}} {mean_std.item():.2f}\n"""
                          f"""{'Mean reward/step:':>{pad}} {locs['mean_reward']:.2f}\n"""
                          f"""{'Mean episode length/episode:':>{pad}} {locs['mean_trajectory_length']:.2f}\n""")

        log_string += ep_string
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {self.tot_time / (locs['it'] + 1) * (
                               locs['num_learning_iterations'] - locs['it']):.1f}s\n""")
        print(log_string)

    def update(self, avg_cost):
        mean_value_loss = 0
        mean_surrogate_loss = 0

        batch = self.storage.mini_batch_generator(self.num_mini_batches)
        for epoch in range(self.num_learning_epochs):
            for indices in batch:
                obs_batch = self.storage.observations.view(-1, *self.storage.observations.size()[2:])[indices]
                if self.asymmetric:
                    states_batch = self.storage.states.view(-1, *self.storage.states.size()[2:])[indices]
                else:
                    states_batch = None

  
                actions_batch = self.storage.actions.view(-1, self.storage.actions.size(-1))[indices]
                target_values_batch = self.storage.values.view(-1, 1)[indices]
                # For cost
                cost_target_values_batch = self.storage.cost_values.view(-1, 1)[indices]

                returns_batch = self.storage.returns.view(-1, 1)[indices]
                # For cost
                cost_returns_batch = self.storage.creturns.view(-1, 1)[indices]
                old_actions_log_prob_batch = self.storage.actions_log_prob.view(-1, 1)[indices]

                advantages_batch = self.storage.advantages.view(-1, 1)[indices]
                # For cost
                cadvantages_batch = self.storage.cadvantages.view(-1, 1)[indices]
                old_mean_batch = self.storage.mu.view(-1, self.storage.actions.size(-1))[indices]
                old_std_batch = self.storage.sigma.view(-1, self.storage.actions.size(-1))[indices]

                actions_log_prob_batch, entropy_batch, mean_batch, std_batch = self.actor.evaluate(obs_batch,states_batch,actions_batch)
                value_batch = self.critic.evaluate(obs_batch,states_batch)
                cost_value_batch = self.cost_critic.evaluate(obs_batch,states_batch)

                # Surrogate loss
                ratio = torch.exp(actions_log_prob_batch - torch.squeeze(old_actions_log_prob_batch).detach())
                # print("ratio", ratio)
                surrogate_loss = (torch.squeeze(advantages_batch) * ratio).mean()
                surrogate_cost_loss = (torch.squeeze(cadvantages_batch) * ratio).mean()

                self.penalty_item = softplus(self.penalty_param).detach()
                # exit(0)
                loss = surrogate_loss - self.penalty_item * surrogate_cost_loss + self.entropy_coef * entropy_batch.mean()
                loss /= (1 + self.penalty_item)
                loss = -loss
                
                # Value function loss
                value_loss = (returns_batch - value_batch).pow(2).mean()

                for param in self.critic.parameters():
                    value_loss += param.pow(2).sum() * 0.001
                # value critic
                self.critic_optimizer.zero_grad()
                value_loss.backward()
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.max_grad_norm)
                self.critic_optimizer.step()

                # cost function loss
                cost_value_loss = (cost_returns_batch - cost_value_batch).pow(2).mean()

                # value critic
                for param in self.cost_critic.parameters():
                    cost_value_loss += param.pow(2).sum() * 0.001
                self.cost_critic_optimizer.zero_grad()
                cost_value_loss.backward()
                nn.utils.clip_grad_norm_(self.cost_critic.parameters(), self.max_grad_norm)
                self.cost_critic_optimizer.step()


                # Policy

                # _, _, mean, std = self.actor.evaluate(obs_batch, states_batch, actions_batch)

                self.actor.zero_grad()
                g = flat_grad(loss, self.actor.parameters(),retain_graph=True)

                
                def Fvp(v):
                    kl = gaussian_kl(mean_batch, std_batch, mean_batch.detach(), std_batch.detach()).mean()
                    self.actor.zero_grad()
                    grads = torch.autograd.grad(kl, self.actor.parameters(), create_graph=True)
                    flat_grad_kl = torch.cat([grad.view(-1) for grad in grads])
                    kl_v = (flat_grad_kl * v).sum()
                    self.actor.zero_grad()
                    grads = torch.autograd.grad(kl_v, self.actor.parameters(), create_graph=True)
                    flat_grad_grad_kl = torch.cat([grad.contiguous().view(-1) for grad in grads]).data
                    return flat_grad_grad_kl + v * self.damping

                x = self.cg_solver(Fvp, g)  # H_inv_g
                approx_g = Fvp(x)
                q = torch.matmul(x, approx_g) # g^T * H^-1 * g
                alpha = torch.sqrt(2*self.max_kl/(q+EPS))

                print("q" , q)
                # 
                # print("search_dir", search_dir)
                current_policy = get_flat_params(self.actor)   
                # Linear Search
                def set_and_eval(step):
                    new_policy = current_policy - step * alpha * x
                    set_params(self.actor, new_policy)

                    logprob_eval, _, mean_eval, std_eval = self.actor.evaluate(obs_batch, states_batch, actions_batch)
                    # kl_value = calc_kl(mean_eval, std_eval, old_mu_batch, old_sigma_batch)
                    kl_value = gaussian_kl(mean_eval, std_eval, mean_batch.detach(), std_batch.detach()).mean()
                    #print("kl", kl_value)
                    assert kl_value.item() >= 0, "kl_mean is negative!!!"
                    ratio_eval = torch.exp(logprob_eval - torch.squeeze(old_actions_log_prob_batch).detach()).detach()
                    surr_cost = ratio_eval * torch.squeeze(cadvantages_batch).detach()
                    surr_cost = surr_cost.mean()
                    surr_adv = ratio_eval * torch.squeeze(advantages_batch).detach()
                    pi_l_new = surr_adv.mean()
                    
                    
                    improve_loss = pi_l_new - surrogate_loss

                    if kl_value <= self.max_kl and improve_loss >= 0:
                        return True
                    else:
                        return False

                pi_linear_search_list = np.linspace(0, 0.002, 10, endpoint=True, dtype=np.float32)
                step_len = 0
                pi_linear_search_list = pi_linear_search_list[::-1]
                for e in pi_linear_search_list:
                    if set_and_eval(step=e):
                        step_len = e
                        break
                # print('Step Len.:', step_len, '\n')
                # assert step_len == 0.0 , "update!!!"
                new_policy = current_policy - step_len * alpha * x
                set_params(self.actor, new_policy)
                # nn.utils.clip_grad_norm_(self.actor.parameters(), self.max_grad_norm)

                mean_value_loss += value_loss.item()
                mean_surrogate_loss += surrogate_loss.item()

        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_value_loss /= num_updates
        mean_surrogate_loss /= num_updates

        return mean_value_loss, mean_surrogate_loss


    def cg_solver(self, Avp_fun, b, max_iter=10):
        """
        Finds an approximate solution to a set of linear equations Ax = b
        Parameters
        ----------
        Avp_fun : callable
            a function that right multiplies a matrix A by a vector
        b : torch.FloatTensor
            the right hand term in the set of linear equations Ax = b
        max_iter : int
            the maximum number of iterations (default is 10)
        Returns
        -------
        x : torch.FloatTensor
            the approximate solution to the system of equations defined by Avp_fun
            and b
        """
        x = torch.zeros_like(b).to(self.device)
        r = b.clone()
        p = b.clone()

        for i in range(max_iter):
            Avp = Avp_fun(p)
            # sclars
            alpha = torch.matmul(r, r) / (torch.matmul(p, Avp) + EPS)
            x += alpha * p
            if i == max_iter - 1:
                return x

            r_new = r - alpha * Avp
            beta = torch.matmul(r_new, r_new) / torch.matmul(r, r)
            r = r_new
            p = r + beta * p

    def kl_divergence(self, obs_batch, states_batch, actions_batch, new_actor, old_actor):
    
        _, _, mu, std = new_actor.evaluate(obs_batch, states_batch, actions_batch)
        _, _, mu_old, std_old = old_actor.evaluate(obs_batch, states_batch, actions_batch)

        # print("xxxxx",mu, std)
        # print("xxx", mu_old, std_old)

        logstd = torch.log(std)
        mu_old = mu_old.detach()
        std_old = std_old.detach()
        logstd_old = torch.log(std_old)

        # kl divergence between old policy and new policy : D( pi_old || pi_new )
        # pi_old -> mu0, logstd0, std0 / pi_new -> mu, logstd, std
        # be careful of calculating KL-divergence. It is not symmetric metric
        kl = logstd_old - logstd + (std_old.pow(2) + (mu_old - mu).pow(2)) / \
             (EPS + 2.0 * std.pow(2)) - 0.5
        kl_value = kl.sum(1, keepdim=True)
        return kl_value.mean()