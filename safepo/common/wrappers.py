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

from abc import ABC, abstractmethod
from multiprocessing import Pipe, Process

import torch
import numpy as np
from gymnasium.vector.vector_env import VectorEnv
from gymnasium.spaces import Box
from gymnasium.wrappers.normalize import NormalizeObservation

from safety_gymnasium.vector.utils.tile_images import tile_images
try:
    from safety_gymnasium.tasks.safe_multi_agent.safe_mujoco_multi import SafeMAEnv
except ImportError:
    from safety_gymnasium.tasks.safe_multi_agent.tasks.velocity.safe_mujoco_multi import SafeMAEnv
from typing import Optional
try :
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.hand_base.vec_task import VecTaskPython
    from safety_gymnasium.tasks.safe_isaac_gym.envs.tasks.base.vec_task import VecTaskPython as FrankaVecTaskPython
except ImportError:
    pass

class SafeNormalizeObservation(NormalizeObservation):
    """This wrapper will normalize observations as Gymnasium's NormalizeObservation wrapper does."""

    def step(self, action):
        """Steps through the environment and normalizes the observation."""
        obs, rews, costs, terminateds, truncateds, infos = self.env.step(action)
        obs = self.normalize(obs) if self.is_vector_env else self.normalize(np.array([obs]))[0]
        return obs, rews, costs, terminateds, truncateds, infos

try:
    class GymnasiumIsaacEnv(VecTaskPython):
        """This wrapper will use Gymnasium API to wrap IsaacGym environment."""

        def step(self, action):
            """Steps through the environment."""
            obs, rews, costs, terminated, infos = super().step(action)
            truncated = terminated
            return obs, rews, costs, terminated, truncated, infos
        
        def reset(self):
            """Resets the environment."""
            obs = super().reset()
            return obs, {}
except NameError:
    pass

class ShareEnv(SafeMAEnv):
    
    def __init__(
        self,
        scenario: str,
        agent_conf: str | None,
        agent_obsk: int | None = 1,
        agent_factorization: dict | None = None,
        local_categories: list[list[str]] | None = None,
        global_categories: tuple[str, ...] | None = None,
        render_mode: str | None = None,
        **kwargs,
    ):
        super().__init__(
            scenario=scenario,
            agent_conf=agent_conf,
            agent_obsk=agent_obsk,
            agent_factorization=agent_factorization,
            local_categories=local_categories,
            global_categories=global_categories,
            render_mode=render_mode,
            **kwargs,
        )
        self.num_agents = len(self.agent_action_partitions)
        self.n_actions = max([len(l) for l in self.agent_action_partitions])
        self.share_obs_size = self._get_share_obs_size()
        self.obs_size=self._get_obs_size()
        self.share_observation_spaces = {}
        self.observation_spaces={}
        for agent in range(self.num_agents):
            self.share_observation_spaces[f"agent_{agent}"] = Box(low=-10, high=10, shape=(self.share_obs_size,)) 
            self.observation_spaces[f"agent_{agent}"] = Box(low=-10, high=10, shape=(self.obs_size,)) 

    def _get_obs(self):
        state = self.env.state()
        obs_n = []
        for a in range(self.num_agents):
            agent_id_feats = np.zeros(self.num_agents, dtype=np.float32)
            agent_id_feats[a] = 1.0
            obs_i = np.concatenate([state, agent_id_feats])
            obs_i = (obs_i - np.mean(obs_i)) / np.std(obs_i)
            obs_n.append(obs_i)
        return obs_n

    def _get_obs_size(self):
        return len(self._get_obs()[0])

    def _get_share_obs(self):
        state = self.env.state()
        state_normed = (state - np.mean(state)) / (np.std(state)+1e-8)
        share_obs = []
        for _ in range(self.num_agents):
            share_obs.append(state_normed)
        return share_obs

    def _get_share_obs_size(self):
        return len(self._get_share_obs()[0])

    def _get_avail_actions(self):
        return np.ones(
            shape=(
                self.num_agents,
                self.n_actions,
            )
        )

    def reset(self, seed=None):
        super().reset(seed=seed)
        return self._get_obs(), self._get_share_obs(), self._get_avail_actions()

    
    def step(
        self, actions: dict[str, np.ndarray]
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, str],
    ]:
        dict_actions={}
        for agent_id, agent in enumerate(self.possible_agents):
            dict_actions[agent]=actions[agent_id].cpu().numpy()
        _, rewards, costs, terminations, truncations, infos = super().step(dict_actions)
        dones={}
        for agent_id, agent in enumerate(self.possible_agents):
            dones[agent] = terminations[agent] or truncations[agent]
            rewards[agent] = [rewards[agent]]
            costs[agent]=[costs[agent]]
        rewards, costs, dones, infos = list(rewards.values()), list(costs.values()), list(dones.values()), list(infos.values())
        return self._get_obs(), self._get_share_obs(), rewards, costs, dones, infos, self._get_avail_actions()


class CloudpickleWrapper:

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle

        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle

        self.x = pickle.loads(ob)


class ShareVecEnv(ABC):

    closed = False
    viewer = None

    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, num_envs, observation_space, share_observation_space, action_space):
        self.num_envs = num_envs
        self._observation_space = observation_space
        self._share_observation_space = share_observation_space
        self._action_space = action_space

    @property
    def observation_space(self, idx: Optional[int] = None):
        if idx is None:
            return list(self._observation_space.values())
        return self._observation_space[f"agent_{idx}"]
    
    @property
    def share_observation_space(self, idx: Optional[int] = None):
        if idx is None:
            return list(self._share_observation_space.values())
        return self._share_observation_space[f"agent_{idx}"]
    
    @property
    def action_space(self, idx: Optional[int] = None):
        if idx is None:
            return list(self._action_space.values())
        return self._action_space[f"agent_{idx}"]

    @abstractmethod
    def reset(self):
        """
        Reset all the environments and return an array of
        observations, or a dict of observation arrays.

        If step_async is still doing work, that work will
        be cancelled and step_wait() should not be called
        until step_async() is invoked again.
        """
        pass

    @abstractmethod
    def step_async(self, actions):
        """
        Tell all the environments to start taking a step
        with the given actions.
        Call step_wait() to get the results of the step.

        You should not call this if a step_async run is
        already pending.
        """
        pass

    @abstractmethod
    def step_wait(self):
        """
        Wait for the step taken with step_async().

        Returns (obs, rews, cos, dones, infos):
         - obs: an array of observations, or a dict of
                arrays of observations.
         - rews: an array of rewards
         - cos: an array of costs
         - dones: an array of "episode done" booleans
         - infos: a sequence of info objects
        """
        pass

    def close_extras(self):
        """
        Clean up the  extra resources, beyond what's in this base class.
        Only runs when not self.closed.
        """
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        """
        Step the environments synchronously.

        This is available for backwards compatibility.
        """
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        """
        Return RGB images from each environment
        """
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VectorEnv):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gymnasium.envs.classic_control import rendering

            self.viewer = rendering.SimpleImageViewer()
        return self.viewer



def shareworker(remote, parent_remote, env_fn_wrapper):
    parent_remote.close()
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, s_ob, reward, cost, done, info, available_actions = env.step(data)
            if 'bool' in done.__class__.__name__:
                if done:
                    ob, s_ob, available_actions = env.reset()
            else:
                if np.all(done):
                    ob, s_ob, available_actions = env.reset()

            remote.send((ob, s_ob, reward, cost, done, info, available_actions))
        elif cmd == 'reset':
            ob, s_ob, available_actions = env.reset()
            remote.send((ob, s_ob, available_actions))
        elif cmd == 'reset_task':
            ob = env.reset_task()
            remote.send(ob)
        elif cmd == 'render':
            if data == 'rgb_array':
                fr = env.render(mode=data)
                remote.send(fr)
            elif data == 'human':
                env.render(mode=data)
        elif cmd == 'close':
            env.close()
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.observation_spaces, env.share_observation_spaces, env.action_spaces))
        elif cmd == 'render_vulnerability':
            fr = env.render_vulnerability(data)
            remote.send(fr)
        elif cmd == 'get_num_agents':
            remote.send(env.num_agents)
        else:
            raise NotImplementedError


class ShareSubprocVecEnv(ShareVecEnv):
    def __init__(self, env_fns, device=torch.device("cpu")):
        self.waiting = False
        self.closed = False
        self.device = device
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [
            Process(target=shareworker, args=(work_remote, remote, CloudpickleWrapper(env_fn)))
            for (work_remote, remote, env_fn) in zip(self.work_remotes, self.remotes, env_fns)
        ]
        for p in self.ps:
            p.daemon = True  # if the main process crashes, we should not cause things to hang
            p.start()
        for remote in self.work_remotes:
            remote.close()
        self.remotes[0].send(('get_num_agents', None))
        self.num_agents = self.remotes[0].recv()
        self.remotes[0].send(('get_spaces', None))
        observation_space, share_observation_space, action_space = self.remotes[0].recv()
        ShareVecEnv.__init__(
            self, len(env_fns), observation_space, share_observation_space, action_space
        )

    def step_async(self, actions):
        env_actions = torch.transpose(torch.stack(actions), 1, 0)
        for remote, action in zip(self.remotes, env_actions):
            remote.send(('step', action))
        self.waiting = True

    def step_wait(self):
        results = [remote.recv() for remote in self.remotes]
        self.waiting = False
        obs, share_obs, rews, costs, dones, infos, available_actions = zip(*results)
        obs, share_obs, rews, costs, dones, available_actions = map(
            lambda x: torch.tensor(np.stack(x), device=self.device), (obs, share_obs, rews, costs, dones, available_actions)
        )
        return obs, share_obs, rews, costs, dones, infos, available_actions

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        results = [remote.recv() for remote in self.remotes]
        obs, share_obs, available_actions = map(
            lambda x: torch.tensor(np.stack(x), device=self.device), zip(*results)
        )
        return obs, share_obs, available_actions

    

class ShareDummyVecEnv(ShareVecEnv):
    def __init__(self, env_fns, device=torch.device("cpu")):
        self.envs = [fn() for fn in env_fns]
        env = self.envs[0]
        self.device = device
        self.num_agents=env.num_agents
        ShareVecEnv.__init__(
            self, len(env_fns), env.observation_spaces, env.share_observation_spaces, env.action_spaces
        )
        self.actions = None

    def step_async(self, actions):
        env_actions = torch.transpose(torch.stack(actions), 1, 0)
        self.actions = env_actions

    def step_wait(self):
        results = [env.step(a) for (a, env) in zip(self.actions, self.envs)]
        obs, share_obs, rews, cos, dones, infos, available_actions = map(np.array, zip(*results))

        for i, done in enumerate(dones):
            if np.all(done):
                obs[i], share_obs[i], available_actions[i] = self.envs[i].reset()
        self.actions = None

        obs, share_obs, rews, cos, dones, available_actions = map(
            lambda x: torch.tensor(x).to(self.device), (obs, share_obs, rews, cos, dones, available_actions)
        )

        return obs, share_obs, rews, cos, dones, infos, available_actions

    def reset(self):
        results = [env.reset() for env in self.envs]
        obs, share_obs, available_actions = map(
            lambda x: torch.tensor(np.stack(x), device=self.device), zip(*results)
        )
        return obs, share_obs, available_actions

    def render(self):
        return self.envs[0].render()