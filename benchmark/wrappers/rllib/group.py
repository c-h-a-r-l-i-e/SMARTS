# MIT License
#
# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
import copy

import gym

from ray.rllib.agents.dqn import DQNTrainer
from ray.rllib.env.constants import GROUP_INFO, GROUP_REWARDS
from ray.rllib.env.group_agents_wrapper import _GroupAgentsWrapper
from benchmark.wrappers.rllib import FrameStack


class Group(_GroupAgentsWrapper):
    def __init__(self, config):
        groups = config["groups"]

        env_config = config["custom_config"]
        obs_space = env_config["observation_space"]
        act_space = env_config["action_space"]

        self.observation_adapter = env_config.get("observation_adapter")
        self.info_adapter = env_config.get("info_adapter")
        self.reward_adapter = env_config.get("reward_adapter")

        # self._agent_keys = list(config["agent_specs"].keys())
        # self._last_observations = {k: None for k in self._agent_keys}
        self._last_observations = {}
        self._done_agents = []

        base_env_cls = config["custom_config"]["base_env_cls"]
        for key in ["observation_adapter", "action_adapter", "reward_adapter"]:
            config["custom_config"][key] = config["custom_config"][f"base_{key}"]

        env = base_env_cls(config)

        super(Group, self).__init__(env, groups, obs_space, act_space)

    @staticmethod
    def get_reward_adapter(observation_adapter):
        return FrameStack.get_reward_adapter(observation_adapter)

    @staticmethod
    def get_observation_space(observation_space, wrapper_config):
        obs_space = FrameStack.get_observation_space(observation_space, wrapper_config)
        return gym.spaces.Tuple([obs_space] * wrapper_config["agent_count"])

    @staticmethod
    def get_action_space(action_space, wrapper_config=None):
        act_space = FrameStack.get_action_space(action_space, wrapper_config)
        return gym.spaces.Tuple([act_space] * wrapper_config["agent_count"])

    @staticmethod
    def get_observation_adapter(
        observation_space, feature_configs, wrapper_config=None
    ):
        def func(env_obs):
            return env_obs
        return func

    @staticmethod
    def get_action_adapter(action_type, action_space, wrapper_config=None):
        return FrameStack.get_action_adapter(action_type, action_space, wrapper_config)

    @staticmethod
    def get_preprocessor():
        return FrameStack.get_preprocessor()

    def _get_infos(self, obs, rewards, infos):
        return infos

    def _get_rewards(self, last_obs, obs, rewards):
        res = {}
        for key in obs:
            res[key] = rewards[key]
        return res

    def _get_observations(self, obs):
        res = {}
        for key, _obs in obs.items():
            res[key] = self.observation_adapter(_obs)
        return res

    def _update_last_observation(self, obs):
        for k, _obs in obs.items():
            self._last_observations[k] = copy.copy(_obs)

    def keep_done_agents(self, agent_ids, obs, infos, rewards, dones):
        for k, done in dones.items():
            if done:
                self._done_agents.append(k)
        for k in agent_ids:
            if k not in dones.keys():
                rewards[k] = 0
                infos[k] = {'score': 0}
                obs[k] = self._last_observations[k]
                dones[k] = True

    def filter_actions(self, action_dict):
        new_action_dict = {}
        for k in action_dict.keys():
            if k not in self._done_agents:
                new_action_dict[k] = action_dict[k]

        return new_action_dict


    def step(self, action_dict):
        action_dict = self._ungroup_items(action_dict)
        agent_ids = action_dict.keys()
        action_dict = self.filter_actions(action_dict)
        obs, rewards, dones, infos = self.env.step(action_dict)

        infos = self._get_infos(obs, rewards, infos)
        rewards = self._get_rewards(self._last_observations, obs, rewards)
        self._update_last_observation(obs)
        obs = self._get_observations(obs)

        self.keep_done_agents(agent_ids, obs, infos, rewards, dones)

        # Apply grouping transforms to the env outputs
        obs = self._group_items(obs)
        rewards = self._group_items(rewards, agg_fn=lambda gvals: list(gvals.values()))
        dones = self._group_items(dones, agg_fn=lambda gvals: all(gvals.values()))

        infos = self._group_items(
            infos, agg_fn=lambda gvals: {GROUP_INFO: list(gvals.values())}
        )

        # Aggregate rewards, but preserve the original values in infos
        for agent_id, rew in rewards.items():
            if isinstance(rew, list):
                rewards[agent_id] = sum(rew)
                if agent_id not in infos:
                    infos[agent_id] = {}
                infos[agent_id][GROUP_REWARDS] = rew

        dones["__all__"] = all(dones.values())
        return obs, rewards, dones, infos

    def reset(self):
        obs = self.env.reset()
        self._update_last_observation(obs)
        obs = self._get_observations(obs)
        self._done_agents = []
        return self._group_items(obs)
