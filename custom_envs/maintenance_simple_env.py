#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Simulate the simplified maintenance scheduling environment

Each episode is a maintenance cycle over time horizon T, e.g. 10 years
"""


import numpy as np
import tensorflow as tf

import gym
from gym.spaces import Discrete, Box
from gym.envs.registration import EnvSpec
from gym import logger

from custom_envs.machines.machine_park_simple import MachineParkVecSimple

ENV_CONFIG = dict(
    # general
    number_of_machines=100,
        time_horizon=365,
    episodes=100,
    number_of_workers=1,
    max_rul=365,
    # costs
    cost_unavailable_workers_penalty=10000,
    cost_penalty_base=1,
    cost_price_t=1,
    cost_worker_hourly_wage=1,
    # machine output capacity parameter
    loc_mean_output=10,
    scale_mean_output=4,
    # corrective
    loc_cost_material_cor=8,
    scale_cost_material_cor=2,

    max_time_spend_working_cor=24,
    min_time_spend_working_cor=4,
    loc_time_spend_working_cor=8,
    scale_time_spend_working_cor=8.16,

    # preventive
    loc_cost_material_pre=6,
    scale_cost_material_pre=2,

    max_time_spend_working_pre=20,
    min_time_spend_working_pre=4,
    loc_time_spend_working_pre=7,
    scale_time_spend_working_pre=8.14,

    # machine initial parameters
    min_machine_shape=2,
    min_machine_loc=10,
    min_machine_scale=24,
    max_machine_shape=5,
    max_machine_loc=20,
    max_machine_scale=38,
    loc_machine_shape=3,
    scale_machine_shape=1,
    loc_machine_loc=15,
    scale_machine_loc=2,
    loc_machine_scale=30,
    scale_machine_scale=2,
    loc_machine_noise_level=2,
    scale_machine_noise_level=1,

    # choose rul estimation
    rul_estimation_method='clipped',

    # keras expert model for worker
    path_to_keras_expert_model='./custom_envs/pretrained_models/model_f_wsupervisor_D3BND3BND3BND1_adam.h5',

    # boost "save 0action" reward
    enable_0action_boost=False,
)


class MaintenanceEnv(gym.Env):
    """
    Define a complex maintenance scheduling environment

    The environment defines the actions space and the reward function.
    """

    metadata = {'render.modes': ['human']}

    def __init__(self, config=ENV_CONFIG):
        self.__version__ = '0.0.1'
        logger.info('RawMaintenanceEnv - Version {}'.format(self.__version__))
        self._spec = EnvSpec("RawMaintenanceEnv-Worker-{}-v0".format(config["number_of_workers"]))

        self.time = 1
        self.time_horizon = config["time_horizon"]

        # initialize machines
        self.machine_park = MachineParkVecSimple(config)

        self.action_space = self.machine_park.action_space
        self.observation_space = self.machine_park.observation_space

        self.number_of_machines = self.machine_park.number_of_machines
        self.number_of_workers = self.machine_park.number_of_workers

    def step(self, action):
        reward, invalid_action, stats = self.machine_park.perform_action(action)
        ob = self.machine_park.get_ob_state()
        self.time = self.time + 1
        episode_over = self.time >= self.time_horizon or invalid_action

        reward += self.machine_park.get_episode_over_reward() if episode_over else 0.0

        return ob, reward, episode_over, stats

    def reset(self):
        self.time = 1
        self.machine_park.reset()
        return self.machine_park.get_ob_state()

    def render(self, mode='human', close=False):
        pass

    @property
    def machine_park_info_df(self):
        return self.machine_park.machine_park_info_df

    @property
    def cheat_state_true_rul(self):
        return self.machine_park.state_true_rul

    def __str__(self):
        return 'RawMaintenanceEnv'


class WorkerMaintenanceEnv(gym.Env):
    """Example of a custom env in which you have to walk down a corridor.
    You can configure the length of the corridor via the env config."""

    def __init__(self, config):
        self.env = MaintenanceEnv(config)
        self.config = config
        self.n_worker = config["number_of_workers"]
        self.action_space = Discrete(self.n_worker + 1)
        self.observation_space = Box(
                0,
                1,
                shape=[config["number_of_machines"], ],
                dtype='float32')

        self.model_expert = tf.keras.models.load_model(config["path_to_keras_expert_model"])
        self._spec = EnvSpec("WorkerMaintenanceEnv-Worker-{}-v0".format(self.n_worker))
        self.ranking = 0

    def reset(self):
        obs = self.env.reset()
        obs, self.ranking = self._transform_obs(obs)
        return obs

    def step(self, action):
        actions = self._map_to_env_action(action)
        ob, reward, episode_over, stats = self.env.step(actions)
        ob, self.ranking = self._transform_obs(ob)
        return ob, reward, episode_over, stats

    def render(self, mode='human', close=False):
        pass

    def _transform_obs(self, obs):
        # new_obs = self._obs_to_np_array(obs)
        new_obs = self.model_expert.predict(obs, batch_size=self.config["number_of_machines"])
        new_obs = new_obs.reshape(new_obs.shape[0])
        priority_ranking = np.argsort(-new_obs,axis=0)
        return new_obs, priority_ranking

    def _map_to_env_action(self, action):
        actions = np.zeros(shape=[self.config["number_of_machines"],])
        actions[self.ranking[:action]] = 1
        return actions

    def _obs_to_np_array(self, obs):
        return np.vstack((obs['rul_estimation'], obs['time_since_last_maintenance'], obs['is_running'])).T

    def __str__(self):
        return 'WorkerMaintenanceEnv'


if __name__ == '__main__':
    env = MaintenanceEnv()
    print(env.machine_park.machine_park_info_df)
