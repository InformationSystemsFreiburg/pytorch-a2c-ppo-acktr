#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Managing the machine park, e.g. a set of machines.
"""

import numpy as np
import pandas as pd

from gym import spaces
from scipy.stats import norm, weibull_min


def linear_rul(config, nsamples, true_rul, initial_true_rul, time_since_last_maintenance):
    noise_level = np.random.normal(
        loc=np.full(nsamples, config['loc_machine_noise_level']),
        scale=np.full(nsamples, config['scale_machine_noise_level']),
        size=nsamples
    )
    noise_level[noise_level < 0.5] = 0.5
    rv_noise_rul_estimation = norm(loc=np.zeros(shape=[nsamples, ]), scale=noise_level)

    error_noise = rv_noise_rul_estimation.rvs(nsamples)
    rul = true_rul + error_noise
    return rul


def clipped_rul_single(true_rul):
    if true_rul <= 0:
        return 0.0

    threshold = 20
    base_epsilon = 0.9
    base_rate = (1 - base_epsilon) / (threshold - 1)

    sigma_start = 2.00
    sigma_end = 1.01
    sigma_rate = (sigma_start - sigma_end) / (threshold - 2)

    base_rul = threshold
    sigma = sigma_start

    if true_rul <= threshold:
        epsilon = base_epsilon + ((threshold - true_rul) * base_rate)
        base_rul = true_rul if np.random.uniform() <= epsilon else true_rul + 1

        sigma = sigma_start - ((threshold - true_rul) * sigma_rate)
        if true_rul == 2.0:
            sigma = 1.0
        elif true_rul == 1.0:
            sigma = 0.95

    rul = base_rul + np.random.normal(loc=0, scale=sigma, size=1)
    return rul[0] if rul[0] >= 0.0 else 0.0


v_clipped_rul_single = np.vectorize(clipped_rul_single)


def clipped_rul(config, nsamples, true_rul, initial_true_rul, time_since_last_maintenance):
    return v_clipped_rul_single(true_rul)


RUL_ESTIMATION_FUNCTIONS = {
    "linear": linear_rul,
    "clipped": clipped_rul,
}


class MachineParkVecSimple(object):
    """
        Defines a machine park within the environment - Vectorized Version, .

        The machine park holds list of machines and handles group interactions as getting te overall state as well as
        the observation state
    """

    def __init__(self, config):
        self.config = config
        self.number_of_machines = self.config['number_of_machines']
        self.number_of_workers = self.config['number_of_workers']

        # define action space
        action_param = np.full(shape=[self.number_of_machines, ], fill_value=2, dtype='int32')
        self._action_space = spaces.MultiDiscrete(action_param)

        # define observation space
        box_space_max = np.inf
        # self._observation_space = spaces.Dict(dict(
        #     rul_estimation=spaces.Box(
        #         0,
        #         box_space_max,
        #         shape=[self.number_of_machines, ],
        #         dtype='float32'),
        #     is_running=spaces.MultiDiscrete(
        #         np.full(shape=[self.number_of_machines, ], fill_value=2, dtype='int32')),
        #     time_since_last_maintenance=spaces.Box(
        #         0,
        #         box_space_max,
        #         shape=[self.number_of_machines, ],
        #         dtype='float32'),
        # ))

        self._observation_space = spaces.Box(
            0,
            box_space_max,
            shape=[self.number_of_machines, 3],
            dtype='float32',
        )

        # machine info
        self._got_repaired = np.full(shape=[self.number_of_machines, ], fill_value=False, dtype=bool)

        # states defining a machine
        self._state_time_since_last_maintenance = np.zeros(
            shape=[self.number_of_machines, ], dtype='float32')

        shape = np.random.normal(
            loc=np.full(self.number_of_machines, self.config['loc_machine_shape']),
            scale=np.full(self.number_of_machines, self.config['scale_machine_shape'])
        )
        shape[shape < self.config['min_machine_shape']] = self.config['min_machine_shape']
        scale = np.random.normal(
            loc=np.full(self.number_of_machines, self.config['loc_machine_scale']),
            scale=np.full(self.number_of_machines, self.config['scale_machine_scale'])
        )
        scale[scale < self.config['min_machine_scale']] = self.config['min_machine_scale']
        loc = np.random.normal(
            loc=np.full(self.number_of_machines, self.config['loc_machine_loc']),
            scale=np.full(self.number_of_machines, self.config['scale_machine_loc'])
        )
        loc[loc < self.config['min_machine_loc']] = self.config['min_machine_loc']
        self._rv_rul = weibull_min(c=shape, scale=scale, loc=loc)
        self._state_initial_true_rul = np.ceil(self._rv_rul.rvs(self.number_of_machines))
        self._state_initial_true_rul[ self._state_initial_true_rul > self.config['max_rul']] = self.config['max_rul']
        self._state_true_rul = self._state_initial_true_rul.copy()
        self._state_rul = self.compute_state_rul(
            self.number_of_machines,
            self._state_true_rul,
            self._state_initial_true_rul,
            self._state_time_since_last_maintenance)

        # machine park info
        self._machine_park_info_df = pd.DataFrame(
            {
                'machine_id': np.arange(0, self.number_of_machines),
                'weibull_shape': shape,
                'weibull_scale': scale,
                'weibull_loc': loc,
                'weibull_mean': self._rv_rul.mean(),
            },
            columns=[
                'machine_id', 'weibull_shape', 'weibull_scale', 'weibull_loc', 'weibull_mean',
            ]
        )
        self.internal_state_days_with_maintenance = 0.0

    def reset(self):
        self._got_repaired = np.full(shape=[self.number_of_machines, ], fill_value=False, dtype=bool)
        self._state_time_since_last_maintenance = np.zeros(shape=[self.number_of_machines, ], dtype='float32')
        self._state_initial_true_rul = np.ceil(self._rv_rul.rvs(self.number_of_machines).astype('float32'))
        self._state_true_rul = self._state_initial_true_rul.copy()
        self._state_rul = self.compute_state_rul(
            self.number_of_machines,
            self._state_true_rul,
            self._state_initial_true_rul,
            self._state_time_since_last_maintenance)
        self.internal_state_days_with_maintenance = 0.0

    # action: vector of integers, here simple binary. action == 1 -> repair machine, action == 0 -> do nothing
    # ATTENTION: the mapping between action index and corresponding cannot change!
    def perform_action(self, actions):
        stats = dict(
            stats_nr_maintenance_actions=actions.sum(),
            stats_nr_unavailable_workers_penalty=0.0,
            stats_nr_machines_in_good_state_before_action=self.state_is_running.sum(),
            stats_nr_machines_in_good_state_after_action=0.0,
            stats_absolute_reward_regret=0.0,
            stats_absolute_reward_penalty=0.0,
            stats_relative_reward_regret=0.0,
            stats_relative_reward_penalty=0.0,
            stats_nr_preventive_actions=0.0,
            stats_nr_corrective_actions=0.0,
            stats_nr_days_with_machines_in_failure_state_after_maintenance=0.0,
            stats_nr_maintenance_days=0.0,
        )
        reward = 0.0
        invalid_action = actions.sum() > self.number_of_workers

        if invalid_action:
            reward = -1000 * (actions.sum() - self.number_of_workers)
            stats['stats_nr_unavailable_workers_penalty'] = (actions.sum() - self.number_of_workers)

        calc_reward, stats = self.calculate_reward(actions, stats)
        reward +=calc_reward

        idx_to_repair = np.where(actions == 1)[0]
        idx_to_noop = np.where(actions == 0)[0]
        stats = self.repair(idx_to_repair, stats)
        stats = self.noop(idx_to_noop, stats)

        stats['stats_nr_machines_in_good_state_after_action'] = self.state_is_running.sum()
        stats['stats_nr_days_with_machines_in_failure_state_after_maintenance'] = 1.0 if \
            self.state_is_running.any() == 0 else 0.0

        return reward, invalid_action, stats

    def calculate_reward(self, actions, stats):
        reward_0action_boost = 0.0
        reward_regret = 0.0
        reward_penalty = 0.0
        n_good_repairs = 0
        n_machines_in_failure_state = self.number_of_machines - self.state_is_running.sum()

        if actions.sum() == 0 and self.state_is_running.all() == 1 and self.config['enable_0action_boost']:
            reward_0action_boost = 1.0

        if actions.sum() > 0:
            self.internal_state_days_with_maintenance += 1
            stats['stats_nr_maintenance_days'] = 1.0

        for i in range(actions.size):
            if actions[i] == 1 and self.state_is_running[i] == 0 or actions[i] == 0 and self.state_is_running[i] == 1:
                continue
            elif actions[i] == 1 and self.state_is_running[i] == 1:
                n_good_repairs += 1
                reward_regret += 1 - ((self.state_true_rul[i]-1) / self.config['max_rul'])
            elif actions[i] == 0 and self.state_is_running[i] == 0:
                reward_penalty += -1

        stats['stats_absolute_reward_regret'] = reward_regret
        stats['stats_absolute_reward_penalty'] = reward_penalty
        reward_regret = reward_regret / n_good_repairs if n_good_repairs > 0 else 0
        reward_penalty = reward_penalty / n_machines_in_failure_state if n_machines_in_failure_state > 0 else 0
        stats['stats_relative_reward_regret'] = reward_regret
        stats['stats_relative_reward_penalty'] = reward_penalty

        reward = reward_0action_boost + reward_regret + reward_penalty
        return reward, stats

    def repair(self, idx, stats):
        if idx.size > 0:
            # statistics
            idx_is_preventive = np.where(self.state_is_running[idx] == 1)[0]
            idx_is_corrective = np.where(self.state_is_running[idx] == 0)[0]
            stats['stats_nr_preventive_actions'] = idx_is_preventive.size
            stats['stats_nr_corrective_actions'] = idx_is_corrective.size
            # actual repair
            self._state_initial_true_rul[idx] = np.ceil(self._rv_rul.rvs(self.number_of_machines)[idx])
            self._state_true_rul[idx] = self._state_initial_true_rul[idx].copy()
            self._state_time_since_last_maintenance[idx] = 0.0
            self._state_rul[idx] = self.compute_state_rul(
                idx.size,
                self._state_true_rul[idx],
                self._state_initial_true_rul[idx],
                self._state_time_since_last_maintenance[idx])
            self._got_repaired[idx] = True
        return stats

    def noop(self, idx, stats):
        if idx.size > 0:
            self._state_true_rul[idx] = self._state_true_rul[idx] - 1.0
            self._state_true_rul[self._state_true_rul < 0] = 0
            self._state_time_since_last_maintenance[idx] = self._state_time_since_last_maintenance[idx] + 1.0
            self._state_rul[idx] = self.compute_state_rul(
                idx.size,
                self._state_true_rul[idx],
                self._state_initial_true_rul[idx],
                self._state_time_since_last_maintenance[idx])
            self._got_repaired[idx] = False
        return stats

    def get_episode_over_reward(self):
        # return -(self.internal_state_days_with_maintenance/self.config["time_horizon"])
        return -self.internal_state_days_with_maintenance

    def compute_state_rul(self, nsamples, true_rul, initial_true_rul, time_since_last_maintenance):
        return RUL_ESTIMATION_FUNCTIONS[self.config['rul_estimation_method']](
            self.config, nsamples, true_rul, initial_true_rul, time_since_last_maintenance)

    def get_ob_state(self):
        return np.vstack((self.state_rul, self.state_time_since_last_maintenance, self.state_is_running)).T
        # return dict(rul_estimation=self.state_rul,
        #             is_running=self.state_is_running,
        #             time_since_last_maintenance=self.state_time_since_last_maintenance)

    @property
    def action_space(self):
        return self._action_space

    @property
    def observation_space(self):
        return self._observation_space

    @property
    def state_rul(self):
        return self._state_rul

    @property
    def state_time_since_last_maintenance(self):
        return self._state_time_since_last_maintenance

    @property
    def state_is_running(self):
        return np.array(self._state_true_rul > 0.0).astype('float32')

    @property
    def state_true_rul(self):
        return self._state_true_rul

    @property
    def machine_park_info_df(self):
        return self._machine_park_info_df
