import argparse
import os
import types

import pandas as pd
import numpy as np
import torch
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.vec_normalize import VecNormalize

from envs import make_env

from custom_envs.maintenance_simple_env import ENV_CONFIG


parser = argparse.ArgumentParser(description='RL')
parser.add_argument('--seed', type=int, default=1,
                    help='random seed (default: 1)')
parser.add_argument('--num-stack', type=int, default=4,
                    help='number of frames to stack (default: 4)')
parser.add_argument('--log-interval', type=int, default=10,
                    help='log interval, one log per n updates (default: 10)')
parser.add_argument('--env-name', default='PongNoFrameskip-v4',
                    help='environment to train on (default: PongNoFrameskip-v4)')
parser.add_argument('--load-dir', default='./trained_models/',
                    help='directory to save agent logs (default: ./trained_models/)')
parser.add_argument('--add-timestep', action='store_true', default=False,
                    help='add timestep to observations')
## NG: custom arguments
parser.add_argument('--number-of-workers', type=int, default=10,
                    help='number of maintenance workers tha agent can schedule in the maintenance env')
parser.add_argument('--path-to-keras-expert-model', default='./custom_envs/pretrained_models/',
                    help='path to keras expert model that the worker uses to transform the original env '
                         'observation into a worker space')
parser.add_argument('--save-model-postfix', default='',
                    help='add some string to the model name')
parser.add_argument('--enable-debug-info-print', action='store_true', default=False,
                    help='enable debug info print')
parser.add_argument('--disable-env-normalize-ob', action='store_true', default=False,
                    help='disable normalization of env observation see VecNormalize()')
parser.add_argument('--disable-env-normalize-rw', action='store_true', default=False,
                    help='disable normalization of env reward see VecNormalize()')
parser.add_argument('--path-to-ac', default=None,
                    help='path to trained pytorch actor critic model')
parser.add_argument('--path-to-results-dir', default='./results/',
                    help='path to results')
parser.add_argument('--strategy-name', default='default',
                    help='name of the strategy')
parser.add_argument('--number-of-episodes', type=int, default=100,
                    help='number of episodes')
args = parser.parse_args()


def init_statistics_vec_df(maintenance_strategy, nworker, nmachines, max_exp_nr):
    stats = pd.DataFrame(
        {
            'maintenance_strategy': np.asarray([maintenance_strategy] * max_exp_nr, dtype=np.dtype('U25')),
            'nworker': [nworker] * max_exp_nr,
            'nmachines': [nmachines] * max_exp_nr,
            'wm_ratio': [nworker / nmachines] * max_exp_nr,
            'episode': np.zeros(max_exp_nr, dtype=np.int64),
            'nr_maintenance_actions': np.zeros(max_exp_nr, dtype=np.float64),
            'nr_preventive_actions': np.zeros(max_exp_nr, dtype=np.float64),
            'nr_corrective_actions': np.zeros(max_exp_nr, dtype=np.float64),
            'nr_machines_in_good_state_before_action': np.zeros(max_exp_nr, dtype=np.float64),
            'nr_machines_in_good_state_after_action': np.zeros(max_exp_nr, dtype=np.float64),
            'nr_unavailable_workers_penalty': np.zeros(max_exp_nr, dtype=np.float64),
            'absolute_reward_regret': np.zeros(max_exp_nr, dtype=np.float64),
            'absolute_reward_penalty': np.zeros(max_exp_nr, dtype=np.float64),
            'relative_reward_regret': np.zeros(max_exp_nr, dtype=np.float64),
            'relative_reward_penalty': np.zeros(max_exp_nr, dtype=np.float64),
            'nr_days_with_machines_in_failure_state_after_maintenance': np.zeros(max_exp_nr, dtype=np.float64),
            'nr_maintenance_days': np.zeros(max_exp_nr, dtype=np.float64),
            'reward_total': np.zeros(max_exp_nr, dtype=np.float64),
        },
        columns=[
            'maintenance_strategy', 'nworker', 'nmachines', 'wm_ratio', 'episode',
            'nr_maintenance_actions', 'nr_preventive_actions', 'nr_corrective_actions',
            'nr_machines_in_good_state_before_action', 'nr_machines_in_good_state_after_action',
            'nr_unavailable_workers_penalty', 'absolute_reward_regret', 'absolute_reward_penalty',
            'relative_reward_regret', 'relative_reward_penalty',
            'nr_days_with_machines_in_failure_state_after_maintenance', 'nr_maintenance_days',
            'reward_total',
        ]
    )
    return stats


# Done: change make_env behaviour such that simple env is created; see custom_envs.py
# args.env_name has to start with ng_ currently only WorkerMaintenanceEnv is working
env_config = ENV_CONFIG.copy()
# env_config['path_to_keras_expert_model'] = args.path_to_keras_expert_model
env_config['number_of_workers'] = args.number_of_workers

# exchange None with args.log_dir
env = make_env(args.env_name, args.seed, 0, None, args.add_timestep, env_config)
env = DummyVecEnv([env])

# save_path = os.path.join(args.save_dir, args.algo)
# model_name = "{}-{}-{}-{}.pt".format(args.env_name, args.algo, args.save_model_postfix, j)
#             torch.save(save_model, os.path.join(save_path, model_name))

if args.path_to_ac:
    actor_critic, ob_rms = \
        torch.load(args.path_to_ac)
else:
    actor_critic, ob_rms = \
            torch.load(os.path.join(args.load_dir, args.env_name + ".pt"))


if len(env.observation_space.shape) == 1:
    env = VecNormalize(env, ob=not args.disable_env_normalize_ob, ret=not args.disable_env_normalize_rw)
    env.ob_rms = ob_rms

    # An ugly hack to remove updates
    def _obfilt(self, obs):
        if self.ob_rms:
            obs = np.clip((obs - self.ob_rms.mean) / np.sqrt(self.ob_rms.var + self.epsilon), -self.clipob, self.clipob)
            return obs
        else:
            return obs
    env._obfilt = types.MethodType(_obfilt, env)
    render_func = env.venv.envs[0].render
else:
    render_func = env.envs[0].render

obs_shape = env.observation_space.shape
obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])
current_obs = torch.zeros(1, *obs_shape)
states = torch.zeros(1, actor_critic.state_size)
masks = torch.zeros(1, 1)


def update_current_obs(obs):
    shape_dim0 = env.observation_space.shape[0]
    obs = torch.from_numpy(obs).float()
    if args.num_stack > 1:
        current_obs[:, :-shape_dim0] = current_obs[:, shape_dim0:]
    current_obs[:, -shape_dim0:] = obs


if args.env_name.find('Bullet') > -1:
    import pybullet as p

    torsoId = -1
    for i in range(p.getNumBodies()):
        if (p.getBodyInfo(i)[0].decode() == "torso"):
            torsoId = i

statistics = init_statistics_vec_df(
                    maintenance_strategy=args.strategy_name,
                    nworker=args.number_of_workers,
                    nmachines=env_config['number_of_machines'],
                    max_exp_nr=args.number_of_episodes)

# this ill break the code, if we use an other environment than ng_...
action_seq = np.zeros([args.number_of_episodes, env_config["time_horizon"]-1])

for i in range(args.number_of_episodes):

    # render_func('human')
    obs = env.reset()
    update_current_obs(obs)

    reward = 0
    done = False
    sum_reward = 0.0
    sum_nr_maintenance_actions = 0.0
    sum_nr_preventive_actions = 0.0
    sum_nr_corrective_actions = 0.0
    sum_nr_machines_in_good_state_before_action = 0.0
    sum_nr_machines_in_good_state_after_action = 0.0
    sum_nr_unavailable_workers_penalty = 0.0
    sum_absolute_reward_regret = 0.0
    sum_absolute_reward_penalty = 0.0
    sum_relative_reward_regret = 0.0
    sum_relative_reward_penalty = 0.0
    sum_nr_days_with_machines_in_failure_state_after_maintenance = 0.0
    sum_nr_maintenance_days = 0.0
    acsec = []
    while True:

        with torch.no_grad():
            value, action, _, states = actor_critic.act(current_obs,
                                                        states,
                                                        masks,
                                                        deterministic=True)
        cpu_actions = action.squeeze(1).cpu().numpy()
        acsec.extend(cpu_actions[0][0])
        # Obser reward and next obs
        obs, reward, done, info_ = env.step(cpu_actions)
        info = info_[0]

        sum_reward += reward
        sum_nr_maintenance_actions += info['stats_nr_maintenance_actions']
        sum_nr_preventive_actions += info['stats_nr_preventive_actions']
        sum_nr_corrective_actions += info['stats_nr_corrective_actions']
        sum_nr_machines_in_good_state_before_action += info[
            'stats_nr_machines_in_good_state_before_action']
        sum_nr_machines_in_good_state_after_action += info[
            'stats_nr_machines_in_good_state_after_action']
        sum_nr_unavailable_workers_penalty += info['stats_nr_unavailable_workers_penalty']
        sum_absolute_reward_regret += info['stats_absolute_reward_regret']
        sum_absolute_reward_penalty += info['stats_absolute_reward_penalty']
        sum_relative_reward_regret += info['stats_relative_reward_regret']
        sum_relative_reward_penalty += info['stats_relative_reward_penalty']
        sum_nr_days_with_machines_in_failure_state_after_maintenance += \
            info['stats_nr_days_with_machines_in_failure_state_after_maintenance']
        sum_nr_maintenance_days += info['stats_nr_maintenance_days']


        masks.fill_(0.0 if done else 1.0)

        if current_obs.dim() == 4:
            current_obs *= masks.unsqueeze(2).unsqueeze(2)
        else:
            current_obs *= masks
        update_current_obs(obs)

        if args.env_name.find('Bullet') > -1:
            if torsoId > -1:
                distance = 5
                yaw = 0
                humanPos, humanOrn = p.getBasePositionAndOrientation(torsoId)
                p.resetDebugVisualizerCamera(distance, yaw, -20, humanPos)

        if done:
            statistics.loc[i, 'reward_total'] = sum_reward
            statistics.loc[i, 'episode'] = i
            statistics.loc[i, 'nr_maintenance_actions'] = sum_nr_maintenance_actions
            statistics.loc[i, 'nr_preventive_actions'] = sum_nr_preventive_actions
            statistics.loc[i, 'nr_corrective_actions'] = sum_nr_corrective_actions
            statistics.loc[i, 'nr_machines_in_good_state_before_action'] = \
                sum_nr_machines_in_good_state_before_action
            statistics.loc[i, 'nr_machines_in_good_state_after_action'] = \
                sum_nr_machines_in_good_state_after_action
            statistics.loc[i, 'nr_unavailable_workers_penalty'] = sum_nr_unavailable_workers_penalty
            statistics.loc[i, 'absolute_reward_regret'] = sum_absolute_reward_regret
            statistics.loc[i, 'absolute_reward_penalty'] = sum_absolute_reward_penalty
            statistics.loc[i, 'relative_reward_regret'] = sum_relative_reward_regret
            statistics.loc[i, 'relative_reward_penalty'] = sum_relative_reward_penalty
            statistics.loc[i, 'nr_days_with_machines_in_failure_state_after_maintenance'] = \
                sum_nr_days_with_machines_in_failure_state_after_maintenance
            statistics.loc[i, 'nr_maintenance_days'] = sum_nr_maintenance_days
            action_seq[i,:] = acsec
            break

try:
    os.makedirs(args.path_to_results_dir)
except OSError:
    pass

# env.machine_park_info_df.to_csv(
#     os.path.join(
#         args.path_to_results_dir,
#         'machine_park_info_{}_w{}.csv'.format(
#             args.strategy_name,
#             args.number_of_workers)))
# env.close()
np.savetxt(
    os.path.join(
        args.path_to_results_dir,
        'action_sequence_{}_w{}.csv'.format(
            args.strategy_name,
            args.number_of_workers)),
    action_seq,
    delimiter=",")
statistics.to_csv(
    os.path.join(
        args.path_to_results_dir,
        'statistics_{}_w{}.csv'.format(
            args.strategy_name,
            args.number_of_workers)))

    # render_func('human')
