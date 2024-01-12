import os
import json
import random

import numpy
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from MultiSystemIdentification.VariableCheetahEnv import *
import cv2
from tqdm import tqdm, trange

# seed everything
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


# directory of policies
log_dir = "logs/policy/"
data_dir = "data/"
os.makedirs(data_dir, exist_ok=True)

# hyper parameters
transitions_per_policy = 20_000
num_random_envs = 200

# load all policies
policies = []
dynamics_variables_per_policy = []
for policy_dir in os.listdir(log_dir):
    # load policy
    model = PPO.load(os.path.join(log_dir, policy_dir, "ppo_policy"))
    policies.append(model)
    # get dynamics variables
    with open(os.path.join(log_dir, policy_dir, "dynamics_variables.json"), 'r') as f:
        dynamics_variables = json.load(f)
    dynamics_variables_per_policy.append(dynamics_variables)

# gather data from the exact dynamics they are trained on. Store in a huge list
state_actions = []
for index in trange(len(policies), desc="Part 1: Gathering State-Actions"):
    policy = policies[index]
    dynamics_variables = dynamics_variables_per_policy[index]

    # create env
    make_env = lambda: VariableCheetahEnv(dynamics_variables,)
    vec_env = make_vec_env(make_env, n_envs=1, )

    # prepare to run an episode
    obs = vec_env.reset()
    for current_index in range(transitions_per_policy):
        # get action
        action, _states = policy.predict(obs)

        # fetch mujoco state, so that we can load it exactly for any environment
        state = vec_env.envs[0].env.env.sim.get_state()
        state_actions.append((state, action))

        # step env, go to next step
        n_obs, rewards, dones, info = vec_env.step(action)
        obs = n_obs
        if dones.any():
            obs = vec_env.reset()
del policies, dynamics_variables_per_policy # unload to save memory

# now that we have #policy * #transitions_per_policy state actions,
# we want to simulate the transition for many test envs to get a distribution of each state-action-next_state
dynamics_variable_ranges={  'friction':(DEFAULT_FRICTION*0.5, DEFAULT_FRICTION*2),
                            'torso_length':(DEFAULT_TORSO_LENGTH * 0.5, DEFAULT_TORSO_LENGTH * 1.5),
                            'bthigh_length':(DEFAULT_BTHIGH_LENGTH * 0.5, DEFAULT_BTHIGH_LENGTH * 1.5),
                            'bshin_length':(DEFAULT_BSHIN_LENGTH * 0.5, DEFAULT_BSHIN_LENGTH * 1.5),
                            'bfoot_length':(DEFAULT_BFOOT_LENGTH * 0.5, DEFAULT_BFOOT_LENGTH * 1.5),
                            'fthigh_length':(DEFAULT_FTHIGH_LENGTH * 0.5, DEFAULT_FTHIGH_LENGTH * 1.5),
                            'fshin_length':(DEFAULT_FSHIN_LENGTH * 0.5, DEFAULT_FSHIN_LENGTH * 1.5),
                            'ffoot_length':(DEFAULT_FFOOT_LENGTH * 0.5, DEFAULT_FFOOT_LENGTH * 1.5),
                            'bthigh_gear':(DEFAULT_BTHIGH_GEAR * 0.0, DEFAULT_BTHIGH_GEAR * 2.0),
                            'bshin_gear':(DEFAULT_BSHIN_GEAR * 0.0, DEFAULT_BSHIN_GEAR * 2.0),
                            'bfoot_gear':(DEFAULT_BFOOT_GEAR * 0.0, DEFAULT_BFOOT_GEAR * 2.0),
                            'fthigh_gear':(DEFAULT_FTHIGH_GEAR * 0.0, DEFAULT_FTHIGH_GEAR * 2.0),
                            'fshin_gear':(DEFAULT_FSHIN_GEAR * 0.0, DEFAULT_FSHIN_GEAR * 2.0),
                            'ffoot_gear':(DEFAULT_FFOOT_GEAR * 0.0, DEFAULT_FFOOT_GEAR * 2.0),
                           }
dynamics_variable_defaults = {  'friction':DEFAULT_FRICTION,
                                'torso_length':DEFAULT_TORSO_LENGTH,
                                'bthigh_length':DEFAULT_BTHIGH_LENGTH,
                                'bshin_length':DEFAULT_BSHIN_LENGTH,
                                'bfoot_length':DEFAULT_BFOOT_LENGTH,
                                'fthigh_length':DEFAULT_FTHIGH_LENGTH,
                                'fshin_length':DEFAULT_FSHIN_LENGTH,
                                'ffoot_length':DEFAULT_FFOOT_LENGTH,
                                'bthigh_gear':DEFAULT_BTHIGH_GEAR,
                                'bshin_gear':DEFAULT_BSHIN_GEAR,
                                'bfoot_gear':DEFAULT_BFOOT_GEAR,
                                'fthigh_gear':DEFAULT_FTHIGH_GEAR,
                                'fshin_gear':DEFAULT_FSHIN_GEAR,
                                'ffoot_gear':DEFAULT_FFOOT_GEAR,
                                 }

# initalize buffers for transitions
states = numpy.zeros((num_random_envs, len(state_actions), vec_env.observation_space.shape[0]))
actions = numpy.zeros((num_random_envs, len(state_actions), vec_env.action_space.shape[0]))
next_states = numpy.zeros((num_random_envs, len(state_actions), vec_env.observation_space.shape[0]))

# for a large sample of possible dynamics
dyns = numpy.zeros((num_random_envs, len(dynamics_variable_ranges.keys())))
for example in trange(num_random_envs, desc="Part 2: Gathering Next States for random dynamics"):
    # generate random dynamics constants
    random_dynamics = {}
    for key in dynamics_variable_ranges.keys():
        val = np.random.uniform(dynamics_variable_ranges[key][0], dynamics_variable_ranges[key][1])
        random_dynamics[key] = (val, val)
        dyns[example, list(dynamics_variable_ranges.keys()).index(key)] = val / dynamics_variable_defaults[key]

    # create env
    make_env = lambda: VariableCheetahEnv(random_dynamics)
    vec_env = make_vec_env(make_env, n_envs=1, )

    # prepare test all transitions
    obs = vec_env.reset()
    for current_index in range(len(state_actions)):
        mujoco_state, action = state_actions[current_index]

        # load mujoco state
        vec_env.envs[0].env.env.sim.set_state(mujoco_state)

        # fetch observation
        obs = vec_env.envs[0].env.env.unwrapped._get_obs()

        # transition
        n_obs, rewards, dones, info = vec_env.step(action)

        # store in buffer
        states[example, current_index] = obs
        actions[example, current_index] = action
        next_states[example, current_index] = n_obs
#
# # save buffers to data
numpy.save(os.path.join(data_dir, "states.npy"), states)
numpy.save(os.path.join(data_dir, "actions.npy"), actions)
numpy.save(os.path.join(data_dir, "next_states.npy"), next_states)
numpy.save(os.path.join(data_dir, "dyns.npy"), dyns)



