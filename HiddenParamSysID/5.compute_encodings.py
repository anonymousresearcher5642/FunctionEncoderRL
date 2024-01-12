import argparse
import os
import json
import random

import numpy
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from MultiSystemIdentification.FE import FE
from MultiSystemIdentification.VariableCheetahEnv import *
import cv2
from tqdm import tqdm, trange


parser = argparse.ArgumentParser(
                    prog='6.compute_encodings.py',
                    description='Computes the encodings of the dynamics variables')
parser.add_argument('--dimensions_to_investigate', type=str, default="torso_length",
                    help='The dimensions to investigate')
parser.add_argument('--encoder_to_load', type=str, default="logs/predictors/2023-11-08 15:32:22/FE",
                    help='The encoder to load')
args = parser.parse_args()

dimensions_to_investigate = args.dimensions_to_investigate
encoder_to_load = args.encoder_to_load

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
transitions_per_policy = 5000 # TODO change back to 20_000 or maybe just 2000

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

# create a sweep over the two dimensions
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
grid_size = 11
dim_range = np.linspace(*dynamics_variable_ranges[dimensions_to_investigate], grid_size)

# generate the grid
env_variables = []
for dim in dim_range:
    dynamics_variables = {}
    dynamics_variables[dimensions_to_investigate] = (dim, dim)
    env_variables.append(dynamics_variables)

# initalize buffers for transitions
# we will do a sweep of a 10x10 grid of dynamics variables
states = numpy.zeros((len(env_variables), len(state_actions), vec_env.observation_space.shape[0]))
actions = numpy.zeros((len(env_variables), len(state_actions), vec_env.action_space.shape[0]))
next_states = numpy.zeros((len(env_variables), len(state_actions), vec_env.observation_space.shape[0]))

# for a large sample of possible dynamics
for example in trange(len(env_variables), desc="Part 2: Gathering Next States for random dynamics"):
    # generate random dynamics constants
    dynamics = env_variables[example]

    # create env
    make_env = lambda: VariableCheetahEnv(dynamics)
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




# Now we compute encodings for all envs. First load the model
function_encoder = FE(states.shape[2] + actions.shape[2], next_states.shape[2], embed_size=100, device="cuda:0")
function_encoder.load(encoder_to_load)

# compute encodings via the loaded encoder
inputs = torch.tensor(np.concatenate((states, actions), axis=2), dtype=torch.float32).to("cuda:0")
outputs = torch.tensor(next_states, dtype=torch.float32).to("cuda:0")

# normalize input
mean = torch.mean(inputs[:, :, :17], dim=(0, 1))
stds = torch.std(inputs[:, :, :17], dim=(0, 1))
assert mean.shape == (17,), f"mean is wrong shape, got {mean.shape}, expected {(17,)}"
assert stds.shape == (17,), f"stds is wrong shape, got {stds.shape}, expected {(17,)}"
eps = 1e-8
inputs[:, :, :17] = (inputs[:, :, :17] - mean) / (stds + eps)
outputs[:, :, :17] = (outputs[:, :, :17] - mean) / (stds + eps)

# get encodings
encodings = function_encoder.get_encodings(inputs, outputs)

# first save dynamics variables and encodings so we can reuse them later if needed
savedir = os.path.join(data_dir, dimensions_to_investigate)
os.makedirs(savedir, exist_ok=True)
np.save(os.path.join(savedir, "dynamics_variables.npy"), env_variables)
np.save(os.path.join(savedir, "encodings.npy"), encodings.cpu().numpy())
