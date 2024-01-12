import os
import json
import random

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from MultiSystemIdentification.VariableCheetahEnv import *
import cv2
from tqdm import tqdm

# seed everything
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)

# directory of policies
log_dir = "logs/policy/"

# video specs
width = 2560
height = 1440

# iterate through all policies in dir
for policy_dir in tqdm(os.listdir(log_dir)):
    # load policy
    model = PPO.load(os.path.join(log_dir, policy_dir, "ppo_policy"))

    # get dynamics variables
    with open(os.path.join(log_dir, policy_dir, "dynamics_variables.json"), 'r') as f:
        dynamics_variables = json.load(f)

    # create env
    make_env = lambda: VariableCheetahEnv(dynamics_variables, render_mode='rgb_array', width=width, height=height)
    vec_env = make_vec_env(make_env, n_envs=1, )

    # run policy. Write a mp4 to the log dir
    # Define the codec and create a VideoWriter object
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(os.path.join(log_dir, policy_dir, "episode.mp4"), fourcc, 30, (width, height))

    # prepare to run an episode
    obs = vec_env.reset()
    while True:
        action, _states = model.predict(obs)
        obs, rewards, dones, info = vec_env.step(action)
        img = vec_env.render()

        # render image to screen. Switch RGB channel order to BGR for cv2
        img = img[:, :, ::-1]
        video_writer.write(img)

        if dones.any():
            break
    video_writer.release()
    cv2.destroyAllWindows()
