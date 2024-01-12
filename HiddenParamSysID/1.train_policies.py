import json
import random

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from MultiSystemIdentification.VariableCheetahEnv import *

# seed everything
seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)



# Env dynamics parameters
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

# sample 10 sets of dynamic parameters into a list
dynamics_variable_list = []
for i in range(10):
    new_dynamics_variable = {}
    for key in dynamics_variable_ranges.keys():
        val = np.random.uniform(dynamics_variable_ranges[key][0], dynamics_variable_ranges[key][1])
        new_dynamics_variable[key] = (val, val)
    dynamics_variable_list.append(new_dynamics_variable)

for dynamics_variables in dynamics_variable_list:
    # Parallel environments
    make_env = lambda: VariableCheetahEnv(dynamics_variables)
    vec_env = make_vec_env(make_env, n_envs=4)

    # current date timestring for logging
    date_time_string = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    log_dir = "logs/policy/" + date_time_string + "/"
    os.makedirs(log_dir, exist_ok=True)

    # train for some number of steps
    model = PPO("MlpPolicy", vec_env, verbose=0, tensorboard_log=log_dir)
    model.learn(total_timesteps=2_000_000, progress_bar=True)
    model.save(os.path.join(log_dir, "ppo_policy"))

    # write dynamics variables to file as well in case we need them
    with open(os.path.join(log_dir, "dynamics_variables.json"), 'w') as f:
        json.dump(dynamics_variables, f)
