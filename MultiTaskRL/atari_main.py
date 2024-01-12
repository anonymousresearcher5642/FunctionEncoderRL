import os
import pickle

import numpy as np
from arguments import get_args
from atari_modules.dqn_agent import dqn_agent
from atari_modules.fb_agent import FBAgent
from atari_modules.her_dqn_agent import HerDQNAgent
from atari_modules.reward_encoder_ablation_agent import RewardEncoderAblationAgent
from atari_modules.reward_encoder_agent import RewardEncoderAgent
from atari_modules.transformer_agent import TransformerAgent
from atari_modules.transformer_agent2 import TransformerAgent2
from atari_modules.wrappers import make_goalPacman
import random
import torch



def get_env_params(env):
    params = {'obs': env.observation_space['observation'].shape,
              'goal': 2,
              'action': 5,
              }
    params['max_timesteps'] = 50
    return params


def launch(args):

    env = make_goalPacman()
    # set random seeds for reproduce
    env.seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.cuda:
        torch.cuda.manual_seed(args.seed)
    # get the environment parameters
    env_params = get_env_params(env)
    # create the agent to interact with the environment
    if args.agent == 'DQN':
        dqn_trainer = dqn_agent(args, env, env_params)
        dqn_trainer.learn()
    elif args.agent == 'FB':
        fb_trainer = FBAgent(args, env, env_params)
        fb_trainer.learn()
    elif args.agent == 'HerDQN':
        her_agent = HerDQNAgent(args, env, env_params)
        her_agent.learn()
    elif args.agent == 'RE':
        learn = True
        reward_encoding_agent = RewardEncoderAgent(args, env, env_params)
        if learn:
            reward_encoding_agent.learn()
        else:
            dt = '2023-09-28 15:13:41'
            reward_encoding_agent.critic_network.load_state_dict(torch.load(f'data/{dt}/critic.pt', map_location="cpu"))
            reward_encoding_agent.reward_encoder.load_state_dict(torch.load(f'data/{dt}/reward_encoder.pt', map_location="cpu"))
            # with open(os.path.join(f'data/{dt}', 'uniform_buffer.pickle'), "rb") as file:
            #     reward_encoding_agent.uniform_buffer = pickle.load(file)

        reward_encoding_agent.render_episodes()
    elif args.agent == 'REA':
        learn = True
        reward_encoding_agent = RewardEncoderAblationAgent(args, env, env_params)
        if learn:
            reward_encoding_agent.learn()
        else:
            dt = '2023-09-28 15:13:41'
            reward_encoding_agent.critic_network.load_state_dict(torch.load(f'data/{dt}/critic.pt', map_location="cpu"))
            reward_encoding_agent.reward_encoder.load_state_dict(torch.load(f'data/{dt}/reward_encoder.pt', map_location="cpu"))
            # with open(os.path.join(f'data/{dt}', 'uniform_buffer.pickle'), "rb") as file:
            #     reward_encoding_agent.uniform_buffer = pickle.load(file)

        reward_encoding_agent.render_episodes()
    elif args.agent == 'Transformer':
        agent = TransformerAgent(args, env, env_params)
        agent.learn()
        agent.render_episodes()
    elif args.agent == 'Transformer2':
        agent = TransformerAgent2(args, env, env_params)
        # agent.get_data()
        # agent.critic_network.load_state_dict(torch.load(f'data/2023-11-30 12:25:09/critic.pt', map_location="cpu"))
        agent.learn()
        agent.render_episodes()
    else:
        raise NotImplementedError()


if __name__ == '__main__':
    # get the params
    args = get_args()
    launch(args)
