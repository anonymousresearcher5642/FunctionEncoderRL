import dataclasses
import time

import cv2
import torch
import os
import numpy as np
import random
import pickle
import csv
from tqdm import trange
from matplotlib import pyplot as plt

from continuous_world_modules.geometry import Point
from atari_modules.replay_buffer import ReplayBuffer
from atari_modules.models import TaskAwareCritic, RewardEncoder # , RewardEncoderTranslator
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
"""
Reward encoder agent agent
Similiar to DQN, but also encodes the reward function via a deep set.
This is learned via back propagation from value loss. 

"""
def get_reward_function(gs, num_actions):
    assert (gs >= 1).any()
    gs = torch.tensor(gs, device="cuda:0") / 170.
    if len(gs.shape) == 1: # make a single goal into multiple
        gs = gs.unsqueeze(0)
    number_goals = gs.shape[0]
    gs = gs.unsqueeze(1) # now gs = [batch, 1, 2]
    assert gs.shape[0] == number_goals
    assert gs.shape[1] == 1
    assert gs.shape[2] == 2


    # dense
    def reward_function(achieved_goals):
        assert (achieved_goals <= 1).all()
        achieved_goals = achieved_goals.unsqueeze(0) # now achieved_goals = [1, batch2, 2]
        assert achieved_goals.shape[0] == 1
        assert achieved_goals.shape[2] == 2
        distances = torch.sum(torch.abs(gs - achieved_goals), dim=-1)
        rewards = -distances
        rewards = rewards.unsqueeze(2).expand(rewards.shape[0], rewards.shape[1], num_actions).to(torch.float32)
        assert rewards.shape[0] == number_goals
        assert rewards.shape[1] == achieved_goals.shape[1]
        assert rewards.shape[2] == num_actions
        return rewards # should be (number goals x number achieved goals x  number actions)

    # sparse
    # distance_threshold = 6 / 170.
    # def reward_function(achieved_goals):
    #     assert (achieved_goals <= 1).all()
    #     achieved_goals = achieved_goals.unsqueeze(0)  # now achieved_goals = [1, batch2, 2]
    #     distances = torch.max(torch.abs(gs - achieved_goals), dim=-1)[0]
    #     rewards = torch.where(distances < distance_threshold, torch.tensor(1.0), torch.tensor(0.0))
    #     rewards = rewards.unsqueeze(2).expand(rewards.shape[0], rewards.shape[1], num_actions).to(torch.float32)
    #     return rewards

    return reward_function

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.max(np.abs(goal_a - goal_b), axis=-1)

class RewardEncoderAblationAgent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.reward_encoding_dim = args.embed_dim
        self.num_actions = env_params['action']
        self.device = "cuda" if self.args.cuda else "cpu"

        # create the networks
        self.critic_network = TaskAwareCritic(env_params, args.embed_dim)
        self.reward_encoder = RewardEncoder(env_params, args.embed_dim)
        # self.reward_encoder_translator = RewardEncoderTranslator(env_params, args.embed_dim)
        # self.train_reward_encoder = False
        # self.update_eps = 1.0 # unfirom random actions for exploration
        # save_dir_date_time = '2023-09-19 10:15:17'
        # if not self.train_reward_encoder:
        #     self.reward_encoder.load_state_dict(torch.load(f'data/{save_dir_date_time}/reward_encoder.pt'))
        self.update_eps = 0.05 # epsilon greedy actions for exploration in RL case
        self.policy_type = "boltzmann" # epsilon or boltzmann
        # self.use_translator = False

        # build up the target networks
        self.target_critic_network = TaskAwareCritic(env_params, args.embed_dim)
        # self.target_reward_encoder = RewardEncoder(env_params, args.embed_dim)

        # load the weights into the target networks
        self.target_critic_network.load_state_dict(self.critic_network.state_dict())
        # self.target_reward_encoder.load_state_dict(self.reward_encoder.state_dict())

        # if use gpu
        if self.args.cuda:
            self.critic_network.cuda()
            self.reward_encoder.cuda()
            self.target_critic_network.cuda()
            # self.reward_encoder_translator.cuda()
            # self.target_reward_encoder.cuda()
        # create the optimizer
        self.optim = torch.optim.Adam([*self.critic_network.parameters(),
                                               *self.reward_encoder.parameters(),
                                               # *self.reward_encoder_translator.parameters()
                                       ], lr=self.args.lr)
        # create the replay buffer
        self.training_buffer = ReplayBuffer(self.args.buffer_size)
        # self.uniform_buffer = ReplayBuffer(self.args.buffer_size)
        # if not self.train_reward_encoder:
        #     with open(os.path.join(f'data/{save_dir_date_time}', 'uniform_buffer.pickle'), "rb") as file:
        #         self.uniform_buffer = pickle.load(file)


        # create logger for losses
        current_datetime = datetime.now()
        date_time_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        dir = f"./data/" + date_time_string
        self.dir = dir
        self.logger = SummaryWriter(dir)
        self.update_iteration = 0

        # create a file called transformer.txt in log
        with open(os.path.join(dir, "reward_encoder_ablation.txt"), "wt") as file:
            file.write("")

        if args.save_dir is not None:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)

            print(' ' * 26 + 'Options')
            for k, v in vars(self.args).items():
                print(' ' * 26 + k + ': ' + str(v))

            with open(self.args.save_dir + "/arguments.pkl", 'wb') as f:
                pickle.dump(self.args, f)

            with open('{}/score_monitor.csv'.format(self.args.save_dir), "wt") as monitor_file:
                monitor = csv.writer(monitor_file)
                monitor.writerow(['epoch', 'avg. reward', 'avg. dist'])
    def learn(self):
        """
        train the network

        """
        self.pretrain_reward_encoder()


        best_average_reward = -1e9
        # start to collect samples
        for epoch in trange(self.args.n_epochs):
            for cycle in range(self.args.n_cycles):
                for _ in range(self.args.num_rollouts_per_cycle):
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # create artificial reward function
                    with torch.no_grad():
                        reward_function = get_reward_function(g, self.num_actions)
                        reward_encoding_flat, reward_encoding = self.compute_reward_encoding(reward_function)

                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            obs_tensor = self._preproc_o(obs)
                            action = self.act(obs_tensor, reward_encoding)
                        # feed the actions into the environment
                        observation_new, reward, done, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']

                        # add transition
                        # if self.train_reward_encoder:
                        #     self.uniform_buffer.add(obs, ag, g, action, reward, obs_new, done)
                        self.training_buffer.add(obs, ag, g, action, reward, obs_new, done)
                        if done:
                            observation = self.env.reset()
                            obs = observation['observation']
                            ag = observation['achieved_goal']
                            g = observation['desired_goal']
                        else:
                            obs = obs_new
                            ag = ag_new


                # each update network backpropagates loss, but does not update parameters
                # This is because each update_network uses only one sampled reward functino
                # and therefore is biased. So we accumulate gradients to try to remove that bias
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()

                # torch.nn.utils.clip_grad_norm_([*self.critic_network.parameters(), *self.reward_encoder.parameters()], 1.0)
                # self.optim.step()
                # self.optim.zero_grad()




                # soft update
                self._soft_update_target_network(self.target_critic_network, self.critic_network)
                # self._soft_update_target_network(self.target_reward_encoder, self.reward_encoder)

            # start to do the evaluation
            average_reward, average_dist, success_rate = self._eval_agent()

            # print('[{}] epoch is: {}, eval: {:.3f}, dist: {:.3f}'.format(datetime.now(), epoch, average_reward, average_dist))
            self.logger.add_scalar('rl/total_reward', average_reward, self.update_iteration)
            self.logger.add_scalar('rl/final_distance', average_dist, self.update_iteration)
            self.logger.add_scalar('rl/success_rate', success_rate, self.update_iteration)
            with open('{}/score_monitor.csv'.format(self.args.save_dir), "a") as monitor_file:
                monitor = csv.writer(monitor_file)
                monitor.writerow([epoch, average_reward, average_dist])
            torch.save(self.critic_network.state_dict(), os.path.join(self.dir, 'critic.pt'))
            torch.save(self.reward_encoder.state_dict(), os.path.join(self.dir, 'reward_encoder.pt'))

            if average_reward > best_average_reward:
                best_average_reward = average_reward
                torch.save(self.critic_network.state_dict(), os.path.join(self.dir, 'best_critic.pt'))
                torch.save(self.reward_encoder.state_dict(), os.path.join(self.dir, 'best_reward_encoder.pt'))

            # maybe save uniform buffer
            # with open(os.path.join(self.dir, 'uniform_buffer.pickle'), "wb") as file:
            #     pickle.dump(self.uniform_buffer, file)


    # pre_process the inputs
    def _preproc_o(self, obs):
        if type(obs) is not torch.Tensor:
            obs = np.array(obs)
            obs = torch.tensor(obs, dtype=torch.float32, device="cuda:0" if self.args.cuda else "cpu")

        if len(obs.shape) == 3:
            obs = obs[None] # add batch dim of 1 if needed
        obs = obs/255.
        # obs = torch.transpose(obs, [0, 3, 1, 2])
        obs = obs.permute(0, 3, 1, 2)
        return obs

    def _preproc_g(self, g):
        if len(g.shape) == 1:
            g = g[None] # add batch dim of 1 if needed
        g_tensor = torch.tensor(g / 170, dtype=torch.float32)
        if self.args.cuda:
            g_tensor = g_tensor.cuda()
        return g_tensor


    # Acts based on single state (no batch)
    def act(self, obs, reward_encoding, target_network=False, print_q=False):
        # if epsilon greedy, then do random action with prob epsilon
        if self.policy_type == "epsilon" and random.random() < self.update_eps:
            return random.randrange(self.env_params['action'])
        if target_network:
            q = self.target_critic_network(obs, reward_encoding)
        else:
            q = self.critic_network(obs, reward_encoding)
            if print_q:
                print(f"Q: {q}")
        if self.policy_type == "epsilon":
            return q.max(1)[1].item()
        elif self.policy_type == "boltzmann":
            q = q / self.args.temp
            return torch.distributions.Categorical(logits=q).sample().item()
        else:
            raise NotImplementedError()

    # Acts with an epsilon-greedy policy
    # def act_e_greedy(self, obs, reward_encoding, update_eps=0.2, print_q=False):
    #     return random.randrange(self.env_params['action']) if random.random() < update_eps else self.act(obs, reward_encoding, print_q=print_q).item()

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def _hard_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    # def _reward_encoder_loss(self):
    #     # sample a random reward function
    #     with torch.no_grad():
    #         g = self.env._sample_goal()
    #         reward_function = get_reward_function(g, self.num_actions)
    #         uniform_transitions = self.uniform_buffer.sample(self.args.batch_size)
    #         # uniform_achieved_goal_tensor = torch.tensor(uniform_transitions['ag'], dtype=torch.float32, device=self.device) / 170
    #         uniform_achieved_goal_tensor = self._preproc_g(uniform_transitions['ag'])
    #         rewards = reward_function(uniform_achieved_goal_tensor)
    #
    #     # compute reward encoding from the obs
    #     individual_encodings = self.reward_encoder(uniform_achieved_goal_tensor)
    #     encoding = torch.mean(rewards.unsqueeze(1) * individual_encodings, dim=0)
    #     reward_encoding_normed = self.critic_network.norm_encoding(encoding)
    #
    #     # compute losses
    #     # normalization loss to be orthonormal. Just for interesting stat
    #     with torch.no_grad():
    #         flattened_individual_encodings = individual_encodings.reshape(individual_encodings.shape[0], -1)
    #         expanded_matrix = flattened_individual_encodings.unsqueeze(2)  # Shape: SxEx1
    #         gi_times_gj = expanded_matrix * expanded_matrix.transpose(1, 2)  # Shape: SxExE
    #         estimated_inner_products = torch.mean(gi_times_gj, dim=0)
    #         orthogonal_loss = torch.mean((estimated_inner_products - torch.eye(estimated_inner_products.shape[0], device=estimated_inner_products.device))**2)
    #         self.logger.add_scalar('Loss/orthogonal', orthogonal_loss.item(), self.update_iteration)
    #
    #     # normalization loss so it does not change too much. Just to see how it drifts
    #     with torch.no_grad():
    #         target_individual_encodings = self.target_reward_encoder(uniform_achieved_goal_tensor)
    #         target_encoding = torch.mean(rewards.unsqueeze(1) * target_individual_encodings, dim=0)
    #         target_reward_encoding_normed = self.critic_network.norm_encoding(target_encoding)
    #         normalization_loss = torch.sum((reward_encoding_normed - target_reward_encoding_normed)**2).pow(0.5)
    #         normalization_loss2 = torch.sum((encoding - target_encoding)**2).pow(0.5)
    #         self.logger.add_scalar('Loss/drift_normed', normalization_loss.item(), self.update_iteration)
    #         self.logger.add_scalar('Loss/drift_not_normed', normalization_loss2.item(), self.update_iteration)
    #
    #
    #     # normalization loss to satisfy definition of a basis
    #     estimated_reward = torch.sum(encoding * individual_encodings, dim=1)
    #     reward_prediction_loss = torch.mean((estimated_reward - rewards)**2)
    #
    #     # error of individual rewards for fun
    #     with torch.no_grad():
    #         self.logger.add_scalar('reward_loss/encoding_magnitude', torch.linalg.norm(encoding.flatten()), self.update_iteration)
    #
    #     # ret
    #     return encoding, reward_encoding_normed, reward_function, reward_prediction_loss


    def _value_loss(self, goals, reward_encoding, reward_function):
        with torch.no_grad():
            online_transitions = self.training_buffer.sample(self.args.batch_size)

            obs_tensor = torch.tensor(online_transitions['obs'], dtype=torch.float32, device=self.device)
            obs_next_tensor = torch.tensor(online_transitions['obs_next'], dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(online_transitions['action'], dtype=torch.long, device=self.device)
            achieved_goals = torch.tensor(online_transitions['ag'], dtype=torch.long, device=self.device)
            dones_tensor = torch.tensor(online_transitions['done'], dtype=torch.float32, device=self.device)
            achieved_goals = achieved_goals/170. # normalize it to be betwreen 0 and 1
            
            rewards = reward_function(achieved_goals)
            # if done, then we hit a ghost. Want to avoid that, so penalize
            rewards = rewards - 1.0 * dones_tensor.reshape(-1, 1) # TODO dense reward

            # want to check if we have arrived at goal. If so, we are done (for achieving goal)
            # only add reward if dense
            goals = self._preproc_g(goals)
            distance = torch.max(torch.abs(achieved_goals.unsqueeze(0) - goals.unsqueeze(1)), dim=-1)[0]
            at_goal =  distance <= (self.env.distance_threshold/170.)
            rewards = rewards + 1.0 * at_goal.unsqueeze(2).to(torch.float32) # TODO dense reward
            dones_tensor = torch.logical_or(at_goal, dones_tensor).to(torch.float32)

            # preprocess
            obs_tensor = self._preproc_o(obs_tensor)
            obs_next_tensor = self._preproc_o(obs_next_tensor)
            dones_tensor = dones_tensor.unsqueeze(-1)

            # calculate the target Q value function
            q_next_values = self.target_critic_network(obs_next_tensor, reward_encoding)
            action_probs = torch.nn.functional.softmax(q_next_values/self.args.temp, dim=-1)
            q_next_value = torch.sum(action_probs * q_next_values, dim=-1).unsqueeze(-1)
            # q_next_value = q_next_values.max(1)[0].reshape(-1, 1) # nromal best action
            rewards = rewards[:, torch.arange(rewards.shape[1]), actions_tensor].unsqueeze(-1)
            q_next_value = q_next_value.detach()
            target_q_value = rewards + (1-dones_tensor) * self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            # clip_return = 1 / (1 - self.args.gamma) # TODO why do this?
            # target_q_value = torch.clamp(target_q_value, 0, clip_return)
            # the q loss
        real_q_value = self.critic_network(obs_tensor, reward_encoding.detach())
        real_q_value = real_q_value[:, torch.arange(rewards.shape[1]), actions_tensor].unsqueeze(-1)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        return critic_loss


    # update the network
    def _update_network(self):
        goals = self.env.all_goals
        num_goals_each_time = 40  # 148 max
        goals2 = goals[torch.randperm(goals.shape[0])[:num_goals_each_time]]
        reward_function = get_reward_function(goals2, self.num_actions)
        reward_encoding_flat, reward_encoding = self.compute_reward_encoding(reward_function)
        critic_loss = self._value_loss(goals2, reward_encoding, reward_function)

        # if not self.train_reward_encoder:
        #     loss = critic_loss
        # else:
        #     loss = critic_loss + reward_prediction_loss
        loss = critic_loss

        loss.backward()
        # norm = torch.nn.utils.clip_grad_norm_(self.reward_encoder.parameters(), 1.0)
        self.optim.step()
        self.optim.zero_grad()
        with torch.no_grad():
            # if self.train_reward_encoder:
            #     norm = torch.linalg.norm(torch.tensor([torch.linalg.norm(p.grad) for p in self.reward_encoder.parameters()]))
            # else:
            #     norm = torch.tensor(0.0)

            self.logger.add_scalar('Loss/critic', critic_loss.item(), self.update_iteration)
            # self.logger.add_scalar('Loss/reward', reward_prediction_loss.item(), self.update_iteration)
            # self.logger.add_scalar('reward_loss/gradient_magnitude', norm.item(), self.update_iteration)
            self.update_iteration += 1


    # do the evaluation
    def _eval_agent(self):
        total_rewards = []
        total_dist = []
        total_successes = []
        for _ in range(self.args.n_test_rollouts):
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            ag = observation['achieved_goal']
            reward_function = get_reward_function(g, self.num_actions)
            reward_encoding_flat, reward_encoding = self.compute_reward_encoding(reward_function)

            total_reward = 0
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    obs_norm_tensor = self._preproc_o(obs)
                    action = self.act(obs_norm_tensor, reward_encoding)
                observation, _, d, info = self.env.step(action)
                obs = observation['observation']
                g = observation['desired_goal']
                ag = observation['achieved_goal']
                reward = reward_function(self._preproc_g(ag))[0,0,action].item()
                if d: # caught by ghost # TODO dense rewards
                    reward -= 1
                if info['is_success'] > 0: # at goal
                    reward += 1

                total_reward += reward
                if info['is_success'] > 0 or d:
                    break
            dist = goal_distance(ag, g)
            total_rewards.append(total_reward)
            total_dist.append(dist)
            total_successes.append(info['is_success'])

        total_rewards = np.array(total_rewards)
        total_rewards = np.mean(total_rewards)

        total_dist = np.array(total_dist)
        total_dist = np.mean(total_dist)

        total_successes = np.array(total_successes)
        total_successes = np.mean(total_successes)

        return total_rewards, total_dist, total_successes

    def compute_reward_encoding(self, reward_function, batch_size=10_000, use_target=False):
        rewards = reward_function(self.uniform_inputs)
        individual_encodings = self.reward_encoder(self.uniform_inputs)
        encoding = torch.mean(rewards.unsqueeze(2) * individual_encodings, dim=1)
        # if self.use_translator:
        #     translated_encoding = self.reward_encoder_translator(encoding)
        #     translated_encoding_flat = translated_encoding.reshape(translated_encoding.shape[0], -1)
        #     return translated_encoding_flat, translated_encoding
        # else:
        encoding_flat = encoding.reshape(encoding.shape[0], -1)
        return encoding_flat, encoding

        # with torch.no_grad():
        #     batch = min(len(self.uniform_buffer), batch_size)
        #     if batch == 0:
        #         random_encoding = torch.rand(self.reward_encoding_dim//self.num_actions, self.num_actions, device=self.device)
        #         return random_encoding.flatten(), (None, None, random_encoding)
        #     achieved_goals = self.uniform_buffer.sample(batch_size=batch)['ag']
        #     achieved_goals = self._preproc_g(achieved_goals)
        #     rewards = reward_function(achieved_goals)
        # encoder = self.target_reward_encoder if use_target else self.reward_encoder
        # individual_encodings = encoder(achieved_goals)
        # encoding = torch.mean(rewards.unsqueeze(1) * individual_encodings, dim=0)
        # encoding_flat = encoding.flatten()
        # # if (encoding_flat != 0).any():
        # #     print("Here")
        # return encoding_flat, (individual_encodings, rewards, encoding)

    def render_episodes(self):
        self.uniform_inputs = self.env.get_uniform_inputs()
        self.uniform_inputs = self._preproc_g(self.uniform_inputs)
        # Define the codec and create a VideoWriter object
        width, height = 160*2, 210*2
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # You can change the codec as needed
        out = cv2.VideoWriter(os.path.join(self.dir, "atari.mp4"), fourcc, 30.0, (width, height))

        for _ in range(100):
            # fig.clear
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            reward_function = get_reward_function(g, self.num_actions)
            reward_encoding_flat, reward_encoding = self.compute_reward_encoding(reward_function)

            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    obs_norm_tensor = self._preproc_o(obs)
                    action = self.act(obs_norm_tensor, reward_encoding, print_q=False)
                observation, r, d, info = self.env.step(action)
                obs = observation['observation']
                img = self.env.render(mode="rgb_array")
                # TODO add goal
                x_min, x_max = 0,160
                y_min, y_max = 0,170
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # add a star at the goal location
                center_x, center_y = g[0], g[1]
                size = 7
                # Define the coordinates of the points to draw the star
                points = [
                    (center_x, center_y - size),
                    (int(center_x + size * np.sin(np.deg2rad(72))), int(center_y - size * np.cos(np.deg2rad(72)))),
                    (int(center_x + size * np.sin(np.deg2rad(144))), int(center_y - size * np.cos(np.deg2rad(144)))),
                    (int(center_x + size * np.sin(np.deg2rad(216))), int(center_y - size * np.cos(np.deg2rad(216)))),
                    (int(center_x + size * np.sin(np.deg2rad(288))), int(center_y - size * np.cos(np.deg2rad(288)))),
                ]

                # Draw the star by connecting the points with lines
                for i in range(5):
                    cv2.line(img, points[i], points[(i + 2) % 5], (255, 255, 0), 2)

                print("AG: ", observation['achieved_goal'], "\tDG:", observation['desired_goal'], "Time remaining: ", self.env_params['max_timesteps'] - _)
                img = cv2.resize(img, (160*2, 210*2))
                out.write(img)
                # cv2.imshow("pac", img)
                # k = cv2.waitKey(200)


                if d or info['is_success'] > 0:
                    break
        out.release()



    def pretrain_reward_encoder(self):
        self.uniform_inputs = self.env.get_uniform_inputs()
        self.uniform_inputs = self._preproc_g(self.uniform_inputs)
        goals = self.env.all_goals
        num_goals_each_time = 40 # 148 max

        # create a new network that takes in the encoding and the state as input and predicts reward
        output_net = torch.nn.Sequential(
            torch.nn.Linear(self.reward_encoding_dim * self.num_actions + 2, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, 256),
            torch.nn.ReLU(),
            torch.nn.Linear(256, self.num_actions),
        ).to(self.device)


        for descent_step in trange(15_000):
            # randomly select half of goals
            goals2 = goals[torch.randperm(goals.shape[0])[:num_goals_each_time]]
            reward_functions = get_reward_function(goals2, self.num_actions)
            rewards = reward_functions(self.uniform_inputs)
            individual_encodings = self.reward_encoder(self.uniform_inputs)
            encoding = torch.mean(rewards.unsqueeze(2) * individual_encodings, dim=1)
            encoding = encoding.reshape(encoding.shape[0], -1)
            encoding = encoding.unsqueeze(1).expand(-1, rewards.shape[1], -1)

            states = self.uniform_inputs.unsqueeze(0).expand(40, -1, -1)
            # compute estimation loss
            representation = torch.cat((encoding, states), dim=2)
            estimated_reward = output_net(representation)

            assert estimated_reward.shape == rewards.shape
            estimation_loss = torch.mean((estimated_reward - rewards)**2)

            # now do translator loss. its task is to spread out the coordinates so that they are as far apart as possible without losing information
            # uses cos similiary
            # translations = self.reward_encoder_translator(encoding.detach())
            # translations = translations.reshape(translations.shape[0], -1)
            # cos_sims = torch.nn.functional.cosine_similarity(translations.unsqueeze(1), translations.unsqueeze(0), dim=2)
            # cos_sim = torch.mean(cos_sims)
            # (estimation_loss + cos_sim).backward()
            (estimation_loss).backward()
            self.optim.step()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.reward_encoder.parameters(), 0.1)
            self.optim.zero_grad()
            with torch.no_grad():
                self.logger.add_scalar('loss/reward', estimation_loss.item(), descent_step)
                # self.logger.add_scalar('loss/regularization', cos_sim.item(), descent_step)
                self.logger.add_scalar('loss/grad_norm', grad_norm.item(), descent_step)

        torch.save(self.reward_encoder.state_dict(), os.path.join(self.dir, 'reward_encoder.pt'))



