import dataclasses
import time

import torch
import os
from datetime import datetime
import numpy as np
import random
import pickle
import csv

from matplotlib import pyplot as plt

from continuous_world_modules.env import visualize_environment
from continuous_world_modules.geometry import Point
from grid_modules.replay_buffer import ReplayBuffer, her_replay_buffer
from grid_modules.her import her_sampler
from discrete_action_robots_modules.models import critic, TaskAwareCritic, RewardEncoder
from continuous_world_modules.featurizer import RadialBasisFunction2D
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
"""
Reward encoder agent agent
Similiar to DQN, but also encodes the reward function via a deep set.
This is learned via back propagation from value loss. 

"""
def get_reward_function(g, num_actions, featurizer):
    g = torch.tensor(g, device=featurizer.XX.device, dtype=torch.float32)

    def reward_function(obs):
        obs = featurizer.inverse_transform(obs)
        distances = torch.sum((obs - g) ** 2, dim=1) ** 0.5
        rewards = torch.where(distances < 0.07, torch.tensor([1.0], device=obs.device), torch.tensor([0.0], device=obs.device))
        rewards = rewards.unsqueeze(1).expand(rewards.shape[0], num_actions).to(torch.float32)
        return rewards
        # g_canonical = featurizer.transform(g[None, :])
        # distances = torch.sum((obs - g_canonical) ** 2, dim=1) ** 0.5
        # # rewards = torch.where(distances < 0.07, torch.tensor([0.0], device=obs.device), torch.tensor([-1.0], device=obs.device))
        # rewards = -distances
        # rewards = rewards.unsqueeze(1).expand(rewards.shape[0], num_actions).to(torch.float32)
        # return rewards


    return reward_function

def goal_distance(goal_a, goal_b):
    assert goal_a.shape == goal_b.shape
    return np.max(np.abs(goal_a - goal_b), axis=-1)


class RewardEncoderAgent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.featuriser = RadialBasisFunction2D(1, 21, 0.05, cuda=args.cuda)
        self.reward_encoding_dim = args.embed_dim
        self.num_actions = env_params['action']
        self.device = "cuda" if self.args.cuda else "cpu"

        # create the networks
        self.critic_network = TaskAwareCritic(env_params, args)
        self.reward_encoder = RewardEncoder(env_params, args)
        self.train_reward_encoder = False
        self.update_eps = 1.0 # unfirom random actions for exploration
        save_dir_date_time = '2023-09-18 10:15:44'
        if not self.train_reward_encoder:
            self.reward_encoder.load_state_dict(torch.load(f'data/{save_dir_date_time}/reward_encoder.pt'))
            self.update_eps = 0.05 # epsilon greedy actions for exploration in RL case


        # build up the target networks
        self.target_critic_network = TaskAwareCritic(env_params, args)
        self.target_reward_encoder = RewardEncoder(env_params, args)

        # load the weights into the target networks
        self.target_critic_network.load_state_dict(self.critic_network.state_dict())
        self.target_reward_encoder.load_state_dict(self.reward_encoder.state_dict())




        # if use gpu
        if self.args.cuda:
            self.critic_network.cuda()
            self.reward_encoder.cuda()
            self.target_critic_network.cuda()
            self.target_reward_encoder.cuda()
        # create the optimizer
        self.optim = torch.optim.Adam([*self.critic_network.parameters(), *self.reward_encoder.parameters()], lr=self.args.lr)
        # create the replay buffer
        self.training_buffer = ReplayBuffer(self.args.buffer_size)
        self.uniform_buffer = ReplayBuffer(self.args.buffer_size)
        if not self.train_reward_encoder:
            with open(os.path.join(f'data/{save_dir_date_time}', 'uniform_buffer.pickle'), "rb") as file:
                self.uniform_buffer = pickle.load(file)


        # create logger for losses
        current_datetime = datetime.now()
        date_time_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        dir = f"./data/" + date_time_string
        self.dir = dir
        self.logger = SummaryWriter(dir)
        self.update_iteration = 0

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
        best_average_reward = -1e9
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for cycle in range(self.args.n_cycles):
                for _ in range(self.args.num_rollouts_per_cycle):
                    # reset the environment
                    obs = self.env.reset()
                    g = self.env.goal
                    # create artificial reward function
                    with torch.no_grad():
                        reward_function = get_reward_function(g, self.num_actions, self.featuriser)
                        reward_encoding_flat, (_, __, reward_encoding) = self.compute_reward_encoding(reward_function)

                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            obs_tensor = self._preproc_o(obs)
                            action = self.act_e_greedy(obs_tensor, reward_encoding, update_eps=self.update_eps)
                        # feed the actions into the environment
                        obs_new, reward, done, info = self.env.step(action)

                        # save to buffers
                        if self.train_reward_encoder:
                            self.uniform_buffer.add(obs, g, action, reward, obs_new, done)
                        self.training_buffer.add(obs, g, action, reward, obs_new, done)

                        # do next iter
                        obs = obs_new

                # each update network backpropagates loss, but does not update parameters
                # This is because each update_network uses only one sampled reward functino
                # and therefore is biased. So we accumulate gradients to try to remove that bias
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                    if (_ + 1) % 10 == 0:
                        norm = torch.nn.utils.clip_grad_norm_(self.reward_encoder.parameters(), 1.0)
                        self.optim.step()
                        self.optim.zero_grad()
                # torch.nn.utils.clip_grad_norm_([*self.critic_network.parameters(), *self.reward_encoder.parameters()], 1.0)
                # self.optim.step()
                # self.optim.zero_grad()




                # soft update
                self._soft_update_target_network(self.target_critic_network, self.critic_network)
                self._soft_update_target_network(self.target_reward_encoder, self.reward_encoder)

            # start to do the evaluation
            average_reward, average_dist = self._eval_agent()

            print('[{}] epoch is: {}, eval: {:.3f}, dist: {:.3f}'.format(datetime.now(), epoch, average_reward, average_dist))
            self.logger.add_scalar('rl/total_reward', average_reward, self.update_iteration)
            self.logger.add_scalar('rl/final_distance', average_dist, self.update_iteration)
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
            with open(os.path.join(self.dir, 'uniform_buffer.pickle'), "wb") as file:
                pickle.dump(self.uniform_buffer, file)


    # pre_process the inputs
    def _preproc_o(self, obs):
        obs_tensor = torch.tensor(obs, dtype=torch.float32, device=self.device)
        if len(obs_tensor.shape) == 1:
            obs_tensor = obs_tensor.unsqueeze(0)
        obs_tensor = self.featuriser.transform(obs_tensor)
        return obs_tensor

    def _preproc_g(self, g):
        g_tensor = torch.tensor(g, dtype=torch.float32, device=self.device).unsqueeze(0)
        g_tensor = self.featuriser.transform(g_tensor)
        return g_tensor


    # Acts based on single state (no batch)
    def act(self, obs, reward_encoding, target_network=False, print_q=False):
        if target_network:
            q = self.target_critic_network(obs, reward_encoding)
        else:
            q = self.critic_network(obs, reward_encoding)
            if print_q:
                print(f"Q: {q}")
        return q.max(1)[1]

    # Acts with an epsilon-greedy policy
    def act_e_greedy(self, obs, reward_encoding, update_eps=0.2, print_q=False):
        return random.randrange(self.env_params['action']) if random.random() < update_eps else self.act(obs, reward_encoding, print_q=print_q).item()

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def _hard_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)

    def _reward_encoder_loss(self):
        # sample a random reward function
        with torch.no_grad():
            g = self.env.sample_goal()
            g = np.array(dataclasses.astuple(g))
            reward_function = get_reward_function(g, self.num_actions, self.featuriser)
            uniform_transitions = self.uniform_buffer.sample(self.args.batch_size)

            uniform_obs_tensor = torch.tensor(uniform_transitions['obs'], dtype=torch.float32, device=self.device)

            # convert to canonical obs, next_obs, and rewards
            uniform_obs_tensor = self.featuriser.transform(uniform_obs_tensor)
            rewards = reward_function(uniform_obs_tensor)

        # compute reward encoding from the obs
        individual_encodings = self.reward_encoder(uniform_obs_tensor)
        encoding = torch.mean(rewards.unsqueeze(1) * individual_encodings, dim=0)
        reward_encoding_normed = self.critic_network.norm_encoding(encoding)

        # compute losses
        # normalization loss to be orthonormal. Just for interesting stat
        with torch.no_grad():
            flattened_individual_encodings = individual_encodings.reshape(individual_encodings.shape[0], -1)
            expanded_matrix = flattened_individual_encodings.unsqueeze(2)  # Shape: SxEx1
            gi_times_gj = expanded_matrix * expanded_matrix.transpose(1, 2)  # Shape: SxExE
            estimated_inner_products = torch.mean(gi_times_gj, dim=0)
            orthogonal_loss = torch.mean((estimated_inner_products - torch.eye(estimated_inner_products.shape[0], device=estimated_inner_products.device))**2)
            self.logger.add_scalar('Loss/orthogonal', orthogonal_loss.item(), self.update_iteration)

        # normalization loss so it does not change too much. Just to see how it drifts
        with torch.no_grad():
            target_individual_encodings = self.target_reward_encoder(uniform_obs_tensor)
            target_encoding = torch.mean(rewards.unsqueeze(1) * target_individual_encodings, dim=0)
            target_reward_encoding_normed = self.critic_network.norm_encoding(target_encoding)
            normalization_loss = torch.sum((reward_encoding_normed - target_reward_encoding_normed)**2).pow(0.5)
            normalization_loss2 = torch.sum((encoding - target_encoding)**2).pow(0.5)
            self.logger.add_scalar('Loss/drift_normed', normalization_loss.item(), self.update_iteration)
            self.logger.add_scalar('Loss/drift_not_normed', normalization_loss2.item(), self.update_iteration)


        # normalization loss to satisfy definition of a basis
        estimated_reward = torch.sum(encoding * individual_encodings, dim=1)
        reward_prediction_loss = torch.mean((estimated_reward - rewards)**2)

        # error of individual rewards for fun
        with torch.no_grad():
            ones = estimated_reward[torch.where(rewards == 1)]
            zeros = estimated_reward[torch.where(rewards == 0)]
            self.logger.add_scalar('reward_loss/1_mean', torch.mean(ones).item() if ones.shape[0] != 0 else torch.nan,self.update_iteration)
            self.logger.add_scalar('reward_loss/1_max', torch.max(ones).item() if ones.shape[0] != 0 else torch.nan, self.update_iteration)
            self.logger.add_scalar('reward_loss/1_min', torch.min(ones).item() if ones.shape[0] != 0 else torch.nan,self.update_iteration)
            self.logger.add_scalar('reward_loss/0_mean', torch.mean(zeros).item() if zeros.shape[0] != 0 else torch.nan, self.update_iteration)
            self.logger.add_scalar('reward_loss/0_max', torch.max(zeros).item() if zeros.shape[0] != 0 else torch.nan, self.update_iteration)
            self.logger.add_scalar('reward_loss/0_min', torch.min(zeros).item() if zeros.shape[0] != 0 else torch.nan, self.update_iteration)
            self.logger.add_scalar('reward_loss/count', (torch.sum(rewards[torch.where(rewards == 1)])/(rewards.shape[0] * rewards.shape[1])).item(), self.update_iteration)
            self.logger.add_scalar('reward_loss/encoding_magnitude', torch.linalg.norm(encoding.flatten()), self.update_iteration)
            # compute gap between encodings
            goals = [Point(0.75, 0.25), Point(0.05, 0.25)]
            g1, g2 = np.array(dataclasses.astuple(goals[0])), np.array(dataclasses.astuple(goals[1]))
            r1, r2 = get_reward_function(g1, self.num_actions, self.featuriser), get_reward_function(g2, self.num_actions, self.featuriser)
            rs1, rs2 = r1(uniform_obs_tensor), r2(uniform_obs_tensor)
            individual_encodings = self.reward_encoder(uniform_obs_tensor)
            encoding1 = torch.mean(rs1.unsqueeze(1) * individual_encodings, dim=0)
            encoding2 = torch.mean(rs2.unsqueeze(1) * individual_encodings, dim=0)
            distance = torch.sum((encoding1 - encoding2)**2)**0.5
            self.logger.add_scalar('reward_loss/encoding_separation', distance, self.update_iteration)

        # ret
        return encoding, reward_encoding_normed, reward_function, reward_prediction_loss


    def _value_loss(self, reward_encoding, reward_function):
        with torch.no_grad():
            online_transitions = self.training_buffer.sample(self.args.batch_size)

            obs_tensor = torch.tensor(online_transitions['obs'], dtype=torch.float32, device=self.device)
            obs_next_tensor = torch.tensor(online_transitions['obs_next'], dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(online_transitions['action'], dtype=torch.long, device=self.device)


            # convert to canonical obs, next_obs, and rewards
            obs_tensor = self.featuriser.transform(obs_tensor)
            obs_next_tensor = self.featuriser.transform(obs_next_tensor)
            rewards = reward_function(obs_tensor)
            rewards = rewards.gather(1, actions_tensor.reshape(-1, 1))

            # calculate the target Q value function
            q_next_values = self.target_critic_network(obs_next_tensor, reward_encoding)
            q_next_value = q_next_values.max(1)[0].reshape(-1, 1)
            q_next_value = q_next_value.detach()
            target_q_value = rewards + self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            # clip_return = 1 / (1 - self.args.gamma) # TODO why do this?
            # target_q_value = torch.clamp(target_q_value, 0, clip_return)
            # the q loss
        real_q_value = self.critic_network(obs_tensor, reward_encoding.detach())
        real_q_value = real_q_value.gather(1, actions_tensor.reshape(-1, 1))
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        return critic_loss


    # update the network
    def _update_network(self):
        reward_encoding, reward_encoding_normed, reward_function, reward_prediction_loss = self._reward_encoder_loss()
        critic_loss = self._value_loss(reward_encoding, reward_function)

        if not self.train_reward_encoder:
            loss = critic_loss
        else:
            loss = critic_loss + reward_prediction_loss

        loss.backward()
        with torch.no_grad():
            if self.train_reward_encoder:
                norm = torch.linalg.norm(torch.tensor([torch.linalg.norm(p.grad) for p in self.reward_encoder.parameters()]))
            else:
                norm = torch.tensor(0.0)

            self.logger.add_scalar('Loss/critic', critic_loss.item(), self.update_iteration)
            self.logger.add_scalar('Loss/reward', reward_prediction_loss.item(), self.update_iteration)
            self.logger.add_scalar('reward_loss/gradient_magnitude', norm.item(), self.update_iteration)
            self.update_iteration += 1


    # do the evaluation
    def _eval_agent(self):
        total_rewards = []
        total_dist = []
        for _ in range(self.args.n_test_rollouts):
            obs = self.env.reset()
            g = self.env.goal
            reward_function = get_reward_function(g, self.num_actions, self.featuriser)
            reward_encoding_flat, (_, __, reward_encoding) = self.compute_reward_encoding(reward_function)

            # self.env.set_initial_position(Point(0.2, 0.1))
            # self.env.set_goal(Point(0.9, 0.9))
            # obs = self.env.current_position
            # g = self.env.goal
            total_reward = 0
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    obs_norm_tensor = self._preproc_o(obs)
                    action = self.act_e_greedy(obs_norm_tensor, reward_encoding, update_eps=0.02)
                obs, r, d, info = self.env.step(action)
                reward = reward_function(obs_norm_tensor)[0][action].item()
                total_reward += reward
                if d:
                    break

            total_rewards.append(total_reward)
            dist = goal_distance(obs, g)
            total_dist.append(dist)

        total_rewards = np.array(total_rewards)
        total_rewards = np.mean(total_rewards)

        total_dist = np.array(total_dist)
        total_dist = np.mean(total_dist)

        return total_rewards, total_dist

    def compute_reward_encoding(self, reward_function, batch_size=10_000, use_target=False):
        with torch.no_grad():
            batch = min(len(self.uniform_buffer), batch_size)
            if batch == 0:
                random_encoding = torch.rand(self.reward_encoding_dim//self.num_actions, self.num_actions, device=self.device)
                return random_encoding.flatten(), (None, None, random_encoding)
            obs = self.uniform_buffer.sample(batch_size=batch)['obs']
            obs = self._preproc_o(obs)
            rewards = reward_function(obs)
        encoder = self.target_reward_encoder if use_target else self.reward_encoder
        individual_encodings = encoder(obs)
        encoding = torch.mean(rewards.unsqueeze(1) * individual_encodings, dim=0)
        encoding_flat = encoding.flatten()
        # if (encoding_flat != 0).any():
        #     print("Here")
        return encoding_flat, (individual_encodings, rewards, encoding)

    def render_episodes(self):
        # compute gap between encodings
        goals = [Point(0.75, 0.25), Point(0.05, 0.25)]
        g1, g2 = np.array(dataclasses.astuple(goals[0])), np.array(dataclasses.astuple(goals[1]))
        r1, r2 = get_reward_function(g1, self.num_actions, self.featuriser), get_reward_function(g2, self.num_actions, self.featuriser)

        re1, re2 = self.compute_reward_encoding(r1)[0], self.compute_reward_encoding(r2)[0]

        print("Individual dims")
        for dim1, dim2 in zip(re1, re2):
            print(f"{dim1.item():0.3f}, {dim2.item():0.3f}")
        print(f"\n\ndot product\n{torch.dot(re1, re2).item():0.3f}\n\ncos sim.")

        # re1, re2 = self.norm_encoding(re1), self.norm_encoding(re2)
        for i in range(10):
            new_re1, _ = self.compute_reward_encoding(r1)
            new_re2, _ = self.compute_reward_encoding(r2)
            # new_re1, new_re2 = self.norm_encoding(new_re1), self.norm_encoding(new_re2)

            # print(re1.shape, new_re1.shape)
            # print(re2.shape, new_re2.shape)

            r1_sim = torch.nn.functional.cosine_similarity(re1, new_re1, dim=0)
            r2_sim = torch.nn.functional.cosine_similarity(re2, new_re2, dim=0)
            r1_r2_sim = torch.nn.functional.cosine_similarity(new_re1, new_re2, dim=0)
            print(f"{r1_sim.item():0.5f}, {r2_sim.item():0.5f}, {r1_r2_sim.item():0.5f}")
            # print(f"{torch.linalg.norm(new_re1 - re1).item():0.2f}, {torch.linalg.norm(new_re2 - re2).item():0.2f}, {torch.linalg.norm(new_re1 - new_re2).item():0.2f}")




        re1, re2 = self.compute_reward_encoding(r1)[1][2], self.compute_reward_encoding(r2)[1][2]

        fig, ax = plt.subplots()
        for _ in range(100):
            # fig.clear
            obs = self.env.reset()
            g = self.env.goal
            reward_function = get_reward_function(g, self.num_actions, self.featuriser)
            reward_encoding_flat, (_, __, reward_encoding) = self.compute_reward_encoding(reward_function)
            print(reward_encoding)

            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    obs_norm_tensor = self._preproc_o(obs)
                    action = self.act_e_greedy(obs_norm_tensor, reward_encoding, update_eps=0.02, print_q=False)
                    self.act_e_greedy(obs_norm_tensor, re1, update_eps=0.02, print_q=True)
                    self.act_e_greedy(obs_norm_tensor, re2, update_eps=0.02, print_q=True)
                n_obs, r, d, info = self.env.step(action)
                print(reward_function(obs_norm_tensor)[0][action].item())
                self.env.render(fig, ax)
                plt.draw()
                plt.pause(0.1)
                ax.clear()


                obs = n_obs



