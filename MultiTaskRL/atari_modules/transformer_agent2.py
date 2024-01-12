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
from atari_modules.models import TaskAwareCritic, RewardEncoder, TransformerCritic, \
    TransformerOracle  # , RewardEncoderTranslator
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
"""
Reward encoder agent agent
Similiar to DQN, but also encodes the reward function via a deep set.
This is learned via back propagation from value loss. 

"""
def get_reward_function(gs, num_actions):
    assert (gs >= 1).any()
    gs = torch.tensor(gs, device="cuda") / 170.
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

class TransformerAgent2:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        self.num_actions = env_params['action']
        self.device = "cuda" if self.args.cuda else "cpu"
        self.max_examples = 100 # this is due to memory concerns

        # create the networks
        state_space = env.observation_space['achieved_goal'].shape
        observation_space = env.observation_space['observation'].shape
        action_space = (env.action_space.n,)
        self.critic_network = TransformerCritic(observation_space=observation_space,
                                                state_space=state_space,
                                                action_space=action_space,
                                                device=self.device)
        self.update_eps = 0.05 # epsilon greedy actions for exploration in RL case
        self.policy_type = "boltzmann" # epsilon or boltzmann

        # build up the target networks
        self.target_critic_network = TransformerCritic(observation_space=observation_space,
                                                        state_space=state_space,
                                                        action_space=action_space,
                                                        device=self.device)

        # load the weights into the target networks
        self.target_critic_network.load_state_dict(self.critic_network.state_dict())

        # if use gpu
        if self.args.cuda:
            self.critic_network.cuda()
            self.target_critic_network.cuda()


        # create the optimizer
        self.optim = torch.optim.Adam([*self.critic_network.parameters(),], lr=self.args.lr)

        # create the replay buffer
        self.training_buffer = ReplayBuffer(self.args.buffer_size)

        # create logger for losses
        current_datetime = datetime.now()
        date_time_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        dir = f"./data/" + date_time_string
        self.dir = dir
        self.logger = SummaryWriter(dir)
        self.update_iteration = 0

        # create a file called transformer.txt in log
        with open(os.path.join(dir, "transformer2.txt"), "wt") as file:
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
        self.get_data()


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

                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            obs_tensor = self._preproc_o(obs)
                            action = self.act(obs_tensor, g)
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

            if average_reward > best_average_reward:
                best_average_reward = average_reward
                torch.save(self.critic_network.state_dict(), os.path.join(self.dir, 'best_critic.pt'))

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
    def act(self, obs, goal, target_network=False, print_q=False):
        # fetch goal index from list of all goals
        if len(goal.shape) == 1:
            goal = goal[None]

        # For oracle model
        # goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
        # goal = self._preproc_g(goal)

        # for data mode
        # get indices
        goal = torch.tensor(goal, dtype=torch.float32, device=self.device)
        goal_indicies = torch.argmin(torch.sum(torch.abs(self.goals - goal), dim=-1), dim=-1)

        # get permutation of data for each goal index # TODO  make permutation different for every goal
        permutation = torch.randperm(self.states.shape[1])[:self.max_examples] 

        example_states = self.states[:, permutation, :][goal_indicies]
        example_rewards = self.rewards[:, permutation, :][goal_indicies]

        if len(example_states.shape) == 2:
            example_states = example_states.unsqueeze(0)
            example_rewards = example_rewards.unsqueeze(0)


        # if epsilon greedy, then do random action with prob epsilon
        if self.policy_type == "epsilon" and random.random() < self.update_eps:
            return random.randrange(self.env_params['action'])
        if target_network:
            q = self.target_critic_network(obs, example_states, example_rewards)
        else:
            q = self.critic_network(obs, example_states, example_rewards)
            if print_q:
                print(f"Q: {q}")
        if self.policy_type == "epsilon":
            return q.max(1)[1].item()
        elif self.policy_type == "boltzmann":
            q = q / self.args.temp
            return torch.distributions.Categorical(logits=q).sample().item()
        else:
            raise NotImplementedError()

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    def _hard_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(param.data)


    def _value_loss(self, goals, reward_function):
        with torch.no_grad():
            online_transitions = self.training_buffer.sample(self.args.batch_size)

            obs_tensor = torch.tensor(online_transitions['obs'], dtype=torch.float32, device=self.device)
            obs_next_tensor = torch.tensor(online_transitions['obs_next'], dtype=torch.float32, device=self.device)
            actions_tensor = torch.tensor(online_transitions['action'], dtype=torch.long, device=self.device)
            achieved_goals = torch.tensor(online_transitions['ag'], dtype=torch.long, device=self.device)
            dones_tensor = torch.tensor(online_transitions['done'], dtype=torch.float32, device=self.device)
            achieved_goals = achieved_goals/170. # normalize it to be betwreen 0 and 1

            # get example data
            goals = torch.tensor(goals, device=self.device)

            # gather example states and rewards
            goal_indicies = torch.argmin(torch.sum(torch.abs(self.goals.unsqueeze(1) - goals.unsqueeze(0)), dim=-1), dim=0)

            # get permutation of data for each goal index # TODO  make permutation different for every goal
            permutation = torch.randperm(self.states.shape[1])[:self.max_examples] 

            example_states = self.states[:, permutation, :][goal_indicies]
            example_rewards = self.rewards[:, permutation, :][goal_indicies]


            # compute reward
            rewards = reward_function(achieved_goals)

            # if done, then we hit a ghost. Want to avoid that, so penalize
            rewards = rewards - 1.0 * dones_tensor.reshape(1, -1, 1) # TODO dense reward

            # want to check if we have arrived at goal. If so, we are done (for achieving goal)
            # only add reward if dense
            goals = self._preproc_g(goals)
            distance = torch.max(torch.abs(achieved_goals.unsqueeze(0) - goals.unsqueeze(1)), dim=-1)[0]
            at_goal =  distance <= (self.env.distance_threshold/170.)
            rewards = rewards + 1.0 * at_goal.unsqueeze(-1).to(torch.float32) # TODO dense reward
            dones_tensor = torch.logical_or(at_goal, dones_tensor).to(torch.float32)

            # preprocess
            obs_tensor = self._preproc_o(obs_tensor)
            obs_next_tensor = self._preproc_o(obs_next_tensor)
            dones_tensor = dones_tensor



            # calculate the target Q value function
            q_next_values = self.target_critic_network(obs_next_tensor, example_states, example_rewards)
            action_probs = torch.nn.functional.softmax(q_next_values/self.args.temp, dim=-1)
            q_next_value = torch.sum(action_probs * q_next_values, dim=-1)
            # q_next_value = q_next_values.max(1)[0].reshape(-1, 1) # nromal best action
            rewards = rewards[:, torch.arange(rewards.shape[1]), actions_tensor]
            q_next_value = q_next_value.detach()
            target_q_value = rewards + (1-dones_tensor) * self.args.gamma * q_next_value.transpose(1,0)
            target_q_value = target_q_value.detach()
            # clip the q value
            # clip_return = 1 / (1 - self.args.gamma) # TODO why do this?
            # target_q_value = torch.clamp(target_q_value, 0, clip_return)
            # the q loss
        real_q_value = self.critic_network(obs_tensor, example_states, example_rewards).squeeze(0)
        real_q_value = real_q_value[torch.arange(real_q_value.shape[0]), :, actions_tensor].transpose(1,0)
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        return critic_loss


    # update the network
    def _update_network(self):
        self.optim.zero_grad()
        # fetch goals to train on
        # goals = self.env.all_goals[0][None] # TODO delete this
        goals = self.env.all_goals
        # num_goals_each_time = self.args.batch_size
        num_goals_each_time = 30 # due to memory consideratiosn

        # We are extremely limited by memory requirements of transformer
        # We can only afford to process 1 goal each time if we want it to finish within a day
        total_loss = 0
        goals2 = goals[torch.randperm(goals.shape[0])[:num_goals_each_time]]
        if len(goals2.shape) == 1:
            goals2 = goals2[None]

        # start accumulating
        # for index in range(num_goals_each_time):
        #     goals3 = goals2[index:index+1]
        reward_function = get_reward_function(goals2, self.num_actions)
        loss = self._value_loss(goals2, reward_function)
        loss.backward()
        with torch.no_grad():
            total_loss += loss.item()

        # actually step now
        norm = torch.nn.utils.clip_grad_norm_(self.critic_network.parameters(), 1.0)
        self.optim.step()
        with torch.no_grad():
            self.logger.add_scalar('Loss/critic', total_loss, self.update_iteration)
            self.logger.add_scalar('Loss/norm', norm.item(), self.update_iteration)
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

            total_reward = 0
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    obs_norm_tensor = self._preproc_o(obs)
                    action = self.act(obs_norm_tensor, g)
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

            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    obs_norm_tensor = self._preproc_o(obs)
                    action = self.act(obs_norm_tensor, g, print_q=False)
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

                img = cv2.resize(img, (160*2, 210*2))
                out.write(img)
                # cv2.imshow("pac", img)
                # k = cv2.waitKey(200)


                if d or info['is_success'] > 0:
                    break
        out.release()



    def get_data(self):
        # get all states and goals
        states = self._preproc_g(self.env.get_uniform_inputs())
        goals = self.env.all_goals

        # compute reward for all states for all goals
        reward_functions = get_reward_function(goals, self.num_actions)
        rewards = reward_functions(states)

        # process the data into tensors
        self.states = states.unsqueeze(0).repeat(rewards.shape[0], 1, 1)
        self.rewards = rewards
        self.goals = torch.tensor(goals).to(self.device)

        # sample a random set of states-rewards so there is no data noise, at the cost of losing information
        # permutation = torch.randperm(self.states.shape[1])[:self.max_examples]
        # self.states = self.states[:, permutation, :]
        # self.rewards = self.rewards[:, permutation, :]

