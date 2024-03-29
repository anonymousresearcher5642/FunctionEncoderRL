import torch
import os
from datetime import datetime
import numpy as np
import random

from torch.utils.tensorboard import SummaryWriter

from atari_modules.replay_buffer import her_replay_buffer
from atari_modules.models import critic
from atari_modules.her import her_sampler
import pickle
import csv

from atari_modules.wrappers import goal_distance


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


class HerDQNAgent:
    def __init__(self, args, env, env_params):
        self.args = args
        self.env = env
        self.env_params = env_params
        # create the network
        self.critic_network = critic(env_params)
        # build up the target network
        self.critic_target_network = critic(env_params)
        # load the weights into the target networks
        self.critic_target_network.load_state_dict(self.critic_network.state_dict())
        # if use gpu
        if self.args.cuda:
            self.critic_network.cuda()
            self.critic_target_network.cuda()
        # create the optimizer
        self.critic_optim = torch.optim.Adam(self.critic_network.parameters(), lr=self.args.lr)
        # her sampler
        self.her_module = her_sampler('future', self.args.replay_k, self.env.compute_reward)
        # create the replay buffer
        self.buffer = her_replay_buffer(self.env_params, self.args.buffer_size, self.her_module.sample_her_transitions)
        # self.buffer = ReplayBuffer(self.args.buffer_size)
        # create the dict for store the model
        if self.args.save_dir is not None:
            if not os.path.exists(self.args.save_dir):
                os.mkdir(self.args.save_dir)

            print(' ' * 26 + 'Options')
            for k, v in vars(self.args).items():
                print(' ' * 26 + k + ': ' + str(v))

            with open(self.args.save_dir + "/arguments.pkl", 'wb') as f:
                pickle.dump(self.args, f)

            with open('{}/score_monitor.csv'.format(self.args.save_dir), "wt") as monitor_file:
                monitor = csv.writer(monitor_file)
                monitor.writerow(['epoch', 'eval', 'avg dist'])

        # create logger for losses
        current_datetime = datetime.now()
        date_time_string = current_datetime.strftime("%Y-%m-%d %H:%M:%S")
        dir = f"./data/" + date_time_string
        self.dir = dir
        self.logger = SummaryWriter(dir)

        # create a file called transformer.txt in log
        with open(os.path.join(dir, "her_dqn.txt"), "wt") as file:
            file.write("")


        self.update_iteration = 0
        self.num_actions = env.action_space.n

    def learn(self):
        """
        train the network

        """
        # start to collect samples
        for epoch in range(self.args.n_epochs):
            for _ in range(self.args.n_cycles):
                mb_obs, mb_ag, mb_g, mb_actions, mb_dones = [], [], [], [], []
                for _ in range(self.args.num_rollouts_per_cycle):
                    # reset the rollouts
                    ep_obs, ep_ag, ep_g, ep_actions, ep_dones = [], [], [], [], []
                    # reset the environment
                    observation = self.env.reset()
                    obs = observation['observation']
                    ag = observation['achieved_goal']
                    g = observation['desired_goal']
                    # start to collect samples
                    for t in range(self.env_params['max_timesteps']):
                        with torch.no_grad():
                            obs_tensor = self._preproc_o(obs)
                            g_tensor = self._preproc_g(g)
                            action = self.act_e_greedy(obs_tensor, g_tensor, update_eps=0.2)
                        # feed the actions into the environment
                        observation_new, reward, done, info = self.env.step(action)
                        obs_new = observation_new['observation']
                        ag_new = observation_new['achieved_goal']
                        # add transition to replay buffer
                        # self.buffer.add(obs, ag, g, action, reward, obs_new, done)
                        # append rollouts
                        ep_obs.append(np.array(obs, dtype=np.uint8))
                        ep_ag.append(ag.copy())
                        ep_g.append(g.copy())
                        ep_actions.append(action)
                        ep_dones.append(float(done))
                        # re-assign the observation
                        if done:
                            observation = self.env.reset()
                            obs = observation['observation']
                            ag = observation['achieved_goal']
                            g = observation['desired_goal']
                        else:
                            obs = obs_new
                            ag = ag_new
                    ep_obs.append(np.array(obs, dtype=np.uint8))
                    ep_ag.append(ag.copy())
                    mb_obs.append(ep_obs)
                    mb_ag.append(ep_ag)
                    mb_g.append(ep_g)
                    mb_actions.append(ep_actions)
                    mb_dones.append(ep_dones)
                # convert them into arrays
                mb_obs = np.array(mb_obs, dtype=np.uint8)
                mb_ag = np.array(mb_ag)
                mb_g = np.array(mb_g)
                mb_actions = np.array(mb_actions)
                mb_dones = np.array(mb_dones)
                # store the episodes
                self.buffer.store_episode([mb_obs, mb_ag, mb_g, mb_actions, mb_dones])
                for _ in range(self.args.n_batches):
                    # train the network
                    self._update_network()
                # soft update
                self._soft_update_target_network(self.critic_target_network, self.critic_network)
            # start to do the evaluation
            average_reward, average_dist, success_rate  = self._eval_agent()
            self.logger.add_scalar('rl/total_reward', average_reward, self.update_iteration)
            self.logger.add_scalar('rl/final_distance', average_dist, self.update_iteration)
            self.logger.add_scalar('rl/success_rate', success_rate, self.update_iteration)

            print('[{}] epoch is: {}, eval success rate is: {:.3f}, avg dist: {:.3f}'.format(datetime.now(), epoch, success_rate, average_dist))
            with open('{}/score_monitor.csv'.format(self.args.save_dir), "a") as monitor_file:
                monitor = csv.writer(monitor_file)
                monitor.writerow([epoch, success_rate, average_dist])
            torch.save([self.critic_network.state_dict()],
                       os.path.join(self.dir, 'model.pt'))
                # print('n_transitions_stored: {}'.format(self.buffer.n_transitions_stored))
                # print('current replay size: {}, percentage: {}'.format(self.buffer.current_size, self.buffer.current_size / self.buffer.size * 100))
                # torch.save([self.o_norm.mean, self.o_norm.std, self.g_norm.mean, self.g_norm.std,
                #             self.actor_network.state_dict()], \
                #            self.model_path + '/model.pt')


    # pre_process the inputs
    def _preproc_o(self, obs):
        obs = np.transpose(np.array(obs)[None] / 255., [0, 3, 1, 2])
        obs_tensor = torch.tensor(obs, dtype=torch.float32)
        if self.args.cuda:
            obs_tensor = obs_tensor.cuda()
        return obs_tensor

    def _preproc_g(self, g):
        g_tensor = torch.tensor(g[None] / 170, dtype=torch.float32)
        if self.args.cuda:
            g_tensor = g_tensor.cuda()
        return g_tensor


    # Acts based on single state (no batch)
    def act(self, obs, g):
        return self.critic_network(obs, g).data.max(1)[1]

    # Acts with an epsilon-greedy policy
    def act_e_greedy(self, obs, g, update_eps=0.2):
        return random.randrange(self.env_params['action']) if random.random() < update_eps else self.act(obs, g).item()

    # soft update
    def _soft_update_target_network(self, target, source):
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_((1 - self.args.polyak) * param.data + self.args.polyak * target_param.data)

    # update the network
    def _update_network(self):
        # sample the episodes
        transitions = self.buffer.sample(self.args.batch_size)
        # pre-process the observation and goal
        obs_tensor = torch.tensor(np.transpose(transitions['obs'], [0, 3, 1, 2]) / 255, dtype=torch.float32)
        obs_next_tensor = torch.tensor(np.transpose(transitions['obs_next'], [0, 3, 1, 2]) / 255, dtype=torch.float32)
        g_tensor = torch.tensor(transitions['g'] / 170, dtype=torch.float32)
        ag_tensor = torch.tensor(transitions['ag'] / 170, dtype=torch.float32)
        # import pdb
        # pdb.set_trace()
        dones_tensor = torch.tensor(transitions['done'], dtype=torch.float32).reshape(-1, 1)
        actions_tensor = torch.tensor(transitions['actions'], dtype=torch.long)
        if self.args.cuda:
            obs_tensor = obs_tensor.cuda()
            obs_next_tensor = obs_next_tensor.cuda()
            g_tensor = g_tensor.cuda()
            dones_tensor = dones_tensor.cuda()
            actions_tensor = actions_tensor.cuda()
            ag_tensor = ag_tensor.cuda()
        # calculate the target Q value function
        with torch.no_grad():
            # caculate reward
            reward_function = get_reward_function(g_tensor * 170, num_actions=5)
            rewards = reward_function(ag_tensor)
            rewards = rewards[torch.arange(rewards.shape[0]), torch.arange(rewards.shape[0]), :]
            # if done, then we hit a ghost. Want to avoid that, so penalize
            rewards = rewards - 1.0 * dones_tensor.reshape(-1, 1)

            # want to check if we have arrived at goal. If so, we are done (for achieving goal)
            # only add reward if dense
            distance = torch.max(torch.abs(ag_tensor - g_tensor), dim=-1)[0]
            at_goal = distance <= (self.env.distance_threshold / 170.)
            r_tensor = rewards + 1.0 * at_goal.unsqueeze(1).to(torch.float32)
            r_tensor = r_tensor[torch.arange(r_tensor.shape[0]), actions_tensor].unsqueeze(-1)

            # do the normalization
            q_next_value = self.critic_target_network(obs_next_tensor, g_tensor).max(1)[0].reshape(-1, 1)
            q_next_value = q_next_value.detach()
            target_q_value = r_tensor + (1 - dones_tensor) * self.args.gamma * q_next_value
            target_q_value = target_q_value.detach()
            # clip the q value
            clip_return = 1 / (1 - self.args.gamma)
            target_q_value = torch.clamp(target_q_value, 0, clip_return)
        # the q loss
        real_q_value = self.critic_network(obs_tensor, g_tensor).gather(1, actions_tensor.reshape(-1, 1))
        critic_loss = (target_q_value - real_q_value).pow(2).mean()
        # update the critic_network
        self.critic_optim.zero_grad()
        critic_loss.backward()
        self.critic_optim.step()

        # log
        self.logger.add_scalar('Loss/critic', critic_loss.item(), self.update_iteration)
        self.update_iteration += 1

    # do the evaluation
    def _eval_agent(self):
        total_success_rate = []
        total_dist = []
        total_rewards = []

        for _ in range(self.args.n_test_rollouts):
            observation = self.env.reset()
            obs = observation['observation']
            g = observation['desired_goal']
            reward_function = get_reward_function(g, self.num_actions)
            total_reward = 0
            for _ in range(self.env_params['max_timesteps']):
                with torch.no_grad():
                    # import pdb
                    # pdb.set_trace()
                    obs_tensor = self._preproc_o(obs)
                    g_tensor = self._preproc_g(g)
                    action = self.act_e_greedy(obs_tensor, g_tensor, update_eps=0.01)
                observation_new, _, done, info = self.env.step(action)
                obs = observation_new['observation']
                g = observation_new['desired_goal']
                dist = goal_distance(observation_new['achieved_goal'], observation_new['desired_goal'])
                reward = reward_function(self._preproc_g(observation_new['achieved_goal']))[0, 0, action].item()
                if done:  # caught by ghost # TODO dense rewards
                    reward -= 1
                if info['is_success'] > 0:  # at goal
                    reward += 1

                total_reward += reward

                if info['is_success'] > 0 or done:
                    break
            total_rewards.append(total_reward)
            total_success_rate.append(info['is_success'])
            total_dist.append(dist)

        total_rewards = np.array(total_rewards)
        total_rewards = np.mean(total_rewards)

        total_dist = np.array(total_dist)
        total_dist = np.mean(total_dist)

        total_successes = np.array(total_success_rate)
        total_successes = np.mean(total_successes)
        # return success_rate, dist
        return total_rewards, total_dist, total_successes