import torch
import torch.nn as nn
import torch.nn.functional as F

"""
the input x in both networks should be [o, g], where o is the observation and g is the goal.

"""


class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.q_out = nn.Linear(256, env_params['action'])

    def forward(self, obs, g):
        x = torch.cat([obs, g], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        q_value = self.q_out(x)
        return q_value


class VMap(nn.Module):
    def __init__(self, env_params, embed_dim):
        super(VMap, self).__init__()
        self.embed_dim = embed_dim
        self.fc1 = nn.Linear(env_params['obs'] + embed_dim + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.v_out = nn.Linear(256, 1)

    def forward(self, obs, w, g):
        w = w / torch.sqrt(1 + torch.norm(w, dim=-1, keepdim=True) ** 2 / self.embed_dim)
        x = torch.cat([obs, w, g], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        v_value = self.v_out(x)
        return v_value


class ZMap(nn.Module):
    def __init__(self, env_params):
        super(ZMap, self).__init__()
        self.fc1 = nn.Linear(env_params['obs'] + env_params['goal'] + env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.z_out = nn.Linear(256, env_params['action'])

    def forward(self, obs, g, g_other):
        assert g.shape[-1] == g_other.shape[-1]
        x = torch.cat([obs, g, g_other], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        z_value = self.z_out(x)
        return z_value


class BackwardMap(nn.Module):
    def __init__(self, env_params, embed_dim):
        super(BackwardMap, self).__init__()
        self.fc1 = nn.Linear(env_params['goal'], 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.backward_out = nn.Linear(256, embed_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        backward_value = self.backward_out(x)
        return backward_value


class ForwardMap(nn.Module):
    def __init__(self, env_params, embed_dim):
        super(ForwardMap, self).__init__()
        self.embed_dim = embed_dim
        self.num_actions = env_params['action']
        self.fc1 = nn.Linear(env_params['obs'] + embed_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.forward_out = nn.Linear(256, embed_dim * env_params['action'])

    def forward(self, obs, w):
        w = w / torch.sqrt(1 + torch.norm(w, dim=-1, keepdim=True) ** 2 / self.embed_dim)
        x = torch.cat([obs, w], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        forward_value = self.forward_out(x)

        return forward_value.reshape(-1, self.embed_dim, self.num_actions)



class RewardEncoder(nn.Module):
    def __init__(self, env_params, args):
        super(RewardEncoder, self).__init__()
        self.embed_dim = args.embed_dim
        self.num_actions = env_params['action']
        assert self.embed_dim % self.num_actions == 0
        self.nn = nn.Sequential(
            nn.Linear(env_params['obs'], 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, self.embed_dim),
        )
    def forward(self, obs):
        reward_encoding = self.nn(obs)
        reward_encoding = reward_encoding.reshape(-1, self.embed_dim//self.num_actions, self.num_actions)
        return reward_encoding

class TaskAwareCritic(nn.Module):
    def __init__(self, env_params,args):
        super(TaskAwareCritic, self).__init__()
        self.obs_dim = env_params['obs']
        self.embed_dim = args.embed_dim
        self.num_actions = env_params['action']
        self.input_dim = self.obs_dim + self.embed_dim
        self.output_dim = self.num_actions * self.embed_dim//self.num_actions # outputs a matrix, which we multiple by the embedding to get a value for each action

        self.nn = nn.Sequential(
            nn.Linear(self.input_dim, 256),
            nn.Linear(256, 256),
            nn.Linear(256, 256),
            nn.Linear(256, self.output_dim),
        )

    def norm_encoding(self, encoding):
        if (encoding != 0).any():
            encoding =  self.embed_dim **0.5  * encoding / (torch.norm(encoding))
        return encoding.flatten()

    def forward(self, obs, reward_encoding):
        assert obs.shape[1] == self.obs_dim
        assert reward_encoding.shape[0] == self.embed_dim//self.num_actions
        assert reward_encoding.shape[1] == self.num_actions
        # w = w / torch.sqrt(1 + torch.norm(w, dim=-1, keepdim=True) ** 2 / self.embed_dim) Some sort of scaling
        normed_reward_encoding = self.norm_encoding(reward_encoding)
        x = torch.cat([obs, normed_reward_encoding.expand(obs.shape[0], normed_reward_encoding.shape[0])], dim=1)
        q_values = self.nn(x)
        q_values = q_values.reshape(-1, self.num_actions, self.embed_dim//self.num_actions)
        q_values = torch.sum(q_values * reward_encoding.transpose(0, 1), dim=2)
        return q_values

