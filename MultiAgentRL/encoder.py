import torch
from torch import nn
from torch.nn import Sequential


class PolicyEncoder(nn.Module):
    def __init__(self, state_size, action_size, embed_dim):
        super(PolicyEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_actions = action_size[0]
        self.nn = Sequential(
            nn.Linear(state_size[0], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim * action_size[0]),
        )
    def forward(self, x):
        encoding = self.nn(x)
        reward_encoding = encoding.reshape(-1, self.embed_dim, self.num_actions)
        return reward_encoding