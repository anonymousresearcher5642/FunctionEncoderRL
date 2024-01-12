import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Sequential


# define the critic network
class critic(nn.Module):
    def __init__(self, env_params):
        super(critic, self).__init__()
        self.conv1 = nn.Conv2d(env_params['obs'][-1], 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(3136 + env_params['goal'], 512)
        self.fc2 = nn.Linear(512, env_params['action'])

    def forward(self, obs, goal):
        assert (goal < 1.0).all()
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1, 3136)
        x = torch.cat([x, goal], dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


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
        self.conv1 = nn.Conv2d(env_params['obs'][-1], 32, 8, stride=4, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(3136 + embed_dim, 512)
        self.forward_out = nn.Linear(512, embed_dim * env_params['action'])

    def forward(self, obs, w):
        w = w / torch.sqrt(1 + torch.norm(w, dim=-1, keepdim=True) ** 2 / self.embed_dim)
        x = F.relu(self.conv1(obs))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.reshape(-1, 3136)
        x = torch.cat([x, w], dim=1)
        x = F.relu(self.fc1(x))
        forward_value = self.forward_out(x)
        return forward_value.reshape(-1, self.embed_dim, self.num_actions)


class RewardEncoder(nn.Module):
    def __init__(self, env_params, embed_dim):
        super(RewardEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.num_actions = env_params['action']
        self.nn = Sequential(
            nn.Linear(env_params['goal'], 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, embed_dim * env_params['action']),
        )
    def forward(self, x):
        encoding = self.nn(x)
        reward_encoding = encoding.reshape(-1, self.embed_dim, self.num_actions)
        return reward_encoding
class RewardEncoderTranslator(nn.Module): # This was a test to see if we could increase the difference between encodings by adding a translator,
    # whiches only purpose is to increase the distance. However, this turns out to not be needed, so this is unused.
    def __init__(self, env_params, embed_dim):
        super(RewardEncoderTranslator, self).__init__()
        self.embed_dim = embed_dim
        self.num_actions = env_params['action']
        self.nn = nn.Linear(embed_dim * env_params['action'], embed_dim * env_params['action'], bias=False)
    def forward(self, x):
        x = x.reshape(-1, self.embed_dim * self.num_actions)
        translation = self.nn(x)
        translation = translation.reshape(-1, self.embed_dim, self.num_actions)
        return translation

class TaskAwareCritic(nn.Module):
    def __init__(self, env_params, embed_dim):
        super(TaskAwareCritic, self).__init__()
        self.embed_dim = embed_dim
        self.num_actions = env_params['action']

        self.image = Sequential(
            nn.Conv2d(env_params['obs'][-1], 32, 8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
        )
        self.nn = Sequential(
            nn.Linear(3136 + embed_dim * self.num_actions, 512),
            nn.ReLU(),
            # nn.Linear(512, self.num_actions), # scalar value function
            nn.Linear(512, embed_dim * env_params['action']), # vector value function
        )
    # def norm_encoding(self, encoding):
    #     if (encoding != 0).any():
    #         encoding =  self.embed_dim **0.5  * encoding / (torch.norm(encoding))
    #     return encoding.flatten()
    def forward(self, obs, encoding):
        norms = torch.linalg.vector_norm(encoding.reshape(encoding.shape[0], -1), dim=1)
        norms = torch.where(norms == 0, torch.ones_like(norms), norms)
        assert norms.shape[0] == encoding.shape[0]
        norm_encoding = encoding / norms.reshape(-1, 1, 1)
        assert norm_encoding.shape[0] == encoding.shape[0]
        assert norm_encoding.shape[1] == encoding.shape[1]
        assert norm_encoding.shape[2] == encoding.shape[2]
        norm_encoding = norm_encoding.reshape(-1, self.embed_dim * self.num_actions)
        image_encoding = self.image(obs)
        image_encoding = image_encoding.reshape(-1, 3136)
        norm_encoding = norm_encoding.unsqueeze(1).repeat_interleave(image_encoding.shape[0], dim=1)
        image_encoding = image_encoding.unsqueeze(0).repeat_interleave(norm_encoding.shape[0], dim=0)
        values = self.nn(torch.cat([image_encoding, norm_encoding], dim=2))

        # scalar value function
        # q_values = values

        # Vector value function
        q_values = values.reshape(values.shape[0], values.shape[1], self.num_actions, self.embed_dim)
        q_values = torch.sum(q_values * encoding.unsqueeze(1).transpose(-1, -2), dim=3)
        return q_values

class TransformerCritic(nn.Module):

    def __init__(self, observation_space, state_space, action_space, d_model=64, device="cpu"):
        super().__init__()

        # save the input and output sizes
        self.observation_space = observation_space # note this is the image input state
        self.state_space = state_space # this is the x,y state used for reward
        self.action_space = action_space # this is the action space
        self.d_model = d_model
        self.device = device

        self.transformer = torch.nn.Transformer(d_model=d_model,
                                                nhead=4,
                                                num_encoder_layers=3,
                                                num_decoder_layers=3,
                                                dim_feedforward=d_model,
                                                dropout=0.0,
                                                batch_first=True).to(device)
        input_size = state_space[0] + action_space[0]  # state, 1 reward per action
        self.encoder_goal = torch.nn.Sequential(  # encodes inputs (state x action)
            torch.nn.Linear(input_size, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
        ).to(device)
        self.encoder_observation = Sequential(
            nn.Conv2d(observation_space[-1], 32, 8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            torch.nn.Flatten(),
            nn.Linear(3136, d_model),
        ).to(device)
        self.decoder = torch.nn.Sequential(  # outputs the next state
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, action_space[0]),
        ).to(device)
        # note we need to make the values approximately equal in the beginning. This is done by scaling the weights
        # of the decoder linear layer by 1/sqrt(d_model)
        # I think this is super important.
        for p in self.decoder.parameters():
            p.data = p.data / (d_model ** 0.5)


    def forward(self, obs, example_state, example_reward):
        assert len(example_state.shape) == 3, "Example state should have dimension 3: (num_reward_functions, num_datapoints, state_dim)"
        assert len(example_reward.shape) == 3, "Example reward should have dimension 3: (num_reward_functions, num_datapoints, 1)"
        assert example_state.shape[-1] == self.state_space[0], "Example state must have the same dimension as the state space"
        assert example_reward.shape[-1] == self.action_space[0], "Example reward must be a scalar for each action"

        # compute encodings of inputs
        state_action_reward_example = torch.cat([example_state, example_reward], dim=-1)
        example_encoding = self.encoder_goal(state_action_reward_example)
        observation_encoding = self.encoder_observation(obs)
        observation_encoding = observation_encoding.unsqueeze(1) # need to add extra sequence dimension

        num_obs = observation_encoding.shape[0]
        num_goals = example_encoding.shape[0]

        if example_encoding.shape[0] != observation_encoding.shape[0]:
            example_encoding = example_encoding.unsqueeze(0)
            observation_encoding = observation_encoding.unsqueeze(1)
            example_encoding = example_encoding.repeat(num_obs, 1, 1, 1)
            observation_encoding = observation_encoding.repeat(1, num_goals, 1, 1)
            example_encoding = example_encoding.view(-1, example_encoding.shape[-2], example_encoding.shape[-1])
            observation_encoding = observation_encoding.view(-1, observation_encoding.shape[-2], observation_encoding.shape[-1])

        # pass through
        # num_reward_functions = example_encoding.shape[0]
        # example_encoding = example_encoding.repeat(num_obs, 1, 1) # repeat for each observation
        # observation_encoding = observation_encoding.repeat(num_reward_functions, 1, 1) # repeat for each example
        output_embedding = self.transformer(example_encoding, observation_encoding)

        # compute output
        output = self.decoder(output_embedding)
        output = output.reshape(num_obs, num_goals, -1)
        return output

    def get_latent_embedding(self, obs, example_state, example_reward):
        assert len(
            example_state.shape) == 3, "Example state should have dimension 3: (num_reward_functions, num_datapoints, state_dim)"
        assert len(
            example_reward.shape) == 3, "Example reward should have dimension 3: (num_reward_functions, num_datapoints, 1)"
        assert example_state.shape[-1] == self.state_space[
            0], "Example state must have the same dimension as the state space"
        assert example_reward.shape[-1] == self.action_space[0], "Example reward must be a scalar for each action"

        # compute encodings of inputs
        state_action_reward_example = torch.cat([example_state, example_reward], dim=-1)
        example_encoding = self.encoder_goal(state_action_reward_example)
        return example_encoding
        # observation_encoding = self.encoder_observation(obs)
        # observation_encoding = observation_encoding.unsqueeze(1)  # need to add extra sequence dimension
        #
        # # pass through
        # output_embedding = self.transformer(example_encoding, observation_encoding)
        # return output_embedding

class TransformerOracle(nn.Module):

    def __init__(self, observation_space, state_space, action_space, d_model=512, device="cpu"):
        super().__init__()

        # save the input and output sizes
        self.observation_space = observation_space # note this is the image input state
        self.state_space = state_space # this is the x,y state used for reward
        self.action_space = action_space # this is the action space
        self.d_model = d_model
        self.device = device

        self.transformer = torch.nn.Transformer(d_model=d_model,
                                                nhead=4,
                                                num_encoder_layers=4,
                                                num_decoder_layers=4,
                                                dim_feedforward=d_model,
                                                dropout=0.0,
                                                batch_first=True).to(device)
       
        input_size = state_space[0]
        self.encoder_goal = torch.nn.Sequential(  # encodes inputs (state x action)
            torch.nn.Linear(input_size, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
        ).to(device)
        self.encoder_observation = Sequential(
            nn.Conv2d(observation_space[-1], 32, 8, stride=4, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3),
            nn.ReLU(),
            torch.nn.Flatten(),
            nn.Linear(3136, d_model),
        ).to(device)
        self.decoder = torch.nn.Sequential(  # outputs the next state
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, action_space[0]),
        ).to(device)
        # note we need to make the values approximately equal in the beginning. This is done by scaling the weights
        # of the decoder linear layer by 1/sqrt(d_model)
        # I think this is super important.
        for p in self.decoder.parameters():
            p.data = p.data / (d_model ** 0.5)


    def forward(self, obs, goal):
        assert goal.shape[-1] == self.state_space[0], "goal must have the same dimension as the state space"
        if len(goal.shape) == 2:
            goal = goal.unsqueeze(1)
        # compute encodings of inputs
        example_encoding = self.encoder_goal(goal)
        observation_encoding = self.encoder_observation(obs)
        observation_encoding = observation_encoding.unsqueeze(1) # need to add extra sequence dimension

        # pass through
        # num_reward_functions = example_encoding.shape[0]
        num_obs = observation_encoding.shape[0]
        num_goals = example_encoding.shape[0]
        # example_encoding = example_encoding.repeat(num_obs, 1, 1) # repeat for each observation
        # observation_encoding = observation_encoding.repeat(num_reward_functions, 1, 1) # repeat for each example
        if example_encoding.shape[0] != observation_encoding.shape[0]:
            example_encoding = example_encoding.unsqueeze(0)
            observation_encoding = observation_encoding.unsqueeze(1)
            example_encoding = example_encoding.repeat(num_obs, 1, 1, 1)
            observation_encoding = observation_encoding.repeat(1, num_goals, 1, 1)
            example_encoding = example_encoding.view(-1, example_encoding.shape[-2], example_encoding.shape[-1])
            observation_encoding = observation_encoding.view(-1, observation_encoding.shape[-2], observation_encoding.shape[-1])
        # ins = torch.cat((observation_encoding, example_encoding), dim=-2)
        output_embedding = self.transformer(example_encoding, observation_encoding) # [:, -1, :]
        # compute output
        output = self.decoder(output_embedding)

        output = output.reshape(num_obs, num_goals, -1)
        return output

    def get_latent_embedding(self, obs, goal):
        # compute encodings of inputs
        example_encoding = self.encoder_goal(goal)
        observation_encoding = self.encoder_observation(obs)
        observation_encoding = observation_encoding.unsqueeze(1)  # need to add extra sequence dimension

        # pass through
        output_embedding = self.transformer(example_encoding, observation_encoding)
        return output_embedding
