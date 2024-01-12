import warnings
from typing import Sequence, Union, Optional, Type, Dict, Any, Tuple

import numpy as np
import torch
from tianshou.utils.net.common import MLP
from tianshou.utils.net.continuous import SIGMA_MIN, SIGMA_MAX
from torch import nn


class TransformerCritic(nn.Module):
    """Simple critic network. Will create an actor operated in continuous \
    action space with structure of preprocess_net ---> 1(q value).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.
    :param linear_layer: use this module as linear layer. Default to nn.Linear.
    :param bool flatten_input: whether to flatten input data for the last layer.
        Default to True.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        state_space,
        action_space,
        example_data:torch.tensor,
        device: Union[str, int, torch.device] = "cpu",
        d_model=64,
        number_datapoints: int = 200,

    ) -> None:
        super().__init__()

        # get input and output sizes
        input_size = state_space[0]
        output_size = 1
        self.number_datapoints = number_datapoints

        self.device = device
        self.transformer = torch.nn.Transformer(d_model=d_model,
                                                nhead=4,
                                                num_encoder_layers=3,
                                                num_decoder_layers=3,
                                                dim_feedforward=d_model,
                                                dropout=0.0,
                                                batch_first=True).to(device)
        self.encoder_examples = torch.nn.Sequential( # encodes examples (state x action)
            torch.nn.Linear(state_space[0] + action_space.shape[0], d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
        ).to(device)
        self.encoder_states = torch.nn.Sequential( # encodes the current state (state)
            torch.nn.Linear(state_space[0], d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
        ).to(device)
        self.decoder = torch.nn.Sequential( # outputs the next state
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, output_size),
        ).to(device)
        self.example_states = example_data[0]
        self.example_actions = example_data[1]
        self.d_model = d_model


    def forward_pass(self,
                example_states: torch.tensor, # F x B1 x S size
                example_actions: torch.tensor, # F x B1 x A size
                states: torch.tensor # F X B2 x S size
                ) -> torch.Tensor:
        # convert all data to encodings
        examples = torch.cat((example_states, example_actions), dim=-1)
        example_encoding = self.encoder_examples(examples) # F x B1 x D size
        # example_state_encodings = self.encoder_states(example_states)  # F x B1 x D size
        # example_action_encodings = self.encoder_actions(example_actions)  # F x B1 x D size
        state_encoding = self.encoder_states(states).unsqueeze(1)  # F x B2 x D size
        # assert example_state_encodings.shape == (
        # example_state_encodings.shape[0], example_state_encodings.shape[1], self.d_model)
        # assert example_action_encodings.shape == (
        # example_action_encodings.shape[0], example_action_encodings.shape[1], self.d_model)
        # assert state_encoding.shape == (state_encoding.shape[0], 1, self.d_model)

        # convert example encodings to something we can feed into model
        # combined_encoder_inputs = torch.zeros((example_state_encodings.shape[0], 2 * example_state_encodings.shape[1], self.d_model)).to(self.device)
        # combined_encoder_inputs[:, 0::2, :] = example_state_encodings
        # combined_encoder_inputs[:, 1::2, :] = example_action_encodings
        # combined_encoder_inputs = combined_encoder_inputs.expand(states.shape[1], combined_encoder_inputs.shape[1], combined_encoder_inputs.shape[2])
        # assert combined_encoder_inputs.shape == (states.shape[0], 2 * example_state_encodings.shape[1], self.d_model)

        # convert real encodings to something we can feed into model
        # real_input_encoding = state_encoding[0]
        # real_input_encoding = real_input_encoding.unsqueeze(1)
        # assert real_input_encoding.shape == (state_encoding.shape[1], 1, self.d_model)

        output_embedding = self.transformer(example_encoding, state_encoding)
        output = self.decoder(output_embedding)
        return output.squeeze(1)

    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        act: Optional[Union[np.ndarray, torch.Tensor]] = None,
        info: Dict[str, Any] = {},
    ) -> torch.Tensor:
        assert act is None, "This should be none I think for PPO, since it learns V instead of Q"
        """Mapping: (s, a) -> logits -> Q(s, a)."""
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        ).flatten(1)
        # if act is not None:
        #     act = torch.as_tensor(
        #         act,
        #         device=self.device,
        #         dtype=torch.float32,
        #     ).flatten(1)
        #     obs = torch.cat([obs, act], dim=1)

        # get example data
        with torch.no_grad(): # dont want to change the example data. It is not trainable.
            assert torch.isnan(obs).any() == False
            env_indicies = info.get('env_id', np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])[:obs.shape[0]])  # the very first time, info is empty for some reason,but it is only the 10 envs, not training
            assert torch.isnan(torch.tensor(env_indicies)).any() == False
            random_permutation = torch.randperm(self.example_states.shape[1])[:self.number_datapoints]
            example_states = self.example_states[:, random_permutation, :]
            example_actions = self.example_actions[:, random_permutation, :]
            example_states = example_states[env_indicies]
            example_actions = example_actions[env_indicies]

        # pass through network and get output
        values = self.forward_pass(example_states, example_actions, obs)
        return values


class TransformerActorProb(nn.Module):
    """Simple actor network (output with a Gauss distribution).

    :param preprocess_net: a self-defined preprocess_net which output a
        flattened hidden state.
    :param action_shape: a sequence of int for the shape of action.
    :param hidden_sizes: a sequence of int for constructing the MLP after
        preprocess_net. Default to empty sequence (where the MLP now contains
        only a single linear layer).
    :param float max_action: the scale for the final action logits. Default to
        1.
    :param bool unbounded: whether to apply tanh activation on final logits.
        Default to False.
    :param bool conditioned_sigma: True when sigma is calculated from the
        input, False when sigma is an independent parameter. Default to False.
    :param int preprocess_net_output_dim: the output dimension of
        preprocess_net.

    For advanced usage (how to customize the network), please refer to
    :ref:`build_the_network`.

    .. seealso::

        Please refer to :class:`~tianshou.utils.net.common.Net` as an instance
        of how preprocess_net is suggested to be defined.
    """

    def __init__(
        self,
        state_shape,
        action_space: Sequence[int],
        example_data:torch.tensor,
        max_action: float = 1.0,
        device: Union[str, int, torch.device] = "cpu",
        unbounded: bool = False,
        conditioned_sigma: bool = False,
        preprocess_net_output_dim: Optional[int] = None,
        d_model=64,
        hidden_sizes: Sequence[int] = (128, 128),
        number_datapoints: int = 200,
    ) -> None:
        super().__init__()
        if unbounded and not np.isclose(max_action, 1.0):
            warnings.warn(
                "Note that max_action input will be discarded when unbounded is True."
            )
            max_action = 1.0
        self.device = device
        action_dim = action_space.shape[0]
        self.output_dim = int(np.prod(action_space.shape))
        self.input_dim = state_shape[0]
        self.d_model = d_model

        # self.mu = MLP(
        #     input_dim,  # type: ignore
        #     self.output_dim,
        #     hidden_sizes,
        #     device=self.device
        # )
        self._c_sigma = conditioned_sigma
        if conditioned_sigma:
            self.sigma = MLP(
                self.input_dim,  # type: ignore
                self.output_dim,
                hidden_sizes,
                device=self.device
            )
        else:
            self.sigma_param = nn.Parameter(torch.zeros(int(np.prod(action_space.shape)), 1))
        self.max_action = max_action
        self._unbounded = unbounded

        # create models
        self.transformer = torch.nn.Transformer(d_model=d_model,
                                                nhead=4,
                                                num_encoder_layers=3,
                                                num_decoder_layers=3,
                                                dim_feedforward=d_model,
                                                dropout=0.0,
                                                batch_first=True).to(device)
        self.encoder_examples = torch.nn.Sequential( # encodes example data (state, action)
            torch.nn.Linear(state_shape[0] + action_dim, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
        ).to(device)
        self.encoder_states = torch.nn.Sequential( # encodes the current states
            torch.nn.Linear(state_shape[0], d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, d_model),
        ).to(device)
        self.decoder = torch.nn.Sequential( # outputs the next state
            torch.nn.Linear(d_model, d_model),
            torch.nn.ReLU(),
            torch.nn.Linear(d_model, self.output_dim),
        ).to(device)
        self.example_states = example_data[0]
        self.example_actions = example_data[1]


        self.number_datapoints = number_datapoints


    def forward(
        self,
        obs: Union[np.ndarray, torch.Tensor],
        state: Any = None,
        info: Dict[str, Any] = {},
    ) -> Tuple[Tuple[torch.Tensor, torch.Tensor], Any]:
        """Mapping: obs -> logits -> (mu, sigma)."""
        obs = torch.as_tensor(
            obs,
            device=self.device,
            dtype=torch.float32,
        ).flatten(1)

        # get example data
        with torch.no_grad(): # dont want to change the example data. It is not trainable.
            assert torch.isnan(obs).any() == False
            env_indicies = info.get('env_id', np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9])[:obs.shape[0]])  # the very first time, info is empty for some reason,but it is only the 10 envs, not training
            assert torch.isnan(torch.tensor(env_indicies)).any() == False
            random_permutation = torch.randperm(self.example_states.shape[1])[:self.number_datapoints]
            example_states = self.example_states[:, random_permutation, :]
            example_actions = self.example_actions[:, random_permutation, :]
            example_states = example_states[env_indicies]
            example_actions = example_actions[env_indicies]


        # do forwrad pass
        mu = self.forward_pass(example_states, example_actions, obs)
        # mu = self.mu(logits)

        if not self._unbounded:
            mu = self.max_action * torch.tanh(mu)
        if self._c_sigma:
            raise NotImplementedError
            sigma = torch.clamp(self.sigma(mu), min=SIGMA_MIN, max=SIGMA_MAX).exp()
        else:
            shape = [1] * len(mu.shape)
            shape[1] = -1
            sigma = (self.sigma_param.view(shape) + torch.zeros_like(mu)).exp()
        if torch.isnan(mu).any() or torch.isnan(sigma).any():
            print("NANS")

        return (mu, sigma), state

    def forward_pass(self,
                example_states: torch.tensor, # F x B1 x S size
                example_actions: torch.tensor, # F x B1 x A size
                states: torch.tensor # F X B2 x S size
                ) -> torch.Tensor:
        # convert all data to encodings
        examples = torch.cat((example_states, example_actions), dim=-1)
        example_encoding = self.encoder_examples(examples)
        # example_state_encodings = self.encoder_states(example_states) # F x B1 x D size
        # example_action_encodings = self.encoder_actions(example_actions) # F x B1 x D size
        state_encoding = self.encoder_states(states).unsqueeze(1) # F x B2 x D size
        # assert example_state_encodings.shape == (example_state_encodings.shape[0], example_state_encodings.shape[1], self.d_model)
        # assert example_action_encodings.shape == (example_action_encodings.shape[0], example_action_encodings.shape[1], self.d_model)
        # assert state_encoding.shape == (state_encoding.shape[0], 1, self.d_model)

        # convert example encodings to something we can feed into model
        # combined_encoder_inputs = torch.zeros((example_state_encodings.shape[0], 2 * example_state_encodings.shape[1], self.d_model)).to(self.device)
        # combined_encoder_inputs[:, 0::2, :] = example_state_encodings
        # combined_encoder_inputs[:, 1::2, :] = example_action_encodings
        # combined_encoder_inputs = combined_encoder_inputs.expand(states.shape[1], combined_encoder_inputs.shape[1], combined_encoder_inputs.shape[2])
        # assert combined_encoder_inputs.shape == (states.shape[0], 2 * example_state_encodings.shape[1], self.d_model)

        # convert real encodings to something we can feed into model
        # real_input_encoding = state_encoding[0]
        # real_input_encoding = real_input_encoding.unsqueeze(1)
        # assert real_input_encoding.shape == (state_encoding.shape[1], 1, self.d_model)

        output_embedding = self.transformer(example_encoding, state_encoding)
        output = self.decoder(output_embedding)
        return output.squeeze(1)
