from typing import Any, Dict, List, Optional, Type

import numpy as np
import torch
from torch import nn

from tianshou.data import Batch, ReplayBuffer, to_torch_as
from tianshou.policy import A2CPolicy, PPOPolicy, BasePolicy
from tianshou.utils.net.common import ActorCritic


class FixedPPOPolicyWrapper:
    """A version of PPO with no update function, to keep a agent fixed.
    """

    def __init__(
        self,
        ppo_agent: PPOPolicy,
    ) -> None:
        # super().__init__()
        self.wrapped_agent = ppo_agent # intentionally not calling super().__init__ to avoid creating a new agent

    def learn(  # type: ignore # Learn does nothing so it does not update
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        return {}

    # all other functions are passed to the agent
    def __getattr__(self, name):
        return getattr(self.wrapped_agent, name)

    def __call__(self, *args, **kwargs):
        return self.wrapped_agent(*args, **kwargs)