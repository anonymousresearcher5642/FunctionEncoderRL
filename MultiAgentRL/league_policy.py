from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch

from tianshou.data import Batch, ReplayBuffer, to_torch, to_torch_as
from tianshou.policy import BasePolicy, PGPolicy
from tianshou.utils import RunningMeanStd


class LeaguePolicy(BasePolicy):
    def __init__(
        self,
        agents: List[PGPolicy],
    ) -> None:
        super(LeaguePolicy, self).__init__()
        self.agents = agents
        self.num_agents = len(agents)
        self.force_agent = -1

    def process_fn(
        self, batch: Batch, buffer: ReplayBuffer, indices: np.ndarray
    ) -> Batch:
        return Batch() # we dont update anyway so just return nothing

    def set_agent(self, agent: int):
        self.force_agent = agent

    def forward(
        self,
        batch: Batch,
        state: Optional[Union[dict, Batch, np.ndarray]] = None,
        **kwargs: Any,
    ) -> Batch:

        outs = []
        for i in range(min(self.num_agents, len(batch))):
            # break into parts
            minibatch = batch[i]
            minibatch.act = minibatch.act.reshape(1, -1)
            minibatch.done = np.array([minibatch.done]) # make sure data is still in batch form even if batch is 1
            minibatch.obs = minibatch.obs.reshape(1, -1)
            minibatch.obs_next = minibatch.obs_next.reshape(1, -1)
            minibatch.rew = np.array([minibatch.rew])
            minibatch.terminated = np.array([minibatch.terminated])
            minibatch.truncated = np.array([minibatch.truncated])


            # pass to each agent
            agent_index = i if self.force_agent == -1 else self.force_agent # allows for 1 agent to always play in testing
            agent = self.agents[agent_index]
            out_minibatch = agent.forward(minibatch, state, **kwargs)
            outs.append(out_minibatch)

        # recombine parts
        ret = Batch(outs)
        ret.act = ret.act.reshape(len(batch), -1) # fix dimension issue, extra 1 dim
        ret.logits = ret.logits.reshape(len(batch), 2, -1)
        return ret

    def learn(  # type: ignore
        self, batch: Batch, batch_size: int, repeat: int, **kwargs: Any
    ) -> Dict[str, List[float]]:
        print("Run")
        return {} # the league is constant, does not update
