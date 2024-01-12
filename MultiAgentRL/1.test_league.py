import argparse
import os
import time
from copy import deepcopy
from typing import Optional, Tuple, List

import gymnasium as gym
import numpy as np
import torch
from pettingzoo.classic import tictactoe_v3
from tianshou.utils.net.continuous import Critic, ActorProb
from torch import nn
from torch.distributions import Independent, Normal
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter

from tianshou.data import Collector, VectorReplayBuffer
from tianshou.env import DummyVectorEnv
from tianshou.env.pettingzoo_env import PettingZooEnv
from tianshou.policy import (
    BasePolicy,
    # DQNPolicy,
    PPOPolicy,
    MultiAgentPolicyManager,
    RandomPolicy,
)
from tianshou.trainer import OffpolicyTrainer, OnpolicyTrainer
from tianshou.utils import TensorboardLogger
from tianshou.utils.net.common import Net, ActorCritic
from tqdm import trange

from dense_tag import dense_tag_env
from video_recorder import VideoRecorder


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=1626)
    parser.add_argument('--eps-test', type=float, default=0.05)
    parser.add_argument('--eps-train', type=float, default=0.1)
    parser.add_argument('--buffer-size', type=int, default=20000)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument(
        '--gamma', type=float, default=0.9, help='a smaller gamma favors earlier win'
    )
    parser.add_argument('--n-step', type=int, default=3)
    parser.add_argument('--target-update-freq', type=int, default=320)
    parser.add_argument('--epoch', type=int, default=500)
    parser.add_argument('--step-per-epoch', type=int, default=1000)
    parser.add_argument('--step-per-collect', type=int, default=2000)
    parser.add_argument('--update-per-step', type=float, default=0.1)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument(
        '--hidden-sizes', type=int, nargs='*', default=[128, 128, 128, 128]
    )
    parser.add_argument('--training-num', type=int, default=10)
    parser.add_argument('--test-num', type=int, default=10)
    parser.add_argument('--logdir', type=str, default='log')
    parser.add_argument('--render', type=float, default=0.1)
    parser.add_argument(
        '--win-rate',
        type=float,
        default=0.6,
        help='the expected winning rate: Optimal policy can get 0.7'
    )
    parser.add_argument(
        '--watch',
        default=False,
        action='store_true',
        help='no training, '
        'watch the play of pre-trained models'
    )
    # parser.add_argument(
    #     '--agent-id',
    #     type=int,
    #     default=2,
    #     help='the learned agent plays as the'
    #     ' agent_id-th player. Choices are 1 and 2.'
    # )
    parser.add_argument(
        '--resume-path',
        type=str,
        default='',
        help='the path of agent pth file '
        'for resuming from a pre-trained agent'
    )
    parser.add_argument(
        '--device', type=str, default='cuda' if torch.cuda.is_available() else 'cpu'
    )
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--vf-coef", type=float, default=0.25)
    parser.add_argument("--ent-coef", type=float, default=0.0)
    parser.add_argument("--eps-clip", type=float, default=0.2)
    parser.add_argument("--dual-clip", type=float, default=None)
    parser.add_argument("--value-clip", type=int, default=0)
    parser.add_argument("--norm-adv", type=int, default=0)
    parser.add_argument("--recompute-adv", type=int, default=1)
    parser.add_argument("--repeat-per-collect", type=int, default=10)
    # parser.add_argument("--step-per-collect", type=int, default=2048)
    return parser

def get_args() -> argparse.Namespace:
    parser = get_parser()
    return parser.parse_known_args()[0]

def make_ppo_agent(args, action_space, optim=None):
    # model
    net_a = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )
    actor = ActorProb(
        net_a,
        args.action_shape,
        unbounded=True,
        device=args.device,
    ).to(args.device)
    net_c = Net(
        args.state_shape,
        hidden_sizes=args.hidden_sizes,
        activation=nn.Tanh,
        device=args.device,
    )
    critic = Critic(net_c, device=args.device).to(args.device)
    actor_critic = ActorCritic(actor, critic)

    torch.nn.init.constant_(actor.sigma_param, -0.5)
    for m in actor_critic.modules():
        if isinstance(m, torch.nn.Linear):
            # orthogonal initialization
            torch.nn.init.orthogonal_(m.weight, gain=np.sqrt(2))
            torch.nn.init.zeros_(m.bias)
    # do last policy layer scaling, this will make initial actions have (close to)
    # 0 mean and std, and will help boost performances,
    # see https://arxiv.org/abs/2006.05990, Fig.24 for details
    for m in actor.mu.modules():
        if isinstance(m, torch.nn.Linear):
            torch.nn.init.zeros_(m.bias)
            m.weight.data.copy_(0.01 * m.weight.data)

    optim = torch.optim.Adam(actor_critic.parameters(), lr=args.lr)

    # lr_scheduler = None
    # if args.lr_decay:
    #     # decay learning rate to 0 linearly
    #     max_update_num = np.ceil(args.step_per_epoch / args.step_per_collect) * args.epoch
    #
    #     lr_scheduler = LambdaLR(optim, lr_lambda=lambda epoch: 1 - epoch / max_update_num)

    def dist(*logits):
        return Independent(Normal(*logits), 1)

    policy = PPOPolicy(
        actor,
        critic,
        optim,
        dist,
        discount_factor=args.gamma,
        gae_lambda=args.gae_lambda,
        max_grad_norm=args.max_grad_norm,
        vf_coef=args.vf_coef,
        ent_coef=args.ent_coef,
        # reward_normalization=args.rew_norm,
        action_scaling=True,
        # action_bound_method=args.bound_action_method,
        # lr_scheduler=lr_scheduler,
        action_space=action_space,
        eps_clip=args.eps_clip,
        value_clip=args.value_clip,
        dual_clip=args.dual_clip,
        advantage_normalization=args.norm_adv,
        recompute_advantage=args.recompute_adv,
    )
    return policy

def get_agents(
    args: argparse.Namespace = get_args(),
    agent_1: BasePolicy = None,
    agent_2: BasePolicy = None,
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[BasePolicy, torch.optim.Optimizer, list]:
    env = get_env()
    observation_space = env.observation_space['observation'] if isinstance(
        env.observation_space, gym.spaces.Dict
    ) else env.observation_space
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = env.action_space.shape or env.action_space.n
    # if agent_1 is None:
    #     if args.resume_path:
    #         agent_1.load_state_dict(torch.load(args.resume_path))
    #     agent_1 = make_ppo_agent(args, env.action_space)
    #
    # if agent_2 is None:
    #     if args.opponent_path:
    #         agent_2 = deepcopy(agent_1)
    #         agent_2.load_state_dict(torch.load(args.opponent_path))
    #     else:
    #         agent_2 = make_ppo_agent(args, env.action_space)

    agents = [agent_1, agent_2]

    policy = MultiAgentPolicyManager(agents, env)

    return policy, optim, env.agents

def get_env(render_mode=None, video_dir=None):
    style = "favors_runner"
    if render_mode == "video":
        path = video_dir
        os.makedirs(path, exist_ok=True)
        return PettingZooEnv(VideoRecorder( dense_tag_env(render_mode="rgb_array", num_good=1, num_adversaries=1, num_obstacles=0, continuous_actions=True, style=style), path))
    else:
        return PettingZooEnv(dense_tag_env(render_mode=render_mode, num_good=1, num_adversaries=1, num_obstacles=0, continuous_actions=True, style=style))

def save_agents(taggers, runners, base_log_path):
    path = os.path.join(base_log_path, "models")
    os.makedirs(path, exist_ok=True)
    for index, agent in enumerate(taggers):
        torch.save(agent.state_dict(), os.path.join(path, f"tagger_{index}.pt"))
    for index, agent in enumerate(runners):
        torch.save(agent.state_dict(), os.path.join(path, f"runner_{index}.pt"))




def train_league(
    args: argparse.Namespace = get_args(),
    optim: Optional[torch.optim.Optimizer] = None,
) -> Tuple[List, List, str]:
    # fetch state and action space info
    throw_away_env = get_env()
    observation_space = throw_away_env.observation_space['observation'] if isinstance(
        throw_away_env.observation_space, gym.spaces.Dict
    ) else throw_away_env.observation_space
    args.state_shape = observation_space.shape or observation_space.n
    args.action_shape = throw_away_env.action_space.shape or throw_away_env.action_space.n


    # ======== environment setup =========
    train_envs = DummyVectorEnv([get_env for _ in range(args.training_num)])
    test_envs = DummyVectorEnv([get_env for _ in range(args.test_num)])
    # seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    train_envs.seed(args.seed)
    test_envs.seed(args.seed)

    # generate N taggers and N runners
    N = 10
    taggers = [make_ppo_agent(args, throw_away_env.action_space) for n in range(N)]
    runners = [make_ppo_agent(args, throw_away_env.action_space) for n in range(N)]
    if args.resume_path: # loads previously trained models
        path = args.resume_path
        if path[-6:] != "models":
            path = os.path.join(path, "models")
        for index, agent in enumerate(taggers):
            agent.load_state_dict(torch.load(os.path.join(path, f"tagger_{index}.pt")))
        for index, agent in enumerate(runners):
            agent.load_state_dict(torch.load(os.path.join(path, f"runner_{index}.pt")))

    # ======== tensorboard logging setup =========
    current_date_time_string = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    base_log_path = os.path.join(args.logdir, 'tag', 'league', current_date_time_string)
    tagger_writers = [SummaryWriter(os.path.join(base_log_path, f"tagger_{index}")) for index in range(len(taggers))]
    runner_writers = [SummaryWriter(os.path.join(base_log_path, f"runner_{index}")) for index in range(len(runners))]

    # reusable variables
    buffer = VectorReplayBuffer(args.buffer_size, len(train_envs))

    for league_step in trange(args.epoch):
        random_permutation_of_taggers = np.random.permutation(len(taggers))
        random_permutation_of_runners = np.random.permutation(len(runners))

        for tagger, runner in zip(random_permutation_of_taggers, random_permutation_of_runners):

            # ======== agent setup =========
            policy, optim, agents = get_agents(
                args, agent_1=taggers[tagger], agent_2=runners[runner], optim=optim
            )

            # ======== collector setup =========
            buffer.reset(keep_statistics=False)
            train_collector = Collector(
                policy,
                train_envs,
                buffer,
                exploration_noise=True
            )
            # policy.set_eps(1)
            train_collector.collect(n_step=args.batch_size * args.training_num)
            test_collector = Collector(policy, test_envs, exploration_noise=True)

            def reward_metric(rews): # this is only for tensorboard purposes. We want to see the runners score to see if they are evading or not.
                return rews[:, 0]

            # trainer
            result = OnpolicyTrainer(
                policy,
                train_collector,
                test_collector,
                max_epoch=1,
                step_per_epoch=args.step_per_epoch,
                repeat_per_collect=args.repeat_per_collect,
                # args.step_per_collect,
                episode_per_test=args.test_num,
                batch_size=args.batch_size,
                # train_fn=train_fn,
                # test_fn=test_fn,
                step_per_collect=args.step_per_collect,
                # stop_fn=stop_fn,
                # save_best_fn=save_best_fn,
                update_per_step=args.update_per_step,
                # logger=logger, # we dont want to log each 1v1 match
                test_in_train=False,
                reward_metric=reward_metric,
                show_progress=False,
                verbose=False,
            ).run()

            result = test_collector.collect(n_episode=10, render=args.render)
            mean_tagger_score = result["rews"][:, 0].mean()
            mean_runner_score = result["rews"][:, 1].mean()

            # update agents
            taggers[tagger] = policy.policies[agents[0]]
            runners[runner] = policy.policies[agents[1]]

            # log results
            tagger_writers[tagger].add_scalar("test/tagger_reward", mean_tagger_score, league_step)
            runner_writers[runner].add_scalar("test/runner_reward", mean_runner_score, league_step)

            # save models
            save_agents(taggers, runners, base_log_path)
    return taggers, runners, base_log_path

    # ======== a test function that tests a pre-trained agent ======
def watch(
    args: argparse.Namespace = get_args(),
    taggers: List[BasePolicy] = None,
    runners: List[BasePolicy] = None,
    video_dir: Optional[str] = None,
) -> None:
    video_dir = os.path.join(video_dir, "videos")
    for tagger_index in range(len(taggers)):
        for runner_index in range(len(runners)):
            special_dir = os.path.join(video_dir, f"tagger_{tagger_index}_runner_{runner_index}")
            os.makedirs(special_dir, exist_ok=True)
            env = get_env(render_mode="video", video_dir=special_dir)
            env = DummyVectorEnv([lambda: env])
            policy, optim, agents = get_agents(
                args, agent_1=taggers[tagger_index], agent_2=runners[runner_index], optim=None
            )
            policy.eval()
            # policy.policies[agents[args.agent_id - 1]].set_eps(args.eps_test)
            collector = Collector(policy, env, exploration_noise=True)
            result = collector.collect(n_episode=10, render=args.render)
            rews, lens = result["rews"], result["lens"]
            # print(f"Final reward: {rews}")
            # print(f"Final reward: {rews[:, 0].mean()}, length: {lens.mean()}")
            # print(f"Final reward: {rews[:, 1].mean()}, length: {lens.mean()}")

# train the agent and watch its performance in a match!
args = get_args()
taggers, runners, base_log_path = train_league(args)
watch(args, taggers, runners,video_dir=base_log_path)
