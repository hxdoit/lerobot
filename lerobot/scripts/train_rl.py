#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import logging
import threading
import time
from collections import deque
from contextlib import nullcontext
from itertools import zip_longest
from multiprocessing import Process
from os import mkdir
from pprint import pformat
import random
from typing import Any, Iterable, List, Type
import torch as th
import safetensors
import torch
from termcolor import colored
from torch.amp import GradScaler
import torch.optim as optim
from torch.optim import Optimizer
from torch import nn
import torch.nn.functional as F
from lerobot.common.datasets.factory import make_dataset
from lerobot.common.datasets.sampler import EpisodeAwareSampler
from lerobot.common.datasets.utils import cycle
from lerobot.common.envs.factory import make_env
from lerobot.common.optim.factory import make_optimizer_and_scheduler
from lerobot.common.policies.act.modeling_act import ACTPolicy
from lerobot.common.policies.factory import make_policy
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.policies.utils import get_device_from_parameters
from lerobot.common.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.common.utils.random_utils import set_seed
from lerobot.common.utils.train_utils import (
    get_step_checkpoint_dir,
    get_step_identifier,
    load_training_state,
    save_td3_checkpoint,
    update_last_checkpoint,
)
from lerobot.common.utils.utils import (
    format_big_number,
    get_safe_torch_device,
    has_method,
    init_logging,
)
from lerobot.common.utils.wandb_utils import WandBLogger
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig
from lerobot.scripts.eval import eval_policy
import copy
def zip_strict(*iterables: Iterable) -> Iterable:
    r"""
    ``zip()`` function but enforces that iterables are of equal length.
    Raises ``ValueError`` if iterables not of equal length.
    Code inspired by Stackoverflow answer for question #32954486.

    :param \*iterables: iterables to ``zip()``
    """
    # As in Stackoverflow #32954486, use
    # new object for "empty" in case we have
    # Nones in iterable.
    sentinel = object()
    for combo in zip_longest(*iterables, fillvalue=sentinel):
        if sentinel in combo:
            raise ValueError("Iterables have different lengths")
        yield combo

def polyak_update(
    params: Iterable[torch.nn.Parameter],
    target_params: Iterable[torch.nn.Parameter],
    tau: float,
) -> None:
    with torch.no_grad():
        # zip does not raise an exception if length of parameters does not match.
        for param, target_param in zip_strict(params, target_params):
            target_param.data.mul_(1 - tau)
            torch.add(target_param.data, param.data, alpha=tau, out=target_param.data)

def create_mlp(
    input_dim: int,
    output_dim: int,
    net_arch: List[int],
    activation_fn: Type[nn.Module] = nn.ReLU,
    squash_output: bool = False,
):

    if len(net_arch) > 0:
        modules = [nn.Linear(input_dim, net_arch[0]), activation_fn()]
    else:
        modules = []

    for idx in range(len(net_arch) - 1):
        modules.append(nn.Linear(net_arch[idx], net_arch[idx + 1]))
        modules.append(activation_fn())

    if output_dim > 0:
        last_layer_dim = net_arch[-1] if len(net_arch) > 0 else input_dim
        modules.append(nn.Linear(last_layer_dim, output_dim))
    if squash_output:
        modules.append(nn.Tanh())
    return modules

class Actor(nn.Module):
    def __init__(
        self,
    ):
        super(Actor, self).__init__()

        self.net_arch = [400, 300]
        self.features_dim = 6 # 6 angles

        self.action_dim = 6 # 6 angles
        actor_net = create_mlp(self.features_dim, self.action_dim, self.net_arch, squash_output=False)
        self.mu = nn.Sequential(*actor_net)
        self.mean = torch.tensor([-27.4765,  86.3493,  92.4536,  67.2350,   5.4615,  -0.2023]).to('cuda:0')
        self.std = torch.tensor([10.0312, 33.6212, 30.0723,  7.6640, 12.1933,  0.2019]).to('cuda:0')

    def forward(self, obs):
        state = obs['observation.state']
        #state = (state - self.mean) / (self.std + 1e-8)
        actions = self.mu(state)
        #return actions * self.std + self.mean
        return actions

class Critic(nn.Module):
    def __init__(
        self,
    ):
        super().__init__()

        self.action_dim = 6 # 6 angles
        self.features_dim = 6  # 6 angles
        self.net_arch = [400, 300]
        self.n_critics = 2
        self.q_networks = []
        self.mean = torch.tensor([-27.4765, 86.3493, 92.4536, 67.2350, 5.4615, -0.2023]).to('cuda:0')
        self.std = torch.tensor([10.0312, 33.6212, 30.0723, 7.6640, 12.1933, 0.2019]).to('cuda:0')
        for idx in range(self.n_critics):
            q_net = create_mlp(self.features_dim + self.action_dim, 1, self.net_arch)
            q_net = nn.Sequential(*q_net)
            self.add_module(f"qf{idx}", q_net)
            self.q_networks.append(q_net)

    def forward(self, obs, action):
        state = obs['observation.state']
        #state = (state - self.mean) / (self.std + 1e-8)
        #action = (action - self.mean) / (self.std + 1e-8)
        qvalue_input = th.cat([state, action], dim=1)
        return tuple(q_net(qvalue_input) for q_net in self.q_networks)

    def init(self):
        for q_net in self.q_networks:
            for module in q_net.modules():
                if isinstance(module, nn.Linear):
                    torch.nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.constant_(module.bias, 0)


    def q1_forward(self, obs, action):
        state = obs['observation.state']
        #state = (state - self.mean) / (self.std + 1e-8)
        #action = (action - self.mean) / (self.std + 1e-8)
        return self.q_networks[0](th.cat([state, action], dim=1))

class Td3Policy(torch.nn.Module):
    def __init__(self, cfg, dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = get_safe_torch_device(cfg.policy.device, log=True)
        lr = 1e-5
        self.actor = Actor()
        self.actor = torch.load('outputs/train/act_so100_rl6/checkpoints/200000/actor.pth', weights_only=False)
        self.actor_target = Actor()
        self.actor_target.load_state_dict(self.actor.state_dict())
        self.actor_target.train(False)
        for p in (self.actor_target.parameters()):
            p.requires_grad = False
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=lr)

        self.critic = Critic()
        #self.critic.init()
        self.critic = torch.load('outputs/train/act_so100_rl7/checkpoints/002400/critic.pth', weights_only=False)
        self.critic_target = Critic()
        self.critic_target.load_state_dict(self.critic.state_dict())
        self.critic_target.train(False)
        for p in (self.critic_target.parameters()):
            p.requires_grad = False
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=lr)

    def save_pretrained(self, pretrained_dir):
        pretrained_dir.mkdir(parents=True, exist_ok=True)
        torch.save(self.actor, pretrained_dir / 'actor.pth')
        torch.save(self.critic, pretrained_dir / 'critic.pth')


def update_policy(
    cur_step,
    train_metrics: MetricsTracker,
    policy,
    batch_tuple: Any,
    grad_clip_norm: float,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    cur_batch = batch_tuple[0]
    next_batch = batch_tuple[1]
    gamma = 0.99
    tau = 0.005
    start_time = time.perf_counter()

    with torch.no_grad():
        noise = next_batch['action'].clone().data.normal_(0, 0.2)
        noise = noise.clamp(-0.5, 0.5)
        next_actions = policy.actor_target(next_batch) + noise

        # Compute the next Q-values: min over all critics targets
        next_q_values = torch.cat(policy.critic_target(next_batch, next_actions), dim=1)
        next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
        target_q_values = cur_batch['reward'].view(-1, 1) + (1 - cur_batch['done'].view(-1, 1)) * gamma * next_q_values

    # Get current Q-values estimates for each critic network
    current_q_values = policy.critic(cur_batch, cur_batch['action'])
    train_metrics.critic_q0 = torch.mean(current_q_values[0]).cpu().item()
    train_metrics.critic_q1 = torch.mean(current_q_values[1]).cpu().item()

    # Compute critic loss
    critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])
    train_metrics.critic_loss = critic_loss.cpu().item()
    # Optimize the critics
    policy.critic_optimizer.zero_grad()
    critic_loss.backward()
    policy.critic_optimizer.step()

    # Delayed policy updates
    #actor_loss = torch.tensor([0.0]).to('cuda:0')
    if cur_step % 2 == 0:
        # Compute actor loss
        next_actions = policy.actor(cur_batch)
        actor_loss = -policy.critic.q1_forward(cur_batch, next_actions).mean()

        # Optimize the actor
        policy.actor_optimizer.zero_grad()
        actor_loss.backward()
        policy.actor_optimizer.step()

        train_metrics.actor_loss = actor_loss.cpu().item()

        polyak_update(policy.critic.parameters(), policy.critic_target.parameters(), tau)
        polyak_update(policy.actor.parameters(), policy.actor_target.parameters(), tau)


    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics


from multiprocessing import Queue
dq = Queue(maxsize=50)
def process_task(dataset, que):
    random.seed(random.randint(1, 1000))
    count = 0
    while True:
        while que.full():
            time.sleep(0.05)
        batch_tuple = dataset.sample(100)
        que.put(batch_tuple)
        count += 1



@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    if cfg.wandb.enable and cfg.wandb.project:
        wandb_logger = WandBLogger(cfg)
    else:
        wandb_logger = None
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed)

    # Check device is available
    device = get_safe_torch_device(cfg.policy.device, log=True)
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    # Create environment used for evaluating checkpoints during training on simulation data.
    # On real-world data, no need to create an environment as evaluations are done outside train.py,
    # using the eval.py instead, with gym_dora environment and dora-rs.
    eval_env = None
    if cfg.eval_freq > 0 and cfg.env is not None:
        logging.info("Creating env")
        eval_env = make_env(cfg.env, n_envs=cfg.eval.batch_size, use_async_envs=cfg.eval.use_async_envs)

    logging.info("Creating policy")

    td3policy = Td3Policy(cfg, dataset)
    td3policy.to(device)

    step = 0  # number of policy updates (forward + backward + optim)

    num_learnable_params = sum(p.numel() for p in td3policy.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in td3policy.parameters())

    logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
    if cfg.env is not None:
        logging.info(f"{cfg.env.task=}")
    logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")
    logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
    logging.info(f"{dataset.num_episodes=}")
    logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
    logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    global dq
    processnum = 16
    processes = []
    for pidx in range(processnum):
        p = Process(target=process_task, args=(dataset, dq))
        p.start()
        processes.append(p)

    train_metrics = {
        "actor_loss": AverageMeter("actor_loss", ":.4f"),
        "critic_loss": AverageMeter("critic_loss", ":.4f"),
        "critic_q0": AverageMeter("critic_q0", ":.4f"),
        "critic_q1": AverageMeter("critic_q1", ":.4f"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )


    logging.info("Start offline training on a fixed dataset")
    for cur_step in range(step, cfg.steps):
        start_time = time.perf_counter()
        while dq.empty():
            time.sleep(0.05)
        batch_tuple = dq.get()

        train_tracker.dataloading_s = time.perf_counter() - start_time
        for batch in batch_tuple:
            for key in batch:
                if isinstance(batch[key], torch.Tensor):
                    batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker = update_policy(
            cur_step,
            train_tracker,
            td3policy,
            batch_tuple,
            cfg.optimizer.grad_clip_norm,
            use_amp=cfg.policy.use_amp,
        )

        # Note: eval and checkpoint happens *after* the `step`th training update has completed, so we
        # increment `step` here.
        step += 1
        train_tracker.step()
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps
        is_eval_step = cfg.eval_freq > 0 and step % cfg.eval_freq == 0

        if is_log_step:
            logging.info(train_tracker)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            logging.info(f"Checkpoint policy after step {step}")
            checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
            td3policy.save_pretrained(checkpoint_dir)


    logging.info("End of training")
    for data_process in processes:
        data_process.join()


if __name__ == "__main__":
    init_logging()
    train()
