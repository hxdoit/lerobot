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
import time
from contextlib import nullcontext
from itertools import zip_longest
from pprint import pformat
from typing import Any, Iterable

import safetensors
import torch
from termcolor import colored
from torch.amp import GradScaler
from torch.optim import Optimizer

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


class Td3Policy(torch.nn.Module):
    def __init__(self, cfg, dataset, *args, **kwargs):
        super().__init__(*args, **kwargs)
        device = get_safe_torch_device(cfg.policy.device, log=True)
        self.actor = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
        )
        safetensors.torch.load_model(self.actor,
                                      'outputs/train/act_so100_rl2/checkpoints/060000/pretrained_model/actor/model.safetensors',
                                      'cuda:0', False)
        self.actor_target = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
        )
        self.actor_target.model.load_state_dict(self.actor.model.state_dict())
        self.actor_target.train(False)
        for p in (self.actor_target.parameters()):
            p.requires_grad = False
        self.actor_optimizer, self.actor_lr_scheduler = make_optimizer_and_scheduler(cfg, self.actor)
        self.actor_grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

        self.critic = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            type='critic'
        )
        #polyak_update(self.actor.feature_extract_params(), self.critic.feature_extract_params(), 1)
        safetensors.torch.load_model(self.critic,
                                     'outputs/train/act_so100_rl2/checkpoints/060000/pretrained_model/critic/model.safetensors',
                                     'cuda:0', False)
        #safetensors.torch.load_model(self.actor,
        #                             'outputs/train/act_so100_rl/checkpoints/080000/pretrained_model/model.safetensors',
        #                             'cuda:0', False)
        #self.critic.copy_from_actor(self.actor)
        self.critic_target = make_policy(
            cfg=cfg.policy,
            ds_meta=dataset.meta,
            type='critic'
        )
        #self.critic_target.copy_from_actor(self.actor_target)
        self.critic_target.model.load_state_dict(self.critic.model.state_dict())
        self.critic_target.train(False)
        for p in (self.critic_target.parameters()):
            p.requires_grad = False
        self.critic_optimizer, self.critic_lr_scheduler = make_optimizer_and_scheduler(cfg, self.critic)
        self.critic_grad_scaler = GradScaler(device.type, enabled=cfg.policy.use_amp)

    def save_pretrained(self, pretrained_dir):
        self.actor.save_pretrained(pretrained_dir / 'actor')
        self.critic.save_pretrained(pretrained_dir / 'critic')

import torch.nn.functional as F

def update_policy(
    cur_step,
    train_metrics: MetricsTracker,
    policy,
    batch: Any,
    grad_clip_norm: float,
    use_amp: bool = False,
    lock=None,
) -> tuple[MetricsTracker, dict]:
    cur_batch = {
        'observation.images.laptop': batch['observation.images.laptop'][:, 0],
        'observation.images.phone': batch['observation.images.phone'][:, 0],
        'observation.state': batch['observation.state'][:, 0],
        'action_is_pad': batch['action_is_pad'],
        'reward_is_pad': batch['reward_is_pad'],
        'action': batch['action'],
        'reward': batch['reward'],
    }
    next_batch = {
        'observation.images.laptop': batch['observation.images.laptop'][:, 1],
        'observation.images.phone': batch['observation.images.phone'][:, 1],
        'observation.state': batch['observation.state'][:, 1],
        'action_is_pad': batch['action_is_pad'],
        'reward_is_pad': batch['reward_is_pad'],
        'action': batch['action'],
        'reward': batch['reward']
    }
    gamma = 0.99
    tau = 0.005
    start_time = time.perf_counter()
    with torch.no_grad():
        _, _, next_actions = policy.actor_target(next_batch)

        # Compute the next Q-values: min over all critics targets
        next_q_values = torch.cat(policy.critic_target.q_forward(next_batch, next_actions), dim=1)
        next_q_values, _ = torch.min(next_q_values, dim=1, keepdim=True)
        cur_reward = (cur_batch['reward'][:, 1:] - cur_batch['reward'][:, :25])
        target_q_values = torch.sum(cur_reward, dim=1, keepdim=True) + gamma * next_q_values

    # Get current Q-values estimates for each critic network
    current_q_values = policy.critic.q_forward(cur_batch, cur_batch['action'])
    train_metrics.critic_q0 = torch.mean(current_q_values[0]).cpu().item()
    train_metrics.critic_q1 = torch.mean(current_q_values[1]).cpu().item()

    # Compute critic loss
    critic_loss = sum([F.mse_loss(current_q, target_q_values) for current_q in current_q_values])

    # Optimize the critics
    policy.critic_optimizer.zero_grad()
    #critic_loss.backward()
    #policy.critic_optimizer.step()

    policy.critic_grad_scaler.scale(critic_loss).backward()

    # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
    policy.critic_grad_scaler.unscale_(policy.critic_optimizer)
    grad_norm_critic = torch.nn.utils.clip_grad_norm_(
        policy.critic.parameters(),
        grad_clip_norm,
        error_if_nonfinite=False,
    )

    policy.critic_grad_scaler.step(policy.critic_optimizer)
    # Updates the scale for next iteration.
    policy.critic_grad_scaler.update()

    # Delayed policy updates
    #actor_loss = torch.tensor([0.0]).to('cuda:0')
    if cur_step % 2 == 0:
        # Compute actor loss
        _,_,next_actions = policy.actor(cur_batch)
        actor_loss = -policy.critic.q1_forward(cur_batch, next_actions).mean()

        # Optimize the actor
        policy.actor_optimizer.zero_grad()
        #actor_loss.backward()
        #policy.actor_optimizer.step()

        policy.actor_grad_scaler.scale(actor_loss).backward()

        # Unscale the gradient of the optimizer's assigned params in-place **prior to gradient clipping**.
        policy.actor_grad_scaler.unscale_(policy.actor_optimizer)
        grad_norm_actor = torch.nn.utils.clip_grad_norm_(
            policy.actor.parameters(),
            grad_clip_norm,
            error_if_nonfinite=False,
        )
        train_metrics.grad_norm_actor = grad_norm_actor.cpu().item()
        train_metrics.actor_loss = actor_loss.cpu().item()

        policy.actor_grad_scaler.step(policy.actor_optimizer)
        # Updates the scale for next iteration.
        policy.actor_grad_scaler.update()

        polyak_update(policy.actor.feature_extract_params(), policy.critic.feature_extract_params(), 1)
        polyak_update(policy.critic.parameters(), policy.critic_target.parameters(), tau)
        polyak_update(policy.actor.parameters(), policy.actor_target.parameters(), tau)

    #policy.train()




    train_metrics.critic_loss = critic_loss.cpu().item()
    train_metrics.grad_norm_critic = grad_norm_critic.cpu().item()
    train_metrics.update_s = time.perf_counter() - start_time
    return train_metrics


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

    #if cfg.resume:
    #    step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)

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

    # create dataloader for offline training
    if hasattr(cfg.policy, "drop_n_last_frames"):
        shuffle = False
        sampler = EpisodeAwareSampler(
            dataset.episode_data_index,
            drop_n_last_frames=cfg.policy.drop_n_last_frames,
            shuffle=True,
        )
    else:
        shuffle = True
        sampler = None

    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=shuffle,
        sampler=sampler,
        pin_memory=device.type != "cpu",
        drop_last=False,
    )
    dl_iter = cycle(dataloader)

    #td3policy.train()

    train_metrics = {
        "actor_loss": AverageMeter("actor_loss", ":.4f"),
        "critic_loss": AverageMeter("critic_loss", ":.4f"),
        "critic_q0": AverageMeter("critic_q0", ":.4f"),
        "critic_q1": AverageMeter("critic_q1", ":.4f"),
        "grad_norm_actor": AverageMeter("grad_norm_actor", ":.3f"),
        "grad_norm_critic": AverageMeter("grad_norm_critic", ":.3f"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    train_tracker = MetricsTracker(
        cfg.batch_size, dataset.num_frames, dataset.num_episodes, train_metrics, initial_step=step
    )

    logging.info("Start offline training on a fixed dataset")
    for cur_step in range(step, cfg.steps):
        start_time = time.perf_counter()
        batch = next(dl_iter)
        train_tracker.dataloading_s = time.perf_counter() - start_time

        for key in batch:
            if isinstance(batch[key], torch.Tensor):
                batch[key] = batch[key].to(device, non_blocking=True)

        train_tracker = update_policy(
            cur_step,
            train_tracker,
            td3policy,
            batch,
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
            save_td3_checkpoint(checkpoint_dir, step, cfg, td3policy)


    logging.info("End of training")


if __name__ == "__main__":
    init_logging()
    train()
