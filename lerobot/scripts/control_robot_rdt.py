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
from dataclasses import asdict
from pprint import pformat

import numpy as np
import torch
import yaml
from lerobot.scripts.agilex_model import create_model
from PIL import Image as PImage

# from safetensors.torch import load_file, save_file
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.policies.factory import make_policy
from lerobot.common.robot_devices.control_configs import (
    CalibrateControlConfig,
    ControlPipelineConfig,
    RecordControlConfig,
    RemoteRobotConfig,
    ReplayControlConfig,
    TeleoperateControlConfig,
)
from lerobot.common.robot_devices.control_utils import (
    control_loop,
    init_keyboard_listener,
    log_control_info,
    record_episode,
    reset_environment,
    sanity_check_dataset_name,
    sanity_check_dataset_robot_compatibility,
    stop_recording,
    warmup_record,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser
from collections import deque
########################################################################################
# Control modes
########################################################################################
_action_queue = deque([], maxlen=64)

def get_one_action(policy, proprio, images, lang_embeddings):
    global _action_queue
    if len(_action_queue) == 0:
        # actions shaped as [1, 64, 6]
        actions = policy.step(
            proprio=proprio,
            images=images,
            text_embeds=lang_embeddings
        )
        actions = actions.squeeze(0).cpu().numpy()
        a_min = [-0.212890625,
                 0.18896484375,
                 0.14013671875,
                 0.17919921875,
                 -0.6171875,
                 -0.004390166203180949]
        a_min = [x * 180 - 20 for x in a_min]
        a_max = [
            0.4248046875,
            1.0830078125,
            0.994140625,
            0.5654296875,
            0.322265625,
            0.20753512912326388,
        ]
        a_max = [x * 180 + 20 for x in a_max]

        if np.any(actions < a_min) or np.any(actions > a_max):
            #raise ValueError("actions out of range")
            pass

        _action_queue.extend(actions)
    return _action_queue.popleft()

#python lerobot/scripts/control_robot.py   --robot.type=so100   --control.type=record   --control.fps=30   --control.single_task="Put the yellow toy block in a stainless steel bowl."   --control.repo_id=hxdoso/so100_test   --control.tags='["so100","tutorial"]'   --control.warmup_time_s=5   --control.episode_time_s=300   --control.reset_time_s=30   --control.num_episodes=1 --control.resume=true
@safe_disconnect
def record(
    robot: Robot
) -> LeRobotDataset:
    with open("configs/base.yaml", "r") as fp:
        config = yaml.safe_load(fp)

    pretrained_vision_encoder_name_or_path = "google/siglip-so400m-patch14-384"
    policy = create_model(
        args=config,
        dtype=torch.bfloat16,
        #pretrained="robotics-diffusion-transformer/rdt-1b",
        pretrained="/home/ubuntu/Downloads/rdt/RoboticsDiffusionTransformer/101episode_1200_1200_800_800.pytorch.bin",
        pretrained_vision_encoder_name_or_path=pretrained_vision_encoder_name_or_path,
        control_frequency=30,
    )
    fps = 30
    if not robot.is_connected:
        robot.connect()
    control_time_s = float("inf")
    timestamp = 0
    start_episode_t = time.perf_counter()
    last_img_high = None
    last_img_right = None
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        observation = robot.capture_observation()

        lang_dict = torch.load("/home/ubuntu/Downloads/rdt/RoboticsDiffusionTransformer/outs/handover_pan.pt.new")
        # (20, 4096)->(1, 20, 4096)
        lang_embeddings = lang_dict.unsqueeze(0)

        img_high = observation['observation.images.phone']
        img_right = observation['observation.images.laptop']
        image_arrs = [
            last_img_high,
            last_img_right,
            None,
            img_high,
            img_right,
            None
        ]
        last_img_high = img_high
        last_img_right = img_right
        images = [PImage.fromarray(arr.numpy()) if arr is not None else None
                  for arr in image_arrs]

        proprio = observation['observation.state']
        # unsqueeze to [1, 6]
        proprio = proprio.unsqueeze(0)

        action = get_one_action(policy, proprio, images, lang_embeddings)

        robot.send_action(torch.tensor(action))

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t



@parser.wrap()
def control_robot(cfg: ControlPipelineConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    robot = make_robot_from_config(cfg.robot)
    record(robot)

    if robot.is_connected:
        # Disconnect manually to avoid a "Core dump" during process
        # termination due to camera threads not properly exiting.
        robot.disconnect()


if __name__ == "__main__":
    control_robot()
