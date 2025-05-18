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
import json
import logging
import os
import time
import traceback
from dataclasses import asdict
from pathlib import Path
from pprint import pformat
from typing import Union, Optional

import PIL
import numpy as np
import torch
from PIL import Image as PImage
from transformers import AutoProcessor, AutoModelForVision2Seq


from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

from lerobot.common.robot_devices.control_configs import (
    ControlPipelineConfig,
)
from lerobot.common.robot_devices.control_utils import (
    log_control_info,
)
from lerobot.common.robot_devices.robots.utils import Robot, make_robot_from_config
from lerobot.common.robot_devices.utils import busy_wait, safe_disconnect
from lerobot.common.utils.utils import has_method, init_logging, log_say
from lerobot.configs import parser

########################################################################################
# Control modes
########################################################################################
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_openvla_prompt(instruction: str, openvla_path: Union[str, Path]) -> str:
    if "v01" in openvla_path:
        return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"
    else:
        return f"In: What action should the robot take to {instruction.lower()}?\nOut:"


# === Server Interface ===
class OpenVLAServer:
    def __init__(self, openvla_path: Union[str, Path], attn_implementation: Optional[str] = "flash_attention_2") -> Path:
        """
        A simple server for OpenVLA models; exposes `/act` to predict an action for a given image + instruction.
            => Takes in {"image": np.ndarray, "instruction": str, "unnorm_key": Optional[str]}
            => Returns  {"action": np.ndarray}
        """
        self.openvla_path, self.attn_implementation = openvla_path, attn_implementation
        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # Load VLA Model using HF AutoClasses
        self.processor = AutoProcessor.from_pretrained(self.openvla_path, trust_remote_code=True)
        self.vla = AutoModelForVision2Seq.from_pretrained(
            self.openvla_path,
            #attn_implementation=attn_implementation,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            trust_remote_code=True  ,
        ).to(self.device)

        # [Hacky] Load Dataset Statistics from Disk (if passing a path to a fine-tuned model)
        if os.path.isdir(self.openvla_path):
            with open(Path(self.openvla_path) / "dataset_statistics.json", "r") as f:
                self.vla.norm_stats = json.load(f)

    def predict_action(self, lang, image, state):
        try:

            # Run VLA Inference
            prompt = get_openvla_prompt(lang, self.openvla_path)
            inputs = self.processor(prompt, image).to(self.device, dtype=torch.bfloat16)
            actions = self.vla.predict_action(**inputs, unnorm_key='lerobot_dataset', do_sample=False)
            print(actions)
            actions = actions[:6] + state
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
            return actions
        except:  # noqa: E722
            logging.error(traceback.format_exc())
            logging.warning(
                "Your request threw an error; make sure your request complies with the expected format:\n"
                "{'image': np.ndarray, 'instruction': str}\n"
                "You can optionally an `unnorm_key: str` to specific the dataset statistics you want to use for "
                "de-normalizing the output actions."
            )
            return "error"




#python lerobot/scripts/control_robot.py   --robot.type=so100   --control.type=record   --control.fps=30   --control.single_task="Put the yellow toy block in a stainless steel bowl."   --control.repo_id=hxdoso/so100_test   --control.tags='["so100","tutorial"]'   --control.warmup_time_s=5   --control.episode_time_s=300   --control.reset_time_s=30   --control.num_episodes=1 --control.resume=true
@safe_disconnect
def record(
    robot: Robot
) -> LeRobotDataset:
    server = OpenVLAServer("/home/ubuntu/Downloads/lerobot/iter1600")

    fps = 5
    if not robot.is_connected:
        robot.connect()
    control_time_s = float("inf")
    timestamp = 0
    start_episode_t = time.perf_counter()

    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        observation = robot.capture_observation()

        img_right = observation['observation.images.phone']
        img = PImage.fromarray(img_right.numpy())
        lang = 'put the yellow toy block in a stainless steel bowl'
        action = server.predict_action(lang, img, observation['observation.state'].numpy())
        print(action)
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
