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
import os

from pprint import pformat


import cv2
import h5py
import numpy as np
import torch

from lerobot.common.datasets.factory import make_dataset

from lerobot.common.utils.utils import (
    init_logging,
)
from lerobot.configs import parser
from lerobot.configs.train import TrainPipelineConfig


def images_encoding(imgs):
    encode_data = []
    padded_data = []
    max_len = 0
    for i in range(len(imgs)):
        temp = imgs[i].to(torch.uint8)
        #rgb->brg
        temp = torch.cat((temp[2:3,:], temp[1:2,:], temp[0:1,:]), dim=0)
        #chw->hwc
        success, encoded_image = cv2.imencode('.png', temp.permute(1,2,0).numpy())
        jpeg_data = encoded_image.tobytes()
        encode_data.append(jpeg_data)
        # encode_data.append(np.frombuffer(jpeg_data, dtype='S1'))
        max_len = max(max_len, len(jpeg_data))
    # padding
    for i in range(len(imgs)):
        padded_data.append(encode_data[i].ljust(max_len, b'\0'))
    return encode_data, max_len

@parser.wrap()
def train(cfg: TrainPipelineConfig):
    cfg.validate()
    logging.info(pformat(cfg.to_dict()))

    logging.info("Creating dataset")
    dataset = make_dataset(cfg)

    episode_from, episode_to = dataset.episode_data_index['from'], dataset.episode_data_index['to']
    idx = 0
    for start in episode_from:
        start = start.item()
        end = episode_to[idx].item()
        qpos = []
        actions = []
        cam_high = []
        cam_low = []
        while (start < end):
            item = dataset[start]
            cam_low.append(item['observation.images.laptop'] * 255)
            cam_high.append(item['observation.images.phone'] * 255)
            actions.append(item['action'][0].numpy())
            qpos.append(item['observation.state'].numpy())
            start += 1
        hdf5path = os.path.join('/home/ubuntu/Downloads/lerobot/hdf5_right_left/', f'episode_{idx}.hdf5')
        print(hdf5path)
        with h5py.File(hdf5path, 'w') as f:
            f.create_dataset('action', data=np.array(actions))
            obs = f.create_group('observations')
            image = obs.create_group('images')
            obs.create_dataset('qpos', data=qpos)
            # 图像编码后按顺序存储

            cam_high_enc, len_high = images_encoding(cam_high)
            cam_right_wrist_enc, len_right = images_encoding(cam_low)
            image.create_dataset('cam_high', data=cam_high_enc, dtype=f'S{len_high}')
            image.create_dataset('cam_right_wrist', data=cam_right_wrist_enc, dtype=f'S{len_right}')
        idx += 1




if __name__ == "__main__":
    init_logging()
    train()
