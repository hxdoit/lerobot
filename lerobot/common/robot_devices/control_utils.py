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

########################################################################################
# Utilities
########################################################################################


import logging
import time
import traceback
from contextlib import nullcontext
from copy import copy
from functools import cache
from typing import List, Type

import cv2
import torch
from deepdiff import DeepDiff
from sympy.codegen.ast import float32
from termcolor import colored
from torch import nn, device
from torchvision import transforms

from lerobot.common.datasets.image_writer import safe_stop_image_writer
from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
from lerobot.common.datasets.utils import get_features_from_robot
from lerobot.common.policies.pretrained import PreTrainedPolicy
from lerobot.common.robot_devices.robots.utils import Robot
from lerobot.common.robot_devices.utils import busy_wait
from lerobot.common.utils.utils import get_safe_torch_device, has_method

from math import pi
import apriltag
import numpy as np
from scipy.spatial import ConvexHull
from shapely.geometry import Polygon

def log_control_info(robot: Robot, dt_s, episode_index=None, frame_index=None, fps=None):
    log_items = []
    if episode_index is not None:
        log_items.append(f"ep:{episode_index}")
    if frame_index is not None:
        log_items.append(f"frame:{frame_index}")

    def log_dt(shortname, dt_val_s):
        nonlocal log_items, fps
        info_str = f"{shortname}:{dt_val_s * 1000:5.2f} ({1 / dt_val_s:3.1f}hz)"
        if fps is not None:
            actual_fps = 1 / dt_val_s
            if actual_fps < fps - 1:
                info_str = colored(info_str, "yellow")
        log_items.append(info_str)

    # total step time displayed in milliseconds and its frequency
    log_dt("dt", dt_s)

    # TODO(aliberts): move robot-specific logs logic in robot.print_logs()
    if not robot.robot_type.startswith("stretch"):
        for name in robot.leader_arms:
            key = f"read_leader_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRlead", robot.logs[key])

        for name in robot.follower_arms:
            key = f"write_follower_{name}_goal_pos_dt_s"
            if key in robot.logs:
                log_dt("dtWfoll", robot.logs[key])

            key = f"read_follower_{name}_pos_dt_s"
            if key in robot.logs:
                log_dt("dtRfoll", robot.logs[key])

        for name in robot.cameras:
            key = f"read_camera_{name}_dt_s"
            if key in robot.logs:
                log_dt(f"dtR{name}", robot.logs[key])

    info_str = " ".join(log_items)
    logging.info(info_str)


@cache
def is_headless():
    """Detects if python is running without a monitor."""
    try:
        import pynput  # noqa

        return False
    except Exception:
        print(
            "Error trying to import pynput. Switching to headless mode. "
            "As a result, the video stream from the cameras won't be shown, "
            "and you won't be able to change the control flow with keyboards. "
            "For more info, see traceback below.\n"
        )
        traceback.print_exc()
        print()
        return True

def predict_action(observation, policy, device, use_amp):
    observation = copy(observation)
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type) if device.type == "cuda" and use_amp else nullcontext(),
    ):
        # Convert to pytorch format: channel first and float32 in [0,1] with batch dimension
        resize_fun = transforms.Resize((480, 640))
        for name in observation:
            if "image" in name:
                observation[name] = observation[name].permute(2, 0, 1).contiguous()
                observation[name] = resize_fun(observation[name])
                observation[name] = observation[name].type(torch.float32) / 255
            observation[name] = observation[name].unsqueeze(0)
            observation[name] = observation[name].to(device)

        # Compute the next action with the policy
        # based on the current observation
        action = policy(observation)

        # Remove batch dimension
        action = action.squeeze(0)

        # Move to cpu, if not already the case
        action = action.to("cpu")

    return action


def init_keyboard_listener():
    # Allow to exit early while recording an episode or resetting the environment,
    # by tapping the right arrow key '->'. This might require a sudo permission
    # to allow your terminal to monitor keyboard events.
    events = {}
    events["exit_early"] = False
    events["rerecord_episode"] = False
    events["stop_recording"] = False

    if is_headless():
        logging.warning(
            "Headless environment detected. On-screen cameras display and keyboard inputs will not be available."
        )
        listener = None
        return listener, events

    # Only import pynput if not in a headless environment
    from pynput import keyboard

    def on_press(key):
        try:
            if key == keyboard.Key.right:
                print("Right arrow key pressed. Exiting loop...")
                events["exit_early"] = True
            elif key == keyboard.Key.left:
                print("Left arrow key pressed. Exiting loop and rerecord the last episode...")
                events["rerecord_episode"] = True
                events["exit_early"] = True
            elif key == keyboard.Key.esc:
                print("Escape key pressed. Stopping data recording...")
                events["stop_recording"] = True
                events["exit_early"] = True
        except Exception as e:
            print(f"Error handling key press: {e}")

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    return listener, events


def warmup_record(
    robot,
    events,
    enable_teleoperation,
    warmup_time_s,
    display_cameras,
    fps,
):
    control_loop(
        robot=robot,
        control_time_s=warmup_time_s,
        display_cameras=display_cameras,
        events=events,
        fps=fps,
        teleoperate=enable_teleoperation,
    )


def record_episode(
    robot,
    dataset,
    events,
    episode_time_s,
    display_cameras,
    policy,
    fps,
    single_task,
):
    control_loop(
        robot=robot,
        control_time_s=episode_time_s,
        display_cameras=display_cameras,
        dataset=dataset,
        events=events,
        policy=policy,
        fps=fps,
        teleoperate=policy is None,
        single_task=single_task,
    )

def rotationMatrixToEulerAngles(R):
    sy = np.sqrt(R[0,0] * R[0,0] + R[1,0] * R[1,0])
    singular = sy < 1e-6
    if not singular :
        x = np.arctan2(R[2,1] , R[2,2])
        y = np.arctan2(-R[2,0], sy)
        z = np.arctan2(R[1,0], R[0,0])
    else :
        x = np.arctan2(-R[1,2], R[1,1])
        y = np.arctan2(-R[2,0], sy)
        z = 0
    return np.array([x, y, z])

def compute_transform(T_A_to_cam, T_B_to_cam):
    # 计算cam→B的逆变换
    T_cam_to_B = np.linalg.inv(T_B_to_cam[0])
    # 组合变换矩阵
    T_A_to_B = T_cam_to_B @ T_A_to_cam[0]
    return T_A_to_B

def draw_image_points(frame, img_pts):
    img_pts = img_pts.reshape(-1, 2).astype(int)
    # 绘制前后面
    for i in range(4):
        cv2.line(frame, tuple(img_pts[i]), tuple(img_pts[(i + 1) % 4]), (0, 255, 0), 1)
        cv2.line(frame, tuple(img_pts[i + 4]), tuple(img_pts[(i + 1) % 4 + 4]), (0, 0, 255), 1)
        cv2.line(frame, tuple(img_pts[i]), tuple(img_pts[i + 4]), (255, 0, 0), 1)

def compute_iou(poly1_coords, poly2_coords):
    # 创建多边形对象
    poly1 = Polygon(poly1_coords)
    poly2 = Polygon(poly2_coords)
    # 计算交集面积
    intersection = poly1.intersection(poly2)
    area_intersection = intersection.area
    # 计算各自面积
    area_poly1 = poly1.area
    area_poly2 = poly2.area
    # 计算并集面积
    area_union = area_poly1 + area_poly2 - area_intersection
    # 防止除以零
    if area_union == 0:
        return 0.0
    iou = area_intersection / area_union
    return iou

def get_convex_hull(points):
    """计算凸包顶点并按顺时针排序"""
    yz_points = points[:, 1:3] # y,z平面
    hull = ConvexHull(yz_points)
    ordered_points = yz_points[hull.vertices]
    return ordered_points

def draw_text(frame, text, height_coord, show_text):
    if show_text:
        cv2.putText(frame, text, (50, height_coord), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 1,
                cv2.LINE_AA)

@safe_stop_image_writer
def control_loop(
    robot,
    control_time_s=None,
    teleoperate=False,
    display_cameras=False,
    dataset: LeRobotDataset | None = None,
    events=None,
    policy: PreTrainedPolicy = None,
    fps: int | None = None,
    single_task: str | None = None,
):
    # TODO(rcadene): Add option to record logs
    if not robot.is_connected:
        robot.connect()

    if events is None:
        events = {"exit_early": False}

    if control_time_s is None:
        control_time_s = float("inf")

    if teleoperate and policy is not None:
        raise ValueError("When `teleoperate` is True, `policy` should be None.")

    if dataset is not None and single_task is None:
        raise ValueError("You need to provide a task as argument in `single_task`.")

    if dataset is not None and fps is not None and dataset.fps != fps:
        raise ValueError(f"The dataset fps should be equal to requested fps ({dataset['fps']} != {fps}).")

    camera_name = 'desktop'
    tag_obj_id = 11
    tag_size = 0.017  # m
    # 目标物体的size
    obj_width = 0.03
    obj_height = 0.03
    obj_length = 0.056

    show_obj_tag_cube = True
    show_grasp_tag_cube = True
    show_text = True

    virtual_obj_3d_corners = np.array([
        [-obj_width / 2, -obj_length / 2, -obj_height / 2],  # 左下前
        [obj_width / 2, -obj_length / 2, -obj_height / 2],  # 右下前
        [obj_width / 2, obj_length / 2, -obj_height / 2],  # 右上前
        [-obj_width / 2, obj_length / 2, -obj_height / 2],  # 左上前
        [-obj_width / 2, -obj_length / 2, obj_height / 2],  # 左下后
        [obj_width / 2, -obj_length / 2, obj_height / 2],  # 右下后
        [obj_width / 2, obj_length / 2, obj_height / 2],  # 右上后
        [-obj_width / 2, obj_length / 2, obj_height / 2]  # 左上后
    ])

    if camera_name == 'desktop':
        print('desktop')
        K = np.array([[1592.776294, 0.000000, 655.148278],
                      [0.000000, 1595.337130, 402.386737],
                      [0.000000, 0.000000, 1.000000]])
        distCoeffs = np.array([0.089836, 0.058097, 0.029668, 0.028918, 0.000000])
    else:
        print('other')
        K = np.array([[671.907962, 0.000000, 639.211904],
                      [0.000000, 674.667340, 360.929253],
                      [0.000000, 0.000000, 1.000000]])
        distCoeffs = np.array([0.109473, -0.127263, 0.000050, 0.003589, 0.000000])
    cameraparam = [K[0, 0], K[1, 1], K[0, 2], K[1, 2]]
    # 不用效果更好
    distCoeffs = None

    # 设置AprilTag检测器的参数
    options = apriltag.DetectorOptions(families='tag36h11', border=0)
    # 初始化AprilTag检测器
    at_detector = apriltag.Detector(options)


    timestamp = 0
    start_episode_t = time.perf_counter()
    while timestamp < control_time_s:
        start_loop_t = time.perf_counter()

        if teleoperate:
            observation, action = robot.teleop_step(record_data=True)
        else:
            observation = robot.capture_observation()

        laptop_frame = observation['observation.images.laptop'].numpy()
        gray = cv2.cvtColor(laptop_frame, cv2.COLOR_BGR2GRAY)
        # 检测AprilTag
        tags = at_detector.detect(gray)
        tag_obj_result = None
        for tag in tags:
            if tag.tag_id == tag_obj_id:
                tag_obj_result = at_detector.detection_pose(tag, cameraparam, tag_size=tag_size)
                if show_obj_tag_cube:
                    img_pts, _ = cv2.projectPoints(
                        virtual_obj_3d_corners, tag_obj_result[0][:3, :3], tag_obj_result[0][:3, 3], K, distCoeffs=distCoeffs
                    )
                    draw_image_points(laptop_frame, img_pts)
        total_reward = 0.0
        trans_between_2tags = np.array([0.0, 0.0, 0.0])
        eulerangles = np.array([0.0, 0.0, 0.0])
        for tag in tags:
            if tag.tag_id == tag_obj_id:
                continue
            result = at_detector.detection_pose(tag, cameraparam, tag_size=tag_size)
            if show_grasp_tag_cube:
                img_pts, _ = cv2.projectPoints(virtual_obj_3d_corners, result[0][:3, :3], result[0][:3, 3], K,
                                               distCoeffs=distCoeffs)
                draw_image_points(laptop_frame, img_pts)

            if tag_obj_result is not None:
                tag_grasp_2_obj = compute_transform(result, tag_obj_result)
                rot_between_2tags = tag_grasp_2_obj[:3, :3]
                trans_between_2tags = tag_grasp_2_obj[:3, 3]
                distance_between_2tags = np.sqrt(np.sum(trans_between_2tags ** 2))
                image_text = "Tag ID: %d, distance to tag_obj: %.2f" % (tag.tag_id, distance_between_2tags)
                draw_text(laptop_frame, image_text, 50, show_text)
                image_text = "coord in tag_obj: %s" % ["%.2f" % item for item in trans_between_2tags]
                draw_text(laptop_frame, image_text, 100, show_text)
                eulerangles = rotationMatrixToEulerAngles(rot_between_2tags) * 180.0 / pi
                image_text = "euler in tag_obj: %s" % ["%.2f" % item for item in eulerangles]
                draw_text(laptop_frame, image_text, 150, show_text)
                temp = (rot_between_2tags @ virtual_obj_3d_corners.T).T + trans_between_2tags

                iou = compute_iou(get_convex_hull(temp), get_convex_hull(virtual_obj_3d_corners))
                image_text = "iou: %.2f" % iou
                draw_text(laptop_frame, image_text, 200, show_text)

                distance_between_2tags *= 100
                distance_between_2tags = (9 - distance_between_2tags) + 9 if distance_between_2tags < 9 else distance_between_2tags
                distance_reward = 0.6 * np.exp(-0.12 * (distance_between_2tags - 9))
                euler_sum = np.sum(np.abs(eulerangles))
                euler_reward = 0.2 * (
                            1 - euler_sum / 45) if distance_between_2tags < 13 and euler_sum < 45 else 0
                iou_reward = 0.2 * iou
                total_reward = distance_reward + euler_reward + iou_reward
                image_text = ("reward: %.2f, dis_reward: %.2f, eul_reward: %.2f, iou_reward: %.2f"
                                % (total_reward, distance_reward, euler_reward, iou_reward))
                draw_text(laptop_frame, image_text, 250, show_text)

        if policy is not None:
            # pred_action = predict_action(
            #    observation, policy, get_safe_torch_device(policy.config.device), policy.config.use_amp
            # )
            #observation['desired_goal'] = torch.tensor([-4.57481870e-03,-9.22452551e-02, 5.56919394e-02,-6.21978684e+00,
 #-2.94220078e+00,-8.39413667e-01]).view(1, -1)
            #observation['desired_goal'] = torch.tensor(np.concatenate((trans_between_2tags, eulerangles)), dtype=torch.float32).view(1, -1)
            #observation['desired_goal'][0,0] -= 0.1
            #print(observation['desired_goal'])
            observation['observation.state'] = observation['observation.state'].view(1, -1)
            for name in observation:
                observation[name] = observation[name].to(torch.device('cuda:0'))
            with torch.no_grad():
                prediction = policy(observation)
                noise = prediction.clone().data.normal_(0, 0.2)
                noise = noise.clamp(-0.3, 0.3)
                next_actions = prediction + noise
                #print(prediction)
            # Action can eventually be clipped using `max_relative_target`,
            # so action actually sent is saved in the dataset.
            action = robot.send_action(next_actions.squeeze().cpu())
            # print(pred_action)
            action = {"action": action}

        if dataset is not None:
            #observation.pop('desired_goal')
            for name in observation:
                observation[name] = observation[name].cpu().squeeze()
            frame = {**observation, **action, "task": single_task,
                     'reward': np.array([total_reward], dtype=np.float32)}

            dataset.add_frame(frame)

        if display_cameras and not is_headless():
            image_keys = [key for key in observation if "image" in key]
            for key in image_keys:
                cv2.imshow(key, cv2.cvtColor(observation[key].numpy(), cv2.COLOR_RGB2BGR))
            cv2.waitKey(1)

        if fps is not None:
            dt_s = time.perf_counter() - start_loop_t
            busy_wait(1 / fps - dt_s)

        dt_s = time.perf_counter() - start_loop_t
        log_control_info(robot, dt_s, fps=fps)

        timestamp = time.perf_counter() - start_episode_t
        if events["exit_early"]:
            events["exit_early"] = False
            break


def reset_environment(robot, events, reset_time_s, fps):
    # TODO(rcadene): refactor warmup_record and reset_environment
    if has_method(robot, "teleop_safety_stop"):
        robot.teleop_safety_stop()

    control_loop(
        robot=robot,
        control_time_s=reset_time_s,
        events=events,
        fps=fps,
        teleoperate=True,
    )


def stop_recording(robot, listener, display_cameras):
    robot.disconnect()

    if not is_headless():
        if listener is not None:
            listener.stop()

        if display_cameras:
            cv2.destroyAllWindows()


def sanity_check_dataset_name(repo_id, policy_cfg):
    _, dataset_name = repo_id.split("/")
    # either repo_id doesnt start with "eval_" and there is no policy
    # or repo_id starts with "eval_" and there is a policy

    # Check if dataset_name starts with "eval_" but policy is missing
    if dataset_name.startswith("eval_") and policy_cfg is None:
        raise ValueError(
            f"Your dataset name begins with 'eval_' ({dataset_name}), but no policy is provided ({policy_cfg.type})."
        )

    # Check if dataset_name does not start with "eval_" but policy is provided
    if not dataset_name.startswith("eval_") and policy_cfg is not None:
        raise ValueError(
            f"Your dataset name does not begin with 'eval_' ({dataset_name}), but a policy is provided ({policy_cfg.type})."
        )


def sanity_check_dataset_robot_compatibility(
    dataset: LeRobotDataset, robot: Robot, fps: int, use_videos: bool
) -> None:
    fields = [
        ("robot_type", dataset.meta.robot_type, robot.robot_type),
        ("fps", dataset.fps, fps),
        ("features", dataset.features, get_features_from_robot(robot, use_videos)),
    ]

    mismatches = []
    for field, dataset_value, present_value in fields:
        diff = DeepDiff(dataset_value, present_value, exclude_regex_paths=[r".*\['info'\]$"])
        if diff:
            mismatches.append(f"{field}: expected {present_value}, got {dataset_value}")

    if mismatches:
        raise ValueError(
            "Dataset metadata compatibility check failed with mismatches:\n" + "\n".join(mismatches)
        )
