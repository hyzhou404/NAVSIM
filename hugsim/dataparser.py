import cv2
import torch
import numpy as np
from scipy.spatial.transform import Rotation as SCR
import math
from navsim.common.dataclasses import AgentInput, EgoStatus, Cameras, Camera, Lidar

OPENCV2IMU = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])
OPENCV2LIDAR = np.array([[1, 0, 0], [0, 0, 1], [0, -1, 0]])
nusc_cameras = ['CAM_FRONT', 'CAM_FRONT_RIGHT', 'CAM_FRONT_LEFT']
nuplan_cameras = ["cam_f0", "cam_r0", "cam_l0"]

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def get_intrinsic(intrin_dict):
    fovx, fovy = intrin_dict['fovx'], intrin_dict['fovy']
    h, w = intrin_dict['H'], intrin_dict['W']
    K = np.eye(4)
    K[0, 0], K[1, 1] = fov2focal(fovx, w), fov2focal(fovy, h)
    K[0, 2], K[1, 2] = intrin_dict['cx'], intrin_dict['cy']
    return K

def parse_raw(raw_data):
    obs, info = raw_data
    
    imgs = {}
    raw_imgs = {}
    
    for cam in nusc_cameras:
        im = obs['rgb'][cam]
        resize_im = cv2.resize(im, (1920, 1080))
        raw_imgs[cam] = im
        imgs[cam] = resize_im

    velo, acc = np.zeros(2), np.zeros(2)
    command = np.zeros(4)
    # print('cmd', info['command'])
    # command[info['command']] = 1
    command[1] = 1
    
    ego_pose = OPENCV2IMU @ info['ego_pos']
    # yaw = -info['ego_rot'][1]
    yaw = -info['ego_steer']
    forward_velo = info['ego_velo']
    forward_acc = info['accelerate']
    velo[0] = forward_velo * np.cos(yaw)
    velo[1] = forward_velo * np.sin(yaw)
    acc[0] = forward_acc * np.cos(yaw)
    acc[1] = forward_acc * np.sin(yaw)
    ego_status = EgoStatus(ego_pose, velo, acc, command)
    
    cameras_dict = {}
    for nusc_cam, nuplan_cam in zip(nusc_cameras, nuplan_cameras):
        cameras_dict[nuplan_cam] = Camera(image=imgs[nusc_cam])
        
    cameras = Cameras(*([None] * 8))
    for cam_name, cam in cameras_dict.items():
        setattr(cameras, cam_name, cam)
        
    lidar = Lidar(np.zeros((6, 100), dtype=np.float32))
    
    agent_input = AgentInput([ego_status], [cameras], [lidar])
    
    data = {
        'input': agent_input,
        'raw_imgs': raw_imgs,
    }

    return data

