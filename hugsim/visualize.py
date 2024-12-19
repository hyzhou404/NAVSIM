import cv2
import os
import glob
import numpy as np
import math

nusc_cameras = ['CAM_FRONT_LEFT', 'CAM_FRONT', 'CAM_FRONT_RIGHT']
OPENCV2IMU = np.array([[0, 0, 1], [-1, 0, 0], [0, -1, 0]])

def fov2focal(fov, pixels):
    return pixels / (2 * math.tan(fov / 2))

def save_frame(raw_image, traj_poses, cam_params, save_fn):
    plan_traj = np.stack([
        traj_poses[:, 0],
        1.4*np.ones_like(traj_poses[:, 2]),
        traj_poses[:, 1],
    ], axis=1)
    
    intrinsic_dict = cam_params['CAM_FRONT']['intrinsic']
    intrinsic = np.eye(4)
    fx = fov2focal(intrinsic_dict['fovx'], intrinsic_dict['W'])
    fy = fov2focal(intrinsic_dict['fovy'], intrinsic_dict['H'])
    intrinsic[0, 0], intrinsic[1, 1] = fx, fy
    intrinsic[0, 2], intrinsic[1, 2] = intrinsic_dict['cx'], intrinsic_dict['cy']
    uvz_traj = (intrinsic[:3, :3] @ plan_traj.T).T
    uv_traj = uvz_traj[:, :2] / uvz_traj[:, 2][:, None]
    uv_traj_int = uv_traj.astype(int)
    
    images = []
    for cam in nusc_cameras:
        im = raw_image[cam]
        if cam == 'CAM_FRONT':
            for point in uv_traj_int:
                im = cv2.circle(im, point, radius=5, color=(192, 152, 7), thickness=-1)
        images.append(im)
    images = cv2.cvtColor(cv2.hconcat(images), cv2.COLOR_RGB2BGR)
    cv2.imwrite(save_fn+'.jpg', images)

def to_video(folder_path, fps=5, downsample=1):
    imgs_path = glob.glob(os.path.join(folder_path, '*.jpg'))
    imgs_path = sorted(imgs_path)
    img_array = []
    for img_path in imgs_path:
        img = cv2.imread(img_path)
        height, width, channel = img.shape
        img = cv2.resize(img, (width//downsample, height//downsample))
        height, width, channel = img.shape
        size = (width, height)
        img_array.append(img)
        
    # media.write_video(os.path.join(folder_path, 'video.mp4'), img_array, fps=10)
    mp4_path = os.path.join(folder_path, 'video.mp4')
    if os.path.exists(mp4_path): 
        os.remove(mp4_path)
    out = cv2.VideoWriter(
        mp4_path, 
        cv2.VideoWriter_fourcc(*'DIVX'), fps, size
    )
    for i in range(len(img_array)):
        out.write(img_array[i])
    out.release()