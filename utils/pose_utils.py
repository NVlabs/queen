# Adapted from 4DGaussians https://github.com/hustvl/4DGaussians

# Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

import numpy as np
from scipy.spatial.transform import Rotation as R
from scene.utils import Camera
from copy import deepcopy
from scene.colmap_loader import qvec2rotmat, rotmat2qvec
def rotation_matrix_to_quaternion(rotation_matrix):
    """将旋转矩阵转换为四元数"""
    return R.from_matrix(rotation_matrix).as_quat()

def quaternion_to_rotation_matrix(quat):
    """将四元数转换为旋转矩阵"""
    return R.from_quat(quat).as_matrix()

def quaternion_slerp(q1, q2, t):
    """在两个四元数之间进行球面线性插值（SLERP）"""
    # 计算两个四元数之间的点积
    dot = np.dot(q1, q2)

    # 如果点积为负，取反一个四元数以保证最短路径插值
    if dot < 0.0:
        q1 = -q1
        dot = -dot

    # 防止数值误差导致的问题
    dot = np.clip(dot, -1.0, 1.0)

    # 计算插值参数
    theta = np.arccos(dot) * t
    q3 = q2 - q1 * dot
    q3 = q3 / np.linalg.norm(q3)

    # 计算插值结果
    return np.cos(theta) * q1 + np.sin(theta) * q3

def bezier_interpolation(p1, p2, t):
    """在两点之间使用贝塞尔曲线进行插值"""
    return (1 - t) * p1 + t * p2
def linear_interpolation(v1, v2, t):
    """线性插值"""
    return (1 - t) * v1 + t * v2
def smooth_camera_poses(cameras, num_interpolations=5):
    """对一系列相机位姿进行平滑处理，通过在每对位姿之间插入额外的位姿"""
    smoothed_cameras = []
    smoothed_times = []
    total_poses = len(cameras) - 1 + (len(cameras) - 1) * num_interpolations
    time_increment = 10 / total_poses

    for i in range(len(cameras) - 1):
        cam1 = cameras[i]
        cam2 = cameras[i + 1]

        # 将旋转矩阵转换为四元数
        quat1 = rotation_matrix_to_quaternion(cam1.orientation)
        quat2 = rotation_matrix_to_quaternion(cam2.orientation)

        for j in range(num_interpolations + 1):
            t = j / (num_interpolations + 1)

            # 插值方向
            interp_orientation_quat = quaternion_slerp(quat1, quat2, t)
            interp_orientation_matrix = quaternion_to_rotation_matrix(interp_orientation_quat)

            # 插值位置
            interp_position = linear_interpolation(cam1.position, cam2.position, t)

            # 计算插值时间戳
            interp_time = i*10 / (len(cameras) - 1) + time_increment * j

            # 添加新的相机位姿和时间戳
            newcam = deepcopy(cam1)
            newcam.orientation = interp_orientation_matrix
            newcam.position = interp_position
            smoothed_cameras.append(newcam)
            smoothed_times.append(interp_time)

    # 添加最后一个原始位姿和时间戳
    smoothed_cameras.append(cameras[-1])
    smoothed_times.append(1.0)
    print(smoothed_times)
    return smoothed_cameras, smoothed_times

# # 示例：使用两个相机位姿
# cam1 = Camera(np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]), np.array([0, 0, 0]))
# cam2 = Camera(np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]), np.array([1, 1, 1]))

# # 应用平滑处理
# smoothed_cameras = smooth_camera_poses([cam1, cam2], num_interpolations=5)

# # 打印结果
# for cam in smoothed_cameras:
#     print("Orientation:\n", cam.orientation)
#     print("Position:", cam.position)



def normalize(v):
    """Normalize a vector."""
    return v / np.linalg.norm(v)


def average_poses(poses):
    """
    Calculate the average pose, which is then used to center all poses
    using @center_poses. Its computation is as follows:
    1. Compute the center: the average of pose centers.
    2. Compute the z axis: the normalized average z axis.
    3. Compute axis y': the average y axis.
    4. Compute x' = y' cross product z, then normalize it as the x axis.
    5. Compute the y axis: z cross product x.

    Note that at step 3, we cannot directly use y' as y axis since it's
    not necessarily orthogonal to z axis. We need to pass from x to y.
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        pose_avg: (3, 4) the average pose
    """
    # 1. Compute the center
    center = poses[..., 3].mean(0)  # (3)

    # 2. Compute the z axis
    z = normalize(poses[..., 2].mean(0))  # (3)

    # 3. Compute axis y' (no need to normalize as it's not the final output)
    y_ = poses[..., 1].mean(0)  # (3)

    # 4. Compute the x axis
    x = normalize(np.cross(z, y_))  # (3)

    # 5. Compute the y axis (as z and x are normalized, y is already of norm 1)
    y = np.cross(x, z)  # (3)

    pose_avg = np.stack([x, y, z, center], 1)  # (3, 4)

    return pose_avg


def eul2rot(angles):
    rz = np.array([[np.cos(angles[2]), -np.sin(angles[2]), 0],
                   [np.sin(angles[2]), np.cos(angles[2]), 0],
                   [0, 0, 1]])
    ry = np.array([[np.cos(angles[1]), 0, np.sin(angles[1])],
                   [0, 1, 0],
                   [-np.sin(angles[1]), 0, np.cos(angles[1])]])
    rx = np.array([[1, 0, 0],
                   [0, np.cos(angles[0]), -np.sin(angles[0])],
                   [0, np.sin(angles[0]), np.cos(angles[0])]])
    
    return rz @ ry @ rx

def center_poses(poses, blender2opencv):
    """
    Center the poses so that we can use NDC.
    See https://github.com/bmild/nerf/issues/34
    Inputs:
        poses: (N_images, 3, 4)
    Outputs:
        poses_centered: (N_images, 3, 4) the centered poses
        pose_avg: (3, 4) the average pose
    """
    poses = poses @ blender2opencv
    pose_avg = average_poses(poses)  # (3, 4)
    pose_avg_homo = np.eye(4)
    pose_avg_homo[
        :3
    ] = pose_avg  # convert to homogeneous coordinate for faster computation
    pose_avg_homo = pose_avg_homo
    # by simply adding 0, 0, 0, 1 as the last row
    last_row = np.tile(np.array([0, 0, 0, 1]), (len(poses), 1, 1))  # (N_images, 1, 4)
    poses_homo = np.concatenate(
        [poses, last_row], 1
    )  # (N_images, 4, 4) homogeneous coordinate

    poses_centered = np.linalg.inv(pose_avg_homo) @ poses_homo  # (N_images, 4, 4)
    #     poses_centered = poses_centered  @ blender2opencv
    poses_centered = poses_centered[:, :3]  # (N_images, 3, 4)

    return poses_centered, pose_avg_homo


def viewmatrix(z, up, pos):
    vec2 = normalize(z)
    vec1_avg = up
    vec0 = normalize(np.cross(vec1_avg, vec2))
    vec1 = normalize(np.cross(vec2, vec0))
    m = np.eye(4)
    m[:3] = np.stack([-vec0, vec1, vec2, pos], 1)
    return m


def render_path_spiral(c2w, up, rads, focal, zdelta, zrate, N_rots=2, N=120, flip=False):
    render_poses = []
    rads = np.array(list(rads) + [1.0])

    for theta in np.linspace(0.0, 2.0 * np.pi * N_rots, N + 1)[:-1]:
        c = np.dot(
            c2w[:3, :4],
            np.array([np.cos(theta), -np.sin(theta), -np.sin(theta * zrate), 1.0])
            * rads,
        )
        if flip:
            z = normalize(np.dot(c2w[:3,:4], np.array([0, 0, focal, 1.])) - c)
        else:
            z = normalize(c - np.dot(c2w[:3,:4], np.array([0, 0, -focal, 1.])))
        render_poses.append(viewmatrix(z, up, c))
    return render_poses


def get_spiral(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """
    # center pose
    c2w = average_poses(c2ws_all)

    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))

    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

    # Get radii for spiral path
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 90, 0) * rads_scale
    render_poses = render_path_spiral(
        c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views
    )
    return np.stack(render_poses)



def get_spiral_immersive(c2ws_all, near_fars, rads_scale=1.0, N_views=120):
    """
    Generate a set of poses using NeRF's spiral camera trajectory as validation poses.
    """

    c2w = average_poses(c2ws_all)


    # Get average pose
    up = normalize(c2ws_all[:, :3, 1].sum(0))
    # Find a reasonable "focus depth" for this dataset
    dt = 0.75
    close_depth, inf_depth = near_fars.min() * 0.9, near_fars.max() * 5.0
    focal = 1.0 / ((1.0 - dt) / close_depth + dt / inf_depth)

    # breakpoint()
    # Get radii for spiral path
    zdelta = near_fars.min() * 0.2
    tt = c2ws_all[:, :3, 3]
    rads = np.percentile(np.abs(tt), 50, 0) * rads_scale
    rads[..., :-1] *= 1.0
    rads[..., -1] *= 0.05
    render_poses = render_path_spiral(
        c2w, up, rads, focal, zdelta, zrate=0.5, N=N_views, flip=True, N_rots=4
    )
    return np.stack(render_poses)
