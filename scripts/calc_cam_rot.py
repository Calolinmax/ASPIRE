#!/usr/bin/env python3
"""计算相机绕视线方向旋转后的新quat。"""
import numpy as np

def quat2mat(q):
    """wxyz -> 3x3 rotation matrix"""
    w, x, y, z = q
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]
    ])

def mat2quat(R):
    """3x3 rotation matrix -> wxyz"""
    tr = np.trace(R)
    if tr > 0:
        S = np.sqrt(tr + 1.0) * 2
        w = 0.25 * S
        x = (R[2,1] - R[1,2]) / S
        y = (R[0,2] - R[2,0]) / S
        z = (R[1,0] - R[0,1]) / S
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        S = np.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2]) * 2
        w = (R[2,1] - R[1,2]) / S
        x = 0.25 * S
        y = (R[0,1] + R[1,0]) / S
        z = (R[0,2] + R[2,0]) / S
    elif R[1,1] > R[2,2]:
        S = np.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2]) * 2
        w = (R[0,2] - R[2,0]) / S
        x = (R[0,1] + R[1,0]) / S
        y = 0.25 * S
        z = (R[1,2] + R[2,1]) / S
    else:
        S = np.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1]) * 2
        w = (R[1,0] - R[0,1]) / S
        x = (R[0,2] + R[2,0]) / S
        y = (R[1,2] + R[2,1]) / S
        z = 0.25 * S
    return np.array([w, x, y, z])

def rotate_camera_along_optical_axis(quat, angle_deg):
    """
    绕相机光轴（Z轴）旋转指定角度。
    这会使图像旋转相应角度。
    正角度 = 逆时针（从相机看出去）
    """
    theta = np.deg2rad(angle_deg)
    # 绕Z轴旋转的旋转矩阵（在相机坐标系中）
    Rz = np.array([
        [np.cos(theta), -np.sin(theta), 0],
        [np.sin(theta), np.cos(theta), 0],
        [0, 0, 1]
    ])
    # 当前相机姿态（世界到相机）
    R_cam = quat2mat(quat)
    # 新姿态 = 当前姿态 @ 绕Z轴旋转
    # 这样保持Z轴（视线）方向不变，只旋转XY平面
    R_new = R_cam @ Rz
    return mat2quat(R_new)

def verify_z_axis_direction(quat, label):
    """检查相机的Z轴指向（视线方向）"""
    R = quat2mat(quat)
    z_axis = R[:, 2]  # 第三列是Z轴在世界坐标系中的方向
    print(f"{label}: Z axis = {z_axis.round(4)}")
    return z_axis

# 1. robotview: 顺时针180度
robotview_orig = np.array([0.653, 0.271, 0.271, 0.653])
robotview_orig = robotview_orig / np.linalg.norm(robotview_orig)
robotview_new = rotate_camera_along_optical_axis(robotview_orig, -180)  # 顺时针 = 负角度
print(f"robotview orig: {robotview_orig.round(4)}")
print(f"robotview new (CW 180°): {robotview_new.round(4)}")

# 2. wrist: 逆时针90度
wrist_orig = np.array([0.0, 0.0, 1.0, 0.0])
wrist_new = rotate_camera_along_optical_axis(wrist_orig, 90)  # 逆时针 = 正角度
print(f"\nwrist orig: {wrist_orig.round(4)}")
print(f"wrist new (CCW 90°): {wrist_new.round(4)}")

# 验证Z轴方向是否保持不变
print("\n=== 验证视线方向 ===")
verify_z_axis_direction(robotview_orig, "robotview orig")
verify_z_axis_direction(robotview_new, "robotview new")
verify_z_axis_direction(wrist_orig, "wrist orig")
verify_z_axis_direction(wrist_new, "wrist new")
