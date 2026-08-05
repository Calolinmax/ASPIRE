#!/usr/bin/env python3
"""A0.5 相机标定验证脚本。

渲染 robotview (top) 与 eye_in_hand (wrist) 在三种位形下的图像：
  1) reset / home 位形 (全0关节角，竖直指天)
  2) 末端竖直朝下，位于物块正上方（预抓位姿）
  3) 末端竖直朝下，下降到物块表面（抓取位姿）

输出: outputs/cam_check/home_{top,wrist}.png,
      pregrasp_{top,wrist}.png, grasp_{top,wrist}.png
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import mujoco
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)

import aspire.robots  # noqa: F401

_ASSETS = os.path.join(os.path.dirname(__file__), "..", "aspire", "robots", "assets", "piper")
_OUTDIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "cam_check")
os.makedirs(_OUTDIR, exist_ok=True)


def make_env():
    return suite.make(
        "Stack",
        robots="Piper",
        has_renderer=False,
        has_offscreen_renderer=True,
        use_camera_obs=True,
        camera_names=["robot0_robotview", "robot0_eye_in_hand"],
        camera_heights=256,
        camera_widths=256,
        camera_depths=True,
        control_freq=20,
        horizon=4000,
        reward_shaping=True,
        controller_configs=load_composite_controller_config(
            controller=os.path.join(_ASSETS, "default_piper.json")
        ),
    )


def save_frames(env, prefix: str):
    """obs 中已有图像，直接保存。"""
    obs = env._get_observations(force_update=True)
    for cam in ("robot0_robotview", "robot0_eye_in_hand"):
        rgb = obs[f"{cam}_image"]
        name = "top" if "robotview" in cam else "wrist"
        path = os.path.join(_OUTDIR, f"{prefix}_{name}.png")
        import imageio.v3 as iio
        iio.imwrite(path, rgb)
        print(f"  saved {path}")


def set_arm_qpos(env, q6: np.ndarray):
    """直接设置6个臂关节角。"""
    model = env.sim.model._model
    data = env.sim.data._data
    arm_joints = [f"robot0_joint{i}" for i in range(1, 7)]
    for j, q in zip(arm_joints, q6):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        adr = model.jnt_qposadr[jid]
        data.qpos[adr] = q
    mujoco.mj_forward(model, data)


def get_cube_pos(env, cube_name: str = "cubeA"):
    """获取物块在世界坐标系中的位置。"""
    model = env.sim.model._model
    data = env.sim.data._data
    try:
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, cube_name)
        return np.array(data.xpos[bid])
    except Exception:
        return None


def set_tcp_topdown(env, pos: np.ndarray, yaw_deg: float = 0.0):
    """将TCP设置为竖直朝下姿态，位于指定位置。

    使用IK库查找最接近的关节角解。
    """
    lib_path = os.path.join(_ASSETS, "ik_library.npz")
    if not os.path.exists(lib_path):
        print(f"[警告] IK库不存在: {lib_path}")
        return None

    # 加载IK库
    lib = np.load(lib_path)
    Q = lib["Q"]  # (N, 6) 关节角
    P = lib["P"]  # (N, 3) TCP位置

    # 在IK库中找最接近目标位置的解
    pos_errors = np.linalg.norm(P - pos, axis=1)
    best_idx = np.argmin(pos_errors)
    min_pos_err = pos_errors[best_idx]

    # 尝试不同的yaw角度来找更好的解
    best_q = Q[best_idx]
    best_err = min_pos_err

    # 在位置误差较小的候选中找（误差 < 5cm的）
    close_indices = np.where(pos_errors < 0.05)[0]
    if len(close_indices) > 1:
        # 从这些候选中选一个joint5最接近0的（竖直姿态时joint5通常为0）
        j5_values = np.abs(Q[close_indices, 4])
        best_j5_idx = close_indices[np.argmin(j5_values)]
        best_q = Q[best_j5_idx]
        best_err = pos_errors[best_j5_idx]

    print(f"  位置误差: {best_err*100:.1f}cm")
    set_arm_qpos(env, best_q)
    env.sim.step()
    return best_q


def main():
    env = make_env()
    env.reset()

    # ---- 位形 1: home (标准 Piper home = 全关节 0, 竖直指天) ----
    print("[check_cameras] 位形 1: home (全0关节角)")
    home_q = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    set_arm_qpos(env, home_q)
    env.sim.step()
    save_frames(env, "home")

    # ---- 位形 2 & 3: 末端竖直朝下，位于物块上方 ----
    # 获取红方块(cubeA)位置
    cube_pos = get_cube_pos(env, "cubeA")
    if cube_pos is None:
        # 回退：使用固定位置
        cube_pos = np.array([0.35, 0.0, 0.83])  # 桌面高度约0.83
        print(f"[check_cameras] 未找到cubeA，使用默认位置: {cube_pos.round(3)}")
    else:
        print(f"[check_cameras] cubeA 位置: {cube_pos.round(3)}")

    # 预抓位姿：物块正上方10cm，竖直朝下
    pregrasp_z = cube_pos[2] + 0.10  # 方块顶部上方10cm
    pregrasp_pos = np.array([cube_pos[0], cube_pos[1], pregrasp_z])

    print(f"[check_cameras] 位形 2: 预抓位姿 (竖直朝下，上方10cm)")
    print(f"  目标位置: {pregrasp_pos.round(3)}")
    q_pregrasp = set_tcp_topdown(env, pregrasp_pos, yaw_deg=0.0)
    if q_pregrasp is not None:
        print(f"  IK解: q={q_pregrasp.round(3)}")
        save_frames(env, "pregrasp")
    else:
        print("  未找到IK解，跳过")

    # 抓取位姿：物块正上方2cm（接近抓取）
    grasp_z = cube_pos[2] + 0.02  # 方块顶部上方2cm
    grasp_pos = np.array([cube_pos[0], cube_pos[1], grasp_z])

    print(f"[check_cameras] 位形 3: 抓取位姿 (竖直朝下，上方2cm)")
    print(f"  目标位置: {grasp_pos.round(3)}")
    q_grasp = set_tcp_topdown(env, grasp_pos, yaw_deg=0.0)
    if q_grasp is not None:
        print(f"  IK解: q={q_grasp.round(3)}")
        save_frames(env, "grasp")
    else:
        print("  未找到IK解，跳过")

    env.close()
    print(f"[check_cameras] 全部图像已输出到 {_OUTDIR}")


if __name__ == "__main__":
    main()
