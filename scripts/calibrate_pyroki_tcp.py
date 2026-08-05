#!/usr/bin/env python3
"""Pyroki TCP 标定脚本：对比 pyroki FK(link6) 与 MuJoCo grip_site 位姿，
确定固定偏移（应 <1mm）。

运行:
    MUJOCO_GL=egl /home/stouching/anaconda3/envs/ASPIRE/bin/python scripts/calibrate_pyroki_tcp.py
前置: pyroki_server_minimal.py 已在端口 8116 运行。
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
from aspire.engine_capx import ExecutionEngineCapx
from aspire.pyroki_client import _post

FK_URL = "http://127.0.0.1:8116/fk"

def pyroki_fk(q6):
    out = _post("/fk", {"joint_positions": np.asarray(q6, dtype=float).tolist()})
    return np.array(out["position"], dtype=np.float64), np.array(out["quat_wxyz"], dtype=np.float64)


def main():
    engine = ExecutionEngineCapx(task="Stack", seed=0, trace_root="traces")
    model = engine.env.sim.model._model
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site")
    arm_joints = [f"robot0_joint{i}" for i in range(1, 7)]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in arm_joints]
    qposadr = [int(model.jnt_qposadr[j]) for j in jids]

    # 标定构型：home + 几个工作区典型构型
    seeds = [
        np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.2, -1.0, 0.0, -1.2, 0.0]),
        np.array([0.5, 1.57, -1.35, 0.0, 0.0, 0.0]),
        np.array([-0.5, 1.57, -1.35, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.57, -1.35, 1.0, 0.0, 0.0]),
        np.array([0.0, 1.57, -1.35, -1.0, 0.0, 0.0]),
    ]

    diffs_pos = []
    diffs_quat = []
    for q in seeds:
        # MuJoCo FK
        data = mujoco.MjData(model)
        data.qpos[:] = engine.env.sim.data.qpos
        for i, adr in enumerate(qposadr):
            data.qpos[adr] = q[i]
        mujoco.mj_forward(model, data)
        mj_pos_world = data.site_xpos[site_id].copy()
        mj_pos_base = (engine.T_base_world @ np.append(mj_pos_world, 1.0))[:3]

        # pyroki FK
        pk_pos, pk_quat = pyroki_fk(q)
        # pyroki 输出是世界系？URDF 的 base_link 是原点，与 MuJoCo 世界系原点不同
        # 但 pyroki 的 FK 基于 URDF 的 base_link，而 MuJoCo 的 grip_site 在世界系
        # 需要用 engine.T_world_base 转换 pyroki 结果到世界系？
        # 实际上 pyroki FK 返回的是基于 URDF base_link 的坐标，而 URDF 的 base_link
        # 在 MuJoCo 中就是 robot0_base_link（世界系中的位置由 base_link 的 xpos 决定）
        # 所以 pyroki 的 FK 输出需要加上 base_link 的世界位置
        base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_base_link")
        base_pos_world = engine.env.sim.data.xpos[base_bid]
        pk_pos_world = pk_pos + base_pos_world  # pyroki FK 是相对于 base_link 的
        pk_pos_base = (engine.T_base_world @ np.append(pk_pos_world, 1.0))[:3]

        dpos = float(np.linalg.norm(pk_pos_base - mj_pos_base))
        # quat 差异 (wxyz)
        mj_mat = data.site_xmat[site_id].reshape(3, 3)
        from robosuite.utils import transform_utils as T
        mj_q_xyzw = T.mat2quat(mj_mat)
        mj_q_wxyz = np.array([mj_q_xyzw[3], mj_q_xyzw[0], mj_q_xyzw[1], mj_q_xyzw[2]])
        dquat = float(np.arccos(np.clip(np.abs(np.dot(pk_quat, mj_q_wxyz)), -1.0, 1.0)))

        diffs_pos.append(dpos)
        diffs_quat.append(dquat)
        print(f"  q={q.round(2)}  dpos={dpos*1000:.2f}mm  dquat={np.rad2deg(dquat):.2f}°")

    print(f"\n=== 标定结果 ===")
    print(f"位置偏差: mean={np.mean(diffs_pos)*1000:.2f}mm max={np.max(diffs_pos)*1000:.2f}mm")
    print(f"姿态偏差: mean={np.rad2deg(np.mean(diffs_quat)):.2f}° max={np.rad2deg(np.max(diffs_quat)):.2f}°")
    if np.max(diffs_pos) < 0.001:
        print("✓ 误差 <1mm，TCP_OFFSET=0 直接可用")
    else:
        print(f"⚠ 误差 >1mm，需记录固定偏移")
    engine.close()


if __name__ == "__main__":
    main()
