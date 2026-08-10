#!/usr/bin/env python3
"""验证 MJCF 派生 URDF 的 FK 与 MuJoCo MJCF 的 FK 一致性。

对比 pyroki FK(link6) 与 MuJoCo FK(robot0_link6)，在 robot0_base_link 系下计算
位置和姿态偏差。

运行:
    # 1) 先启动 pyroki server（使用新生成的 URDF）
    PYTHONPATH=/home/stouching/Desktop/ASPIRE/external/cap-x \
    ~/venvs/pyroki/bin/python scripts/tools/pyroki_server_minimal.py \
    --urdf external/piper_description/piper_mjcf/urdf/piper_mjcf.urdf \
    --target-link link6 --port 8116

    # 2) 再运行本验证
    /home/stouching/anaconda3/envs/ASPIRE/bin/python scripts/archive/verify_mjcf_urdf_fk.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mujoco
from robosuite.utils import transform_utils as T
from aspire.engine.engine_capx import ExecutionEngineCapx
from aspire.planning.pyroki_client import _post


def pyroki_fk(q6):
    q8 = list(q6) + [0.0, 0.0]
    out = _post("/fk", {"joint_positions": q8})
    return np.array(out["position"], dtype=np.float64), np.array(out["quat_wxyz"], dtype=np.float64)


def mj_fk_grip_site(engine, model, q6):
    """返回 robot0_base_link 系下的 gripper0_right_grip_site 位姿：pos, quat_wxyz."""
    arm_joints = [f"robot0_joint{i}" for i in range(1, 7)]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in arm_joints]
    qposadr = [int(model.jnt_qposadr[j]) for j in jids]
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_base_link")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site")

    data = mujoco.MjData(model)
    data.qpos[:] = engine.env.sim.data.qpos
    for i, adr in enumerate(qposadr):
        data.qpos[adr] = q6[i]
    mujoco.mj_forward(model, data)

    # 位置：site 在世界系 - base_link 在世界系
    pos_world = data.site_xpos[site_id] - data.xpos[base_bid]

    # 姿态：site 在世界系的旋转矩阵，再转到 base_link 系
    base_mat = data.xmat[base_bid].reshape(3, 3)
    site_mat = data.site_xmat[site_id].reshape(3, 3)
    rel_mat = base_mat.T @ site_mat
    q_xyzw = T.mat2quat(rel_mat)
    quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=np.float64)
    return pos_world, quat_wxyz


def main():
    engine = ExecutionEngineCapx(task="Stack", seed=0, trace_root="traces")
    model = engine.env.sim.model._model

    # 测试构型：home + 随机采样
    seeds = [
        np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.2, -1.0, 0.0, -1.2, 0.0]),
        np.array([0.5, 1.57, -1.35, 0.0, 0.0, 0.0]),
        np.array([-0.5, 1.57, -1.35, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.57, -1.35, 1.0, 0.0, 0.0]),
        np.array([0.0, 1.57, -1.35, -1.0, 0.0, 0.0]),
    ]
    # 在限位内随机采样 100 组
    mj_ranges = []
    for i in range(6):
        mj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i+1}")
        mj_ranges.append(model.jnt_range[mj_jid].copy())
    np.random.seed(1)
    for _ in range(100):
        q6 = np.array([np.random.uniform(lo, hi) for lo, hi in mj_ranges])
        seeds.append(q6)

    pos_errs, quat_errs = [], []
    fails = []
    for idx, q in enumerate(seeds):
        try:
            pk_pos, pk_quat = pyroki_fk(q)
        except Exception as e:
            fails.append((idx, q, f"pyroki FK failed: {e}"))
            continue
        mj_pos, mj_quat = mj_fk_grip_site(engine, model, q)
        dpos = float(np.linalg.norm(pk_pos - mj_pos))
        qdot = np.clip(np.abs(np.dot(pk_quat, mj_quat)), -1.0, 1.0)
        dquat = float(np.arccos(qdot))
        pos_errs.append(dpos)
        quat_errs.append(dquat)
        if idx < 6:
            print(f"  seed {idx}: q={q.round(3)}  dpos={dpos*1000:.3f}mm  dquat={np.rad2deg(dquat):.3f}°")

    pos_errs = np.array(pos_errs)
    quat_errs = np.array(quat_errs)
    print(f"\n=== 统计 ({len(pos_errs)} valid / {len(seeds)} total) ===")
    print(f"位置偏差: mean={pos_errs.mean()*1000:.3f}mm max={pos_errs.max()*1000:.3f}mm")
    print(f"姿态偏差: mean={np.rad2deg(quat_errs.mean()):.3f}° max={np.rad2deg(quat_errs.max()):.3f}°")
    if fails:
        print(f"失败 {len(fails)} 次")
        for idx, q, msg in fails[:3]:
            print(f"  idx={idx} q={q.round(3)}: {msg}")

    if pos_errs.max() < 0.001 and quat_errs.max() < 0.01:
        print("✓ FK 对拍通过 (<1mm / <0.01rad)")
    else:
        print("✗ FK 对拍未通过")

    engine.close()


if __name__ == "__main__":
    main()
