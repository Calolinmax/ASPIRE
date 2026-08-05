#!/usr/bin/env python3
"""URDF ↔ MJCF 关节映射标定。

几何核对确认两套模型杆长一致，411mm 偏差来自零位/轴系约定。
对每个关节单独转动，采样 FK(link6) 轨迹，拟合：
    q_mjcf_i = sign_i * q_urdf_i + offset_i
验证：100 组随机 q（取限位交集），映射后 FK 位置差 <1mm、姿态差 <0.01rad。
"""
import os
import sys
import json

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
from robosuite.utils import transform_utils as T
from aspire.engine_capx import ExecutionEngineCapx
from aspire.pyroki_client import _post

FK_URL = "http://127.0.0.1:8116/fk"


def pyroki_fk(q6):
    """返回 URDF base_link 系下的 link6 位姿：pos, quat_wxyz."""
    out = _post("/fk", {"joint_positions": np.asarray(q6, dtype=float).tolist()})
    return np.array(out["position"], dtype=np.float64), np.array(out["quat_wxyz"], dtype=np.float64)


def mj_fk(engine, model, q6):
    """返回 robot0_base_link 系下的 robot0_link6 位姿：pos, quat_wxyz."""
    arm_joints = [f"robot0_joint{i}" for i in range(1, 7)]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in arm_joints]
    qposadr = [int(model.jnt_qposadr[j]) for j in jids]
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_base_link")
    base_pos_world = engine.env.sim.data.xpos[base_bid]
    link6_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_link6")

    data = mujoco.MjData(model)
    data.qpos[:] = engine.env.sim.data.qpos
    for i, adr in enumerate(qposadr):
        data.qpos[adr] = q6[i]
    mujoco.mj_forward(model, data)
    pos = data.xpos[link6_bid] - base_pos_world
    mat = data.xmat[link6_bid].reshape(3, 3)
    q_xyzw = T.mat2quat(mat)
    quat = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
    return pos, quat


def calibrate_joint(i, engine, model, n_samples=60):
    """标定第 i 个关节（0-based）。返回 sign, offset (rad)。"""
    # URDF 限位
    urdf_lo, urdf_hi = -1.0, 1.0
    # MJCF 限位
    mj_joint_name = f"robot0_joint{i+1}"
    mj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, mj_joint_name)
    mj_lo, mj_hi = model.jnt_range[mj_jid]

    # 采样 URDF 轨迹
    q_urdf = np.linspace(urdf_lo, urdf_hi, n_samples)
    traj_urdf = []
    for a in q_urdf:
        q6 = np.zeros(6)
        q6[i] = a
        pos, quat = pyroki_fk(q6)
        traj_urdf.append(pos)
    traj_urdf = np.array(traj_urdf)

    # 采样 MJCF 轨迹
    q_mj = np.linspace(mj_lo, mj_hi, n_samples)
    traj_mj = []
    for a in q_mj:
        q6 = np.zeros(6)
        q6[i] = a
        pos, quat = mj_fk(engine, model, q6)
        traj_mj.append(pos)
    traj_mj = np.array(traj_mj)

    # 搜索 sign 和 offset，使位置轨迹对齐
    best = {"err": np.inf, "sign": 1, "offset": 0.0}
    offsets = np.linspace(-np.pi, np.pi, 361)
    for sign in (-1, 1):
        for off in offsets:
            mapped = sign * q_urdf + off
            # 映射后的 URDF 角度必须在 MJCF 限位内
            if mapped.min() < mj_lo - 0.05 or mapped.max() > mj_hi + 0.05:
                continue
            # 插值：对 URDF 每个角度，在 MJCF 轨迹找最近对应
            errs = []
            for k, mu in enumerate(mapped):
                idx = int(np.clip(np.searchsorted(q_mj, mu), 0, n_samples - 1))
                # 局部细化：在 idx 附近 ±5 搜索最近
                local = slice(max(0, idx - 5), min(n_samples, idx + 6))
                best_idx = local.start + np.argmin(np.linalg.norm(traj_mj[local] - traj_urdf[k], axis=1))
                errs.append(np.linalg.norm(traj_mj[best_idx] - traj_urdf[k]))
            mean_err = np.mean(errs)
            if mean_err < best["err"]:
                best.update(err=mean_err, sign=sign, offset=float(off))
    return best


def main():
    engine = ExecutionEngineCapx(task="Stack", seed=0, trace_root="traces")
    model = engine.env.sim.model._model

    mapping = []
    print("=== 逐关节标定 ===")
    for i in range(6):
        best = calibrate_joint(i, engine, model)
        mapping.append({"sign": best["sign"], "offset": best["offset"], "err_mm": best["err"] * 1000})
        print(f"joint{i+1}: sign={best['sign']:+d} offset={best['offset']:.4f}rad "
              f"mean_err={best['err']*1000:.2f}mm")

    # 100 组随机验证
    print("\n=== 100 组随机验证 ===")
    np.random.seed(0)
    pos_errs, quat_errs = [], []
    for _ in range(100):
        q_urdf = np.random.uniform(-0.8, 0.8, 6)
        q_mj = np.array([mapping[i]["sign"] * q_urdf[i] + mapping[i]["offset"] for i in range(6)])
        # 检查 MJCF 限位
        valid = True
        for i in range(6):
            mj_jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"robot0_joint{i+1}")
            lo, hi = model.jnt_range[mj_jid]
            if not (lo <= q_mj[i] <= hi):
                valid = False
                break
        if not valid:
            continue
        pos_u, quat_u = pyroki_fk(q_urdf)
        pos_m, quat_m = mj_fk(engine, model, q_mj)
        pos_errs.append(np.linalg.norm(pos_u - pos_m))
        qdot = np.clip(np.abs(np.dot(quat_u, quat_m)), -1.0, 1.0)
        quat_errs.append(np.arccos(qdot))

    pos_errs = np.array(pos_errs)
    quat_errs = np.array(quat_errs)
    print(f"位置偏差: mean={pos_errs.mean()*1000:.2f}mm max={pos_errs.max()*1000:.2f}mm")
    print(f"姿态偏差: mean={np.rad2deg(quat_errs.mean()):.3f}° max={np.rad2deg(quat_errs.max()):.3f}°")

    # 保存映射表
    out_path = os.path.join(os.path.dirname(__file__), "..", "aspire", "pyroki_joint_map.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2)
    print(f"\n映射表已保存到: {out_path}")

    if pos_errs.max() < 0.001 and quat_errs.max() < 0.01:
        print("✓ 标定通过 (<1mm / <0.01rad)")
    else:
        print("✗ 标定未通过，需 Fix 2：以 MJCF 为真源生成配套 URDF")

    engine.close()


if __name__ == "__main__":
    main()
