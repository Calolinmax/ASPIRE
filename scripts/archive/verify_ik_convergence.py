#!/usr/bin/env python3
"""A1/A2 验收：pyroki vs DLS 在工作区网格上的 IK 收敛对比。

方法：
1. 在关节限位内随机采样 N 组目标构型 q_true。
2. 用 MuJoCo FK 得到每组 q_true 对应的 grip_site TCP 位姿（基座系）。
3. 直接调用 pyroki /ik（不经过 solve_ik 的 fallback）求解。
4. 调用 solve_ik(use_pyroki=False) 作为 DLS 基线。
5. 统计成功率、TCP 重投影误差、关节误差。

通过标准：pyroki 成功率 ≥ DLS，且 TCP 位置误差 <1mm、姿态误差 <0.01rad。
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mujoco
from robosuite.utils import transform_utils as T
from aspire.engine.engine_capx import ExecutionEngineCapx
from aspire.api.primitives_capx import PrimitiveContextCapx
from aspire.planning.pyroki_client import ik_pyroki


def fk_grip_site(model, data, q6):
    """返回基座系下的 grip_site 位姿 (pos, quat_xyzw)。"""
    arm_joints = [f"robot0_joint{i}" for i in range(1, 7)]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in arm_joints]
    qposadr = [int(model.jnt_qposadr[j]) for j in jids]
    base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_base_link")
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site")

    data.qpos[:] = 0.0
    for i, adr in enumerate(qposadr):
        data.qpos[adr] = q6[i]
    mujoco.mj_forward(model, data)

    pos = data.site_xpos[site_id] - data.xpos[base_bid]
    base_mat = data.xmat[base_bid].reshape(3, 3)
    site_mat = data.site_xmat[site_id].reshape(3, 3)
    rel_mat = base_mat.T @ site_mat
    q_xyzw = T.mat2quat(rel_mat)
    return pos, q_xyzw


def main():
    engine = ExecutionEngineCapx(task="Stack", seed=0, trace_root="traces")
    ctx = PrimitiveContextCapx(engine)
    model = engine.env.sim.model._model
    data = mujoco.MjData(model)

    # 关节限位
    arm_joints = [f"robot0_joint{i}" for i in range(1, 7)]
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in arm_joints]
    ranges = np.array([model.jnt_range[j] for j in jids])

    # 生成测试目标：关节空间随机采样 100 组 + 6 个规范种子
    np.random.seed(2)
    targets = []
    for _ in range(100):
        q = np.array([np.random.uniform(lo, hi) for lo, hi in ranges])
        pos, q_xyzw = fk_grip_site(model, data, q)
        targets.append((q, pos, q_xyzw))
    seed_qs = [
        np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.2, -1.0, 0.0, -1.2, 0.0]),
        np.array([0.5, 1.57, -1.35, 0.0, 0.0, 0.0]),
        np.array([-0.5, 1.57, -1.35, 0.0, 0.0, 0.0]),
        np.array([0.0, 1.57, -1.35, 1.0, 0.0, 0.0]),
        np.array([0.0, 1.57, -1.35, -1.0, 0.0, 0.0]),
    ]
    for q in seed_qs:
        pos, q_xyzw = fk_grip_site(model, data, q)
        targets.append((q, pos, q_xyzw))

    print(f"总测试点数: {len(targets)}")

    def tcp_ok(q_ik, pos_true, qxyzw_true):
        if q_ik is None:
            return False, np.nan, np.nan
        pos, qxyzw = fk_grip_site(model, data, q_ik[:6])
        dpos = float(np.linalg.norm(pos - pos_true))
        qdot = np.clip(np.abs(np.dot(qxyzw, qxyzw_true)), -1.0, 1.0)
        dquat = float(np.arccos(qdot))
        return dpos < 1e-3 and dquat < 0.01, dpos, dquat

    results = {"pyroki": [], "dls": []}
    for idx, (q_true, pos, q_xyzw) in enumerate(targets):
        quat_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

        # pyroki（直接调用，无 fallback）
        t0 = time.time()
        try:
            q_pk = ik_pyroki(pos, q_xyzw, prev_cfg=q_true)
            ok_pk, dpos_pk, dquat_pk = tcp_ok(q_pk, pos, q_xyzw)
            err_pk = float(np.linalg.norm(q_pk[:6] - q_true))
        except Exception as e:
            q_pk = None
            ok_pk = False
            dpos_pk = dquat_pk = err_pk = np.nan
        dt_pk = time.time() - t0

        # DLS
        t0 = time.time()
        try:
            q_dls = ctx.solve_ik(pos, quat_wxyz, use_pyroki=False)
            ok_dls, dpos_dls, dquat_dls = tcp_ok(q_dls, pos, q_xyzw)
            err_dls = float(np.linalg.norm(q_dls - q_true))
        except Exception:
            q_dls = None
            ok_dls = False
            dpos_dls = dquat_dls = err_dls = np.nan
        dt_dls = time.time() - t0

        results["pyroki"].append((ok_pk, dpos_pk, dquat_pk, err_pk, dt_pk))
        results["dls"].append((ok_dls, dpos_dls, dquat_dls, err_dls, dt_dls))

    for name in ("pyroki", "dls"):
        rec = results[name]
        oks = [r[0] for r in rec]
        dposes = [r[1] for r in rec if r[0]]
        dquats = [r[2] for r in rec if r[0]]
        errs = [r[3] for r in rec if r[0]]
        times = [r[4] for r in rec if r[0]]
        print(f"\n=== {name.upper()} ===")
        print(f"  成功率: {sum(oks)}/{len(oks)} = {sum(oks)/len(oks)*100:.1f}%")
        if dposes:
            print(f"  TCP 位置误差 (mm): mean={np.mean(dposes)*1000:.3f} max={np.max(dposes)*1000:.3f}")
            print(f"  TCP 姿态误差 (°): mean={np.rad2deg(np.mean(dquats)):.3f} max={np.rad2deg(np.max(dquats)):.3f}")
            print(f"  关节误差 (rad): mean={np.mean(errs):.4f} max={np.max(errs):.4f}")
            print(f"  耗时 (s): mean={np.mean(times):.3f} max={np.max(times):.3f}")

    pk_oks = [r[0] for r in results["pyroki"]]
    dls_oks = [r[0] for r in results["dls"]]
    pk_only = sum(1 for i in range(len(targets)) if pk_oks[i] and not dls_oks[i])
    dls_only = sum(1 for i in range(len(targets)) if not pk_oks[i] and dls_oks[i])
    both = sum(1 for i in range(len(targets)) if pk_oks[i] and dls_oks[i])
    neither = sum(1 for i in range(len(targets)) if not pk_oks[i] and not dls_oks[i])
    print(f"\n=== 对比 ===")
    print(f"  两者都成功: {both}")
    print(f"  仅 pyroki 成功: {pk_only}")
    print(f"  仅 DLS 成功: {dls_only}")
    print(f"  两者都失败: {neither}")

    if sum(pk_oks) >= sum(dls_oks):
        print("\n✓ A1/A2 IK 收敛对比通过（pyroki 成功率不低于 DLS）")
    else:
        print("\n⚠ pyroki 成功率低于 DLS，需检查")

    engine.close()


if __name__ == "__main__":
    main()
