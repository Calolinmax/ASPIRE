#!/usr/bin/env python
"""幻影收敛捕获（2026-08-03）: solve_ik 报告收敛但 FK 姿态误差 134° 的标本捕获。

背景: seed5 CGN cand1 yaw=270 出现一次 pos 误差 0.0000 / ori 误差 134° 的
"收敛"。solve_ik 内部有三重严格校验（pyroki 后验 1mm/0.01rad、DLS tol 2e-4/3e-3），
理论上不可能放过。本脚本打猴子补丁定位返回路径并 FK 复核每个返回值。

用法: MUJOCO_GL=egl python -u scripts/hunt_phantom_ik.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import aspire.robots  # noqa: F401
import aspire.pyroki_client as pkc
from aspire.engine_capx import ExecutionEngineCapx
from aspire.primitives_capx import PrimitiveContextCapx, cgn_to_gripper, EEF_SITE
from aspire.vision_client import segment_sam3_text_prompt, grasp_cgn
from robosuite.utils import transform_utils as T

# ---- 猴子补丁: 记录 pyroki 直调的返回 ----
_orig_ik = pkc.ik_pyroki
pyroki_log = []


def wrapped_ik(*a, **kw):
    r = _orig_ik(*a, **kw)
    pyroki_log.append(np.asarray(r, dtype=float).reshape(-1)[:6])
    return r


pkc.ik_pyroki = wrapped_ik


def qw_of(Rm):
    q = T.mat2quat(Rm)
    return np.array([q[3], q[0], q[1], q[2]])


def main():
    engine = ExecutionEngineCapx(task='Stack', seed=5)
    ctx = PrimitiveContextCapx(engine)
    sim = engine.env.sim
    m = sim.model._model
    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)

    def fk(q6):
        for i in range(6):
            sim.data.set_joint_qpos(f'robot0_joint{i+1}', q6[i])
        mujoco.mj_forward(m, sim.data._data)
        sp = sim.data.site_xpos[sid].copy()
        sm = sim.data.site_xmat[sid].reshape(3, 3).copy()
        return (engine.T_base_world @ np.append(sp, 1.0))[:3], \
            engine.T_base_world[:3, :3] @ sm

    obs = ctx.get_observation()
    cam = obs['robot0_robotview']
    rgb, depth, K, pose_mat = (cam['images']['rgb'], cam['images']['depth'],
                               cam['intrinsics'], cam['pose_mat'])
    masks = segment_sam3_text_prompt(rgb, 'red cube')
    seg = np.zeros(rgb.shape[:2], dtype=np.int32)
    if masks:
        seg[masks[0]['mask'] > 0] = 1

    n_calls = n_ret = n_phantom = 0
    MAX_CALLS = 300
    for rnd in range(6):
        g, s, o = grasp_cgn(depth, K, seg, return_openings=True)
        for ci in range(len(g)):
            gb = cgn_to_gripper(g[ci], pose_mat)
            for yw in (0, 90, 180, 270):
                c_, s_ = np.cos(np.deg2rad(yw)), np.sin(np.deg2rad(yw))
                Ty = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]])
                Rt = gb[:3, :3] @ Ty
                pos = gb[:3, 3].copy()
                qw = qw_of(Rt)
                n_pyroki_before = len(pyroki_log)
                try:
                    q_sol = ctx.solve_ik(pos, qw)
                except RuntimeError:
                    n_calls += 1
                    if n_calls >= MAX_CALLS:
                        break
                    continue
                n_calls += 1
                n_ret += 1
                pb, Rb = fk(q_sol)
                dp = float(np.linalg.norm(pb - pos))
                da = float(np.rad2deg(np.arccos(
                    np.clip(np.dot(Rb[:, 2], Rt[:, 2]), -1, 1))))
                via = 'pyroki' if len(pyroki_log) > n_pyroki_before and \
                    np.allclose(pyroki_log[-1], q_sol, atol=1e-9) else 'DLS'
                print(f'[ret#{n_ret}] rnd{rnd} cand{ci} yaw={yw} via={via} '
                      f'FK误差 pos={dp:.4f} ori={da:.2f}°', flush=True)
                if da > 5.0 or dp > 0.005:
                    n_phantom += 1
                    print(f'  *** 幻影捕获 #{n_phantom} ***')
                    print(f'  q_sol = {q_sol.round(4)}')
                    print(f'  target pos = {pos.round(4)} quat_wxyz = {qw.round(4)}')
                    print(f'  via = {via}')
                    if len(pyroki_log) > n_pyroki_before:
                        print(f'  pyroki 本轮回传 = {pyroki_log[-1].round(4)}')
                if n_phantom >= 2 or n_calls >= MAX_CALLS:
                    break
            if n_phantom >= 2 or n_calls >= MAX_CALLS:
                break
        if n_phantom >= 2 or n_calls >= MAX_CALLS:
            break
    print(f'\n汇总: solve_ik 调用 {n_calls}, 返回 {n_ret}, 幻影 {n_phantom}')
    engine.close()


if __name__ == '__main__':
    main()
