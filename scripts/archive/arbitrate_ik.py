#!/usr/bin/env python
"""IK 裁决实验（2026-08-03 用户判断"位姿可达, IK 本身有问题"）:
用【独立多起点数值 IK】求解 CGN 真实候选位姿, 绕过我方 solve_ik 全栈
(pyroki/DLS/碰撞门), 直接回答: 该位姿在当前模型限位下是否存在关节解。

判读:
  best ori_err ≈ 0  → 用户正确, 我方 IK 栈有 bug → 定位修复
  best ori_err 几十度 → 运动学墙属实（限位内无解）

同时输出: 位置-only 可达性、pyroki 原始返回的残差（检验 pyroki 路径死活）。

用法: MUJOCO_GL=egl python -u scripts/archive/arbitrate_ik.py [seed]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mujoco
import aspire.robots  # noqa: F401
from aspire.engine.engine_capx import ExecutionEngineCapx, TUCK_Q
from aspire.api.primitives_capx import (
    PrimitiveContextCapx, cgn_to_gripper, EEF_SITE, ARM_JOINT_NAMES,
)
from aspire.perception.vision_client import segment_sam3_text_prompt, grasp_cgn
from robosuite.utils import transform_utils as T

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def main():
    engine = ExecutionEngineCapx(task='Stack', seed=SEED)
    ctx = PrimitiveContextCapx(engine)
    sim = engine.env.sim
    m = sim.model._model
    for i in range(6):
        sim.data.set_joint_qpos(f'robot0_joint{i+1}', TUCK_Q[i])
    mujoco.mj_forward(m, sim.data._data)
    obs = ctx.get_observation()
    cam = obs['robot0_robotview']
    rgb, depth, K, pose_mat = (cam['images']['rgb'], cam['images']['depth'],
                               cam['intrinsics'], cam['pose_mat'])
    masks = segment_sam3_text_prompt(rgb, 'red cube')
    seg = np.zeros(rgb.shape[:2], dtype=np.int32)
    if masks:
        seg[masks[0]['mask'] > 0] = 1
    g, s, o = grasp_cgn(depth, K, seg, return_openings=True)
    gb = cgn_to_gripper(g[0], pose_mat)
    tilt = np.rad2deg(np.arccos(np.clip(-gb[2, 2], -1, 1)))
    print(f'cand0: score={s[0]:.3f} tilt={tilt:.1f}° pos(base)={gb[:3,3].round(4)}')

    # ---- 独立 IK 装置（不经过我方 solve_ik）----
    site_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
    jids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINT_NAMES]
    qposadr = [int(m.jnt_qposadr[j]) for j in jids]
    dofadr = [int(m.jnt_dofadr[j]) for j in jids]
    q_lo = np.array([m.jnt_range[j][0] for j in jids])
    q_hi = np.array([m.jnt_range[j][1] for j in jids])
    data = mujoco.MjData(m)

    def fk(q):
        for i, adr in enumerate(qposadr):
            data.qpos[adr] = q[i]
        mujoco.mj_forward(m, data)
        return (data.site_xpos[site_id].copy(),
                data.site_xmat[site_id].reshape(3, 3).copy())

    def rot_err(Rt, Rc):
        q = T.mat2quat(Rt @ Rc.T)
        if q[3] < 0:
            q = -q
        return T.quat2axisangle(q)

    def dls(q0, tgt_p_w, Rt_w, w_ori=1.0, max_iter=800):
        """独立 DLS: 世界系目标, 限位投影, 返回 (q, pos_err, ori_err_rad)。"""
        q = np.clip(q0.copy(), q_lo, q_hi)
        jacp = np.zeros((3, m.nv))
        jacr = np.zeros((3, m.nv))
        best = (q.copy(), np.inf, np.inf)
        for _ in range(max_iter):
            p, R = fk(q)
            ep = tgt_p_w - p
            er = rot_err(Rt_w, R) * w_ori
            np_, nr_ = np.linalg.norm(ep), np.linalg.norm(rot_err(Rt_w, R))
            if np_ + nr_ < best[1] + best[2]:
                best = (q.copy(), np_, nr_)
            if np_ < 5e-4 and nr_ < 5e-3:
                break
            if np_ > 0.05:
                ep *= 0.05 / np_
            ner = np.linalg.norm(er)
            if ner > 0.3:
                er *= 0.3 / ner
            mujoco.mj_jacSite(m, data, jacp, jacr, site_id)
            J = np.vstack([jacp[:, dofadr], jacr[:, dofadr]])
            e = np.concatenate([ep, er])
            dq = J.T @ np.linalg.solve(J @ J.T + 0.05**2 * np.eye(6), e)
            q = np.clip(q + np.clip(dq, -0.3, 0.3), q_lo, q_hi)
        return best

    rng = np.random.default_rng(7)
    starts = [rng.uniform(q_lo, q_hi) for _ in range(120)]
    starts.append(np.zeros(6))
    starts.append(np.array([0.0, 1.57, -1.3485, 0, 0, 0]))

    for tag, gpos, gR in (
        ('grasp点', gb[:3, 3].copy(), gb[:3, :3]),
        ('pre-grasp(+10cm)', gb[:3, 3] - gb[:3, 2] * 0.10, gb[:3, :3]),
    ):
        tgt_p_w = (engine.T_world_base @ np.append(gpos, 1.0))[:3]
        Rt_w = engine.T_world_base[:3, :3] @ gR
        # 1) 位置-only 可达性
        best_pos = np.inf
        for q0 in starts[:40]:
            _, pe, _ = dls(q0, tgt_p_w, Rt_w, w_ori=0.0, max_iter=300)
            best_pos = min(best_pos, pe)
        # 2) 全 6D
        results = [dls(q0, tgt_p_w, Rt_w, max_iter=800) for q0 in starts]
        results.sort(key=lambda r: r[1] + r[2])
        q_b, pe_b, re_b = results[0]
        print(f'--- {tag} ---')
        print(f'  位置-only 最优误差: {best_pos*1000:.2f} mm')
        print(f'  全6D 最优: pos_err={pe_b*1000:.2f} mm  ori_err={np.degrees(re_b):.2f}°')
        print(f'  最优解 q={q_b.round(3)}')
        top5 = [f'({r[1]*1000:.1f}mm,{np.degrees(r[2]):.1f}°)' for r in results[:5]]
        print(f'  top5 残差: {top5}')

    # 3) pyroki 原始返回的残差（检验 pyroki 路径）
    from aspire.planning.pyroki_client import ik_pyroki
    from aspire.api.primitives_capx import _q_in
    def qw_of(Rm):
        q = T.mat2quat(Rm)
        return np.array([q[3], q[0], q[1], q[2]])
    try:
        q_pk = np.asarray(ik_pyroki(gb[:3, 3].copy(), _q_in(qw_of(gb[:3, :3])),
                                    prev_cfg=engine.current_arm_qpos()))[:6]
        p2, R2 = fk(np.clip(q_pk, q_lo, q_hi))
        pw = (engine.T_base_world @ np.append(p2, 1.0))[:3]
        Rb2 = engine.T_base_world[:3, :3] @ R2
        pe = np.linalg.norm(pw - gb[:3, 3])
        re = np.degrees(np.arccos(np.clip((np.trace(Rb2 @ gb[:3, :3].T) - 1) / 2, -1, 1)))
        print(f'pyroki 原始返回残差: pos={pe*1000:.1f}mm ori={re:.1f}°')
    except Exception as e:
        print('pyroki 调用异常:', type(e).__name__, str(e)[:120])
    engine.close()


if __name__ == '__main__':
    main()
