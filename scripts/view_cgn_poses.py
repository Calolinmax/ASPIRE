#!/usr/bin/env python
"""CGN 候选位姿 3D 可视化（2026-08-03 用户要求）: 交互窗口中把 top-N 候选
画成半透明幽灵 glyph（掌杠+双指, 真实物理朝向: 指尖面=grip_site, 手指沿
-site_z 即真逼近方向）, top-1 品红高亮。

用法: MUJOCO_GL=glfw python scripts/view_cgn_poses.py [seed] [topN]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import aspire.robots  # noqa: F401
from aspire.engine_capx import ExecutionEngineCapx, TUCK_Q
from aspire.primitives_capx import PrimitiveContextCapx, cgn_to_gripper, PIPER_MAX_WIDTH
from aspire.vision_client import segment_sam3_text_prompt, grasp_cgn

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5
TOPN = int(sys.argv[2]) if len(sys.argv) > 2 else 8


def add_connector(scn, A, B, width, rgba):
    """在 user_scn 里加一段连接体（胶囊, 从 A 到 B）。"""
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
        np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(A, dtype=np.float64), np.asarray(B, dtype=np.float64))
    scn.ngeom += 1


def main():
    engine = ExecutionEngineCapx(task='Stack', seed=SEED)
    ctx = PrimitiveContextCapx(engine)
    sim = engine.env.sim
    for i in range(6):
        sim.data.set_joint_qpos(f'robot0_joint{i+1}', TUCK_Q[i])
    mujoco.mj_forward(sim.model._model, sim.data._data)

    obs = ctx.get_observation()
    cam = obs['robot0_robotview']
    rgb, depth, K, pose_mat = (cam['images']['rgb'], cam['images']['depth'],
                               cam['intrinsics'], cam['pose_mat'])
    masks = segment_sam3_text_prompt(rgb, 'red cube')
    seg = np.zeros(rgb.shape[:2], dtype=np.int32)
    if masks:
        seg[masks[0]['mask'] > 0] = 1
    g, s, o = grasp_cgn(depth, K, seg, return_openings=True)
    print(f'seed {SEED}: {len(g)} 候选, top{TOPN} 画入场景')
    print('品红 = top-1（score 最高）, 绿 = 其余')

    # 候选 → 世界系 glyph 段集（物理朝向: 指尖面=site, 手指沿 -site_z）
    glyphs = []  # (segments, is_top1)
    for i in range(min(TOPN, len(g))):
        gb = cgn_to_gripper(g[i], pose_mat)          # base 系
        Tw = engine.T_world_base @ gb                # → 世界系
        O = Tw[:3, 3]                                # 指尖接触面中点(=site)
        Sz = Tw[:3, 2]                               # site +z（指向掌心）
        Sy = Tw[:3, 1]                               # 开合轴
        approach = -Sz                               # 真逼近方向
        w = float(np.clip(o[i], 0.0, PIPER_MAX_WIDTH))  # 物理开度
        C = O - Sz * 0.050                           # 指尖极值（实测 -50mm）
        Pp = O + Sz * 0.023                          # 掌心可见表面（实测 +23mm, 用户审定版）
        segs = [
            (Pp - Sy * w / 2, Pp + Sy * w / 2),      # 掌杠
            (Pp + Sy * w / 2, C + Sy * w / 2),       # 指 +
            (Pp - Sy * w / 2, C - Sy * w / 2),       # 指 −
            (Pp, Pp + Sz * 0.05),                    # 尾刺（臂来方向）
        ]
        glyphs.append((segs, i == 0))
        t_deg = float(np.rad2deg(np.arccos(np.clip(approach[2] * -1, -1, 1))))
        print(f'  cand{i}: score={s[i]:.3f} tilt={t_deg:.1f}° '
              f'pos(base)={gb[:3,3].round(3)}')

    with mujoco.viewer.launch_passive(sim.model._model, sim.data._data) as v:
        for segs, is_top in glyphs:
            rgba = (1.0, 0.0, 1.0, 0.8) if is_top else (0.0, 1.0, 0.2, 0.45)
            for A, B in segs:
                add_connector(v.user_scn, A, B, 0.004, rgba)
        while v.is_running():
            sim_obj = v._sim()
            if getattr(sim_obj, 'run', 1):
                mujoco.mj_step(sim.model._model, sim.data._data)
            else:
                mujoco.mj_forward(sim.model._model, sim.data._data)
            v.sync()
            time.sleep(0.004)
    engine.close()


if __name__ == '__main__':
    main()
