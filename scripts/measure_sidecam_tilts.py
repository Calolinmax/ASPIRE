#!/usr/bin/env python
"""侧相机+细高块+tuck 的 CGN 真实分布（2026-08-03 裁决 4a）。

3 seeds × (tuck 位姿 → 观测 → SAM3 → CGN → 修复链 tilt 分布 + lib_support)。
判据: 分布主体进入 50-80°; lib_support>0 比例。

用法: MUJOCO_GL=egl python -u scripts/measure_sidecam_tilts.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import aspire.robots  # noqa: F401
from aspire.engine_capx import ExecutionEngineCapx, TUCK_Q
from aspire.primitives_capx import (
    PrimitiveContextCapx, cgn_to_gripper, _ik_library,
)
from aspire.vision_client import segment_sam3_text_prompt, grasp_cgn

SEEDS = [5, 12, 16]


def main():
    lib = _ik_library()
    P, Z = lib['P'], lib['Z']
    lib_tilts = np.rad2deg(np.arccos(np.clip(-Z[:, 2], -1, 1)))

    for seed in SEEDS:
        engine = ExecutionEngineCapx(task='Stack', seed=seed)
        ctx = PrimitiveContextCapx(engine)
        sim = engine.env.sim
        m = sim.model._model
        # 收臂让拍
        for i in range(6):
            sim.data.set_joint_qpos(f'robot0_joint{i+1}', TUCK_Q[i])
        mujoco.mj_forward(m, sim.data._data)

        obs = ctx.get_observation()
        cam = obs['robot0_robotview']
        rgb, depth, K, pose_mat = (cam['images']['rgb'], cam['images']['depth'],
                                   cam['intrinsics'], cam['pose_mat'])
        masks = segment_sam3_text_prompt(rgb, 'red cube')
        seg = np.zeros(rgb.shape[:2], dtype=np.int32)
        npx = 0
        if masks:
            seg[masks[0]['mask'] > 0] = 1
            npx = int((seg > 0).sum())

        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'cubeA_main')
        cube_b = (engine.T_base_world @ np.append(sim.data.xpos[cid], 1.0))[:3]
        sel = np.linalg.norm(P - cube_b, axis=1) < 0.06
        min_t = float(lib_tilts[sel].min()) if sel.sum() else float('nan')

        g, s, o = grasp_cgn(depth, K, seg, return_openings=True)
        tilts, supps = [], []
        for i in range(len(g)):
            gb = cgn_to_gripper(g[i], pose_mat)
            t = float(np.rad2deg(np.arccos(np.clip(-gb[2, 2], -1, 1))))
            dp = np.linalg.norm(P - gb[:3, 3], axis=1)
            da = np.arccos(np.clip(Z @ gb[:3, 2], -1, 1))
            supp = int(((dp < 0.06) & (da < 0.25)).sum())
            tilts.append(t)
            supps.append(supp)
        tilts = np.array(tilts)
        supps = np.array(supps)
        print(f'=== seed {seed}: cube_z(base)={cube_b[2]:.3f} mask_px={npx} '
              f'候选={len(g)} 库minTilt@该位={min_t:.1f}°')
        if len(g):
            body = ((tilts >= 50) & (tilts <= 80)).mean()
            print(f'  tilt: min={tilts.min():.1f} p25={np.percentile(tilts,25):.1f} '
                  f'med={np.median(tilts):.1f} p75={np.percentile(tilts,75):.1f} '
                  f'max={tilts.max():.1f}')
            print(f'  50-80°主体占比: {body:.0%}  lib_support>0: '
                  f'{int((supps > 0).sum())}/{len(g)}')
            for i in range(min(8, len(g))):
                print(f'    cand{i:2d} score={s[i]:.3f} tilt={tilts[i]:5.1f}° '
                      f'support={supps[i]}')
        engine.close()


if __name__ == '__main__':
    main()
