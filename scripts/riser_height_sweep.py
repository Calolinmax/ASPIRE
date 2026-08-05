#!/usr/bin/env python
"""双高度 CGN 真实分布对比（2026-08-03 裁决 1）: RISER_H ∈ {0.28, 0.34}
× seeds {5,12,16}, 测修复链真实 tilt 分布与可达带重叠。

重叠判据（双指标）:
  band_overlap: tilt ≥ 该 seed 方块位该高度的库最小可达 tilt（6cm 球）
  lib_support>0: (dp<6cm & da<0.25rad) 的 FK 认证条目数>0 —— 直接可达性

用法: MUJOCO_GL=egl python -u scripts/riser_height_sweep.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import aspire.robots  # noqa: F401
import aspire.engine_capx as ec
from aspire.primitives_capx import (
    PrimitiveContextCapx, cgn_to_gripper, _ik_library,
)
from aspire.vision_client import segment_sam3_text_prompt, grasp_cgn

HEIGHTS = [0.28, 0.34]
SEEDS = [5, 12, 16]


def main():
    lib = _ik_library()
    P, Z = lib['P'], lib['Z']
    lib_tilts = np.rad2deg(np.arccos(np.clip(-Z[:, 2], -1, 1)))

    results = {}
    for H in HEIGHTS:
        ec.RISER_H = H  # 模块常量在 _load_model 运行时读取 → 覆盖生效
        for seed in SEEDS:
            engine = ec.ExecutionEngineCapx(task='Stack', seed=seed)
            ctx = PrimitiveContextCapx(engine)
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
            # 方块实际位（基座系）
            cid = mujoco.mj_name2id(engine.env.sim.model._model,
                                    mujoco.mjtObj.mjOBJ_BODY, 'cubeA_main')
            cube_b = (engine.T_base_world @ np.append(
                engine.env.sim.data.xpos[cid], 1.0))[:3]
            # 该位该 z 的库最小可达 tilt（6cm 球）
            sel = np.linalg.norm(P - cube_b, axis=1) < 0.06
            min_t = float(lib_tilts[sel].min()) if sel.sum() else float('nan')

            g, s, o = grasp_cgn(depth, K, seg, return_openings=True)
            tilts = []
            n_band = n_supp = 0
            for i in range(len(g)):
                gb = cgn_to_gripper(g[i], pose_mat)
                t = float(np.rad2deg(np.arccos(np.clip(-gb[2, 2], -1, 1))))
                tilts.append(t)
                dp = np.linalg.norm(P - gb[:3, 3], axis=1)
                da = np.arccos(np.clip(Z @ gb[:3, 2], -1, 1))
                sup = int(((dp < 0.06) & (da < 0.25)).sum())
                if t >= min_t - 1.0:
                    n_band += 1
                if sup > 0:
                    n_supp += 1
            tilts = np.array(tilts)
            results[(H, seed)] = (len(g), n_band, n_supp, min_t, tilts)
            print(f'H={H} seed={seed}: cube_z={cube_b[2]:.3f} mask_px={npx} '
                  f'候选={len(g)} 库minTilt={min_t:.1f}°')
            if len(g):
                print(f'  tilt 分布: min={tilts.min():.1f} p25='
                      f'{np.percentile(tilts,25):.1f} med={np.median(tilts):.1f} '
                      f'p75={np.percentile(tilts,75):.1f} max={tilts.max():.1f}')
                print(f'  重叠: band(tilt≥{min_t - 1:.1f}°)={n_band}/{len(g)}  '
                      f'lib_support>0={n_supp}/{len(g)}')
            engine.close()

    print('=' * 70)
    print('汇总（重叠候选数/总数）:')
    for H in HEIGHTS:
        tot = sum(results[(H, s)][1] for s in SEEDS)
        n = sum(results[(H, s)][0] for s in SEEDS)
        sup = sum(results[(H, s)][2] for s in SEEDS)
        print(f'  H={H}: band 重叠 {tot}/{n} ({tot / max(n, 1):.0%}), '
              f'lib_support {sup}/{n}')


if __name__ == '__main__':
    main()
