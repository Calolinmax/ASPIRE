#!/usr/bin/env python
"""修复后真实 tilt 分布（裁决任务 2）+ a/b 杠杆数据（任务 3）。

任务 2: 5 seeds 全候选的真实 tilt（修复链 z 列）, 按库地图可达族分桶:
        陡降 <50° / 中间 50-80° / 侧抓 80-100° / 其他。
杠杆 a: 每候选的库支持度（dp<6cm & da<0.25rad 条目数）与 score 排名——
        "可达族候选占多少、排在哪"。
杠杆 b: 库中陡降(tilt<50°)构型的桌面 (x,y) 富集区 + 甜点区干净目标
        solve_ik 实测（x_base≈0.08 带, 库地图 min tilt 48°）。

用法: MUJOCO_GL=egl python -u scripts/archive/measure_true_tilts.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import aspire.robots  # noqa: F401
from aspire.engine.engine_capx import ExecutionEngineCapx
from aspire.api.primitives_capx import (
    PrimitiveContextCapx, cgn_to_gripper, _ik_library,
)
from aspire.perception.vision_client import segment_sam3_text_prompt, grasp_cgn
from robosuite.utils import transform_utils as T

SEEDS = [5, 12, 16, 18, 23]


def qw_of(Rm):
    q = T.mat2quat(Rm)
    return np.array([q[3], q[0], q[1], q[2]])


def bucket(t):
    if t < 50:
        return 'steep<50'
    if t < 80:
        return 'mid50-80'
    if t <= 100:
        return 'side80-100'
    return 'other>100'


def main():
    lib = _ik_library()
    P, Z = lib['P'], lib['Z']
    lib_tilts = np.rad2deg(np.arccos(np.clip(-Z[:, 2], -1, 1)))

    print('=' * 70)
    print('杠杆 b 数据 1/2: 库陡降构型(tilt<50°)的 (x,y) 富集区, z∈[0.08,0.20]')
    sel = (lib_tilts < 50) & (P[:, 2] > 0.08) & (P[:, 2] < 0.20)
    print(f'  条目数: {int(sel.sum())}')
    if sel.sum():
        xs, ys = P[sel, 0], P[sel, 1]
        for x0 in np.arange(-0.10, 0.30, 0.04):
            row = []
            for y0 in np.arange(-0.20, 0.21, 0.05):
                n = int(((xs >= x0) & (xs < x0 + 0.04) & (ys >= y0) & (ys < y0 + 0.05)).sum())
                row.append(f'{n:3d}')
            print(f'  x[{x0:+.2f},{x0 + 0.04:+.2f}): ' + ' '.join(row))
        print('   y 列: ' + ' '.join(f'{y0:+.2f}' for y0 in np.arange(-0.20, 0.21, 0.05)))

    summary = {}
    for seed in SEEDS:
        engine = ExecutionEngineCapx(task='Stack', seed=seed)
        ctx = PrimitiveContextCapx(engine)
        obs = ctx.get_observation()
        cam = obs['robot0_robotview']
        rgb, depth, K, pose_mat = (cam['images']['rgb'], cam['images']['depth'],
                                   cam['intrinsics'], cam['pose_mat'])
        masks = segment_sam3_text_prompt(rgb, 'red cube')
        seg = np.zeros(rgb.shape[:2], dtype=np.int32)
        if masks:
            seg[masks[0]['mask'] > 0] = 1
        g, s, o = grasp_cgn(depth, K, seg, return_openings=True)
        print('=' * 70)
        print(f'seed {seed}: {len(g)} 候选, table_offset(world)='
              f'{np.round(engine.env.table_offset, 3)}')
        counts = {}
        n_support = 0
        for i in range(len(g)):
            gb = cgn_to_gripper(g[i], pose_mat)
            t = float(np.rad2deg(np.arccos(np.clip(-gb[2, 2], -1, 1))))
            b = bucket(t)
            counts[b] = counts.get(b, 0) + 1
            dp = np.linalg.norm(P - gb[:3, 3], axis=1)
            da = np.arccos(np.clip(Z @ gb[:3, 2], -1, 1))
            sup = int(((dp < 0.06) & (da < 0.25)).sum())
            n_support += sup > 0
            if i < 6 or sup > 0:
                print(f'  cand{i:2d} score={s[i]:.3f} tilt={t:6.1f}° [{b}] '
                      f'lib_support={sup}')
        summary[seed] = (counts, n_support, len(g))
        print(f'  分桶: {counts}  库支持>0: {n_support}/{len(g)}')

        # 杠杆 b 数据 2/2: 甜点区干净目标实测（只在 seed 5 的引擎上做一次）
        if seed == 5:
            print('-' * 70)
            print('杠杆 b 数据 2/2: 甜点区 (x_base=0.08, y=0, z=0.12) 干净目标 IK 实测')
            for tilt_deg in (23.0, 48.0):
                hits = []
                for k in range(8):
                    az = 2 * np.pi * k / 8
                    zax = np.array([np.sin(np.deg2rad(tilt_deg)) * np.cos(az),
                                    np.sin(np.deg2rad(tilt_deg)) * np.sin(az),
                                    -np.cos(np.deg2rad(tilt_deg))])
                    yax = np.array([1.0, 0, 0]) - zax[0] * zax
                    yax /= np.linalg.norm(yax)
                    xax = np.cross(yax, zax)
                    Rm = np.column_stack([xax, yax, zax])
                    try:
                        ctx.solve_ik(np.array([0.08, 0.0, 0.12]), qw_of(Rm))
                        hits.append(int(np.rad2deg(az)))
                    except RuntimeError:
                        pass
                print(f'  tilt={tilt_deg:.0f}°: 收敛方位 {hits} ({len(hits)}/8)')
        engine.close()

    print('=' * 70)
    print('汇总:')
    for seed, (counts, nsupp, n) in summary.items():
        print(f'  seed {seed}: {counts}  可达族(lib_support>0) {nsupp}/{n}')


if __name__ == '__main__':
    main()
