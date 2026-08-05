#!/usr/bin/env python
"""场景验收 showcase（2026-08-03 用户三改: 全零 home / 相机拉高 / 摆放移近）。

每 seed: home 观测(入 trace) → 收臂 tuck(入 trace) → tuck 观测 → SAM3+CGN
(标注图入 trace)。产出含 home/tuck 画面与 CGN 标注图的完整 trace 供肉审。

用法: MUJOCO_GL=egl python -u scripts/showcase_scene.py [seed ...]
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aspire.robots  # noqa: F401
from aspire.engine_capx import ExecutionEngineCapx, TUCK_Q
from aspire.primitives_capx import build_namespace, cgn_to_gripper, _ik_library


def main():
    seeds = [int(a) for a in sys.argv[1:]] or [5, 12, 16]
    lib = _ik_library()
    P, Z = lib['P'], lib['Z']

    for seed in seeds:
        engine = ExecutionEngineCapx(task='Stack', seed=seed, trace_root='traces')
        ns = build_namespace(engine)
        print(f'=== seed {seed}  trace: {engine.trace_dir}')
        obs_home = ns['get_observation']()          # home 画面入 trace
        ns['move_to_joints'](TUCK_Q)                # 收臂（入 trace）
        obs = ns['get_observation']()               # tuck 画面入 trace
        cam = obs['robot0_robotview']
        rgb, depth, K, pose_mat = (cam['images']['rgb'], cam['images']['depth'],
                                   cam['intrinsics'], cam['pose_mat'])
        masks = ns['segment_sam3_text_prompt'](rgb, 'red cube')
        seg = np.zeros(rgb.shape[:2], dtype=np.int32)
        npx = 0
        if masks:
            seg[masks[0]['mask'] > 0] = 1
            npx = int((seg > 0).sum())
        grasps, scores, openings = ns['grasp_cgn'](rgb, depth, K, seg)  # 标注图入 trace
        tilts = [float(np.rad2deg(np.arccos(np.clip(
            -cgn_to_gripper(grasps[i], pose_mat)[2, 2], -1, 1))))
            for i in range(len(grasps))]
        print(f'  mask_px={npx} 候选={len(grasps)} '
              f'tilts={[round(t, 1) for t in sorted(tilts)[:6]]}...')
        engine.mark_trace_result(True, None)
        engine.close()


if __name__ == '__main__':
    main()
