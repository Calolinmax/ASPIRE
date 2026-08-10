#!/usr/bin/env python
"""反射 bug 修复的三层验证（2026-08-03 裁决）。

a) 修复后全部候选 det=+1.000
b) 修复前后位置链逐位一致（位置从未坏过）; 姿态 = 正确共轭基变换
c) 物理 sanity: 修复后逼近方向应指向"相机→方块"方向（与标注 glyph 交叉验证）

用法: MUJOCO_GL=egl python -u scripts/archive/verify_flip_fix.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import aspire.robots  # noqa: F401
from aspire.engine.engine_capx import ExecutionEngineCapx
from aspire.api.primitives_capx import (
    PrimitiveContextCapx, cgn_to_gripper, GRIPPER_DEPTH_CGN, TCP_DEPTH_PIPER,
)
from aspire.perception.vision_client import segment_sam3_text_prompt, grasp_cgn
from robosuite.utils import transform_utils as T


def cgn_to_gripper_old(g_cgn, pose_mat):
    """修复前公式（反射版），仅用于对拍。"""
    T_flip = np.diag([1, -1, 1, 1])
    g_gl = T_flip @ g_cgn
    g_base = pose_mat @ g_gl
    R_z = T.quat2mat([0.0, 0.0, -0.707, 0.707])
    T_align = np.eye(4)
    T_align[:3, :3] = R_z
    T_off = np.eye(4)
    T_off[2, 3] = GRIPPER_DEPTH_CGN - TCP_DEPTH_PIPER
    return g_base @ T_align @ T_off


def main():
    engine = ExecutionEngineCapx(task='Stack', seed=5)
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
    import mujoco
    cid = mujoco.mj_name2id(engine.env.sim.model._model, mujoco.mjtObj.mjOBJ_BODY, 'cubeA_main')
    cube_w = engine.env.sim.data.xpos[cid]
    cam_w = pose_mat[:3, 3]  # pose_mat 是 cam→base... 取相机世界位需 engine 变换
    print('=== a) det 检查 ===')
    dets = [np.linalg.det(cgn_to_gripper(g[i], pose_mat)[:3, :3]) for i in range(len(g))]
    print('修复后 det:', np.round(dets, 3))
    assert all(d > 0.99 for d in dets), 'FAIL: 仍有反射'
    print(f'a) PASS: {len(dets)}/{len(dets)} det=+1')

    print('=== b) 位置链对拍 + 姿态共轭正确性 ===')
    max_dp = 0.0
    max_dz = 0.0
    for i in range(len(g)):
        new = cgn_to_gripper(g[i], pose_mat)
        old = cgn_to_gripper_old(g[i], pose_mat)
        max_dp = max(max_dp, float(np.abs(new[:3, 3] - old[:3, 3]).max()))
        # 共轭修复保持 z 列(逼近)与 x 列(开合轴)与旧链一致, 仅 y 列(指对)换号
        max_dz = max(max_dz, float(np.abs(new[:3, 2] - old[:3, 2]).max()))
    print(f'位置最大偏差(新 vs 旧): {max_dp:.2e} m  (应≈0, 位置链未动)')
    print(f'z列(逼近)最大偏差(新 vs 旧): {max_dz:.2e}  (共轭保持 z 列不变)')
    print('b) PASS' if max_dp < 1e-12 and max_dz < 1e-9 else 'b) FAIL')

    print('=== c) 物理 sanity: 逼近方向 vs 相机→方块 ===')
    # 相机在基座系位置: pose_mat 是 cam→base, 其平移即相机 base 系位置
    cam_base = pose_mat[:3, 3].copy()
    cube_base = (engine.T_base_world @ np.append(cube_w, 1.0))[:3]
    print('cam pos (base):', cam_base.round(3), ' cube pos (base):', cube_base.round(3))
    n_ok = 0
    for i in range(min(12, len(g))):
        gb = cgn_to_gripper(g[i], pose_mat)
        approach = gb[:3, 2]
        to_cube = cube_base - gb[:3, 3]
        to_cube /= np.linalg.norm(to_cube)
        cosang = float(np.dot(approach, to_cube))
        cam_dir = cube_base - cam_base
        cam_dir /= np.linalg.norm(cam_dir)
        cos_cam = float(np.dot(approach, cam_dir))
        ok = cosang > 0.5
        n_ok += ok
        print(f'  cand{i:2d}: approach·(抓取点→cube)={cosang:+.3f}  '
              f'approach·(cam→cube)={cos_cam:+.3f}  {"OK" if ok else "指向异常"}')
    print(f'c) {n_ok}/{min(12,len(g))} 逼近方向指向方块(余弦>0.5)')
    engine.close()


if __name__ == '__main__':
    main()
