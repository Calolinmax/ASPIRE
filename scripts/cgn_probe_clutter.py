#!/usr/bin/env python
"""CGN 探针: Stack 杂物场景（engine 原生 clutter=4）端到端验证。

背景: CGN 训练分布 = 多物杂乱场景 + RealSense 噪声。干净单方块 sim 是双重 OOD，
方块面 contact score 被压在阈值下（0.16-0.24 vs first_thres 0.23）→ 0 候选。
加杂物后方块面分数升至 ~0.265，端到端 ~1/3 出候选（姿态相关）。

用法:
    MUJOCO_GL=egl python scripts/cgn_probe_clutter.py [--seed 0] [--clutter 4]
输出:
    服务端候选数 + 宽度过滤前后 top-1 的距GT中心/距表面/D3 + /tmp/cgn_scene_clutter.npz
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aspire.robots  # noqa: F401 注册 Piper
from aspire.engine_capx import ExecutionEngineCapx
from aspire.primitives_capx import (
    PrimitiveContextCapx,
    cgn_to_gripper,
    filter_grasps_by_width,
)
from aspire.vision_client import segment_sam3_text_prompt, grasp_cgn


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--clutter', type=int, default=None,
                        help='干扰物数量 (默认 None=Stack 时 4; 0=干净场景)')
    args = parser.parse_args()

    import mujoco

    engine = ExecutionEngineCapx(task="Stack", seed=args.seed, clutter=args.clutter)
    ctx = PrimitiveContextCapx(engine)
    obs = ctx.get_observation()
    cam = obs["robot0_robotview"]
    rgb = cam["images"]["rgb"]
    depth = cam["images"]["depth"]
    K = cam["intrinsics"]
    pose_mat = cam["pose_mat"]

    masks = segment_sam3_text_prompt(rgb, "red cube")
    seg = np.zeros(rgb.shape[:2], dtype=np.int32)
    if masks:
        seg[masks[0]["mask"] > 0] = 1
        print('SAM3 mask pixels:', int((masks[0]["mask"] > 0).sum()))
    else:
        print('WARNING: SAM3 no mask, fallback seg[:]=1')
        seg[:] = 1

    d = depth[seg == 1]
    print('depth in seg: min %.4f max %.4f mean %.4f' % (d.min(), d.max(), d.mean()))

    grasps, scores, openings = grasp_cgn(depth, K, seg, return_openings=True)
    print('service path: grasps', grasps.shape, 'scores',
          scores.round(4) if len(scores) else scores)
    if len(openings):
        print('openings:', openings.round(4))

    model_ = engine.env.sim.model._model
    cid = mujoco.mj_name2id(model_, mujoco.mjtObj.mjOBJ_BODY, "cubeA_main")
    gt_world = engine.env.sim.data.xpos[cid].copy()
    gt_base = (engine.T_base_world @ np.append(gt_world, 1.0))[:3]
    cube_half = 0.02  # cubeA size_min/max = 0.02 (半边长)
    print('GT cube (base):', gt_base.round(4), ' 半边长: %.3f m' % cube_half)

    def report_top(grasps_, scores_, tag):
        if not len(grasps_):
            print(f'{tag}: N/A (0 candidates)')
            return
        grasps_final = np.array([cgn_to_gripper(g, pose_mat) for g in grasps_])
        for i, g in enumerate(grasps_final[:5]):
            p = g[:3, 3]
            d_center = np.linalg.norm(p - gt_base)
            q = np.abs(p - gt_base) - cube_half
            d_surf = float(np.linalg.norm(np.maximum(q, 0.0)))
            print(f'{tag} G{i}: pos(base)={p.round(4)}  距GT中心={d_center * 100:.2f} cm  '
                  f'距表面={d_surf * 100:.2f} cm  D3={-g[2, 2]:.3f}  score={scores_[i]:.4f}')
        top = grasps_final[0]
        p = top[:3, 3]
        d_center = np.linalg.norm(p - gt_base)
        q = np.abs(p - gt_base) - cube_half
        d_surf = float(np.linalg.norm(np.maximum(q, 0.0)))
        d3 = -top[2, 2]
        ok = (d_surf < 0.02) and (d3 > 0.9)
        print('%s TOP-1: 距GT中心 %.2f cm, 距表面 %.2f cm (门限<2cm), D3 %.3f (门限>0.9) => %s'
              % (tag, d_center * 100, d_surf * 100, d3, 'PASS' if ok else 'FAIL'))

    # 变换链验收：在【宽度过滤前】的 top-1 上测量（过滤与变换无关）
    report_top(grasps, scores, '[pre-filter]')
    # 执行管线视角：宽度过滤后
    grasps_f, scores_f, openings_f = filter_grasps_by_width(grasps, scores, openings)
    print('after width filter: grasps', grasps_f.shape)
    report_top(grasps_f, scores_f, '[post-filter]')

    np.savez('/tmp/cgn_scene_clutter.npz',
             depth=np.asarray(depth, dtype=np.float32),
             K=np.asarray(K, dtype=np.float64),
             seg=seg.astype(np.int32), rgb=rgb)
    print('saved /tmp/cgn_scene_clutter.npz')
    engine.close()


if __name__ == '__main__':
    main()
