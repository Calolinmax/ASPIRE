#!/usr/bin/env python
"""fallback 三种失效模式归因（2026-08-03, 裁决任务 2a）+ 库地图分布（2b 参数依据）。

Part 1 (离线): 解码门禁日志里的 fallback quat → 逼近向量/距竖直倾角/方位角。
Part 2 (在线, seed 16/18): 同 seed 复现场景 → _plan_grasp_geometric 内部量
    (PCA 法线、与竖直夹角、snap 是否触发、8 候选逼近), 并验证前 3 候选 quat
    与门禁日志逐位一致（复现有效性证据）。顺带测场景深度量程（任务 3）。
Part 3 (离线): ik_library 陡降条目 tilt/azimuth 分布 → fallback 采样族参数。

用法: MUJOCO_GL=egl python -u scripts/diagnose_fallback_attribution.py
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DOWN = np.array([0.0, 0.0, -1.0])


def approach_from_quat_wxyz(q):
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
        [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
        [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
    ])
    return R[:, 2]  # 局部 z = 逼近轴


def tilt_az(a):
    tilt = np.rad2deg(np.arccos(np.clip(-a[2], -1, 1)))  # 与竖直向下夹角
    az = np.rad2deg(np.arctan2(a[1], a[0]))
    return tilt, az


# ---- Part 1: 门禁日志 quat 解码（原文照抄自 logs/gate/gate_s*.log）----
LOGGED = {
    5: [[0.0, 0.0, 1.0, 5.9604645e-08],
        [1.3196838e-17, 3.8268343e-01, 9.2387950e-01, -4.4351902e-17],
        [8.7692221e-18, 7.0710677e-01, 7.0710677e-01, -8.7692221e-18]],
    12: [[0.0, 0.0, 1.0, 5.9604645e-08],
         [1.3196838e-17, 3.8268343e-01, 9.2387950e-01, -4.4351902e-17],
         [8.7692221e-18, 7.0710677e-01, 7.0710677e-01, -8.7692221e-18]],
    16: [[5.9604645e-08, 0.0, 0.0, -1.0],
         [3.8268334e-01, 0.0, -1.1920929e-07, -9.2387944e-01],
         [7.0710677e-01, 0.0, 2.9458796e-08, -7.0710683e-01]],
    18: [[0.30914205, 0.0572064, 0.94249225, -0.11343327],
         [0.32901913, 0.41352797, 0.8488574, 0.01350488],
         [0.29880583, 0.70689356, 0.6259915, 0.13838698]],
    23: [[0.0, 0.0, 1.0, 5.9604645e-08],
         [1.3196838e-17, 3.8268343e-01, 9.2387950e-01, -4.4351902e-17],
         [8.7692221e-18, 7.0710677e-01, 7.0710677e-01, -8.7692221e-18]],
}

print('=' * 64)
print('Part 1: 门禁日志 fallback quat 解码（基座系逼近轴 = R @ zhat）')
for s, qs in LOGGED.items():
    for i, q in enumerate(qs):
        a = approach_from_quat_wxyz(np.array(q, dtype=float))
        t, az = tilt_az(a)
        print(f'  seed {s} cand{i}: approach=({a[0]:+.3f},{a[1]:+.3f},{a[2]:+.3f})'
              f'  tilt={t:5.1f}°  azimuth={az:+7.1f}°')

# ---- Part 3: 库地图分布（放 Part 2 前, 无需引擎）----
print('=' * 64)
print('Part 3: ik_library 陡降条目分布')
lib = np.load('aspire/robots/assets/piper/ik_library.npz')
print('  keys:', list(lib.keys()))
P, Z = lib['P'], lib['Z']
print('  总条目:', len(P))
tilts = np.rad2deg(np.arccos(np.clip(-Z[:, 2], -1, 1)))
for th in (25, 35, 45, 55):
    print(f'  tilt<{th}°: {int((tilts < th).sum())}')
steep = tilts < 55
az = np.rad2deg(np.arctan2(Z[:, 1], Z[:, 0]))
print('  陡降集(tilt<55°) tilt 百分位:',
      np.percentile(tilts[steep], [10, 25, 50, 75, 90]).round(1))
# 20-26° 带的方位覆盖（8 扇区, 每 45°）
band = (tilts >= 18) & (tilts <= 30)
print(f'  舒适带 tilt∈[18,30]°: {int(band.sum())} 条')
for k in range(8):
    lo_, hi_ = -180 + 45 * k, -180 + 45 * (k + 1)
    n = int(((az >= lo_) & (az < hi_) & band).sum())
    if band.sum():
        px = P[band & (az >= lo_) & (az < hi_)]
        xr = f'x[{px[:,0].min():.2f},{px[:,0].max():.2f}]' if n else '-'
        print(f'    方位扇区 [{lo_:+4d},{hi_:+4d}): {n:5d} 条  {xr}')
# seed 18 fallback 目标附近的可达性
tgt = np.array([0.254, -0.095, 0.192])
dp = np.linalg.norm(P - tgt, axis=1)
near = dp < 0.06
print(f'  seed18 fallback 目标 {tgt} 6cm 内条目: {int(near.sum())}')
if near.sum():
    print('    其 tilt 百分位:', np.percentile(tilts[near], [0, 50, 100]).round(1),
          ' 最小 tilt 条目的 azimuth:',
          round(float(az[near][np.argmin(tilts[near])]), 1))

# ---- Part 2: 在线复现 seed 16/18 ----
print('=' * 64)
print('Part 2: 在线复现（同 seed, 验证 quat 逐位一致）')
import aspire.robots  # noqa: F401
from aspire.engine_capx import ExecutionEngineCapx
from aspire.primitives_capx import PrimitiveContextCapx
from aspire.vision_client import segment_sam3_text_prompt
from robosuite.utils import transform_utils as T

for seed in (16, 18):
    engine = ExecutionEngineCapx(task='Stack', seed=seed)
    ctx = PrimitiveContextCapx(engine)
    obs = ctx.get_observation()
    cam = obs['robot0_robotview']
    rgb, depth, K = cam['images']['rgb'], cam['images']['depth'], cam['intrinsics']
    masks = segment_sam3_text_prompt(rgb, 'red cube')
    seg = np.zeros(rgb.shape[:2], dtype=np.int32)
    if masks:
        seg[masks[0]['mask'] > 0] = 1
    d = depth[:, :, 0] if depth.ndim == 3 else depth
    valid = np.isfinite(d) & (d > 0.05)
    print(f'--- seed {seed}: mask px={int((seg > 0).sum())}')
    print(f'    深度量程: 全场 valid [{d[valid].min():.3f}, {d[valid].max():.3f}]  '
          f'p1/p50/p99={np.percentile(d[valid], [1, 50, 99]).round(3)}')
    if (seg > 0).any():
        dm = d[(seg > 0) & valid]
        print(f'    mask 区深度 [{dm.min():.3f}, {dm.max():.3f}]')
    g, s = ctx._plan_grasp_geometric(depth, K, seg)
    for i in range(min(3, len(g))):
        q_xyzw = T.mat2quat(g[i][:3, :3])
        q_wxyz = np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])
        a = g[i][:3, 2]
        t_, az_ = tilt_az(a)
        print(f'    cand{i}: quat_wxyz={q_wxyz.round(6)}  tilt={t_:.1f}°  az={az_:+.1f}°')
        logged = np.array(LOGGED[seed][i])
        print(f'           与日志 quat 最大偏差: {np.abs(q_wxyz - logged).max():.2e}')
    engine.close()
