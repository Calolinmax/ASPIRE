#!/usr/bin/env python
"""直接加载 CGN wrapper，用合成数据推理，验证是否复现 illegal address。

宿主机/容器双环境：
- 宿主机：无环境变量时使用硬编码宿主机路径（历史行为不变）
- 容器：ENV CGN_REPO=/workspace, CGN_CKPT=/workspace/checkpoint/model.ckpt-144144
        （由 Dockerfile.cgn 设置），且跳过 pip nvidia 库路径 hack
"""
import os
import sys
import numpy as np

PYTHON = sys.executable

import subprocess
try:
    NVIDIA_BASE = subprocess.check_output(
        [PYTHON, '-c', 'import os, nvidia; print(os.path.dirname(nvidia.__file__))'],
        text=True,
    ).strip()
    import glob
    LD_DIRS = ':'.join(d for d in glob.glob(os.path.join(NVIDIA_BASE, '*', 'lib')) + glob.glob(os.path.join(NVIDIA_BASE, '*', '*', 'lib')) if os.path.isdir(d))
    os.environ['LD_LIBRARY_PATH'] = LD_DIRS + ':' + os.environ.get('LD_LIBRARY_PATH', '')
except subprocess.CalledProcessError:
    pass  # 容器（NGC）内 CUDA 为系统安装，无 pip nvidia 包，无需此 hack
os.environ['TF_USE_LEGACY_KERAS'] = '1'

REPO = os.environ.get('CGN_REPO', '/home/stouching/Desktop/ASPIRE/external/contact_graspnet')
sys.path.insert(0, REPO)
sys.path.insert(0, os.path.join(REPO, 'contact_graspnet'))
sys.path.insert(0, os.path.join(REPO, 'pointnet2'))
sys.path.insert(0, os.path.join(REPO, 'pointnet2', 'utils'))

from wrapper import ContactGraspnet

CKPT = os.environ.get(
    'CGN_CKPT',
    '/home/stouching/Desktop/ASPIRE/external/cgn_models/cgn_models/'
    'scene_test_2048_bs3_hor_sigma_001-20260730T045031Z-1-001/'
    'scene_test_2048_bs3_hor_sigma_001/model.ckpt-144144',
)

model = ContactGraspnet(CKPT)

# 合成深度图和内参（模仿 realsense）
H, W = 480, 640
K = np.array([[600, 0, 320],
              [0, 600, 240],
              [0, 0, 1]], dtype=np.float64)

# 创建一个简单的平面深度图 + 中心方块
depth = np.ones((H, W), dtype=np.float32) * 2.0
# 中心方块深度 0.6m
depth[200:280, 280:360] = 0.6

# 分割图：方块为 1，其余为 0
seg = np.zeros((H, W), dtype=np.int32)
seg[200:280, 280:360] = 1

print('Running CGN predict on synthetic data...')
grasps, scores = model.predict(depth, K, seg, z_range=(0.2, 2.0), forward_passes=1)
print(f'grasps shape: {grasps.shape}, scores shape: {scores.shape}')
print(f'scores (first 5): {scores[:5] if len(scores) else []}')
print('Synthetic CGN OK')
