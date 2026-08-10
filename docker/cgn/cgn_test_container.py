#!/usr/bin/env python
"""容器内版本：直接加载 CGN wrapper，用合成数据推理。"""
import os
import sys
import numpy as np

os.environ['TF_USE_LEGACY_KERAS'] = '1'

sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/contact_graspnet')
sys.path.insert(0, '/workspace/pointnet2')
sys.path.insert(0, '/workspace/pointnet2/utils')

from wrapper import ContactGraspnet

CKPT = '/workspace/checkpoint/model.ckpt-144144'

model = ContactGraspnet(CKPT)

# 合成深度图和内参
H, W = 480, 640
K = np.array([[600, 0, 320],
              [0, 600, 240],
              [0, 0, 1]], dtype=np.float64)

# 创建平面深度图 + 中心方块
depth = np.ones((H, W), dtype=np.float32) * 2.0
depth[200:280, 280:360] = 0.6

seg = np.zeros((H, W), dtype=np.int32)
seg[200:280, 280:360] = 1

print('Running CGN predict on synthetic data...')
grasps, scores = model.predict(depth, K, seg, z_range=(0.2, 2.0), forward_passes=1)
print(f'grasps shape: {grasps.shape}, scores shape: {scores.shape}')
print(f'scores (first 5): {scores[:5] if len(scores) else []}')
print('Synthetic CGN OK')
