#!/usr/bin/env python
"""最小复现：单独调用 pointnet2 GatherPoint，排查 CUDA_ERROR_ILLEGAL_ADDRESS。"""
import os
import sys
import numpy as np

# 使用 cgn_venv 的 Python/TF
PYTHON = '/home/stouching/Desktop/ASPIRE/external/cgn_venv/bin/python'
print(f'Python executable: {sys.executable}')

# 让 TF 运行时找到 pip 安装的 nvidia CUDA 库
import subprocess
NVIDIA_BASE = subprocess.check_output(
    [sys.executable, '-c', 'import os, nvidia; print(os.path.dirname(nvidia.__file__))'],
    text=True,
).strip()
import glob
LD_DIRS = ':'.join(d for d in glob.glob(os.path.join(NVIDIA_BASE, '*', 'lib')) + glob.glob(os.path.join(NVIDIA_BASE, '*', '*', 'lib')) if os.path.isdir(d))
os.environ['LD_LIBRARY_PATH'] = LD_DIRS + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['TF_USE_LEGACY_KERAS'] = '1'
print(f'LD_LIBRARY_PATH: {os.environ["LD_LIBRARY_PATH"]}')

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

# 加载 ops 和 Python wrapper
OPS_DIR = '/home/stouching/Desktop/ASPIRE/external/contact_graspnet/pointnet2/tf_ops/sampling'
sys.path.insert(0, OPS_DIR)
from tf_sampling import gather_point, farthest_point_sample

# 合成数据
B, N, M = 1, 2048, 512
np.random.seed(42)
points = np.random.rand(B, N, 3).astype(np.float32)
indices = np.random.randint(0, N, size=(B, M), dtype=np.int32)

print(f'points shape: {points.shape}, dtype: {points.dtype}')
print(f'indices shape: {indices.shape}, dtype: {indices.dtype}')
print(f'indices range: [{indices.min()}, {indices.max()}] (n_dataset={N})')

with tf.device('/gpu:0'):
    inp = tf.constant(points)
    idx = tf.constant(indices)
    out = gather_point(inp, idx)
    print('--- test 1: gather_point with int32 indices ---')

    # test 2: farthest_point_sample + gather_point (real path)
    fps_idx = farthest_point_sample(M, inp)
    out2 = gather_point(inp, fps_idx)
    print('--- test 2: farthest_point_sample + gather_point ---')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
with tf.Session(config=config) as sess:
    result = sess.run(out)
    print(f'test 1 output shape: {result.shape}, dtype: {result.dtype}')
    result2 = sess.run(out2)
    print(f'test 2 output shape: {result2.shape}, dtype: {result2.dtype}')

print('GatherPoint OK')
