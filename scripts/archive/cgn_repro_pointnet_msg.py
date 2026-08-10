#!/usr/bin/env python
"""复现 CGN PointNet++ MSG 下采样路径，匹配真实 config 参数。"""
import os
import sys
import numpy as np

PYTHON = sys.executable

# 让 TF 运行时找到 pip 安装的 nvidia CUDA 库
import subprocess
NVIDIA_BASE = subprocess.check_output(
    [PYTHON, '-c', 'import os, nvidia; print(os.path.dirname(nvidia.__file__))'],
    text=True,
).strip()
import glob
LD_DIRS = ':'.join(d for d in glob.glob(os.path.join(NVIDIA_BASE, '*', 'lib')) + glob.glob(os.path.join(NVIDIA_BASE, '*', '*', 'lib')) if os.path.isdir(d))
os.environ['LD_LIBRARY_PATH'] = LD_DIRS + ':' + os.environ.get('LD_LIBRARY_PATH', '')
os.environ['TF_USE_LEGACY_KERAS'] = '1'

import tensorflow.compat.v1 as tf
tf.disable_eager_execution()

sys.path.insert(0, '/home/stouching/Desktop/ASPIRE/external/contact_graspnet/pointnet2/utils')
from pointnet_util import pointnet_sa_module_msg, pointnet_sa_module

def build_sa_msg(xyz, points, npoint, radius_list, nsample_list, mlp_list, scope):
    return pointnet_sa_module_msg(
        xyz, points,
        npoint=npoint,
        radius_list=radius_list,
        nsample_list=nsample_list,
        mlp_list=mlp_list,
        is_training=False,
        bn_decay=None,
        scope=scope,
        bn=True,
        use_xyz=True,
        use_nchw=False,
    )

# 模拟 CGN 输入：batch=1, 20000 points, 3 coords + 0 features
B = 1
N = 20000
xyz_np = np.random.rand(B, N, 3).astype(np.float32)

with tf.device('/gpu:0'):
    xyz = tf.constant(xyz_np)
    points = None

    # layer 1: npoint=2048
    xyz1, points1 = build_sa_msg(
        xyz, points,
        npoint=2048,
        radius_list=[0.02, 0.04, 0.08],
        nsample_list=[32, 64, 128],
        mlp_list=[[32, 32, 64], [64, 64, 128], [64, 96, 128]],
        scope='layer1',
    )
    print(f'layer1: xyz={xyz1.shape}, points={points1.shape}')

    # layer 2: npoint=512  ——  实际崩溃位置
    xyz2, points2 = build_sa_msg(
        xyz1, points1,
        npoint=512,
        radius_list=[0.04, 0.08, 0.16],
        nsample_list=[64, 64, 128],
        mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
        scope='layer2',
    )
    print(f'layer2: xyz={xyz2.shape}, points={points2.shape}')

    # layer 3: npoint=128
    xyz3, points3 = build_sa_msg(
        xyz2, points2,
        npoint=128,
        radius_list=[0.08, 0.16, 0.32],
        nsample_list=[64, 64, 128],
        mlp_list=[[64, 64, 128], [128, 128, 256], [128, 128, 256]],
        scope='layer3',
    )
    print(f'layer3: xyz={xyz3.shape}, points={points3.shape}')

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.allow_soft_placement = True
with tf.Session(config=config) as sess:
    sess.run(tf.global_variables_initializer())
    ret = sess.run([xyz1, points1, xyz2, points2, xyz3, points3])
    for i, (name, arr) in enumerate(zip(['xyz1','points1','xyz2','points2','xyz3','points3'], ret)):
        print(f'{name}: shape={arr.shape}, dtype={arr.dtype}, min={arr.min():.4f}, max={arr.max():.4f}')

print('PointNet++ MSG OK')
