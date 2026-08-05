#!/usr/bin/env python
"""展示臂在方块位的【认证极限姿态】（2026-08-03）: 交互窗口中把臂设为
库 FK 地图中离方块 5cm 内最陡的认证构型（tilt=97.4°, entry53807），
供用户 3D 核查"为什么竖直抓不可达"——腕部 j5 顶在限位上。

用法: MUJOCO_GL=glfw python scripts/view_best_effort.py
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import aspire.robots  # noqa: F401
from aspire.engine_capx import ExecutionEngineCapx
from aspire.primitives_capx import _ik_library


def main():
    engine = ExecutionEngineCapx(task='Stack', seed=5)
    sim = engine.env.sim
    m = sim.model._model

    lib = _ik_library()
    Q, P, Z = lib['Q'], lib['P'], lib['Z']
    tilts = np.rad2deg(np.arccos(np.clip(-Z[:, 2], -1, 1)))
    cube = np.array([0.21, -0.02, 0.04])
    sel = np.where(np.linalg.norm(P - cube, axis=1) < 0.05)[0]
    i_best = sel[np.argmin(tilts[sel])]
    for i in range(6):
        sim.data.set_joint_qpos(f'robot0_joint{i+1}', Q[i_best][i])
    mujoco.mj_forward(m, sim.data._data)
    print(f'臂已设为方块位认证极限构型: entry{i_best}, tilt={tilts[i_best]:.1f}°')
    print('（位置锁死下全局优化的最陡可达为 96-110°; CGN 候选需要 3-34°）')
    print('看清后可按空格运行物理（臂会被伺服拉回）, 关窗退出')

    with mujoco.viewer.launch_passive(m, sim.data._data) as v:
        while v.is_running():
            time.sleep(0.1)
    engine.close()


if __name__ == '__main__':
    main()
