#!/usr/bin/env python
"""交互式 MuJoCo 可视化（2026-08-03 用户要求）: 带鼠标互动（旋转/缩放/平移）
查看当前任务场景（桌面+垫高台+细高块+杂物+Piper 全零 home）。

用法: python scripts/view_scene_interactive.py [seed]   （默认 5）
鼠标: 左键旋转 / 右键平移 / 滚轮缩放; 空格运行/暂停物理; Ctrl+Q 或关窗退出。
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aspire.robots  # noqa: F401 注册 Piper
from aspire.engine_capx import ExecutionEngineCapx


def main():
    seed = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    engine = ExecutionEngineCapx(task='Stack', seed=seed)
    m = engine.env.sim.model._model
    d = engine.env.sim.data._data
    print(f'交互窗口启动: Stack seed={seed}（全零 home / 无垫高台 / 细高块 / 新相机位）')
    print('空格=运行/暂停; 暂停时右侧 Joint 滑块直接拖关节角; 运行时 Control 滑块驱动伺服')
    import mujoco
    import mujoco.viewer
    with mujoco.viewer.launch_passive(m, d) as v:
        while v.is_running():
            sim_obj = v._sim()
            if getattr(sim_obj, 'run', 1):
                mujoco.mj_step(m, d)        # 运行态: 物理步进（Control 滑块生效）
            else:
                mujoco.mj_forward(m, d)     # 暂停态: 只更新 FK 显示（Joint 滑块拖动静显示）
            v.sync()
            time.sleep(0.004)
    engine.close()


if __name__ == '__main__':
    main()
