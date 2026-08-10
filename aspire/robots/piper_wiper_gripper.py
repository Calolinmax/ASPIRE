"""Piper 擦拭工具末端的 robosuite GripperModel（2026-08-06 用户指令: 拆夹爪换擦头）。

与 PiperGripper 的关系: 同一 XML 骨架（piper_hand/eef sites/joint7/joint8/
equality/actuator 逐字一致）, 指体几何全拆、挂球形擦头——冻结引擎依赖的
接口名字（grip_site / gripper 执行器 / gripper_qpos 观测）全部保留,
仅 wipe 场景经 gripper_types="PiperWiperGripper" 选用, Stack 不受影响。
"""

from __future__ import annotations

import os

import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel

_ASSETS = os.path.join(os.path.dirname(__file__), "assets", "piper")


class PiperWiperGripper(GripperModel):
    """Piper 擦拭工具（无指, 球形擦头; 关节为接口兼容摆设）。

    Args:
        idn (int or str): gripper instance 编号 (0 → gripper0_ 前缀)
    """

    def __init__(self, idn=0):
        super().__init__(os.path.join(_ASSETS, "wiper_gripper.xml"), idn=idn)

    def format_action(self, action):
        """与 PiperGripper 同语义（本工具无指, action 仅维持接口）。"""
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0]) * self.speed * np.sign(action[0]),
            -1.0,
            1.0,
        )
        return self.current_action

    @property
    def init_qpos(self):
        # 与 PiperGripper 一致（关节为摆设, 保持观测语义）
        return np.array([0.035, -0.035])

    @property
    def speed(self):
        return 0.2

    @property
    def dof(self):
        return 1

    @property
    def _important_geoms(self):
        return {
            "left_finger": [],
            "right_finger": [],
            "left_fingerpad": [],
            "right_fingerpad": [],
            "wiper": ["wiper_pad"],
        }
