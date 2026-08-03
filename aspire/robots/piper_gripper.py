"""AgileX Piper 平行夹爪的 robosuite GripperModel。

关节语义 (与原 MJCF 一致):
  - joint7 (slide, 0..0.035): 0 = 闭合, 0.035 = 全开 (张嘴内净距约 45mm)
  - joint8 由 equality 镜像, 无独立执行器
  - 单个 position 执行器 "gripper" 驱动 joint7
"""

from __future__ import annotations

import os

import numpy as np
from robosuite.models.grippers.gripper_model import GripperModel

_ASSETS = os.path.join(os.path.dirname(__file__), "assets", "piper")


class PiperGripper(GripperModel):
    """Piper 二指平行夹爪 (单自由度, 镜像手指)。

    Args:
        idn (int or str): gripper instance 编号 (0 → gripper0_ 前缀)
    """

    def __init__(self, idn=0):
        super().__init__(os.path.join(_ASSETS, "gripper.xml"), idn=idn)

    def format_action(self, action):
        """连续 action → 二值开合并累积 (与 PandaGripper 同语义: -1 开, +1 合)。"""
        assert len(action) == self.dof
        self.current_action = np.clip(
            self.current_action + np.array([-1.0]) * self.speed * np.sign(action[0]),
            -1.0,
            1.0,
        )
        return self.current_action

    @property
    def init_qpos(self):
        # 初始全开
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
            "left_finger": ["finger1_collision", "finger1_pad_collision"],
            "right_finger": ["finger2_collision", "finger2_pad_collision"],
            "left_fingerpad": ["finger1_pad_collision"],
            "right_fingerpad": ["finger2_pad_collision"],
        }
