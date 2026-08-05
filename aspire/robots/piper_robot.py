"""AgileX Piper 6 轴机械臂的 robosuite ManipulatorModel。

MJCF 来源: external/agilex_arm_mujoco (松灵官方 mujoco 模型), 拆分适配:
  - robot.xml   : 臂杆 (base_link..link6) + robotview/eye_in_hand 相机
  - gripper.xml : 平行夹爪 (link7/link8 + joint7/8 镜像), 由 default_gripper 挂载

注册: 本模块 import 时把 "Piper" 注册进 robosuite 的 ROBOT_CLASS_MAPPING
(FixedBaseRobot), ManipulatorModel 的元类同时把模型注册进 REGISTERED_ROBOTS,
此后 suite.make(..., robots="Piper") 即可用。
"""

from __future__ import annotations

import os

import numpy as np
from robosuite.models.robots.manipulators.manipulator_model import ManipulatorModel
from robosuite.robots import ROBOT_CLASS_MAPPING
from robosuite.robots.fixed_base_robot import FixedBaseRobot

from .piper_mount import PiperPedestal  # noqa: F401  (注册台架见 __init__.py)

_ASSETS = os.path.join(os.path.dirname(__file__), "assets", "piper")

# 桌面安装: 基座 (-0.30, 0, 0.80) 直接落在 Stack 桌面上 (桌面顶面 z=0.80,
# table_arena.xml: table_collision center z=0.775 + half 0.025)。
# 注: 25cm 落地立柱配置 (基座 z=0.25) 只适配 capx 低桌环境 (桌面≈0.55),
# 在标准 Stack 桌面 (0.80) 下整条臂会埋在桌下 (相机实锤)。
# PIPER_BASE_RAISE: 垫高实验旋钮（2026-08-03 descend 墙攻坚, 米）——
# 基座与 robotview 相机（挂在 base_link 上）一起抬高; 定稿后硬编码进本值。
PIPER_BASE_XPOS_TABLE = (-0.30, 0.0, 0.55 + float(os.environ.get("PIPER_BASE_RAISE", "0.0")))  # offset z + pedestal 0.25 = 基座 z=0.80 (桌面)


class Piper(ManipulatorModel):
    """AgileX Piper 单臂 (6 轴 + 平行夹爪)。

    Args:
        idn (int or str): robot instance 编号 (0 → robot0_ 前缀)
    """

    arms = ["right"]

    def __init__(self, idn=0):
        super().__init__(os.path.join(_ASSETS, "robot.xml"), idn=idn)

    @property
    def default_base(self):
        # 25cm 落地立柱 (PiperPedestal)
        return "PiperPedestal"

    @property
    def default_gripper(self):
        return {"right": "PiperGripper"}

    @property
    def default_controller_config(self):
        return {"right": "default_piper"}

    @property
    def init_qpos(self):
        # 【2026-08-03 用户指定】全零位姿 = 官方零位（用户 MuJoCo viewer 截图
        # 确认）: 臂前趴折叠, 连杆不直立。eef base 系 (0.145, 0, 0.211),
        # _collision_free_q 通过。旧值: [0, 1.57, -1.3485, 0, 0, 0]（直立）。
        return np.zeros(6)

    @property
    def base_xpos_offset(self):
        return {
            "bins": (-0.5, -0.1, 0.8),
            "empty": (-0.3, 0.0, 0.0),
            "table": lambda table_length: PIPER_BASE_XPOS_TABLE,
        }

    @property
    def top_offset(self):
        return np.array((0, 0, 1.0))

    @property
    def _horizontal_radius(self):
        return 0.3

    @property
    def arm_type(self):
        return "single"

    @property
    def _eef_name(self):
        # 夹爪挂载体 (XML 原始名, robosuite 自动加 robot0_ 前缀)
        return {"right": "link6"}


# suite.make(robots="Piper") 解析入口 (ManipulatorModel 元类负责 REGISTERED_ROBOTS)
ROBOT_CLASS_MAPPING.setdefault("Piper", FixedBaseRobot)
