"""robosuite 自定义机器人注册入口。

import 本包即完成注册:
  - "Piper" → FixedBaseRobot (ROBOT_CLASS_MAPPING)
  - Piper / PiperGripper 模型类 (元类自动注册; 夹爪另需 GRIPPER_MAPPING 注册)
"""

from robosuite.models.bases import register_base
from robosuite.models.grippers import register_gripper

from .piper_gripper import PiperGripper
from .piper_mount import PiperPedestal
from .piper_robot import Piper  # noqa: F401

register_gripper(PiperGripper)
register_base(PiperPedestal)
