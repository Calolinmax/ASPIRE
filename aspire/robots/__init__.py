"""robosuite 自定义机器人注册入口。

import 本包即完成注册:
  - "Piper" → FixedBaseRobot (ROBOT_CLASS_MAPPING)
  - Piper / PiperGripper 模型类 (元类自动注册; 夹爪另需 GRIPPER_MAPPING 注册)
  - "PiperWipeSpill" → 桌面污渍擦拭场景 (2026-08-06; engine_capx 已冻结,
    本包是其模块加载期唯一的注册钩子, 故场景注册也挂在这里)
"""

from robosuite.models.bases import register_base
from robosuite.models.grippers import register_gripper

from .piper_gripper import PiperGripper
from .piper_mount import PiperPedestal
from .piper_robot import Piper  # noqa: F401
from .piper_wiper_gripper import PiperWiperGripper

register_gripper(PiperGripper)
register_gripper(PiperWiperGripper)   # 擦拭工具末端 (2026-08-06), 仅 wipe 场景选用
register_base(PiperPedestal)

from ..envs.wipe_spill import PiperWipeSpill  # noqa: F401,E402  register_env 于模块内完成
