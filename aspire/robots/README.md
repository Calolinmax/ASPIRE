# aspire/robots/ —— 机器人资产层 🔒

AgileX Piper 的 robosuite 适配包。**import 本包即完成全部注册**
（机器人/两种夹爪/台架/PiperWipeSpill 场景）；engine_capx 冻结后，
`__init__.py` 是其模块加载期**唯一的注册钩子**（新场景/新夹爪挂这里）。

| 文件 | 说明 |
|---|---|
| `__init__.py` | 注册入口（register_gripper×2 + register_base + 场景 import）。 |
| `piper_robot.py` | Piper ManipulatorModel：`PIPER_BASE_XPOS_TABLE`（基座世界系锚点）、`init_qpos=zeros(6)` 🔏（全零 home 用户指定）、默认 base/gripper/controller。 |
| `piper_mount.py` | PiperPedestal 台架包装（25cm 立柱现作桌面增高台）。 |
| `piper_gripper.py` | 平行夹爪 GripperModel（Stack 默认；全开净距 70mm）。 |
| `piper_wiper_gripper.py` | 擦拭工具 GripperModel（Franka 平板擦头；仅 wipe 场景选用；接口与夹爪逐字一致）。 |
| `assets/piper/` | MJCF/网格/IK 种子库/控制器配置，见 [assets/piper/README.md](assets/piper/README.md)。 |

**实测关键事实**（勿重蹈）：hand 系 **+z = 逼近轴方向**（site_z = hand -z，FK 实测；
gripper.xml 头注释有误导）；j5 腕限位 ±70° 是一系列可达性坑的物理根源。

🔒 2026-08-07 用户指令整层加锁（含 assets），只调用不修改。
详见 [../../docs/project_files.md](../../docs/project_files.md) §3。
