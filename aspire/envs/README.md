# aspire/envs/ —— 场景层 ◻（不锁）

| 文件 | 说明 |
|---|---|
| `wipe_spill.py` | `PiperWipeSpill`：桌面棕色污渍擦拭场景（官方 robosuite WipeArena 污渍随机路径 + Stack 锁定桌几何；默认挂 Franka 擦板 `PiperWiperGripper`）。成功判定 = 擦板足迹访问覆盖率 ≥50%（被覆盖 marker 即时淡出）。 |

**新场景照此办理**：

1. 在本目录新建 `<name>.py`——仿照 `wipe_spill.py` 抄官方 arena/env 的函数体
   （官方参考优先，禁止凭签名手搓）；⚠️ 桌几何/相机/home 位姿等 🔏 锁定项不可动。
2. 注册：在 `aspire/robots/__init__.py` 末尾加一行 import（engine_capx 冻结后
   那是唯一注册钩子；**该文件已锁🔒，须用户本人解锁**）。
3. 使用：`python -m aspire.engine.engine_capx --task <类名> ...`。

注意官方 `Wipe` 环境**不可直接用**（WipingGripper 断言会拆掉 Piper 夹爪，
grip_site/执行器全灭）——必须走本目录的直建模式。
