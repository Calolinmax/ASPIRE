# aspire/ —— 核心包

ASPIRE 复现的核心代码包，2026-08-07 起按**层**分 7 个子包：

```
aspire/
├── engine/      🔒 执行引擎层（engine.py 基类 + engine_capx.py cap-x/Piper 主线）
├── api/         🔒 Primitive API 层（primitives_capx.py 契约 15+契约外 5；primitives.py Panda 存档）
├── evidence/    🔒 证据层（trace.py 调用记录/帧流 + annotate.py 视觉标注）
├── planning/    🔒 规划层（motion_planner.py RRT-Connect + pyroki_client.py IK 客户端）
├── perception/  🔒 感知服务层（vision_server/vision_client/vision_sam3 + cgn_server）
├── envs/        ◻  场景层（wipe_spill.py；不锁——新场景扩展位）
└── robots/      🔒 机器人资产层（Piper 模型类 + MJCF/网格/IK 种子库 + 注册钩子）
```

## 关键约定

- **空包哲学**：`__init__.py` 不做任何导入（`__all__=[]`）——避免 `import aspire`
  把 vision_server 等独立进程拖入 robosuite/mujoco/EGL 初始化；子模块按需显式导入
  （如 `from aspire.engine.engine_capx import ExecutionEngineCapx`）。
- **🔒 锁定层只能在 scripts 中调用，禁止修改**——解冻须用户本人
  `chmod +w`（或明确授权代办）；完整锁定清单见 [../docs/project_files.md](../docs/project_files.md) §9。
- **CUDA 与 EGL 同进程互毁**（2026-07-28 实测）——CUDA 推理一律进程隔离
  （perception/ 服务化），引擎进程只做 EGL 渲染。
- 逐文件详细说明：[../docs/project_files.md](../docs/project_files.md)。
