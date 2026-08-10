# aspire/engine/ —— 执行引擎层 🔒

| 文件 | 说明 |
|---|---|
| `engine.py` | `ExecutionEngine` **基类**：`run(code)` 把任务代码 exec 进注入 API 的命名空间执行并全程记录 trace；四个子类钩子（`_env_kwargs/_make_env/_post_reset/_build_namespace`）。核心资产是 **EGL 自愈**（损坏帧双指标检测 + L1/L2 两级渲染管线恢复）。Panda 旧线也用此基类。 |
| `engine_capx.py` | `ExecutionEngineCapx`（**现役主线**）：Piper + 直写 position 执行器（≈真机位置伺服），20Hz tick；`StackClutter` 场景类；`--render` 自管交互窗口（关窗不中断）；`draw_grasp_glyphs`；vision_server 自动拉起。 |

**分工一句话**：基类定"怎么执行+怎么记录"，子类定"用什么机器人/什么 API/什么控制模式"。

CLI：`python -m aspire.engine.engine_capx --code <task.py> --task <场景> --seed N [--render --render-slowdown 0.5 --official-stack --clutter N --target-only]`

🔒 封版冻结，只调用不修改（解冻须用户本人）。详见 [../../docs/project_files.md](../../docs/project_files.md) §2。
