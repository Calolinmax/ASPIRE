# aspire/api/ —— Primitive API 层 🔒

| 文件 | 说明 |
|---|---|
| `primitives_capx.py` | **API 层本体** `PiperControlApiReduced`：cap-x 契约 15 函数（感知 4/几何抓取 3/运动 2/控制 2/工具 4）+ 契约外 5 个（grasp_cgn/move_to_joints_planned/execute_legs_rrt/set_gripper_ramp/draw_grasp_glyphs）。`build_namespace(engine)` 产出任务代码命名空间。后端开关：`IK_BACKEND="pyroki"`、`GRASP_BACKEND="cgn"`。 |
| `primitives.py` | **存档（Panda 旧线）**：`PrimitiveContext` 14 个 API（OSC 闭环）；进程内 import vision_sam3（与 cap-x 线的进程隔离路线分叉）。只读留档。 |

**权威文档**：[../../docs/primitive_api_capx.md](../../docs/primitive_api_capx.md)——坐标系约定
（基座系/wxyz/TCP=grip_site）、三处有意偏差、逐函数签名与示例。任务代码**只准使用**
其中列出的函数，禁止臆造 API。

🔒 封版冻结（测试 33/33 验收），只调用不修改。看板：[../../docs/api_asset_map.md](../../docs/api_asset_map.md)。
