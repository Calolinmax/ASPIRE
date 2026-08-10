# aspire/evidence/ —— 证据层 🔒

每次 primitive 调用的【观测、输入、输出、视觉证据】+ 碰撞事件落盘——
**论文组件 1（闭环执行引擎的多模态 trace）的实现**，也是未来 agent 自主诊断失败的数据基础。

| 文件 | 说明 |
|---|---|
| `trace.py` | `Tracer`：每次执行产出一个 `traces/MMDD_HHMM_任务名/` 目录——trace.json（调用记录）+ images/top\|wrist/（帧流，固定帧率+调用边界补帧）+ 算法标注图文件夹（含"未检到"帧，失败定位关键证据）。 |
| `annotate.py` | `build_annotation`：把算法输出画回输入帧（SAM3 mask 着色/质心/score、molmo 品红十字、深度 VIRIDIS、CGN 3D glyph 🔏F2 定稿）。标注异常返回 None，绝不污染任务执行。 |

🔒 封版冻结，只调用不修改。详见 [../../docs/project_files.md](../../docs/project_files.md) §2。
