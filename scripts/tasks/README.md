# scripts/tasks/ —— 任务代码

引擎注入 API 命名空间后 `exec` 执行的任务代码（**非独立脚本，禁止 import**，
只用契约 15 函数 + 契约外 5 个 + np + print）。

| 文件 | 说明 |
|---|---|
| `stack.py` | Stack：SAM3 定位双块 → CGN 姿态抓红 → 堆绿（先算后动门控 + 滑脱检测 + 验收一次修复）。抓取类模板。 |
| `wipe.py` | Wipe：SAM3 定位污渍 → bbox → 双程蛇形 → Franka 擦板 38° 滚转锁定接触擦拭（预解阶梯 + pad_to_site 补偿）。路径跟踪类模板。 |

编写规则与骨架模板见 [../README.md](../README.md)（scripts/ 任务层规范）；
API 唯一依据 [../../docs/primitive_api_capx.md](../../docs/primitive_api_capx.md)。
