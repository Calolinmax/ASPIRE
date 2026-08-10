# open_details/ —— ASPIRE 官方复现基准

ASPIRE 论文官方公布的任务代码与 skill 样例（论文项目页下载）。
**不能直接运行**——需由执行引擎注入 Primitive API 后执行；
本项目用法 = 复刻时的逻辑参照（任务代码照抄其流程，再做 Piper 本体适配）。

| 文件 | 论文任务 | 本项目复刻状态 |
|---|---|---|
| `primitive_api_cube_reset.py` | Cube Reset（挪绿块、红块叠绿块） | ✅ 已复刻验证 → `scripts/tasks/stack.py`（2026-08-04 端到端 PASS） |
| `primitive_api_wipe.py` | Wipe（擦棕色污渍） | ✅ 已复刻验证 → `scripts/tasks/wipe.py`（2026-08-07 Franka 擦板版 PASS） |
| `primitive_api_nut_assembly.py` | Nut Assembly（方螺母套柱） | ⬜ 未复刻（Phase 1/3 候选任务） |
| `skill_grasp.md` | skill library 格式标杆（5 个 Pattern：Problem/When/Strategy/Evidence 四段式，seed 级实证） | Phase 1 入库格式与验证纪律标杆 |
