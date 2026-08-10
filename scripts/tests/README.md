# scripts/tests/ —— 自检

| 文件 | 说明 |
|---|---|
| `test_piper_capx_api.py` | **契约 15 函数全量自检（33 项 A-K 区）**：观测结构/SAM3/反投影标定/plan_grasp/点提示/OBB/运动/夹爪/类结构/工具/select_top_down。任何环境/结构变动后的验收门——全过 exit 0。 |

用法：

```bash
MUJOCO_GL=egl /home/stouching/anaconda3/envs/ASPIRE/bin/python scripts/tests/test_piper_capx_api.py
```

前提服务：vision_server（自动拉起）+ pyroki :8116 + CGN :8117（plan_grasp 用例需要）。
