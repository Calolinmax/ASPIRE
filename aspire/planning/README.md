# aspire/planning/ —— 规划层 🔒

| 文件 | 说明 |
|---|---|
| `motion_planner.py` | RRT-Connect 关节空间规划器：接触级判对（臂体 `robot0_g*_col` vs 场景）+ 净距级判对（手-方块 3mm，仅 cubeA_main 在场时启用）。被 `execute_legs_rrt`/`move_to_joints_planned` 调用（凡动必避障规划的执行者）。cuRobo 决策关闭后这是唯一规划器。 |
| `pyroki_client.py` | pyroki IK 服务 HTTP 客户端（:8116，JSON）：`ik_pyroki(pos, quat_xyzw, prev_cfg)`；失败抛 RuntimeError 由 solve_ik 兜底链（DLS+IK 库+滚转扫描）接管。 |
| `pyroki_joint_map.json` | 存档：URDF↔MJCF 关节映射拟合表（⚠️ joint3 残差 238mm 不可信；无代码消费）。 |

**配套常驻服务**：`scripts/tools/pyroki_server_minimal.py`（独立 venv 启动，命令见其 docstring；
无 /health 路由，任何 HTTP 应答即算活）。

🔒 封版冻结，只调用不修改。详见 [../../docs/project_files.md](../../docs/project_files.md) §2。
