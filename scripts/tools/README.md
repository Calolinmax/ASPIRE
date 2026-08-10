# scripts/tools/ —— 资产生产线 / 查看器 / 常驻服务

| 文件 | 说明 |
|---|---|
| `pyroki_server_minimal.py` | **pyroki IK 常驻服务**（:8116，独立 venv `~/venvs/pyroki`；启动命令见文件 docstring）。每次跑任务都需要。 |
| `cgn_execute_grasp.py` | CGN 抓取端到端验收门（CLI 直驱，非注入式）：SAM3→CGN→宽度过滤→最多 3 候选→悬停不滑落=成功。`--view` 开窗。tasks/stack.py 运动层的验证来源。 |
| `build_piper_ik_library.py` | IK 种子库生产线（FK 稠密采样+俯身筛选）。几何/安装变更后必重建。 |
| `generate_piper_urdf_from_mjcf.py` | pyroki 配套 URDF 生产线（从 MJCF 直出）。模型变更时重跑。 |
| `check_cgn_pose.py` | CGN 位姿对应性肉审（臂控到 top-1 候选保持 + glyph 同框，glfw 窗口）。 |
| `view_cgn_poses.py` | CGN top-N 候选 3D 可视化查看器。 |
| `view_scene_interactive.py` | 通用场景查看器（鼠标交互 + 关节滑块 + 空格暂停）。需显示器。 |
| `watch_grasp_live.py` | 抓-搬-放循环实时围观（3 倍慢放，调试围观用）。 |

均为 CLI 直驱（非引擎注入）；从仓库根目录运行。
