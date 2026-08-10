# scripts/archive/ —— 历史诊断留档（17 个）

**一次性攻坚产物**：结论已固化进文档/代码注释/记忆，日常不运行。
保留价值 = 结论的证据留档（被质疑时可重跑复核）。删除前请先读文件头注释里的结论。

| 文件 | 攻坚对象 → 结论 |
|---|---|
| `arbitrate_ik.py` | IK 裁决实验："IK 有 bug 还是运动学墙属实" |
| `calibrate_pyroki_tcp.py` | pyroki TCP 标定：TCP_OFFSET=0 可用（<1mm） |
| `calibrate_urdf_mjcf_map.py` | URDF↔MJCF 关节映射拟合（产物 joint3 不可信） |
| `cgn_probe_clutter.py` | 干净单方块对 CGN 是双重 OOD（Stack clutter=4 的由来） |
| `cgn_verify_steps.py` | CGN 变换链三段可视化验证（D2/D3 判据） |
| `check_cameras.py` | A0.5 相机标定肉审（三位形渲染） |
| `diagnose_fallback_attribution.py` | fallback 失效三模式归因 |
| `hunt_phantom_ik.py` | 幻影收敛捕获（报收敛但 FK 差 134°）→ 三重校验修复 |
| `measure_sidecam_tilts.py` / `measure_true_tilts.py` | descend 墙攻坚的 tilt 分布数据 |
| `riser_height_sweep.py` | 垫高台负结果（不加高的依据） |
| `showcase_grip_vs_glyph.py` | glyph 定稿同框验证 |
| `showcase_scene.py` | 场景锁定验收 showcase |
| `verify_flip_fix.py` | CGN 反射 bug（det=-1）修复三层验证 |
| `verify_ik_convergence.py` | pyroki vs DLS 收敛对比（pyroki 胜出任默认） |
| `verify_mjcf_urdf_fk.py` | MJCF 派生 URDF 的 FK 一致性验收 |
| `view_best_effort.py` | "认证极限姿态"展品（竖直抓不可达=腕限位） |

2026-08-10 用户批准精简 7 个（test_piper_build / demo_stack_dualcam /
demo_stack_taskcode / cgn_repro×3 / calc_cam_rot），git 历史可恢复。
