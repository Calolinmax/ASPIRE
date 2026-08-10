# ASPIRE 复现路线图（Phase 1 起）

> 2026-08-07 提炼自已删除的 `ASPIRE_REPRO_PLAN.md`（2026-07-28 版）——
> 只保留前瞻内容：Phase 0（API 构建）已完成封版（2026-08-06），其历史基线、
> 交接细节不再留存（攻坚教训在代码注释与各模块文档里）。
> 目标论文：ASPIRE: Agentic Skills Discovery for Robotics（NVIDIA GEAR, 2026）。

## 0. 论文架构对照（本项目现状）

ASPIRE = CaP-X 基底（robot programming APIs）+ 三组件：

| 组件 | 内容 | 本项目状态 |
|---|---|---|
| 基底 | CaP-X Primitive API（感知/几何/运动规划） | ✅ 契约 15 函数封版（2026-08-06） |
| 组件 1 | 闭环执行引擎 + 多模态 trace（观测/输入/输出/视觉证据/碰撞反馈） | ✅ engine + trace 体系封版 |
| 组件 2 | Skill Library（持续扩张的技能库） | ⬜ **Phase 1（下一步）** |
| 组件 3 | 进化搜索（多样任务序列+控制程序, 系统性 debug） | ⬜ Phase 2 |

论文的 coding agent = Claude Code + Claude Opus（1M 上下文）——本项目工作方式同构。

**Skill Library 的论文形态**（Phase 1 需求依据）：
- Skill **不是**可执行函数，而是**结构化知识文档**（SKILL.md），以 in-context guidance
  注入未来 agent 的 prompt；底层 primitive API 人工预定义、固定不变（本项目已封版✓）。
- 每条四要素（论文附录 A）：**Problem**（失败特征）/ **When to Apply**（检索 guard）/
  **Strategy**（验证过的修复策略 + code sketch）/ **Origin task(s)**；
  格式标杆 = `open_details/skill_grasp.md`（五段 Pattern, Evidence 必须 seed 级实证）。
- 入库流程：actor 按 findings schema 上报 → coordinator 审计可复用性 + API 合规性 →
  只把验证通过且可迁移的模式写入共享库。
- sim2real 证据（论文 §1.4）：skill 是本体无关知识；带物理参数的条目（z_offset/yaw 等）
  换新本体需重新验证。

## 1. Phase 1 — Skill Library 闭环（论文组件 2）

**目标**：debug→validate→入库→注入 全链路跑通，见到第一个真实 skill 入库并起效。

- findings.md schema（failure mode / validated repair / transferable patterns / 验证成功率）；
- coordinator 角色审计提炼 SKILL.md（论文附录 E.5 模板）；
- 注入机制（初期全量注入，skill 数 >20 后做检索）；
- 在 Stack + 1-2 个新任务（如 Lift、PickPlace 的 Piper 版）上验证"入库 skill 提升后续任务成功率"。
- **种子素材**（论文附录现成原型，当格式标杆与首批内容）：
  - debugging 表：`plan_grasp` 返回空 → dilate mask、log `mask.sum()` 应 >200px；
    `solve_ik` 返回 None → 目标 z 降 5cm、重查 x/y 可达性；
    抓取成功但抬起掉落 → 试垂直偏航 90° 或沿长轴逼近；
  - `make_topdown_quat(yaw_deg)` 的 scipy 参考实现（xyzw→wxyz 重排）；
  - 论文附录 E.5 的 localize / grasp 两个完整 SKILL.md 样例——首批入库条目按此格式
    蒸馏**我们自己的实测经验**（overhead 钩抓姿态族、x>0.45 不可达、min_z 抓取基准、
    wipe 线的 38° 滚转锁定/软接触/延迟缓存等）。
- **出口标准**：≥3 个 skill 入库；对照实验（有/无注入）成功率有统计意义差异。

## 2. Phase 2 — 进化搜索（论文组件 3）

**目标**：K≥4 候选程序锦标赛 + 多代进化，候选**并行评估**（多进程 sim 实例，AGENTS.md 硬性要求）。

- 候选生成（温度/扰动采样）、并行评估 harness、基于 surviving programs + residual traces
  的下一代条件化；
- 与 Phase 1 联动：repaired program vs evo search 的 held-out 验证（论文 Aspire 列的评测方式）。
- **出口标准**：在 ≥3 个任务上 evo search 候选超过单轨迹修复基线。

## 3. Phase 3 — 长程任务与导航扩展

- BEHAVIOR 风格多阶段任务（多点位巡访、开关容器等）；
- 固定基座下导航以"任务级点位序列"形式纳入，真 mobile base 视硬件到位再议；
- 依赖 Phase 0 的碰撞感知规划与 trace 碰撞反馈。
- **出口标准**：≥1 个长程任务（≥5 阶段）跑通并入库对应 skills。

## 4. Phase 4 — sim2real：Piper 真机

- 真机适配层：API 签名不变，底层走 piper_sdk（IK/关节控制/夹爪）；
- 相机标定（realsense/腕部）、SAM3/CGN 真机推理链路；
- skill 迁移实验：sim 入库 skills 作为 in-context guidance，测真机编程 token 成本降幅。
- **出口标准**：真机完成 ≥1 个 sim 训练过的任务；skill 注入 vs 无注入对照。

## 5. 长期约束（仍然有效）

- **只做单臂任务**（双臂 handover / TwoArmLift 不碰；cap-x bimanual 分支不实现）。
- **cap-x 当插件不当宿主**（复现 LIBERO-Pro/BEHAVIOR 榜单时，cap-x envs 作为另一种基底
  接入同一引擎接口）。
- **每 Phase 结束同步一次文档**（README / docs/project_files.md / 本路线图）。
- 环境铁律：conda ASPIRE（py3.12）绝对路径调 python；CUDA 与 EGL 同进程互毁 →
  CUDA 推理一律进程隔离；mujoco 必须 3.3.*；CGN 只在 docker 容器跑。
