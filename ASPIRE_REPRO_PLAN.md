# ASPIRE 长期复现规划

> 重写日期：2026-07-28（项目已获长期批准，取代 2026-07-23 的 12h 初版计划）
> 目标论文：ASPIRE: Agentic /Skills Discovery for Robotics（NVIDIA GEAR，2026）
> 项目页：https://research.nvidia.com/labs/gear/aspire/
> 基底框架：CaP-X（已开源，本地镜像 [external/cap-x/](external/cap-x/)）

---

## 0. 现状基线（2026-07-28 盘点）

### 已完成

| 资产 | 位置 | 状态 |
|---|---|---|
| 模块 1：仿真环境 + Lift 脚本化策略 | robosuite + Panda | ✅ 5/5 |
| 模块 2：执行引擎 + trace 系统（**= 论文组件 1**） | [aspire/engine.py](aspire/engine.py) + [aspire/trace.py](aspire/trace.py) | ✅ Lift 4/4 |
| 感知迁移：SAM3（transformers 本地权重） | [aspire/vision_sam3.py](aspire/vision_sam3.py) + 服务化 [vision_server.py](aspire/vision_server.py)/[vision_client.py](aspire/vision_client.py) | ✅ |
| 环境迁移：conda `ASPIRE`（py3.12，大写） | — | ✅ 旧 `aspire`（py3.10）已删 |
| CaP-X 源码镜像 | [external/cap-x/](external/cap-x/) | ✅ MIT |
| Piper MJCF 适配 robosuite（4 个 XML 兼容问题已修） | [aspire/robots/](aspire/robots/) | ✅ |
| cap-x 契约 API（Piper，10 函数 1:1） | [aspire/primitives_capx.py](aspire/primitives_capx.py) + [aspire/engine_capx.py](aspire/engine_capx.py) | ⚠️ 80%，Stack 未通 |
| IK 种子库（37,655 条）+ overhead 钩抓姿态族 | [aspire/robots/assets/piper/ik_library.npz](aspire/robots/assets/piper/ik_library.npz) | ✅ |
| API 自检脚本 | [scripts/test_piper_capx_api.py](scripts/test_piper_capx_api.py) | 待全 PASS |

### 进行中（交接状态）

cap-x/Piper 线因上一会话上下文超限中断，交接 prompt 在
[handoff_piper_control_api.md](handoff_piper_control_api.md)（仓库根目录）。
剩余阻塞（P0-P4）：运动可信化 → 工作区收紧 → 抓取闭环 → 感知兜底 →
`PiperControlApiReduced` 类命名收尾 + 补 `docs/primitive_api_capx.md`。

---

## 1. 论文关键结论（复现依据）

### 1.1 ASPIRE 架构 = CaP-X 基底 + 三个组件

论文明确：agent 用 **CaP-X** 的 robot programming APIs（感知/几何/运动规划）编写控制程序；
benchmark 为 LIBERO-Pro、Robosuite 双臂 handover、BEHAVIOR-1K（恰为 cap-x 支持的三个仿真器家族）。
ASPIRE 在此基底上增加三个组件：

1. **闭环执行引擎**：每个 primitive 调用记录【观测、输入、输出、视觉证据】的多模态 trace
   （含 perception overlays、grasp candidates、motion trajectories、**collision feedback**），
   agent 据此自主诊断失败、合成修复、再执行验证 —— **本项目 engine+trace 已复现此组件**；
2. **持续扩张的 skill library**：把验证过的修复蒸馏为可迁移知识（见 1.2）；
3. **进化搜索**：生成多样任务序列与控制程序，系统性地 debug 以超越单轨迹精炼。

论文的 coding agent：**Claude Code + Claude Opus 4.6（1M 上下文）**——本项目的
工作方式（Claude Code 驱动复现）与论文实验设置同构。

### 1.2 Skill Library 的形态

- Skill **不是**可执行函数，而是**结构化知识文档**（`SKILL.md`），以 in-context guidance
  注入未来 agent 的 prompt；底层 primitive API 人工预定义、固定不变。
- 每个 skill 条目四要素（附录 A）：**Problem**（失败特征）/ **When to Apply**（检索 guard）/
  **Strategy**（验证过的修复策略 + code sketch）/ **Origin task(s)**。
- `SKILL.md` 模板（附录 E.5）：YAML frontmatter + 代码骨架 / When to NOT Use /
  Per-Object Registry / Anti-Patterns / Debugging 表。
- 入库流程：actor 按 findings schema 上报 → coordinator 审计可复用性 + API 合规性 →
  只把验证通过且可迁移的模式写入共享库。

### 1.3 开源状态（2026-07-28 更新）

- ASPIRE：**未开源**（项目页无代码链接）；
- CaP-X：**已开源**（github.com/capgym/cap-x，MIT），本地镜像 [external/cap-x/](external/cap-x/)。

### 1.4 论文的 sim2real 证据

论文在 Franka 仿真发现 skills，迁移到双臂 YAM 真机站（本体与 API 均不同），skills 作为
in-context guidance 显著降低真机编程 token 成本。结论：**skill 是本体无关的知识**；
带物理参数的条目（z_offset、yaw 等）需在新本体上重新验证。

---

## 2. 已确认的决策（2026-07-28 修订）

| 项目 | 决策 | 理由 |
|---|---|---|
| 真机 | 松灵 PiPER（6 轴 + 夹爪），后期包同名 primitive API（底层走 piper_sdk） | 现有硬件 |
| 仿真臂 | **松灵 PiPER**（[aspire/robots/](aspire/robots/)，松灵官方 MJCF 适配） | 与真机同本体，skill 物理参数免二次标定 |
| 仿真框架 | robosuite（MuJoCo，自带 benchmark 与 `_check_success()`） | 环境/判定白送 |
| 感知 | **SAM3 主力**（文本+点提示，本地权重进程隔离）+ **Molmo 兜底链**（论文对齐）；Contact-GraspNet 抓取规划（见 §3） | 与 ASPIRE 论文栈对齐 |
| 运动 | **API 构建阶段内三步走**：DLS + 种子库（已建，兜底）→ **pyroki 服务**（Piper URDF）→ **cuRobo**（批量 IK + 碰撞轨迹），对比选默认 | 用户指定 4 必要组件之一，见 §3 |
| 导航与碰撞 | **纳入范围**：碰撞反馈进 trace（论文组件 1 明示）+ 碰撞感知规划（API 构建阶段）；导航随长程任务（BEHAVIOR 风格）在 Phase 3 纳入 | 撤销初版简化 |
| Agent 结构 | 单 agent 循环起步，两段 prompt 分饰 actor/coordinator；Phase 2 起按论文并行化 | 核心闭环优先 |
| Coding agent | Claude Code 本机担任（与论文同构）；无人值守阶段再配 CLI 自动化 | — |
| 视觉证据 | 每 primitive 记录【观测、输入、输出、视觉证据】（AGENTS.md 硬性要求）；存储可压缩，记录不可省 | trace 是 agent 诊断的数据基础 |

---

## 3. 组件覆盖现状与补全计划（用户指定的 4 个必要组件）

> 用户决策（2026-07-28）：以下 4 项**全部属于当前 API 构建阶段（Phase 0）的范畴**，
> 不延后——API 阶段完成 = 15 函数契约 + 4 组件全部集成并通过验收。
> （原第 5 项 OWL-ViT+SAM2 已于 2026-07-28 移出：论文 0 提及，备用配置不再补全。）

| # | 组件 | 现状 | 论文/cap-x 中的角色 | 补全方式（均在 Phase 0） |
|---|---|---|---|---|
| 1 | 碰撞 | ⚠️ 仅 IK 最终构型检查 | trace 的 collision feedback（组件 1）；碰撞感知规划 | 路径碰撞检查 + trace 碰撞字段 + cuRobo/pyroki 碰撞规划 |
| 2 | Contact-GraspNet | ❌ 几何规划器替代 | cap-x `plan_grasp` 真身（server :8115） | CGN 服务（进程隔离，同 vision_server 模式），`plan_grasp` 后端切换 |
| 3 | pyroki | ❌ DLS+种子库替代 | cap-x IK server（:8116），URDF 驱动 | pyroki 服务 + Piper URDF（用户提供），`solve_ik` 后端切换 |
| 4 | cuRobo | ❌ | **cap-x 有完整集成**（`integrations/motion/curobo*.py` + `serving/launch_curobo_server.py`，libero 线默认注释） | 参考 cap-x 集成做 GPU 批量 IK + 碰撞轨迹；与 pyroki 对比选默认 |

---

## 4. 长期路线图

> 里程碑门禁制：每个 Phase 的出口标准全部满足才进入下一个；时间盒为预估，不超时不强推。

### Phase 0 — API 构建阶段（当前，含 4 必要组件）
**目标**：`PiperControlApiReduced` 完整交付 = **15 函数 API 面**（cap-x reduced 10 +
论文/open_details 补充 5，逐函数出处见
[docs/api_asset_map.md](docs/api_asset_map.md)）+ 4 组件全部集成，Stack demo 跑通。
- **运动与 IK 层**：运动可信化（响亮报错 + waypoint 插值 + FK 到位验证）→
  pyroki 服务（Piper URDF）替换 `solve_ik` 后端 → cuRobo 批量 IK / 碰撞轨迹
  （与 pyroki 对比选默认；安装与本机 torch 冲突时 timebox 半天，先用 pyroki 解锁下游）。
- **感知层**：Contact-GraspNet 服务替换 `plan_grasp` 后端（checkpoint 下载先问用户）；
  HSV 兜底 SAM3 漏检（demo 侧）。
- **碰撞**：运动路径逐段碰撞检查；trace 记录碰撞事件（接触 geom 对/位置/时刻），
  对齐论文组件 1 的 "collision feedback"。
- **demo 闭环**：工作区收紧（离线可达性扫掠定 CUBE 范围）+ 抓取后验证/重试。
- **收尾**：`PiperControlApiReduced` 类命名 + `docs/primitive_api_capx.md` 补全 + git 提交。
- 详细交接：[handoff_piper_control_api.md](handoff_piper_control_api.md)（仓库根目录）。
- **出口标准**：`test_piper_capx_api.py` 全 PASS（含新后端与碰撞字段的扩展用例）；
  Stack seed 0 成功 + seed 1/2 ≥2/3；4 组件各自集成测试达标；
  `docs/api_asset_map.md` 看板全部打勾；文档与实现一致。
- 预估：3–4 周。

### Phase 1 — Skill Library 闭环（论文组件 2）
**目标**：debug→validate→入库→注入 全链路跑通，见到第一个真实 skill 入库并起效。
- findings.md schema（failure mode / validated repair / transferable patterns / 验证成功率）；
- coordinator 角色审计提炼 SKILL.md（附录 E.5 模板）；
- 注入机制（初期全量注入，skill 数 >20 后做检索）；
- 在 Stack + 1-2 个新任务（如 Lift、PickPlace 的 Piper 版）上验证"入库 skill 提升后续任务成功率"。
- **种子素材（论文附录现成的 skill 原型，直接当格式标杆与首批内容）**：
  - debugging 表：`plan_grasp` 返回空 → dilate mask、log `mask.sum()` 应 >200px；
    `solve_ik` 返回 None → 目标 z 降 5cm、重查 x/y 可达性；
    抓取成功但抬起掉落（夹爪半闭/细长物）→ 试垂直偏航 90° 或沿长轴逼近；
  - `make_topdown_quat(yaw_deg)` 的 scipy 参考实现（xyzw→wxyz 重排）；
  - localize / grasp 两个初始 SKILL.md 完整样例（论文附录 E.5 贴出全文，
    含 frontmatter 与栏目格式）——首批入库条目按此格式蒸馏我们自己的实测经验
    （如 §6.3 的 overhead 钩抓、x>0.45 不可达、min_z 抓取基准）。
- **出口标准**：≥3 个 skill 入库；对照实验（有/无注入）成功率有统计意义差异。
- 预估：3–4 周。

### Phase 2 — 进化搜索（论文组件 3）
**目标**：K≥4 候选程序锦标赛 + 多代进化，候选**并行评估**（多进程 sim 实例，AGENTS.md 硬性要求）。
- 候选生成（温度/扰动采样）、并行评估 harness、基于 surviving programs + residual traces 的下一代条件化；
- 与 Phase 1 联动：repaired program vs evo search 的 held-out 验证（论文 Aspire 列的评测方式）。
- **出口标准**：在 ≥3 个任务上 evo search 候选超过单轨迹修复基线。
- 预估：3–4 周。

### Phase 3 — 长程任务与导航扩展
**目标**：任务谱扩展到 BEHAVIOR 风格长程任务；导航纳入。
- BEHAVIOR 风格多阶段任务（多点位巡访、开关容器等）；
- 固定基座下导航以"任务级点位序列"形式纳入，真 mobile base 视硬件到位再议；
- 依赖 Phase 0 的碰撞感知规划与 trace 碰撞反馈。
- **出口标准**：≥1 个长程任务（≥5 阶段）跑通并入库对应 skills。
- 预估：3–4 周。

### Phase 4 — sim2real：Piper 真机
**目标**：同名 primitive API 的真机实现，sim 积累的 skill library 上真机验证。
- 真机适配层：API 签名不变，底层走 piper_sdk（IK/关节控制/夹爪）；
- 相机标定（realsense/腕部）、SAM3/CGN 真机推理链路；
- skill 迁移实验：sim 入库 skills 作为 in-context guidance，测真机编程 token 成本降幅（论文 1.4 范式）。
- **出口标准**：真机完成 ≥1 个 sim 训练过的任务；skill 注入 vs 无注入对照。
- 预估：4–6 周（硬件排期另计）。

### 平行线（任意时刻可插入）
- **cap-x 原生环境插件化**：若要复现 LIBERO-Pro/BEHAVIOR 榜单数字，把 cap-x envs 作为
  另一种基底接入同一引擎接口（cap-x 当插件，不当宿主）。
- **AGENTS.md/文档维护**：每 Phase 结束同步一次。

---

## 5. 简化清单（2026-07-28 修订）

**仅保留一条**：

1. **只做单臂任务**（双臂 handover / TwoArmLift 不碰；cap-x 的 bimanual API 分支不实现）。

初版其余简化全部撤销：感知已对齐论文（SAM3 主力 + Molmo 兜底，非 MobileSAM）；
导航与碰撞纳入范围（§4 Phase 0/3）；12h 时间盒作废（§4 长期路线图）；
benchmark 不再"尽可能简单"（Phase 1 起逐步加任务）；primitive API 不做"难复现就略过"
（§3 四组件全部补全，归属 Phase 0）。

---

## 6. 环境配置与踩坑记录（存档）

### 6.1 环境

```bash
# conda env: ASPIRE（大写，py3.12）；python 用绝对路径
/home/stouching/anaconda3/envs/ASPIRE/bin/python script.py
# 无头渲染：EGL 后端
MUJOCO_GL=egl /home/stouching/anaconda3/envs/ASPIRE/bin/python ...
```

mujoco 必须 3.3.*（新版 C API breaking change 与 robosuite 1.5.2 冲突）。
CUDA 与 EGL 同进程冲突（实测 2026-07-28）：torch CUDA 初始化后同进程 EGL 渲染永久损坏
→ SAM3 等 CUDA 推理一律进程隔离（vision_server 模式）。

### 6.2 模块 2 踩坑（2026-07-24，已固化在代码注释）

1. robosuite transform_utils 全套 xyzw，API 边界 wxyz → `_q_in`/`_q_out` 边界转换；
2. `get_camera_segmentation` 多翻转一次 → 调用后再翻回；
3. mujoco 相机系 y 向上 → 反投影 `y = -(v-cy)*z/fy`；
4. `robot0_eef_quat` ≠ `_site` 版本，姿态闭环用 `_site`；
5. 四元数 w 不归一化 → IK 在 π 附近振荡；姿态同伦绕 180° DLS 奇异；
6. horizon 会被闭环吃满 → horizon=1000 + terminated 静默防御；
7. 抓取基准用 min_z（p5）而非 top_z（可见面偏置 +1.3cm，易滑落）。

### 6.3 cap-x/Piper 线踩坑（2026-07-27/28，详见交接文档与代码注释）

1. robosuite XML 解析四坑：嵌套 default class / geom group 仅 0/1 / childclass 不支持 / mesh 前缀冲突；
2. Piper 6 轴 + j5 ±70°：桌面高度严格顶朝下不可达 → 远侧 overhead 钩抓姿态族（倾角 ~25°）；
3. 可达带：悬停高度基座系 x≈0.28–0.40，x>0.45 IK 不收敛；
4. 位置伺服碰撞卡死后静默超时 = 静默失败（TCP 偏差实测 150–300mm）→ Phase 0 P0 修复；
5. EGL offscreen 高频 wedge（512 双相机 → 降 256 + 运动拍不渲染 + L1/L2 自愈）；
6. IK 长距离关节跳转扫碰撞 → 最终构型 collision_free 检查 + Phase 0 路径检查。
7. pyroki 依赖死锁（jaxls→jax≥0.6→numpy 2.x vs 主环境 numpy 1.26）→ **环境隔离**：
   服务住独立 venv，主环境零改动；禁止升级主环境全家桶解依赖冲突。
8. 官方 URDF 与第三方 MJCF（yanyuze1 仓）几何一致（杆长模长精确相等）但**关节零位
   约定不同**（同数值 FK 差 ~411mm）→ 零位映射标定 `q_mjcf = sign·q_urdf + offset`，
   适配层常驻；`mj_saveLastXML` 不能导出 URDF。
9. 相机位姿（2026-07-29 图像实锤）：robotview 只框方块工作区、臂整体出画
   （"臂没动过"是出画假象）；eye_in_hand 朝 link6 +z 看进掌心网格（手指伸向 -z）
   → A0.5 相机标定：改 pos/target + 渲染迭代验收。**教训：观测结构测试
   （shape/dtype）查不出构图错误，相机验收必须看图。**

---

## 7. 后续方向（超出当前路线图，仅记录）

- 多设备/多 worker 并行调度（论文 coordinator 并行 actor 架构）；
- LIBERO-Pro Long zero-shot 迁移实验（论文 31% vs 4% 的招牌对照）；
- skill library 跨本体迁移（Piper ↔ 其他臂，若未来引入）。
