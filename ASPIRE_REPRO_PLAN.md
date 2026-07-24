# ASPIRE 初步复现计划

> 整理日期：2026-07-23
> 目标论文：ASPIRE: Agentic /Skills Discovery for Robotics（NVIDIA GEAR，2026）
> 项目页：https://research.nvidia.com/labs/gear/aspire/
> 论文 PDF：https://research.nvidia.com/labs/gear/aspire/assets/Aspire.pdf

---

## 1. 论文关键结论（复现的依据）

### 1.1 Skill Library 的形态

- Skill **不是**封装好的可执行函数（不是 `handover()` 这种可 import 的 API），而是**结构化的知识文档**（`SKILL.md`），以 in-context guidance 方式注入未来 agent 的 prompt。
- 底层可执行的 primitive API（感知、规划、控制）是**人工预定义、固定不变的**；skill library 不会往 API 里注册新函数。
- 每个 skill 条目四要素（论文附录 A）：
  1. **Problem** — 从触发失败 trace 提取的失败特征（failure signature）
  2. **When to Apply** — 适用/检索条件（guard）
  3. **Strategy** — 验证过的修复策略，可附几行 code sketch
  4. **Origin task(s)** — 来源任务

### 1.2 Skill 范式（附录 E.5 贴出的完整模板）

`SKILL.md` = YAML frontmatter（`name`、`description`）+ 固定栏目：

```
> Purpose / Ownership（coordinator 才能改，subagent 只读）
## 代码骨架（如标准 pick-and-place 流程）
## When to NOT Use This Template（反例边界）
## Per-Object Registry（随经验增长的参数表）
## Anti-Patterns（禁止事项）
## Debugging 表（症状 → 可能原因 → 检查方法）
```

入库流程：actor 按 findings schema 上报（failure mode、validated repair、transferable patterns、task-specific quirks、验证成功率）→ coordinator 审计可复用性 + API 合规性 → 只把验证通过且可迁移的模式写入共享库。

### 1.3 开源状态

**截至 2026-07-23 未开源**：项目页无代码链接，GitHub 搜不到官方仓库，CaP-X 也无公开仓库。论文中 "See code release" 为预留说法。可参考的只有论文附录贴出的部分模板（Figure 3 标注 "Skill Library (Partial)"）。

### 1.4 论文的 sim2real 证据

论文在 **Franka 仿真**中发现 skills，迁移到**双臂 YAM 真机站**（本体和 API 均不同），skills 作为 in-context guidance 显著降低真机编程 token 成本。说明：**skill 是本体无关的知识，同类型机械臂之间可迁移**；带物理参数的条目（z_offset、yaw 等）需在新本体上重新验证。

---

## 2. 已确认的决策

| 项目                | 决策                                                                                                                                                   | 理由                                                                  |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------ | --------------------------------------------------------------------- |
| 真机                | 松灵 PiPER（6 轴 + 夹爪），后期包一层同名 primitive API（底层走 piper_sdk IK/关节控制）                                                                | 现有硬件                                                              |
| 仿真臂              | **Franka Panda**（robosuite 内置 + MuJoCo Menagerie 模型 + 现成 OSC 控制器）                                                                     | 7 轴 vs 6 轴只影响 IK primitive 内部实现，不影响 skill 层；零建模成本 |
| 仿真框架            | **robosuite**（基于 MuJoCo，自带任务 benchmark 和 `_check_success()` 成功判定）                                                                | 环境/benchmark/成功判定全白送，最大省时项                             |
| PiPER 仿真模型      | 暂不需要；松灵官方 GitHub 有`piper_mujoco`，URDF/描述文件可从官方仓库获取转 MJCF，留作后期 sim2real 中间验证 | —                                                                    |
| 感知 primitive      | **sim ground truth 冒充 SAM3**：保留 API 外形 `segment_sam3_text_prompt(rgb, "red_cube")`，内部直接查 sim 状态                                 | 不渲染 depth、不做点云、不算 OBB                                      |
| 运动 primitive      | IK + 路点插值；**不做导航**（固定基座）、**不做碰撞规划**                                                                                  | 砍最大坑                                                              |
| Agent 结构          | 单 agent 循环，两段 prompt 分饰 actor / coordinator 两角，串行执行                                                                                     | 并行多设备调度不影响范式验证                                          |
| Evolutionary search | 作为 stretch goal，退化为 K=2 候选锦标赛（1–2 轮）；**候选评估必须并行**（多进程各起一个 sim 实例，AGENTS.md 硬性要求）                                   | 核心闭环优先；但并行是论文核心机制，不可退化为串行                    |
| Coding agent        | **K3（Claude Code 本机直接担任，不配外部 API）**——CaP-X 的替代品 = K3 + 完整 Primitive API 文档 + open_details few-shot 示例 | CaP-X 本质就是"按 API 文档写控制代码的 LLM"，K3 代码能力足够，差距用 prompt 工程补；CLI 自动化等无人值守阶段再说 |
| 视觉证据            | **每一步都记录**【观测、输入、输出、视觉证据】（AGENTS.md 硬性要求）；存储上可压缩（低分辨率关键帧），但记录不可省略                                                 | 这是 coding agent 抓住失败关键的数据基础 |

**已知代价**：感知走 ground truth 后，ASPIRE 中"感知 prompt 调试"类 skill 练不出来，库里只会长 motion/grasp 类 skill。框架验证阶段可接受。

---

## 3. 环境配置（新机器需重做）

```bash
# Python 3.13 太新，robosuite 兼容性差；环境已建好（2026-07-23）：conda env "aspire"，Python 3.10
conda activate aspire
/home/stouching/anaconda3/envs/aspire/bin/pip install mujoco robosuite

# 无头服务器（无 DISPLAY）渲染验证：EGL 后端
MUJOCO_GL=egl /home/stouching/anaconda3/envs/aspire/bin/python -c "
import robosuite as suite
env = suite.make('Lift', robots='Panda', has_renderer=False,
                 use_camera_obs=True, camera_names='agentview')
obs = env.reset()
print('OK', sorted(obs.keys()))
"
```

注意：若 EGL 报错，检查 NVIDIA 驱动与 `libegl1`；备选 `MUJOCO_GL=osmesa`（需装 osmesa 库，速度慢）。

**已踩过的坑（2026-07-23）**：pip 默认装最新 mujoco（3.10.0），其 C API 有 breaking change，robosuite 1.5.2 初始化 OSC 控制器时报 `TypeError: mj_fullM(): incompatible function arguments`。**必须降级：`pip install "mujoco==3.3.*"`**（实装 3.3.7 验证通过）。

**模块 2 踩坑记录（2026-07-24，Primitive API 实现）**——全部已在代码注释中固化：

1. **robosuite transform_utils 全套是 xyzw**（quat2mat/mat2quat/quat_slerp/quat2axisangle），与 API 文档对用户暴露的 wxyz 约定冲突 → `primitives.py` 中 `_q_in`/`_q_out` 做边界转换，模块内部统一 xyzw。
2. **`get_camera_segmentation` 多翻转一次**：mujoco 3.x render 已返回 row0=top 正向图，obs 的 RGB/depth 未翻转，但 segmentation 函数按老 OpenGL 假设翻了 `[::-1]` → 调用后再翻回一次对齐。
3. **mujoco 相机系 y 轴向上**（图像 v 向下）→ 反投影 `y = -(v-cy)*z/fy`，不处理会致 z 系统性偏高 ~4cm（抓取目标点全错）。
4. **`robot0_eef_quat` ≠ `robot0_eef_quat_site`**（不同 frame），姿态闭环必须用 `_site` 版本。
5. **四元数 w 符号不规范化（w≥0）→ IK 在 π 附近振荡**；另需姿态同伦（slerp 分步）绕开 180° DLS 奇异性。
6. **horizon=300 会被 move_to_pose 闭环吃满**（单次最多 160 步）→ horizon=1000 + engine 对 terminated episode 静默防御。
7. **抓取基准用 min_z（点云 p5）而非 top_z**：可见面偏置使 top_z 系统偏高 ~1.3cm，夹上部易滑落（seed1 失败案例，正是 ASPIRE skill 提炼的典型原料）。

---

## 4. 12 小时计划分解

| # | 模块                           | 内容                                                                                                               | 预估    |
| - | ------------------------------ | ------------------------------------------------------------------------------------------------------------------ | ------- |
| 0 | 环境安装                       | 上节命令 + EGL 渲染验证                                                                                            | 0.5–1h |
| 1 | 仿真环境                       | robosuite hello-world：Lift 任务 + 脚本化策略验证抓取流程                                                          | 1–1.5h |
| 2 | Primitive API                  | **先写完整 API 文档**（每个函数的签名/参数/返回值/副作用/使用示例，作为 coding agent 的 prompt 上下文）→ 实现：`get_observation`、感知 facade（GT 冒充 SAM3）、IK `move_to`、`grasp/lift/place`、每 primitive 的 trace 日志 | 3h      |
| 3 | Agent harness                  | prompt 模板（含 API 文档）→ **K3（Claude Code 交互式）** 生成`task_code.py` → 子进程执行 → 收集 trace/成败 → 带 trace 重试           | 2h      |
| 4 | Skill library                  | findings.md schema、coordinator 角色提炼成 SKILL.md、注入 prompt（初期全量注入，不做检索）                         | 1h      |
| 5 | Evolutionary search（stretch） | K=2 候选、1–2 轮锦标赛选优；**候选并行评估**（多进程 sim 实例，robosuite 非线程安全须用进程）                                        | 1–1.5h |
| 6 | 端到端联调                     | Lift 任务跑通 debug→validate→入库全流程 + 修坑                                                                   | 1.5–2h |

**里程碑**：约 6–8h 时见到第一个 skill 入库（最小闭环 = robosuite Lift + GT 感知 + IK primitives + 单 agent 修复循环 + 1 个种子 SKILL.md + 成功率统计）。剩余时间加 PickPlace 任务和 evo search。

**最大风险点**：

- IK / 抓取鲁棒性（半天级坑都在这；缓解：primitives 里内置 top-down grasp + yaw fallback + argmax fallback）
- LLM 生成代码的抽取与沙箱执行（缓解：强制输出单文件 ```python 块，子进程 + timeout 运行）
- 物理仿真不确定性（缓解：固定 seed 集，debug seeds 与 validation seeds 分开，validation 只跑一次）

---

## 5. 简化清单（已确认）

用户原定的两条：

1. primitive API 中难复现的可简化或直接略过；
2. benchmark 任务尽可能简单，框架能跑通即可（Lift 必做，PickPlace 可选，双臂 TwoArmHandover 第一阶段不碰）。

追加的七条：
3. 环境用 robosuite，不碰 raw MuJoCo XML；
4. 只做单臂、1–2 个任务；
5. 感知全部走 sim ground truth（保留 SAM3 风格 API 外形）；
6. 砍导航与碰撞规划（IK + 插值）；
7. coordinator/subagent 不拆进程，单循环分饰两角；
8. evolutionary search 放最后，退化为 2 候选锦标赛，**但候选评估必须并行（AGENTS.md 硬性要求）**；
9. **执行引擎每一步记录【观测、输入、输出、视觉证据】（AGENTS.md 硬性要求）**，存储可压缩但记录不可省。

---

## 6. 后续（超出 12h 范围，仅记录方向）

- 加 PickPlace / 更多任务，观察 skill library 增长与 zero-shot 迁移；
- 真 SAM3 + depth 点云替换 GT 感知 facade；
- 用松灵官方 `piper_mujoco` 模型建贴近真机的仿真环境做中间验证；
- PiPER primitive 适配层带 sim 积累的 SKILL.md 上真机；
- coordinator/subagent 并行化、多设备调度。
