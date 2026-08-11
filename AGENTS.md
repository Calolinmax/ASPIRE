# ASPIRE 复现项目 - Agent 关键信息文档

> ⚠️ **重要提示**：Claude Code 每次执行前必须先阅读此文档！

---

## 0. 运行环境规则（按任务选环境，不要盲目套用单一环境）

**按任务类型选择运行方式（路由表）：**

| 任务类型 | 用什么跑 |
|---|---|
| 项目主代码（engine / primitives / scripts/ 下任务脚本、SAM3 客户端等） | `/home/stouching/anaconda3/envs/ASPIRE/bin/python`（conda 环境 `ASPIRE`，全大写，py3.12；旧小写 aspire 已删除） |
| 安装主环境依赖 | `/home/stouching/anaconda3/envs/ASPIRE/bin/pip install ...` |
| CGN 抓取服务 | 一律在 Docker 容器内运行（`docker run/exec`，镜像 `cgn-tf:25.02`，容器名 `cgn`），**不要**在宿主机任何 Python 下启动 CGN |
| CGN 宿主机 fallback（仅留档调试用） | `external/cgn_venv/bin/python`（TF 版，正常任务不用） |
| 非 Python 命令（git / docker / curl / nvidia-smi / 文件操作等） | 直接运行，无环境限制 |

- Bash 工具每次调用是新 shell，`conda activate` 不一定生效，用绝对路径最可靠。
- ❌ 唯一禁令：不要用系统 python 或 conda base 环境跑项目主代码。

**开源参考纪律（顾问与执行者共同遵守，违反即返工）：**
凡功能存在开源实现——不限于本仓库 `external/` 下的 cap-x、contact_graspnet，
也包括 PyPI/GitHub 上任何有具体实现的项目——动手前必须先读官方实现的
**完整函数体**，逐一搞清细节（结构、参数、坐标约定、默认值、边界处理），
**能直接照抄就直接照抄**，照抄不了就对齐；**禁止凭签名/文档印象/想象
从 0 到 1 手搓**。即使判定「官方库跑不了」（如 mayavi 在容器里），也必须
继续读它「怎么做的」——实现细节往往可直接移植。
过往教训：CGN glyph 凭想象画三轮不合格，官方 draw_grasps 里开口 Π 结构一直是现成的。

**外部资源下载规则（每次涉及下载必守）：**

- **HuggingFace**：需要拉模型/数据集时，**先请用户人工确认仓库可访问**——HF 存在
  审批门控（gated repo），403/401 容易**误判为"项目不存在"**。拉取失败不得直接下
  "不存在"的结论，先报给用户核查。
- **GitHub 仓库名不要猜**：需要外部仓库时先问用户（用户可能已有本地副本，如 Piper
  URDF 由用户自行提供）。文档中允许写死的 URL 仅限已验证来源（如 cap-x
  `pyproject.toml` 的依赖声明）。

---

## 1. 原始代码与示例资源

### 1.1 Primitive API 任务代码

项目网页端展示的调用 primitive API 的任务代码已存放于：

```
/home/stouching/Desktop/ASPIRE/open_details/
├── primitive_api_cube_reset.py      # Cube Reset 任务
├── primitive_api_nut_assembly.py    # Nut Assembly 任务
└── primitive_api_wipe.py            # Wipe 任务
```

这些文件展示了要部署哪些primitive api，以及如何使用底层 primitive API 构建具体任务，是理解 ASPIRE 执行层的关键参考。

### 1.2 Skills 示例

网页端公布的 skills 示例：

```
/home/stouching/Desktop/ASPIRE/open_details/skill_grasp.md
```

此文件包含了技能定义和实现的参考示例。

---

## 2. Evolutionary Search 模块 - 关键要求

### 2.1 核心机制

ASPIRE 采用**进化搜索过程 (Evolutionary Search Over Programs)** 来生成多样化的任务序列和控制程序：

- 探索超越单轨迹自改进的方法
- 通过**迭代调试 (iterative debugging)** 优化
- 通过**并行精炼 (parallel refinement)** 加速

### 2.2 实现要求 ⚠️

**必须做到并行迭代，提高效率！**

- 不要串行执行搜索过程
- 利用多进程/多线程进行并行评估
- 设计可扩展的并行架构

---

## 3. 执行引擎 (Execution Engine) - 核心职责

### 3.1 功能定义

执行引擎的核心作用是：**执行程序，并记录每一步的完整信息**

### 3.2 需要记录的步骤类型

对于每一个执行步骤，需要捕获以下四类操作：

1. **检测 (Detection)** - 感知模块的调用
2. **规划 (Planning)** - 决策/规划模块的调用
3. **抓取 (Grasping)** - 抓取策略的调用
4. **控制调用 (Control)** - 底层控制命令的调用

### 3.3 需要记录的详细信息

对于每一步，必须记录：

| 字段                                 | 说明                  |
| ------------------------------------ | --------------------- |
| **观测 (Observation)**         | 当前环境状态观测      |
| **输入 (Input)**               | 该步骤的输入参数      |
| **输出 (Output)**              | 该步骤的执行结果      |
| **视觉证据 (Visual Evidence)** | 截图/视觉记录作为证据 |

### 3.4 为什么这很重要

这些记录是：

- 进化搜索评估程序质量的依据
- 失败分析和调试的关键数据
- 生成训练数据的基础

### 3.5 Trace 目录结构规范

每次任务执行生成一个独立目录，命名 `MMDD_HHMM_任务名`（同时刻冲突自动加 `_2` 后缀）。trace 目录只放**执行产物**；任务代码由 agent harness 管理，trace.json 的 `code_ref` 字段仅记录其路径引用：

```
traces/0727_1352_Stack/
├── trace.json              # 全部 API 调用记录：事件顺序 ×【类别/输入/输出/观测】，图像只存链接
└── images/
    ├── top/    s00000.jpg ...   # 顶部原始帧（固定每 5 仿真步一帧 + 调用边界补帧，文件名=仿真步）
    ├── wrist/  s00000.jpg ...   # 腕部原始帧（同帧率，顺序浏览=任务完整过程）
    ├── depth/  s00000.jpg ...   # 顶部深度 colormap（仅 API 调用边界保存——深度只在感知调用时被消费）
    ├── sam3/   005_segment_sam3_text_prompt.jpg ...  # SAM3 标注图（仅调用时保存）
    └── <算法>/ ...              # 任务用到的每个图像算法一个文件夹（molmo/mask_to_world/grasp...）
```

规则：

- **帧流零重复、关键帧不丢**：按 (流, 仿真步) 去重；每次 API 调用边界强制补帧，运动轨迹按文件名（仿真步）顺序可完整复盘。
- **depth 不做固定帧率连拍**：它不是一直被消费的（只有 get_observation/mask_to_world 等感知调用用），只在调用边界保存，每条记录的 depth 链接始终有效。
- **算法标注图每次调用都存**：包括"未检出"的帧（上面画 no mask）——那是失败定位的关键证据；未运行的算法不产生文件夹。
- **图像与记录的关联全部由 trace.json 链接**：`visual_evidence`（调用前帧）/ `visual_evidence_after`（运动结束帧）指向帧流，`annotation` 指向算法标注图。

---

## 4. CaP-X 与 Coding Agent（2026-07-28 修订）

**CaP-X 已开源**（github.com/capgym/cap-x，MIT），本地镜像 `external/cap-x/`。
论文架构：ASPIRE = CaP-X 基底（primitive API + 仿真器）+ 三组件（执行引擎 trace /
skill library / 进化搜索），本项目的定位是**复现三组件，cap-x 当插件不当宿主**。

- ✅ API 基底：`FrankaControlApiReduced` 契约（`external/cap-x/capx/integrations/franka/control_reduced.py`）
  → Piper 实现 `PiperControlApiReduced`（`aspire/api/primitives_capx.py`，10 函数 1:1）
- ✅ Coding agent：Claude Code 本机直接担任（**与论文同构**——论文用 Claude Code + Opus 4.6 1M）
- ✅ **程序化 agentic coding 已落地（2026-08-10，用户指令）**：`aspire/agentic/`
  （LLM client/actor/coordinator/Algorithm 1 进化搜索）+ `aspire/skills/`（技能库）+
  `aspire/web/`（Web UI）。LLM 走 `.env` 配置（`.env.example` 为模板）；
  `--provider file` = 文件桥接模式（无 key 时人工/Claude Code 当模型，与本节交互式
  工作流完全同构）。设计对照表见 `docs/agentic_design.md`。
- ✅ `open_details/` 下三个任务代码作为 few-shot 示例注入 prompt
- 工作模式：交互式循环 —— 生成代码 → 执行引擎运行 → 看 trace → 修代码
  （程序化版：actor.py 的 repair loop 即此循环的自动化）

**当前进展与文档索引**：路线图见 `docs/roadmap.md`（Phase 1 起）；
仓库地图见 `docs/project_files.md`（逐文件说明）；
两条 API 线：Panda 线（`aspire/engine/engine.py` + `aspire/api/primitives.py`，模块 2 存档）
与 **cap-x/Piper 线（`aspire/engine/engine_capx.py` + `aspire/api/primitives_capx.py`，当前主线，已封版）**。

---

## 5. 复现成功的关键因素总结

1. ✅ 部署正确的primitive api和正确使用 primitive API 完成任务代码
2. ✅ **并行实现** Evolutionary Search 模块
3. ✅ 执行引擎完整记录每一步的【观测、输入、输出、视觉证据】，这样coding agent才能抓住失败的关键所在。

---

*最后更新：2026-07-28*
