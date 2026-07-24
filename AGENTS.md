# ASPIRE 复现项目 - Agent 关键信息文档

> ⚠️ **重要提示**：Claude Code 每次执行前必须先阅读此文档！

---

## 0. 运行环境规则（每次执行脚本必守）

**所有 Python 脚本必须在 conda 环境 `aspire`（小写！）中运行。**

- 环境名：`aspire`（⚠️ 全小写，不是 ASPIRE）
- Python 路径：`/home/stouching/anaconda3/envs/aspire/bin/python`
- pip 路径：`/home/stouching/anaconda3/envs/aspire/bin/pip`

**推荐写法**（Bash 工具每次调用是新 shell，`conda activate` 不一定生效，直接用绝对路径最可靠）：

```bash
# ✅ 推荐：直接用环境的 python
/home/stouching/anaconda3/envs/aspire/bin/python script.py

# ✅ 也可以：先 source 再 activate
source /home/stouching/anaconda3/etc/profile.d/conda.sh && conda activate aspire && python script.py

# ❌ 禁止：直接用系统 python 或 base 环境运行项目脚本
python script.py
```

安装依赖同样必须用环境的 pip：`/home/stouching/anaconda3/envs/aspire/bin/pip install ...`

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

---

## 4. CaP-X 替代方案（已确认）

**CaP-X = 按 API 文档生成控制代码的骨干 LLM（未开源）→ 由 Claude Code（K3）本机直接替代**

- ❌ 不配置外部 API / 不写 CLI 调用框架（CLI 自动化留待无人值守阶段）
- ✅ 替代充分条件：**一份完整的 Primitive API 文档**（每个函数的签名/参数/返回值/副作用/使用示例）
- ✅ `open_details/` 下三个任务代码作为 few-shot 示例注入 prompt
- 工作模式：交互式循环 —— 生成代码 → 执行引擎运行 → 看 trace → 修代码

---

## 5. 复现成功的关键因素总结

1. ✅ 部署正确的primitive api和正确使用 primitive API 完成任务代码
2. ✅ **并行实现** Evolutionary Search 模块
3. ✅ 执行引擎完整记录每一步的【观测、输入、输出、视觉证据】，这样coding agent才能抓住失败的关键所在。

---

*最后更新：2026-07-23*
