# aspire/agentic/ —— Agentic Coding 组件（论文组件 3）

> 2026-08-10 新建（不锁）。设计依据：[../../docs/agentic_design.md](../../docs/agentic_design.md)
> （论文 §2.1-2.3/Fig 2-3/Appendix A/E.1-E.5/Algorithm 1 → 实现落点的完整对照表）。

## 这是什么

论文三组件中的后两个之**组件 3 的载体**：让 LLM 作为 coding agent 生成/修复
机器人程序（code-as-policy），在真实执行引擎上闭环验证，并把验证过的经验
蒸馏进 [../skills/](../skills/) 的 Skill Library（组件 2）。

```
TaskSpec → Actor/EvolutionarySearch → evaluate.py(并行子进程跑引擎 CLI)
        → trace.json → trace_digest → LLM 修复/候选生成 → findings.md
        → Coordinator 审计 → skill_library 入库
```

## 文件

| 文件 | 说明 |
|---|---|
| `config.py` | LLM 配置：env/.env 加载（`ASPIRE_LLM_*`），provider = `openai`/`anthropic`/`file`/`mock`。file = prompt/response 文件桥接（无 key 时人工/Claude Code 当模型，与 AGENTS.md §4 交互式工作流同构）；mock = 测试脚本化响应 |
| `llm_client.py` | vendor 自 cap-x `capx/llm/client.py`（MIT）：单次查询/并行集成（并发候选+LLM 综合，非投票）/流式/多模态 data URL/fence 抽取。delta：provider 分派替代模型名分派、5xx 重试加上限、删 OpenRouter 强制改道 |
| `trace_digest.py` | 论文 §2.1 的 LLM 消费侧：trace.json → 全量一行式调用摘要 + 失败信号自动扫描（zero masks/碰撞/IK 失败/空抓）+ **只附失败邻近记录的调用前后帧与算法标注图**（max_images 上限） |
| `prompts.py` | 全部 agent prompt：ACTOR_SYSTEM（任务代码规则+E.2 FORBIDDEN+API 参考注入）、E.3 findings 逐字模板、E.1 coordinator 审计（JSON 输出）、E.4 候选生成（独立假设+docstring+反过拟合条款）与 task_analysis 更新 |
| `task_spec.py` | TaskSpec（场景/指令/成功判定/debug 与 held-out seed 划分/基线路径）；内置 Stack、PiperWipeSpill |
| `evaluate.py` | **并行评估 harness**（AGENTS.md §2 硬性要求）：(候选, seed) 全组合进子进程池跑引擎 CLI；vision_server harness 单例预拉（防端口竞争与误杀）；退出码+stdout 解析 success/trace 路径；cancel_all 支持 web 停止 |
| `actor.py` | E.3 修复闭环：fast path（baseline 先评）→ full debug loop（≤3 轮，=每 seed 3 次 replay 上限）→ Stage 2 held-out 一次性验证 → findings.md |
| `coordinator.py` | E.1 协调者：progress.json 任务队列、分派 actor、只读 findings.md、审计后**串行入库**、已完成任务不重复分派 |
| `evolve.py` | **Algorithm 1 逐行实现** + E.4：task_analysis.md 跨代持久、candidate_A verbatim 精英种子、K 候选并行评估、θ 早停、Stage 2 一次性、最佳不超基线则回落基线、ExtractValidatedPatterns→findings |
| `cli.py` | `python -m aspire.agentic.cli {actor|evosearch|coordinator|skills|digest}` |

## 用法

```bash
cp .env.example .env   # 填入 ASPIRE_LLM_API_KEY（或 --provider file 走人工桥接）
scripts/agentic.sh actor Stack                 # 修复闭环（fast path 直达 Stage 2）
scripts/agentic.sh evosearch Stack -- -K 4 -T 5  # 进化搜索
scripts/agentic.sh coordinator Stack,PiperWipeSpill
scripts/web.sh                                 # Web UI :8200
```

运行产物在 `agent_runs/`（gitignore）：候选代码/leaderboard/task_analysis.md/
findings.md/progress.json；trace 在各 run 目录的 traces/ 下。

## 纪律

- LLM 生成的代码与人工任务代码同规则（scripts/README §1）：禁 import、响亮失败、
  凡动必避障、先算后动、起手 TUCK；外加 E.2：禁 simulator ground truth。
- 本层只经**引擎 CLI 子进程**驱动仿真（封版接口），不 import 引擎内部。
- 测试：`scripts/tests/test_agentic.py`（44 项，mock LLM + fake executor）+
  `test_agentic_e2e.py`（真实引擎端到端）。
