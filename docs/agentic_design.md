# Agentic Coding + Skill Library 模块设计（2026-08-10）

> 组件 2（Skill Library）与组件 3（Agentic Coding / 进化搜索）的复现设计。
> 依据：ASPIRE 论文 §2.1-2.3、Fig 2/3、Appendix A、E.1-E.5、Algorithm 1；
> cap-x 参考实现（`capx/skills`、`capx/llm/client.py`、`capx/web`）；
> 本仓库既有插接点（`engine.run()` / `build_namespace` / trace 体系）。

## 0. 架构总览

```
aspire/
├── agentic/              # 组件 3：agentic coding（新层，不锁）
│   ├── config.py         #   LLM 配置：env/.env 加载，provider = openai|anthropic|file|mock
│   ├── llm_client.py     #   vendor capx/llm/client.py：单次查询/并行集成/流式/多模态
│   ├── trace_digest.py   #   trace.json → LLM 可消费摘要（论文 §2.1：仅原语调用前后紧邻帧+叠加+返回值）
│   ├── prompts.py        #   actor/coordinator/进化搜索 prompt（E.1/E.3/E.4 协议逐项落实）
│   ├── evaluate.py       #   并行评估 harness：子进程跑引擎 CLI（AGENTS.md §2 硬性要求）
│   ├── actor.py          #   actor：生成→执行→诊断→修复闭环 + findings.md（E.3 Stage 1/2）
│   ├── coordinator.py    #   coordinator：任务分派、findings 审计、skill 串行入库（E.1）
│   ├── evolve.py         #   Algorithm 1 + E.4：task_analysis.md 跨代持久，K 候选 × T 代
│   └── cli.py            #   python -m aspire.agentic.cli {actor|evosearch|coordinator|skills}
├── skills/               # 组件 2：skill library（新层，不锁）
│   ├── schema.py         #   SkillEntry = Problem/When to Apply/Strategy/Origin tasks(+Evidence/Limitations)
│   ├── library.py        #   文件系统库 + index.json + prompt 注入格式化（论文：文件系统即索引，无向量检索）
│   └── synthesize.py     #   findings.md → SKILL.md：coordinator 审计（可复用性 + API 合规静态检查）
└── web/                  # Web UI（新层，不锁；cap-x web 最小可用版）
    ├── server.py         #   FastAPI REST + WebSocket 事件流 + 静态托管
    └── static/index.html #   单页应用（Runs/Skills/Traces 三栏，零构建）
skill_library/            # 技能库数据（git 跟踪的共享知识文档，= 论文 .claude/skills/ 的角色）
agent_runs/               # 运行产物（gitignore；候选代码/leaderboard/task_analysis/findings）
```

**关键对应关系**：
- 论文用 Claude Code + Opus 做 coding agent，技能存 `.claude/skills/<name>/SKILL.md` 自动发现。
  本项目技能库存 `skill_library/`（git 可跟踪、跨 agent 共享），文件系统即索引、
  全量 in-context 注入（>20 条后按 category/关键词检索——roadmap 既定路线）。
- 论文 E.1 的 coordinator=Claude Code 主会话、actor=子 agent；本项目用程序化实现：
  coordinator.py 管任务队列与串行入库，actor.py 管单任务修复闭环，二者都只经
  `findings.md` 交换经验（论文 §2：actor 之间不交换原始轨迹，只蒸馏进库）。
- LLM 接入走 `.env`（gitignore 已预留）配置的 OpenAI/Anthropic 兼容端点；
  `provider=file` 时退化为"写 prompt 文件→等响应文件"的人工/Claude Code 桥接模式
  （与 AGENTS.md §4 现行交互式工作流同构，无 key 也能跑全流程）。

## 1. 论文条款 → 实现落点

| 论文条款 | 落点 |
|---|---|
| §2.1 只保留原语调用前后紧邻帧+叠加+返回值 | 引擎 trace 已实现（组件 1）；`trace_digest.py` 负责把 trace.json 裁成 LLM 输入（失败定位 + 失败邻近记录的 before/after 帧与标注图，限图像数防 token 爆炸） |
| §2.2 skill 四要素 + Fig 3 类目不预设 | `schema.py::SkillEntry`（problem/when_to_apply/strategy/code_sketch/origin_tasks/evidence/limitations/category 自由文本）；格式标杆 `open_details/skill_grasp.md` |
| App A 条目结构 + E.5 SKILL.md 模板 | `schema.py` 的 markdown 渲染器：YAML frontmatter + Purpose/Ownership + 正文四段 + Registry 表 + Anti-Patterns + Debugging 表 |
| E.1 actor 上报 findings schema / coordinator 审计可复用性+API 合规 / 串行入库 | `actor.py` 写 findings.md（E.3 模板逐字）；`synthesize.py` 审计（LLM + 代码草图 API 合规 AST 静态检查）；`library.py::admit` 文件锁串行 |
| E.3 Stage 1（fast path→full debug loop，每 seed ≤3 次 replay，BLOCKED 兜底）/ Stage 2（held-out 一次性，禁调试） | `actor.py::repair_loop` / `run_validation` |
| Algorithm 1 逐行（H 累积、Top3 条件化、θ 早停、S_val 终验、ExtractValidatedPatterns） | `evolve.py::EvolutionarySearch.run` |
| E.4 task_analysis.md 跨代持久（场景一次填充/假设账本/已淘汰方向）+ candidate_A verbatim 种子 + 每候选独立假设+docstring + 同 debug seeds + 反过拟合条款 | `evolve.py` + `prompts.py::EVO_*` |
| 并行（AGENTS.md §2 强化：候选并行评估，论文为任务级并行） | `evaluate.py`：子进程池跑 `python -m aspire.engine.engine_capx`，解析退出码+stdout 的 trace 路径；vision_server 由 harness 预检单例拉起避免端口竞争 |
| cap-x 并行集成推理（并发候选+LLM 综合，非投票硬裁决） | `llm_client.py::query_model_ensemble` / `query_single_model_ensemble`（近乎逐行 vendor，重试加上限） |
| cap-x web 最小可用版（start/stop/WS 事件/聊天流/代码高亮/可视化反馈） | `aspire/web/` |

## 2. 与冻结层的边界

- 新层 `aspire/agentic|skills|web` 不碰任何 🔒 文件；执行通道 = 引擎 CLI 子进程
  （退出码 0/1 + stdout 解析 trace 路径，接口已封版）。
- LLM 生成代码的约束 = `scripts/README.md` §1 任务代码规则（禁 import/响亮失败/
  凡动必避障/先算后动/起手 TUCK）+ 论文 E.2 FORBIDDEN（禁 simulator ground truth、
  禁读 XML 资产推断几何）——全部写进 actor system prompt。
- 技能代码草图只允许出现契约 15 + 契约外 5 + np + 自定义 helper
  （`synthesize.py` 静态检查，不合规不得入库）。

## 3. 主从协议（2026-08-11 用户三条指导的固化）

**用户裁决的分工**：coordinator（主）与 agent/actor（从）是主从关系——

1. **agent 的职责**：并行生成不同策略（K 个候选，各自独立假设）→ 一同送真机
   并行评估 → 按成功率论优劣 → 逐代进化（evosearch 为主线工作模式，
   actor 单轨迹修复仅作快速通道/基线建立）。agent **之间不交换经验**，
   唯一出口是 findings.md。
2. **coordinator 的职责**（= 本 Claude Code 顾问窗的化身）：定任务、定入库标准、
   **审核** agent 的入库申请（findings → 审计可复用性 + AST API 合规硬卡 →
   admit/reject 并给理由）；agent 无权直接写库（`SkillLibrary.admit` 只被
   coordinator 调用，文件锁串行）。**不是所有经验都进库**——一次性失败是噪声，
   无 seed 级实证的"模式"一律拒（首个真实审计案例：baseline 全过无修复内容，
   K3 审计拒入库并列出理由，2026-08-11）。
3. **监管可视化**：一切 run 的事件（分派/LLM 调用/逐 seed 结果/审批决定）统一走
   文件总线 `agent_runs/events.jsonl`（CLI 与 Web 发起的 run 同源），
   Web UI（:8200）tail 广播——用户随时可见"现在在干嘛"。

## 4. LLM 端点运维教训（2026-08-11）

- **根因已证实**：Kimi coding 端点的 403/504 并非随机抖动——配额耗尽明文
  `403 permission_error: usage limit for this billing cycle`（14:20 实证）；
  当天上午的 504 episode 是配额逼近上限的网关限流前兆。**仿真评估（本地引擎）
  不受配额影响**——被卡的只有 LLM 调用，故 run 设计必须让 LLM 故障可优雅收口
  （保住已评估数据进 Stage 2/findings，本次两轮 evosearch 均验证）。
- **重试策略**：403/5xx/传输异常统一 60s 短退避 × 6 次封顶（cap-x 的 240s×无限
  策略 = 30 分钟零实验静默，已弃用）；配额类 403 重试无效但能快速确认状态。
- 传输层异常（ReadTimeout/ConnectionError）与 5xx 一样要进重试。
- actor/evolve 的 LLM 调用失败不得杀死整个 run——收口写 findings 保住已产出的
  候选与评估数据。
- trace digest 必须截断（前 8 + 末 20 条记录）：73-record 的完成型 run 会让
  prompt 暴涨到 5 万+ tokens。
- K3 是思考型模型：调用 96–183s 是常态（episode 期间可达 1700s），超时给 600s；
  支持图像输入（实测）。
- findings 的 history 必须带失败信号摘要+模型诊断叙述（否则 LLM 写出
  "失败模式未提供"的空话 findings，首轮真实 run 实测踩坑）。
