# ASPIRE 复现交接报告（2026-08-11）

> 本次会话（顾问+执行者窗口）的工作汇总与交接文档。
> 阅读对象：接任者 / 未来的自己。前置阅读：`AGENTS.md`、`docs/roadmap.md`、
> `docs/agentic_design.md`、`docs/project_files.md`。

---

## 0. 一句话总结

**ASPIRE 论文三组件已全部落地并通过真实验证**：执行引擎 trace（此前封版）、
Skill Library（含首个注入对照实证）、Agentic Coding（coordinator-actor + Algorithm 1
进化搜索系统）——K3 大模型**无参考代码从零生成**的程序已攻克 Stack 与 Wipe 两个任务
（debug 全过）；held-out 泛化与进化搜索真实闭环是留给下一步的核心坑。

---

## 1. 本次任务起点

用户复现 ASPIRE 论文（基于 cap-x）。engine 与 api 已完成封版（Phase 0，33/33 测试），
stack/wipe 两任务脚本可跑。本次任务 = 补齐论文另两个组件：**agentic coding** 与
**skill library**，依据论文 Fig 2/3、Appendix A、E.1/E.3/E.4、Algorithm 1，
参考 cap-x 开源实现（capx/skills、capx/llm/client.py、capx/web）。

## 2. 交付物清单

### 2.1 新代码层（全部未 git 提交，留在工作区供审阅）

| 路径 | 内容 |
|---|---|
| `aspire/agentic/` | 组件 3：config（.env 加载，provider=openai/anthropic/**file**/mock）、llm_client（vendor cap-x：单次/并行集成/流式/多模态；file provider=本会话当 LLM 的桥接，图像落盘可读）、trace_digest（论文 §2.1：只附失败邻近前后帧+标注图，头 8+尾 20 截断）、prompts（E.1-E.4 协议+任务代码规则）、task_spec、evaluate（**并行子进程评估**，逐 seed 事件，cancel 支持）、actor（E.3 修复闭环）、coordinator（E.1 主从：分派/审批/串行入库）、evolve（Algorithm 1 逐行+E.4 task_analysis 跨代）、events（**文件事件总线** events.jsonl）、cli |
| `aspire/skills/` | 组件 2：schema（四要素 SKILL.md 往返）、library（文件系统库+index.json+文件锁串行 admit+全量注入+关键词检索）、synthesize（coordinator 审计：LLM 可复用性审查 + **AST API 合规硬卡**） |
| `aspire/web/` | Web UI（FastAPI+WS+零构建 SPA，:8200）：运行控制台（逐 seed 实时面板/入库审批卡）+ 技能库 + Traces 浏览；CLI 与 Web 发起的 run 经文件总线同源可见 |
| `skill_library/` | 5 条种子技能 + 1 条实战扩充（见 §4.2） |
| `scripts/agentic.sh` / `scripts/web.sh` | 入口 |
| `scripts/tests/test_agentic.py`（45 项）/ `test_agentic_e2e.py` | 测试 |
| `docs/agentic_design.md` | 设计定稿（论文条款→实现落点对照 + 主从协议 §3 + 运维教训 §4） |
| `.env`（gitignore）/ `.env.example` | LLM 配置（Kimi K3 coding 端点已配好并验证） |

### 2.2 文档同步

`docs/project_files.md`（§0 导航+新层登记）、`docs/roadmap.md`（组件 2/3 状态）、
`README.md`（结构+入口+文档索引）、`AGENTS.md`（§4 程序化 agentic 已落地）。

## 3. 验证结果

| 验证 | 结果 |
|---|---|
| 单元测试 `test_agentic.py` | **45/45 PASS**（mock LLM+fake executor，不碰仿真） |
| 端到端 `test_agentic_e2e.py` | **PASS**（真实引擎：baseline seed0 → Stage2 seed1 → findings → 审计入库） |
| 冻结层回归 `test_piper_capx_api.py` | **33/33 PASS**（未碰任何 🔒 文件） |
| Web UI | WS 回放+实时推送实测；REST 全端点实测 |
| 真实 K3 run：Stack 从零生成 | **v1→v4 四代修复通关**：debug 3/3（v1 0/3 IK 腕限位 → v2 0/3 搬运 RRT → v3 2/3 → v4 3/3） |
| 真实 K3 run：Wipe 从零生成 | **技能注入后一击 3/3**（见 §4.2 的 A/B 实证） |
| coordinator 审计（真实） | 拒入一次（baseline 全过无修复内容，守门正确）、获批一次（wipe 几何技能） |

## 4. 关键技术故事

### 4.1 Stack：v4 的进化史（程序在 `agent_runs/actor_Stack_0811_112616/v1.py`）

v1 死于 IK 腕限位（陡姿态）→ v2 修复 IK、夹持成功（开度 0.59）死于搬运 RRT →
v3 修复搬运、放置边缘（2/3）→ v4 的招牌动作：**抓后悬停用 SAM3 重测持块偏移
（tcp→cube 水平偏移与底部间隙），按"CUBE 中心对准绿块顶心"解放置位姿**——
debug 3/3。held-out 三次验证 2/5、4/5、3/5（放置边缘态高方差）。

### 4.2 Wipe：技能库价值的首个 A/B 实证（程序在 `agent_runs/actor_PiperWipeSpill_0811_170054/v1.py`）

- **无技能裸跑**：v1-v3 全灭。K3 把 grip_site 当垫面用（z 直打桌面）、姿态构造缺
  共轭翻转——v3 做了 7 tilt × 4 z-lift × 整/缩 bbox 的网格搜索，2888 次原语调用
  **零收敛**。结论：几何补偿是必要条件，不是优化项。
- **把实测知识补进 `skill_library/manipulation/wipe_serpentine_coverage.md`**：
  `pad_to_site()`（`WIPER_FACE_L=0.090`，38° 时水平错位 5.5cm，site.z=L·cos-press）、
  `_overhead_quat()`（含 `diag([-1,1,-1])` 共轭翻转）、38°=152/152 收敛实测表
  （44°=27/152、50°=0）。**K3 下一 run 一击命中 debug 3/3**。
- 与手写版 wipe.py 的 diff 相似度仅 8.3%（仅共享技能提供的几何 helper）——
  不是抄代码，是复用了蒸馏知识。
- 残留：held-out 2/5，失败集中 seed 5/6/7 的 **SAM3 污渍定位失败**（感知鲁棒性，
  非几何/运动）——下一轮修复/进化的天然靶子。

### 4.3 主从协议已固化（用户 2026-08-11 三条指导）

agent（K3）并行出策略→并行真机评估→成功率进化；coordinator（本窗口化身）定标准、
分派、**审核入库**（agent 只能申请）；一切 run 经 `agent_runs/events.jsonl`
文件总线进 Web UI（:8200）实时监管。详见 `docs/agentic_design.md` §3。

## 5. 运维教训（血泪，勿重蹈）

1. **key 风波**：下午连环 403/504 的真根因是**用户换过一次 API key 未告知**，
   harness 持有的旧 key 计费周期配额耗尽（403 usage limit 明文）。处置：`.env`
   已用当前环境 token 重置。**换 key 必须同步 .env；排障先确认 key 身份再怀疑端点**。
2. **重试策略**：cap-x 的 240s×无限重试 = 30 分钟零实验静默（已弃）；现行为
   60s 短退避 ×6 次 + 传输异常重试 + LLM 失败优雅收口（保住已评估数据进
   Stage 2/findings，两次实战验证）。
3. **file provider 备胎**：直连配额死时，可用本会话（Claude Code 通道）当 harness
   的 LLM——prompt/response 文件交换，图像落盘可 Read 真看。同一 key 的逃生门。
4. **trace digest 必须截断**（头 8+尾 20 条）：73-record 的 run 会把 prompt 撑爆。
5. **findings 的 history 必须带失败信号+模型诊断叙述**——只给 tag/rate 会写出
   "失败模式未提供"的空话 findings。
6. **"机械臂不动"= 规划静默期**：可视化模式下 IK/RRT/SAM3 是墙钟时间，
   抓取后悬停 1-3 分钟在算（v4 的测持块+预解+RRT），不是卡死；看终端日志。
7. **并行评估的服务预检**：harness 单例拉起 vision_server，避免 N 子进程抢端口/
   误杀共享服务。
8. （流程教训）等后台任务时**不要发空命令轮询**——每次工具调用都消耗订阅 token
   并刷屏；用定时唤醒+完成通知。

## 6. 离开时的系统状态

- **后台 run 已全部停止**（evosearch 第 3 轮在 iter 1 候选生成阶段被人工暂停，
  产物在 `agent_runs/evosearch_Stack_0811_*/`，P0 复评 4/6）。
- Web UI 服务进程（:8200）可能仍在跑：`pkill -f aspire.web.server` 可停；
  `scripts/web.sh` 重启。
- 常驻依赖：CGN 容器（:8117）、pyroki（:8116）——跑任务前按
  `scripts/README.md` §5 预检。
- 数据资产：`agent_runs/`（455MB，全部 run 的候选代码+trace+findings+events.jsonl）、
  `traces/`（31MB）。均 gitignore。
- 工作区未提交：`git status` 见 §2.1 清单（用户已推过一版到 GitHub，本次新增
  留待审阅提交）。

## 7. 未竟事项（按优先级）

1. **进化搜索真实闭环**：Algorithm 1 尚未跑完过一个真实迭代（三次尝试均被
   key/配额打断）。续跑命令：
   ```bash
   /home/stouching/anaconda3/envs/ASPIRE/bin/python -m aspire.agentic.cli evosearch \
     --task Stack --baseline-path agent_runs/actor_Stack_0811_112616/v1.py \
     --seeds 0,1,2,8,9,10 --heldout 3,4,5,6,7 -K 4 -T 3 --theta 1.0 --max-workers 4
   ```
2. **held-out 泛化攻关**：Stack v4 放置边缘态（2-4/5 高方差）、Wipe 感知鲁棒性
   （seed 5/6/7 SAM3 定位失败）——正是 evosearch/下一轮修复的靶子。
3. **技能对照实验**（Phase 1 出口标准）：≥3 个 skill + 有/无注入的统计意义差异
   （目前 6 条技能、wipe 一例 A/B）。
4. **多任务扩展**：Nut Assembly（`open_details/primitive_api_nut_assembly.py` 有
   官方参考流程）；新场景要动 `aspire/robots/__init__.py`（🔒，须人类解锁）。
5. **可选**：把 SESSION 产物 git 提交推送（用户原话：改动留工作区待审）。

## 8. 操作手册（速查）

```bash
# 单元测试（不碰仿真/LLM）
/home/stouching/anaconda3/envs/ASPIRE/bin/python scripts/tests/test_agentic.py
# 冻结层回归
/home/stouching/anaconda3/envs/ASPIRE/bin/python scripts/tests/test_piper_capx_api.py
# Web UI
scripts/web.sh                      # http://127.0.0.1:8200
# Stack 可视化（K3 从零生成 v4）
/home/stouching/anaconda3/envs/ASPIRE/bin/python -m aspire.engine.engine_capx \
  --code agent_runs/actor_Stack_0811_112616/v1.py \
  --task Stack --seed 0 --official-stack --render --render-slowdown 0.5
# Wipe 可视化（K3 从零生成 v1）
/home/stouching/anaconda3/envs/ASPIRE/bin/python -m aspire.engine.engine_capx \
  --code agent_runs/actor_PiperWipeSpill_0811_170054/v1.py \
  --task PiperWipeSpill --seed 0 --render --render-slowdown 0.5
# LLM 配置：.env（gitignore）。换 key 流程：改 .env 的 ASPIRE_LLM_API_KEY →
#   跑一条 query_model 冒烟（见 .env.example 注释）→ 继续。
# 无 key 备胎：--provider file（prompt 落 agent_runs/llm_bridge/，人工/Claude Code 回写）
```

## 9. 结语

这一天从"两个模块零代码"走到"三组件真实跑通 + 技能库首个 A/B 实证"，
中间踩了 key 风波、504 风暴、grip_site 几何盲点三个大坑——每个坑的教训都已固化进
`docs/agentic_design.md` §4 与技能库。系统的骨架是健康的：冻层未动、测试全绿、
主从协议清晰、监管可视。剩下的核心仗只有一场：**held-out 泛化**——用进化搜索
和更多技能去打。祝顺利。

—— Claude（顾问+执行者窗口），2026-08-11
