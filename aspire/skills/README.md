# aspire/skills/ —— Skill Library 组件（论文组件 2）

> 2026-08-10 新建（不锁）。格式标杆：`open_details/skill_grasp.md`；
> 结构依据：论文 §2.2 + Appendix A + E.5 模板。

## 核心立场（论文 §2.2）

Skill **不是可执行函数**，而是**结构化知识文档**（SKILL.md），以 in-context
guidance 注入未来 actor 的 prompt。底层 primitive API 人工预定义、固定不变
（本项目已封版）。论文明确不做向量检索——文件系统即索引，全量注入
（库 >20 条后启用 `retrieve()` 关键词/category 检索，roadmap 既定路线）。

## 数据布局

```
skill_library/                  # 仓库根（git 跟踪的共享知识）
├── index.json                  # 索引（admit 时原子更新）
├── grasping/piper_reach_and_grasp.md
├── localization/sam3_prompt_cascade.md
├── motion/rrt_move_discipline.md
├── manipulation/wipe_serpentine_coverage.md
└── debugging/trace_driven_repair.md
```

## 文件

| 文件 | 说明 |
|---|---|
| `schema.py` | `SkillEntry`：四要素（Problem/When to Apply/Strategy/Origin tasks，Appendix A 逐字）+ Evidence（seed 级实证纪律）+ Limitations + Code Sketch；YAML frontmatter markdown 往返；`parse_findings`（E.3 findings schema 解析） |
| `library.py` | `SkillLibrary`：文件系统库 + index.json；`admit()` 文件锁**串行入库**（E.1）；`format_for_prompt()` 全量注入（超长显式截断不静默）；`retrieve()` 关键词检索（when_to_apply 权重 ×2——它是论文定义的检索 guard） |
| `synthesize.py` | findings → SKILL 的 coordinator 审计管线：LLM 审计可复用性（E.1）→ `check_api_compliance` **AST 静态检查**（代码草图只允许契约 15 + 契约外 5 + np + helper，不依赖 LLM 自觉）→ 串行入库 |

## 入库纪律

1. actor 按 E.3 findings schema 上报（failure mode / validated repair /
   transferable patterns / task quirks / Stage 2 成功率）；
2. coordinator 审计：可复用性 + API 合规（静态检查硬卡）；
3. Evidence 必须 seed 级实证（skill_grasp.md 标杆），禁止编造数字；
4. 一次性失败是噪声——≥2 seeds/tasks 同症状才够格入库（App A Fig 7）。
