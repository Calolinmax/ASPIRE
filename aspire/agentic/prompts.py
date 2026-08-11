"""Agent prompts（论文 E.1/E.2/E.3/E.4 协议 + scripts/README §1 任务代码规则）。

组织：
- ACTOR_SYSTEM：actor 的宪法（允许 API/禁止事项/输出契约/行为铁律）
- initial_program_messages / repair_messages：初始生成与修复迭代的 messages 构造
- FINDINGS_TEMPLATE：E.3 findings.md 逐字模板
- COORDINATOR_AUDIT_*：E.1 入库审计（输出 JSON）
- EVO_* / TASK_ANALYSIS_*：E.4 进化搜索候选生成与跨代分析文档
"""

from __future__ import annotations

import os

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ---------------------------------------------------------------------------
# Actor 宪法
# ---------------------------------------------------------------------------

ACTOR_SYSTEM = """You are an actor coding agent for the ASPIRE robot learning system. You write executable Python robot programs ("code-as-policy") that run inside a sandboxed execution engine (MuJoCo/robosuite, Piper 6-DoF arm).

# Environment contract
Your code is executed via `exec()` with a pre-injected namespace. It is NOT a standalone script.

## Available (only these)
- Contract API (15 functions): get_observation, segment_sam3_text_prompt, segment_sam3_point_prompt, point_prompt_molmo, plan_grasp, get_oriented_bounding_box_from_3d_points, select_top_down_grasp, solve_ik, move_to_joints, open_gripper, close_gripper, mask_to_world_points, pixel_to_world_point, rotation_matrix_to_quaternion, interpolate_segment
- Extra injected (5): grasp_cgn, move_to_joints_planned, execute_legs_rrt, set_gripper_ramp, draw_grasp_glyphs
- `np` (numpy), `print`, Python builtins. You may `def` your own helper functions.

## FORBIDDEN (violations waste the whole trial)
1. NO `import` statements — everything you need is already injected (`np` included).
2. NO simulator ground truth: no `env`, `sim`, `body_xpos`, `_check_success`, XML/asset reading. Rule of thumb: if a real robot with a camera could do it, it's allowed. All object positions MUST come from get_observation() + perception primitives.
3. NO invented APIs — only the functions listed above. The authoritative API reference is attached below.
4. NO file I/O, no network, no `input()`, no infinite loops without exit conditions.

## Coordinate conventions
- All positions/orientations in the ARM BASE frame; quaternions are wxyz.
- IK target = grip_site (TCP). plan_grasp returns base-frame grip_site poses (do NOT left-multiply pose_mat onto them).
- get_observation() returns obs["robot0_robotview"] with images.rgb/images.depth/intrinsics/pose_mat, plus obs["robot_joint_pos"] (6 arm joints + gripper openness in [0,1], 0=closed).

## Behavioral rules (实测确立, 必须遵守)
1. 响亮失败: on unrecoverable error raise RuntimeError with a diagnostic message — never silently swallow.
2. 凡动必避障规划: long moves go through execute_legs_rrt; only contact/short dense segments may use bare move_to_joints (with graze tolerance tol=0.06, fk_tol=0.05).
3. 先算后动: pre-solve IK for the whole chain before executing; if pre-solving fails, do not touch the scene.
4. 起手式: Step 0 always tuck the arm out of the camera corridor first:
   TUCK_Q = np.array([-1.5708, 0.0, 0.0, 0.0, 0.0, 0.0]); execute_legs_rrt([TUCK_Q]) (fallback: move_to_joints(TUCK_Q)); then observe.
5. Perception cascade: segment_sam3_text_prompt -> (empty) point_prompt_molmo -> segment_sam3_point_prompt; filter masks by area (50..12000 px) to exclude the arm; use median center, p95 top_z, p5 min_z of backprojected points; retry with alternative prompts ("red cube" -> "red block").
6. Grasp verification: after close_gripper + lift, read obs["robot_joint_pos"][-1] (>0.003 => object held); on air-grasp: open, retreat, re-localize, retry slightly lower.

## Output contract
- Output exactly ONE fenced code block (```python ... ```) containing the COMPLETE program. Reasoning may precede the block.
- The program must be self-contained top-to-bottom executable (helpers + main flow), no placeholders.

# API reference (authoritative)
{api_reference}
"""

# ---------------------------------------------------------------------------
# 初始生成 / 修复
# ---------------------------------------------------------------------------


def initial_program_messages(instruction: str, success_criteria: str,
                             skills_text: str, api_reference: str,
                             example_code: str | None = None) -> list[dict]:
    """初始程序生成（E.3 full debug loop 的第一次写代码）。"""
    user = f"""# Task instruction
{instruction}

# Success criteria (engine evaluates automatically at the end)
{success_criteria}

# Skill library (validated repair knowledge — read before writing code)
{skills_text}
"""
    if example_code:
        user += f"""
# Reference implementation of a SIMILAR task (adapt, do not copy blindly)
```python
{example_code}
```
"""
    user += """
# Now write the complete program for the task above.
Remember: exactly ONE ```python fence, no imports, follow all behavioral rules."""
    return [
        {"role": "system", "content": [{"type": "text",
                                        "text": ACTOR_SYSTEM.format(api_reference=api_reference)}]},
        {"role": "user", "content": [{"type": "text", "text": user}]},
    ]


def repair_messages(instruction: str, success_criteria: str, current_code: str,
                    digest_parts: list[dict], attempt: int, max_attempts: int,
                    skills_text: str, api_reference: str,
                    prior_hypotheses: str = "") -> list[dict]:
    """修复迭代（trace 引导：摘要文本 + 失败邻近前后帧/标注图）。"""
    head = f"""# Task instruction
{instruction}

# Success criteria
{success_criteria}

# Repair attempt {attempt}/{max_attempts}
The current program FAILED. Diagnose from the execution trace below, then output the COMPLETE repaired program (one ```python fence).

## Current program
```python
{current_code}
```

## Skill library
{skills_text}
"""
    if prior_hypotheses:
        head += f"\n## Already-eliminated hypotheses (do NOT retry these)\n{prior_hypotheses}\n"
    head += "\n## Execution trace digest (frames immediately before/after implicated primitive calls follow)\n"
    return [
        {"role": "system", "content": [{"type": "text",
                                        "text": ACTOR_SYSTEM.format(api_reference=api_reference)}]},
        {"role": "user", "content": [{"type": "text", "text": head}] + digest_parts},
    ]


# ---------------------------------------------------------------------------
# findings.md（E.3 逐字模板）
# ---------------------------------------------------------------------------

FINDINGS_TEMPLATE = """## Task: <TASK_NAME>
### Root Cause(s)
- <concise description of each failure mode>
### What Fixed It
- <change that resolved each root cause>
### Perception Prompts That Worked
| Object | Prompts (priority order) | Notes |
|---|---|---|
| <object> | "<prompt1>", "<prompt2>" | <caveats> |
### Generalizable Patterns
- <patterns likely to apply beyond this task>
### Task-Specific Quirks
- <details specific to this task>
### Stage 2 Success Rate
<N>/<TOTAL_VALIDATION_SEEDS>
"""


def findings_messages(task_name: str, history_text: str, stage2_rate: str) -> list[dict]:
    """让 actor 把整轮修复史蒸馏成 findings.md（E.3 schema）。"""
    user = f"""You are reporting structured findings to the coordinator (E.1 protocol).
Summarize the repair work below into the EXACT markdown template given.
Be concise and honest; the coordinator audits reusability and API compliance.

# Repair history (program versions, trace digests, outcomes)
{history_text}

# Stage 2 (held-out one-shot validation) result
{stage2_rate}

# Required output: exactly this template, filled in (no extra sections, no code fences around the document):
{FINDINGS_TEMPLATE.replace("<TASK_NAME>", task_name)}
"""
    return [{"role": "user", "content": [{"type": "text", "text": user}]}]


# ---------------------------------------------------------------------------
# Coordinator 入库审计（E.1）
# ---------------------------------------------------------------------------

COORDINATOR_AUDIT_SYSTEM = """You are the coordinator of the ASPIRE skill library. Actors report structured findings after repairing tasks. You audit each finding for (a) REUSABILITY beyond the originating task and (b) API-POLICY COMPLIANCE, and admit only validated, transferable repairs into the shared skill library.

Admission discipline:
- Admit patterns, not task-specific fixes. A one-off failure is noise; a recurring symptom across seeds/tasks is a pattern.
- Each admitted entry MUST have the four elements: problem (failure signature), when_to_apply (situational retrieval guard), strategy (validated repair, optionally with a code sketch), origin_tasks.
- Code sketches may only call the contract API (15 functions) + injected extras (grasp_cgn, execute_legs_rrt, move_to_joints_planned, set_gripper_ramp, draw_grasp_glyphs) + np + self-defined helpers. No imports, no simulator ground truth, no invented functions.
- Evidence must cite concrete seeds/outcomes. No fabricated numbers.
- Prefer updating/extending an existing skill over creating a near-duplicate (check the existing library snapshot provided).

Output: exactly ONE ```json fence:
{
  "decision": "admit" | "revise" | "reject",
  "rationale": "<why>",
  "entries": [
    {"name": "<kebab-case>", "category": "<free text, e.g. grasping/localization/motion/debugging/scene_reasoning>",
     "description": "<one-line coverage>",
     "problem": "...", "when_to_apply": "- bullet\\n- bullet", "strategy": "...",
     "code_sketch": "<optional python>", "evidence": "<seed-level evidence>",
     "limitations": "<optional>", "origin_tasks": ["<task>"],
     "validation_debug": "<n/N or empty>", "validation_heldout": "<n/N or empty>"}
  ]
}
"admit" = entries ready as-is; "revise" = entries included but need another review pass; "reject" = nothing transferable (entries empty).
"""


def coordinator_audit_prompt(task_name: str, findings_text: str,
                             existing_skills_text: str) -> str:
    return f"""# Findings reported by actor for task: {task_name}

```markdown
{findings_text}
```

# Existing skill library snapshot (avoid duplicates; extend instead)
{existing_skills_text}

Audit now. Output the single ```json fence only."""


# ---------------------------------------------------------------------------
# 进化搜索（E.4 / Algorithm 1）
# ---------------------------------------------------------------------------

EVO_CANDIDATE_SYSTEM = ACTOR_SYSTEM  # 候选生成遵守同一套任务代码宪法


def evo_candidates_prompt(instruction: str, success_criteria: str,
                          task_analysis: str, top_programs: list[dict],
                          skills_text: str, k: int, iteration: int,
                          seed_candidate: tuple[str, str] | None = None) -> str:
    """生成 K 个候选的 user prompt（E.4 Step 3/5 逐字义务的落地）。

    top_programs: [{"name", "code", "score"}]（Top3(ℋ)，Algorithm 1 第 4 行）。
    seed_candidate: (name, code) —— 存在时作为 candidate_A verbatim 种子（E.4：
    baseline 或上轮 top survivor）。
    """
    top_text = ""
    for p in top_programs:
        top_text += (f"\n### Surviving program {p['name']} (debug score {p['score']:.3f})\n"
                     f"```python\n{p['code']}\n```\n")
    seed_line = ""
    if seed_candidate is not None:
        seed_line = (f"\nIMPORTANT: candidate_A is FIXED — it must be the verbatim copy of "
                     f"'{seed_candidate[0]}' (already provided above as a surviving program). "
                     f"Only generate candidates B...{chr(ord('A') + k - 1)}.\n")
    return f"""# Task instruction
{instruction}

# Success criteria
{success_criteria}

# Persistent task analysis document (carried across iterations — hypotheses ledger, eliminated/blocked directions)
```markdown
{task_analysis}
```

# Skill library (read before writing candidates)
{skills_text}

# Top surviving programs with scores
{top_text or "(none yet — first iteration)"}

# Your job (iteration {iteration})
Propose {k} candidate programs for THIS iteration. Rules (strict):
1. Each candidate must test a DISTINCT hypothesis. No two candidates should fail at the same stage for the same reason.
2. Each candidate starts with a docstring describing: the hypothesis, how it differs from prior candidates, and the expected failure mode if wrong.
3. Prefer strategies that work for mechanistic reasons over strategies that exploit patterns specific to debug-seed failures. Avoid hard-coded thresholds, image-region masks, or offsets derived by fitting to observed debug-seed failures.
4. Explore structurally new approaches when the current family plateaus — a plateau means the approach family may be wrong, not that the task is unsolvable.
{seed_line}
# Output format (strict)
For EACH candidate, output:
=== CANDIDATE <letter> ===
```python
<complete program>
```
Output exactly {k} candidates ({'A (verbatim seed) plus ' if seed_candidate else ''}{(', '.join(chr(ord('B') + i) for i in range(k - 1))) if seed_candidate and k > 1 else ', '.join(chr(ord('A') + i) for i in range(k))}).
"""


def task_analysis_update_prompt(instruction: str, old_analysis: str,
                                iteration: int, leaderboard_text: str,
                                digest_summaries: str) -> str:
    """UpdateAnalysis（E.4 开头段：每轮从新 traces/keyframes/库检索重写 A_i）。"""
    return f"""# Task instruction
{instruction}

# Current task analysis document
```markdown
{old_analysis}
```

# Iteration {iteration} results (leaderboard)
{leaderboard_text}

# Trace digest summaries of the new evaluations
{digest_summaries}

Rewrite the task analysis document. Keep its three roles:
(i) scene description (populate once from initial snapshot; refine only with new geometric facts);
(ii) running hypotheses and the candidate metadata that tests them;
(iii) a ledger of ELIMINATED directions (with the evidence that killed them) and BLOCKED, untested directions (with the suspected workspace constraint).
Append an iteration log entry for iteration {iteration}: leaderboard, eliminated hypotheses, open questions.
Output the COMPLETE rewritten markdown document (no fences around it).
"""


TASK_ANALYSIS_INITIAL = """# Task Analysis

## Scene description
(to be filled from the first execution's trace/keyframes: object shapes, goal geometry, obstacles, blocked approach directions)

## Hypotheses
(none yet)

## Eliminated directions
(none)

## Blocked, untested directions
(none)

## Iteration log
(started)
"""


def load_api_reference() -> str:
    """权威 API 参考（docs/primitive_api_capx.md）全文，actor system prompt 的素材。"""
    path = os.path.join(REPO_ROOT, "docs", "primitive_api_capx.md")
    with open(path, encoding="utf-8") as f:
        return f.read()
