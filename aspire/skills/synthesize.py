"""findings.md → SKILL.md 的 coordinator 审计蒸馏（论文 E.1/E.3）。

流程（E.1 逐字义务的落地）：
1. actor 按 findings schema 上报（E.3 模板，actor.py 生成）；
2. coordinator LLM **审计可复用性**（"audits these findings, verifies compliance
   with the allowed API policy, and promotes only reusable repairs that have passed
   debug validation"）→ 产出候选 SkillEntry 草稿（JSON）；
3. **API 合规静态检查**（本模块 `check_api_compliance`）：代码草图 AST 分析，
   只允许契约 15 + 契约外 5 + np + 自定义 helper + 内建函数——不依赖 LLM 自觉；
4. 通过的条目由 `SkillLibrary.admit()` **串行**入库（文件锁，防并发写冲突）。
"""

from __future__ import annotations

import ast
import json
import re

from .library import SkillLibrary
from .schema import SkillEntry, parse_findings

# 契约 15（docs/primitive_api_capx.md 权威清单）+ 契约外 5（build_namespace 注入）
CONTRACT_API = {
    # 感知 4
    "get_observation", "segment_sam3_text_prompt", "segment_sam3_point_prompt",
    "point_prompt_molmo",
    # 几何与抓取 3
    "plan_grasp", "get_oriented_bounding_box_from_3d_points", "select_top_down_grasp",
    # 运动 2
    "solve_ik", "move_to_joints",
    # 控制 2
    "open_gripper", "close_gripper",
    # 工具 4
    "mask_to_world_points", "pixel_to_world_point", "rotation_matrix_to_quaternion",
    "interpolate_segment",
}
EXTRA_API = {"grasp_cgn", "move_to_joints_planned", "execute_legs_rrt",
             "set_gripper_ramp", "draw_grasp_glyphs"}
ALLOWED_API = CONTRACT_API | EXTRA_API

# 任务代码沙盒里天然可用的名字（np/print 由命名空间注入，其余为安全内建）
ALLOWED_BUILTINS = {
    "np", "print", "range", "len", "min", "max", "abs", "float", "int", "str", "bool",
    "list", "dict", "set", "tuple", "sorted", "enumerate", "zip", "sum", "any", "all",
    "isinstance", "round", "repr", "format", "raise", "Exception", "RuntimeError",
    "ValueError", "assert", "True", "False", "None", "map", "filter", "reversed",
}


def check_api_compliance(code: str) -> list[str]:
    """代码草图 API 合规静态检查。返回违规信息列表（空 = 合规）。

    规则：所有裸函数调用名 ∈ ALLOWED_API ∪ ALLOWED_BUILTINS ∪ 草图内 def 的
    helper；禁 import；禁属性调用以外的未知裸名（防臆造 API——scripts/README §1）。
    """
    violations: list[str] = []
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return [f"代码草图语法错误: {e}"]
    defined = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for node in ast.walk(tree):
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            violations.append(f"禁止 import（第 {node.lineno} 行）——np 已注入命名空间")
        elif isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                name = func.id
                if name not in ALLOWED_API | ALLOWED_BUILTINS | defined:
                    violations.append(f"裸调用未授权函数 `{name}`（第 {node.lineno} 行）——"
                                      f"只允许契约 15 + 契约外 5 + np + 自定义 helper")
    return sorted(set(violations))


def audit_findings(cfg, findings_text: str, task_name: str,
                   existing_skills_text: str = "") -> dict:
    """coordinator LLM 审计（E.1）。返回 {"decision", "rationale", "entries": [SkillEntry]}。

    cfg: LLMConfig（agentic.config）；此处延迟 import 避免 skills→agentic 硬依赖成环。
    """
    from ..agentic.llm_client import query_model, text_message
    from ..agentic.prompts import COORDINATOR_AUDIT_SYSTEM, coordinator_audit_prompt

    findings = parse_findings(findings_text)
    messages = [
        text_message("system", COORDINATOR_AUDIT_SYSTEM),
        text_message("user", coordinator_audit_prompt(
            task_name=task_name, findings_text=findings_text,
            existing_skills_text=existing_skills_text)),
    ]
    resp = query_model(cfg, messages)
    raw = resp["content"]
    m = re.search(r"```json\s*\n(.*?)```", raw, re.DOTALL)
    payload = m.group(1) if m else raw
    try:
        data = json.loads(payload)
    except json.JSONDecodeError as e:
        return {"decision": "reject", "rationale": f"coordinator 响应非 JSON: {e}",
                "entries": [], "raw": raw}
    entries: list[SkillEntry] = []
    for e in data.get("entries", []):
        entries.append(SkillEntry(
            name=e.get("name", "").strip(),
            category=e.get("category", "uncategorized").strip(),
            description=e.get("description", "").strip(),
            problem=e.get("problem", "").strip(),
            when_to_apply=e.get("when_to_apply", "").strip(),
            strategy=e.get("strategy", "").strip(),
            code_sketch=e.get("code_sketch", "").strip(),
            evidence=e.get("evidence", "").strip() or findings.get("Stage 2 Success Rate", ""),
            limitations=e.get("limitations", "").strip(),
            origin_tasks=sorted(set(e.get("origin_tasks", []) or [task_name])),
            validation_debug=e.get("validation_debug", "").strip(),
            validation_heldout=e.get("validation_heldout", "").strip()
            or findings.get("Stage 2 Success Rate", "").strip(),
        ))
    return {"decision": data.get("decision", "reject"),
            "rationale": data.get("rationale", ""),
            "entries": entries, "raw": raw}


def synthesize_from_findings(cfg, findings_path: str, library: SkillLibrary,
                             task_name: str) -> dict:
    """完整入库管线：读 findings → LLM 审计 → 静态合规检查 → 串行入库。

    返回审计报告 dict（含每条目的处置与原因），供 coordinator/web 展示。
    """
    with open(findings_path, encoding="utf-8") as f:
        findings_text = f.read()
    existing = library.format_for_prompt(max_chars=12000)
    audit = audit_findings(cfg, findings_text, task_name, existing)
    report = {"decision": audit["decision"], "rationale": audit["rationale"],
              "findings_path": findings_path, "admitted": [], "rejected": []}
    if audit["decision"] not in ("admit", "revise"):
        report["rejected"].append({"name": "(whole findings)", "reason": audit["rationale"]})
        return report
    for entry in audit["entries"]:
        if not entry.name:
            report["rejected"].append({"name": "(unnamed)", "reason": "缺 name"})
            continue
        violations = check_api_compliance(entry.code_sketch) if entry.code_sketch else []
        if violations:
            report["rejected"].append({"name": entry.name,
                                       "reason": "API 不合规: " + "; ".join(violations)})
            continue
        if not entry.problem or not entry.when_to_apply or not entry.strategy:
            report["rejected"].append({"name": entry.name,
                                       "reason": "四要素不全（problem/when_to_apply/strategy 必填）"})
            continue
        path = library.admit(entry)
        report["admitted"].append({"name": entry.name, "path": path})
    return report
