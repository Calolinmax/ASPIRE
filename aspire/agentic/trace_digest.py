"""trace.json → LLM 可消费的失败诊断摘要（论文 §2.1 的实现）。

论文原文义务（Section 2.1）：
"the engine keeps frames immediately before and after each primitive call together
with the corresponding overlays and return values, so the agent can focus on evidence
around calls implicated by the failure. The agent does not receive full video frames."

落地：
- `build_digest()` 读 trace.json，产出 (文本摘要, 多模态 content parts)。
- 文本摘要 = 全部 primitive 调用的一行式记录（类别/输入/输出/耗时/碰撞数/
  标注图有无）+ 失败定位段（首个失败信号 + 其前后窗口）。
- 图像 = **只附失败邻近记录**的调用前帧（top）+ 算法标注图（overlays）+
  运动类结束帧（after），上限 `max_images` 张防 token 爆炸。
  成功 run 则附最后 N 条记录的帧（供总结 findings 用）。

失败信号启发式（E.3 Interpretation examples 的本 API 适配）：
- segment_sam3_* 返回空 list → "zero masks → poor perception prompt or occlusion"
- solve_ik 出现在 error traceback → "missing IK solution → 目标不可达/腕限位"
- close_gripper 后 gripper 开度仍大/小 → 夹住/空抓信号（读 observation）
- collision_events 非空 → 碰撞反馈
- error 非 None → 首个提到契约函数名的 traceback 帧 = 失败调用点
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .llm_client import collapse_text_image_inputs, image_file_to_data_url


@dataclass
class TraceDigest:
    trace_path: str
    task: str
    seed: int
    success: bool
    error: str | None
    n_records: int
    summary_text: str                       # 全量文本摘要（含失败定位段）
    implicated_seqs: list[int] = field(default_factory=list)  # 附图记录的 seq
    content_parts: list[dict] = field(default_factory=list)    # 多模态 parts（可直接进 message）


def _compact(value, max_chars: int = 220) -> str:
    """summarize() 后的值 → 单行紧凑文本。"""
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def _failure_signals(trace: dict) -> list[dict]:
    """扫描 records 找失败信号，返回 [{seq, kind, detail}]（按 seq 升序）。"""
    signals: list[dict] = []
    for rec in trace.get("records", []):
        name = rec.get("name", "")
        out = rec.get("outputs")
        seq = rec.get("seq", -1)
        if name.startswith("segment_sam3") or name == "point_prompt_molmo":
            # 空检出：outputs 为 [] 或 list_summary.len==0 或 {text: [null, null]}
            empty = out == [] or (isinstance(out, dict)
                                  and out.get("list_summary", {}).get("len") == 0)
            if isinstance(out, dict) and any(v == [None, None] or v is None
                                             for v in out.values() if isinstance(v, (list, type(None)))):
                empty = True
            if empty:
                signals.append({"seq": seq, "kind": "zero_masks",
                                "detail": f"{name} 未检出（prompt={_compact(rec.get('inputs', {}))}）"
                                          f" → poor perception prompt or occlusion"})
        if rec.get("collision_events"):
            kinds = sorted({e.get("pair", e.get("type", "?")) if isinstance(e, dict) else str(e)
                            for e in rec["collision_events"]})
            signals.append({"seq": seq, "kind": "collision",
                            "detail": f"{name} 期间 {len(rec['collision_events'])} 次碰撞: "
                                      f"{', '.join(kinds[:4])}"})
        if name == "close_gripper":
            obs = rec.get("observation") or {}
            g = obs.get("gripper") or obs.get("robot_joint_pos") or {}
            width = None
            if isinstance(g, dict):
                width = g.get("ndarray", [None])[-1] if g.get("ndarray") else None
            elif isinstance(g, (int, float)):
                width = g
            if isinstance(width, (int, float)):
                if width > 0.003:
                    signals.append({"seq": seq, "kind": "grasped",
                                    "detail": f"close_gripper 后开度 {width:.4f} → 夹住物体"})
                else:
                    signals.append({"seq": seq, "kind": "air_grasp",
                                    "detail": f"close_gripper 后开度 {width:.4f} → 空抓"})
    return signals


def _error_implicated_seq(trace: dict) -> int | None:
    """error traceback 里首个契约函数名 → 对应 record seq（失败调用点）。"""
    error = trace.get("error") or ""
    if not error:
        return None
    for line in error.splitlines():
        if "<task_code>" in line:
            continue  # 任务代码行不含 primitive 名，跳过
        for rec in trace.get("records", []):
            if rec.get("name") and rec["name"] in line:
                return rec["seq"]
    # traceback 未点名 primitive：失败发生在任务代码层面，取最后一条记录
    records = trace.get("records", [])
    return records[-1]["seq"] if records else None


def build_digest(trace_path: str, max_images: int = 8,
                 window: int = 3, max_stdout: int = 3000) -> TraceDigest:
    """读 trace.json，产出文本摘要 + 多模态 content parts。

    window: 失败信号前后各取多少条记录附图；成功 run 取末尾 window 条。
    """
    trace_dir = os.path.dirname(os.path.abspath(trace_path))
    with open(trace_path, encoding="utf-8") as f:
        trace = json.load(f)
    records = trace.get("records", [])
    signals = _failure_signals(trace)
    err_seq = _error_implicated_seq(trace)

    # ---- 文本摘要 ----
    lines = [
        f"# Execution Trace Digest",
        f"task={trace.get('task')} seed={trace.get('seed')} "
        f"success={trace.get('success')} sim_steps={trace.get('total_sim_steps')}",
        f"primitive_calls={len(records)} code_ref={trace.get('code_ref')}",
        "",
        "## Primitive Records（每次调用：类别/输入/输出/耗时/碰撞）",
    ]
    # 长 trace 防 prompt 爆炸（2026-08-11 实测：73-record 的完成型 run，两 seed
    # 摘要就能把 prompt 撑到 5 万+ tokens）：记录行保留前 8 条（起手+感知建立）
    # + 末 20 条（失败第一现场通常在尾部），中间显式标注省略。
    rec_line_groups: list[list[str]] = []
    for rec in records:
        n_col = len(rec.get("collision_events") or [])
        rec_line_groups.append([
            f"[{rec.get('seq'):03d}] {rec.get('name')} ({rec.get('category')}) "
            f"t={rec.get('wall_time')}s steps {rec.get('sim_step_before')}→{rec.get('sim_step_after')}"
            f"{' COLLISIONS=' + str(n_col) if n_col else ''}"
            f"{' [annotated]' if rec.get('annotation') else ''}",
            f"      in : {_compact(rec.get('inputs'))}",
            f"      out: {_compact(rec.get('outputs'))}",
        ])
    HEAD, TAIL = 8, 20
    if len(rec_line_groups) > HEAD + TAIL + 4:
        shown = rec_line_groups[:HEAD] + [
            [f"... 中间省略 {len(rec_line_groups) - HEAD - TAIL} 条记录"
             f"（完整见 trace.json）..."]
        ] + rec_line_groups[-TAIL:]
    else:
        shown = rec_line_groups
    for group in shown:
        lines.extend(group)
    lines.append("")
    lines.append("## Failure Signals（自动扫描）")
    if signals:
        for s in signals:
            lines.append(f"- [seq {s['seq']:03d}] {s['kind']}: {s['detail']}")
    else:
        lines.append("- （无明显信号）")
    if trace.get("error"):
        lines.append("")
        lines.append("## Error（traceback 尾部）")
        lines.append("```\n" + "\n".join((trace["error"] or "").splitlines()[-12:]) + "\n```")
    stdout = trace.get("stdout") or ""
    if stdout:
        lines.append("")
        lines.append("## Stdout（尾部）")
        lines.append("```\n" + stdout[-max_stdout:] + "\n```")
    summary_text = "\n".join(lines)

    # ---- 附图记录选择：失败信号 ±window，或成功时的末尾 window ----
    if signals or err_seq is not None:
        anchors = {s["seq"] for s in signals} | ({err_seq} if err_seq is not None else set())
        implicated = sorted({seq for a in anchors
                             for seq in range(max(0, a - window), a + window + 1)
                             if 0 <= seq < len(records)})
    else:
        implicated = list(range(max(0, len(records) - window), len(records)))

    # ---- 多模态 parts：摘要文本 + 每个 implicated 记录的前后帧/标注图 ----
    parts: list[dict] = [{"type": "text", "text": summary_text}]
    n_images = 0
    for seq in implicated:
        if n_images >= max_images:
            break
        rec = records[seq]
        imgs = []  # (caption, relpath)
        ve = rec.get("visual_evidence") or {}
        if ve.get("top"):
            imgs.append((f"[seq {seq:03d}] {rec['name']} 调用前帧(top)", ve["top"]))
        if rec.get("annotation"):
            imgs.append((f"[seq {seq:03d}] {rec['name']} 算法标注图(overlay)", rec["annotation"]))
        vea = rec.get("visual_evidence_after") or {}
        if vea.get("top") and vea.get("top") != ve.get("top"):
            imgs.append((f"[seq {seq:03d}] {rec['name']} 结束后帧(top)", vea["top"]))
        for caption, rel in imgs:
            if n_images >= max_images:
                break
            path = os.path.join(trace_dir, rel)
            if not os.path.isfile(path):
                continue
            parts.append({"type": "text", "text": caption})
            parts.append({"type": "image_url",
                          "image_url": {"url": image_file_to_data_url(path)}})
            n_images += 1

    return TraceDigest(
        trace_path=trace_path,
        task=trace.get("task", ""),
        seed=trace.get("seed", -1),
        success=bool(trace.get("success")),
        error=trace.get("error"),
        n_records=len(records),
        summary_text=summary_text,
        implicated_seqs=implicated,
        content_parts=collapse_text_image_inputs(parts),
    )
