"""Actor agent —— 单任务的 生成→执行→诊断→修复 闭环（论文 E.3 fix loop）。

E.3 义务的落地映射：
- Fast Path（Step 1）：baseline 程序先在 debug seeds 上评估；达标直接进 Stage 2。
- Full debug loop（Step 2-3）：读 skill library → trace 诊断 → 写修复 → replay；
  **每 seed 最多 3 次 replay**（本实现 = 最多 3 轮整组评估）→ 超限写 BLOCKED 结论。
- Step 4：合成无 seed 分支的通用程序（由 prompt 契约保证）+ findings.md（E.3 模板）。
- Stage 2：held-out seeds **一次性**验证，不用其结果改代码。
- 反过拟合：修复 prompt 携带"已淘汰假设"清单，防在同一策略上打转（§2.3 local
  repair loop 防线；完整探索由 evolve.py 负责）。
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field

from ..skills.library import SkillLibrary
from . import events
from .config import LLMConfig
from .evaluate import (ProgramResult, ensure_services, evaluate_program,
                       set_event_cb)
from .llm_client import extract_code, query_model
from .prompts import (findings_messages, initial_program_messages,
                      load_api_reference, repair_messages)
from .task_spec import TaskSpec
from .trace_digest import build_digest


@dataclass
class ActorResult:
    task: str
    status: str                     # solved | max_rounds | blocked | error
    stage1_rate: float = 0.0
    stage2_rate: str = ""
    best_code_path: str = ""
    findings_path: str = ""
    run_dir: str = ""
    history: list[dict] = field(default_factory=list)


class Actor:
    """一个 actor 实例负责一个任务的一次完整修复流程（Stage 1 + Stage 2）。"""

    def __init__(self, llm_cfg: LLMConfig, library: SkillLibrary,
                 run_root: str = "agent_runs", max_rounds: int = 3,
                 theta: float = 1.0, max_workers: int = 4,
                 event_cb=None):
        self.cfg = llm_cfg
        self.library = library
        self.run_root = run_root
        self.max_rounds = max_rounds      # E.3: 每 seed 最多 3 次 replay
        self.theta = theta                # debug 成功率阈值（Algorithm 1 的 θ）
        self.max_workers = max_workers
        self.event_cb = events.compose(event_cb)   # 文件总线 + 本地回调（监管可视化）
        set_event_cb(self.event_cb)                # 逐 seed 结果也进事件流
        self.api_reference = load_api_reference()

    # ------------------------------------------------------------------
    def _emit(self, etype: str, **payload):
        self.event_cb(etype, payload)

    def _write_code(self, run_dir: str, tag: str, code: str) -> str:
        path = os.path.join(run_dir, f"{tag}.py")
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        return path

    def _evaluate(self, spec: TaskSpec, name: str, code_path: str,
                  seeds: list[int], run_dir: str) -> ProgramResult:
        self._emit("eval_start", name=name, seeds=seeds)
        result = evaluate_program(
            name=name, code_path=code_path, env_task=spec.env_task, seeds=seeds,
            trace_root=os.path.join(run_dir, "traces", name),
            extra_args=spec.extra_engine_args, timeout=spec.time_budget_per_trial,
            max_workers=self.max_workers)
        self._emit("eval_done", name=name, rate=result.rate,
                   detail=result.leaderboard_line())
        return result

    def _failure_digest_parts(self, result: ProgramResult,
                              max_images: int | None = None) -> tuple[str, list[dict]]:
        """全部失败 seed 的文本摘要 + 首个失败 seed 的前后帧图像（可用
        ASPIRE_DIGEST_MAX_IMAGES=0 关图——文本-only 模型用）。"""
        if max_images is None:
            max_images = self.cfg.digest_max_images
        texts, parts = [], []
        first = True
        for sr in result.seed_results:
            if sr.success or not sr.trace_path:
                if not sr.success:
                    texts.append(f"## seed {sr.seed}: FAIL (无 trace；exit={sr.exit_code}"
                                 f"{' TIMEOUT' if sr.timed_out else ''}) {sr.error_line}")
                continue
            digest = build_digest(sr.trace_path,
                                  max_images=max_images if first else 0)
            texts.append(f"## seed {sr.seed} digest\n{digest.summary_text}")
            if first:
                parts.extend(digest.content_parts[1:])  # 跳过 summary 文本（已并入 texts）
                first = False
        return "\n\n".join(texts), parts

    # ------------------------------------------------------------------
    def run(self, spec: TaskSpec) -> ActorResult:
        run_dir = os.path.join(self.run_root,
                               f"actor_{spec.name}_{time.strftime('%m%d_%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)
        self._emit("actor_start", task=spec.name, run_dir=run_dir)
        services = ensure_services()
        self._emit("services", **services)

        skills_text = self.library.format_for_prompt()
        history: list[dict] = []
        best_code, best_rate, best_tag = None, -1.0, ""
        eliminated: list[str] = []

        # ---- E.3 Step 1: Fast Path —— baseline 先上 debug seeds ----
        if spec.baseline_code and os.path.isfile(spec.baseline_code):
            self._emit("stage", text="Stage 1 fast path: 评估 baseline")
            base = self._evaluate(spec, "baseline", spec.baseline_code,
                                  spec.debug_seeds, run_dir)
            history.append({"round": 0, "tag": "baseline", "code_path": spec.baseline_code,
                            "rate": base.rate})
            best_code, best_rate, best_tag = spec.baseline_code, base.rate, "baseline"
            if base.rate >= self.theta:
                self._emit("stage", text=f"baseline 已达标 (rate={base.rate})，直达 Stage 2")
                return self._finish(spec, run_dir, best_code, best_rate, best_tag,
                                    history, status="solved")
            eliminated.append(f"baseline verbatim: rate={base.rate:.3f}（未达标）")
            base_digest, base_parts = self._failure_digest_parts(base)
        else:
            base = None
            base_digest, base_parts = "", []

        # ---- E.3 Step 2-3: full debug loop（≤ max_rounds 轮）----
        current_code = None
        for round_idx in range(1, self.max_rounds + 1):
            self._emit("stage", text=f"Stage 1 round {round_idx}/{self.max_rounds}")
            if current_code is None and round_idx == 1 and base is None:
                # 无 baseline：从零生成
                messages = initial_program_messages(
                    spec.instruction, spec.success_criteria, skills_text,
                    self.api_reference,
                    example_code=open(spec.baseline_code, encoding="utf-8").read()
                    if spec.baseline_code and os.path.isfile(spec.baseline_code) else None)
                purpose = "initial_generation"
            else:
                # 修复：以上一轮最佳/最新程序为底
                prev_result = history[-1]
                prev_path = prev_result["code_path"]
                with open(prev_path, encoding="utf-8") as f:
                    prev_code = f.read()
                prev_prog = prev_result.get("program_result")
                if prev_prog is not None:
                    digest_text, digest_parts = self._failure_digest_parts(prev_prog)
                else:
                    digest_text, digest_parts = base_digest, base_parts
                messages = repair_messages(
                    spec.instruction, spec.success_criteria, prev_code,
                    [{"type": "text", "text": digest_text}] + digest_parts,
                    attempt=round_idx, max_attempts=self.max_rounds,
                    skills_text=skills_text, api_reference=self.api_reference,
                    prior_hypotheses="\n".join(eliminated))
                purpose = f"repair_round_{round_idx}"
            self._emit("llm_call", purpose=purpose)
            try:
                resp = query_model(self.cfg, messages)
            except Exception as e:
                # LLM 故障（超时/限流重试耗尽）不杀死整个 run：收口落 findings
                self._emit("error", text=f"LLM 调用失败（{purpose}）: {e}")
                eliminated.append(f"round {round_idx}: LLM 调用失败 {type(e).__name__}")
                break
            self._emit("llm_response", content=resp["content"][:4000],
                       reasoning=(resp.get("reasoning") or "")[:2000])

            code = extract_code(resp["content"])
            if code is None:
                eliminated.append(f"round {round_idx}: LLM 响应无代码 fence")
                self._emit("error", text="LLM 响应无代码 fence")
                continue
            tag = f"v{round_idx}"
            code_path = self._write_code(run_dir, tag, code)
            prog = self._evaluate(spec, tag, code_path, spec.debug_seeds, run_dir)
            history.append({"round": round_idx, "tag": tag, "code_path": code_path,
                            "rate": prog.rate, "program_result": prog,
                            "response": resp["content"]})
            if prog.rate > best_rate:
                best_code, best_rate, best_tag = code_path, prog.rate, tag
            else:
                eliminated.append(f"round {round_idx} ({tag}): rate={prog.rate:.3f} "
                                  f"未超过最佳 {best_rate:.3f}")
            if best_rate >= self.theta:
                return self._finish(spec, run_dir, best_code, best_rate, best_tag,
                                    history, status="solved")

        # 超限：E.3 的 BLOCKED 语义（仍有最佳版本则带它进 Stage 2；
        # 但 debug 0 成功 = blocked——"if all debugging seeds are blocked,
        # skip Stage 2"，无有效程序可验证）
        status = "max_rounds" if best_code else "blocked"
        if best_rate <= 0:
            status = "blocked"
        return self._finish(spec, run_dir, best_code, best_rate, best_tag,
                            history, status=status)

    # ------------------------------------------------------------------
    def _finish(self, spec: TaskSpec, run_dir: str, best_code: str | None,
                best_rate: float, best_tag: str, history: list[dict],
                status: str) -> ActorResult:
        """Stage 2 一次性验证 + findings.md（E.3 Step 4 / Stage 2）。

        E.3："If all debugging seeds are blocked, skip Stage 2" —— 最佳程序在
        debug seeds 上 0 成功（blocked）时不跑 held-out。
        """
        stage2_rate = ""
        if best_code and best_rate > 0 and status != "blocked":
            self._emit("stage", text="Stage 2: held-out 一次性验证（禁调试）")
            held = self._evaluate(spec, f"stage2_{best_tag}", best_code,
                                  spec.heldout_seeds, run_dir)
            stage2_rate = f"{held.n_success}/{len(held.seed_results)}"
            self._emit("stage2_done", rate=stage2_rate)
        # findings：LLM 蒸馏整轮历史（无 LLM 可用时落原始历史兜底）
        # history 必须带每轮的失败信号摘要+模型诊断叙述（2026-08-11 教训：
        # 只给 tag/rate/path 的 findings 会写成"失败模式未提供"的空话）
        hist_parts = []
        for h in history:
            part = (f"### {h['tag']} (round {h['round']}) debug rate={h['rate']:.3f}\n"
                    f"code: {h['code_path']}")
            prog = h.get("program_result")
            if prog is not None:
                sigs = []
                for sr in prog.seed_results:
                    if not sr.success and sr.trace_path:
                        try:
                            dg = build_digest(sr.trace_path, max_images=0)
                            fs = dg.summary_text.split("## Failure Signals")[-1]
                            fs = fs.split("## ")[0].strip()
                            sigs.append(f"seed {sr.seed} 失败信号:\n{fs[:600]}")
                        except Exception:
                            pass
                if sigs:
                    part += "\n" + "\n".join(sigs[:2])
            resp = h.get("response") or ""
            if resp:
                part += f"\n该轮修复的诊断与改动说明（模型自述）:\n{resp[:900]}"
            hist_parts.append(part)
        history_text = "\n\n".join(hist_parts)
        findings_path = os.path.join(run_dir, "findings.md")
        try:
            self._emit("llm_call", purpose="findings")
            resp = query_model(self.cfg, findings_messages(spec.name, history_text,
                                                           stage2_rate or "not run"))
            findings = resp["content"]
        except Exception as e:
            findings = (f"## Task: {spec.name}\n### Root Cause(s)\n- (findings 生成失败: {e})\n"
                        f"### Stage 2 Success Rate\n{stage2_rate or 'not run'}\n")
        with open(findings_path, "w", encoding="utf-8") as f:
            f.write(findings)
        self._emit("actor_done", task=spec.name, status=status,
                   stage1_rate=best_rate, stage2_rate=stage2_rate,
                   findings_path=findings_path)
        # 历史落盘（去掉不可 JSON 的 program_result 对象）
        import json
        slim = [{k: v for k, v in h.items() if k not in ("program_result", "response")}
                for h in history]
        with open(os.path.join(run_dir, "history.json"), "w", encoding="utf-8") as f:
            json.dump(slim, f, ensure_ascii=False, indent=2)
        return ActorResult(task=spec.name, status=status, stage1_rate=best_rate,
                           stage2_rate=stage2_rate, best_code_path=best_code or "",
                           findings_path=findings_path, run_dir=run_dir, history=history)
