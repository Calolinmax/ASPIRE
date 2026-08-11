"""进化搜索 —— 论文 Algorithm 1 + E.4 的逐行实现。

Algorithm 1（Evolutionary search over programs）：

    Require: task τ, program P₀, sets S_dbg, S_val, skill library ℒ, agent M,
             budget (K, T), threshold θ
    1:  (r⋆, Z₀) ← Execute(P₀, S_dbg);  P⋆ ← P₀
    2:  ℋ ← {(P₀, r⋆, Z₀)}
    3:  for i = 1, ..., T do
    4:      {P_ik} ← ProposeRepairs(M, τ, Top3(ℋ), ℒ, ℋ)
    5-7:    并行评估 K 候选 on S_dbg（AGENTS.md §2：并行化，论文此处为顺序）
    8:      ℋ ← ℋ ∪ {...};  k⋆ ← arg max r_ik
    9-11:   若 r_ik⋆ > r⋆ 更新 (P⋆, r⋆)
    12-14:  若 r⋆ ≥ θ 则 break
    16: (r_val, Z_val) ← Execute(P⋆, S_val)          # Stage 2 一次性
    17: 𝒢 ← ExtractValidatedPatterns(ℋ, P⋆, r_val, Z_val)
    18: return (P⋆, r_val, 𝒢)

E.4 义务：task_analysis.md 跨代持久（场景描述一次填充/假设账本/已淘汰与受阻方向）；
candidate_A = 基线或上轮 top survivor 的 verbatim 种子；每候选独立假设 + docstring；
同 debug seeds 跨代（公平比较）；反过拟合条款（prompts.py 承载）；收敛判据 =
最佳候选达 θ / 耗尽 T / blocked；Stage 2 不用来改代码；最佳候选不超基线则回落基线。
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass, field

from ..skills.library import SkillLibrary
from . import events
from .config import LLMConfig
from .evaluate import (ProgramResult, ensure_services, evaluate_population,
                       leaderboard, set_event_cb)
from .llm_client import query_model, text_message
from .prompts import (EVO_CANDIDATE_SYSTEM, TASK_ANALYSIS_INITIAL,
                      evo_candidates_prompt, findings_messages,
                      load_api_reference, task_analysis_update_prompt)
from .task_spec import TaskSpec
from .trace_digest import build_digest

_CANDIDATE_RE = re.compile(r"===\s*CANDIDATE\s+([A-Z])\s*===\s*```(?:python|py)?\s*\n(.*?)```",
                           re.DOTALL)


@dataclass
class EvoResult:
    task: str
    status: str                     # solved | max_iterations | blocked | error
    iterations: int = 0
    best_code_path: str = ""
    best_debug_rate: float = 0.0
    stage2_rate: str = ""
    findings_path: str = ""
    run_dir: str = ""
    history: list[dict] = field(default_factory=list)  # ℋ 的轻量视图


class EvolutionarySearch:
    """单任务的进化搜索 actor（E.4 subagent 角色的程序化实现）。"""

    def __init__(self, llm_cfg: LLMConfig, library: SkillLibrary,
                 run_root: str = "agent_runs", K: int = 4, T: int = 5,
                 theta: float = 1.0, max_workers: int = 4, event_cb=None):
        self.cfg = llm_cfg
        self.library = library
        self.run_root = run_root
        self.K, self.T, self.theta = K, T, theta
        self.max_workers = max_workers
        self.event_cb = events.compose(event_cb)   # 文件总线 + 本地回调
        set_event_cb(self.event_cb)
        self.api_reference = load_api_reference()

    def _emit(self, etype: str, **payload):
        self.event_cb(etype, payload)

    # ------------------------------------------------------------------
    def _parse_candidates(self, text: str) -> dict[str, str]:
        """解析 === CANDIDATE X === 代码块（E.4 输出契约）。"""
        return {m.group(1): m.group(2).strip() + "\n"
                for m in _CANDIDATE_RE.finditer(text or "")}

    def _digest_summaries(self, results: dict[str, ProgramResult],
                          max_per_candidate: int = 1200) -> str:
        """本轮全部候选的 trace 文本摘要（喂 UpdateAnalysis；图像不进——E.4
        的 keyframes 检查由 actor 自行看 trace 目录，此处保 token 预算）。"""
        chunks = []
        for name, res in sorted(results.items()):
            chunks.append(f"### {name}: {res.leaderboard_line()}")
            for sr in res.seed_results:
                if sr.success or not sr.trace_path:
                    continue
                try:
                    d = build_digest(sr.trace_path, max_images=0)
                    chunks.append(f"[{name} seed {sr.seed}]\n"
                                  + d.summary_text[:max_per_candidate])
                except Exception as e:
                    chunks.append(f"[{name} seed {sr.seed}] digest 失败: {e}")
        return "\n\n".join(chunks)

    # ------------------------------------------------------------------
    def run(self, spec: TaskSpec) -> EvoResult:
        run_dir = os.path.join(self.run_root,
                               f"evosearch_{spec.name}_{time.strftime('%m%d_%H%M%S')}")
        os.makedirs(run_dir, exist_ok=True)
        analysis_path = os.path.join(run_dir, "task_analysis.md")
        with open(analysis_path, "w", encoding="utf-8") as f:
            f.write(TASK_ANALYSIS_INITIAL)
        self._emit("evo_start", task=spec.name, run_dir=run_dir, K=self.K, T=self.T)
        services = ensure_services()
        self._emit("services", **services)
        skills_text = self.library.format_for_prompt()

        # ---- Algorithm 1 第 1-2 行：P₀ 评估，ℋ 初始化 ----
        H: list[dict] = []   # {name, code, code_path, rate, iteration, traces}
        best_code, best_rate, best_name = None, -1.0, ""
        if spec.baseline_code and os.path.isfile(spec.baseline_code):
            with open(spec.baseline_code, encoding="utf-8") as f:
                base_code = f.read()
            self._emit("stage", text="评估 P₀（baseline）on S_dbg")
            res = evaluate_population({"candidate_A": spec.baseline_code},
                                      spec.env_task, spec.debug_seeds,
                                      os.path.join(run_dir, "traces", "iter_00"),
                                      spec.extra_engine_args,
                                      spec.time_budget_per_trial, self.max_workers)
            r0 = res["candidate_A"]
            H.append({"name": "candidate_A", "code": base_code,
                      "code_path": spec.baseline_code, "rate": r0.rate,
                      "iteration": 0,
                      "traces": [sr.trace_path for sr in r0.seed_results]})
            best_code, best_rate, best_name = base_code, r0.rate, "candidate_A"
            self._emit("eval_done", name="P₀", rate=r0.rate,
                       detail=r0.leaderboard_line())

        # ---- Algorithm 1 第 3-15 行：T 代进化 ----
        stop_reason = "max_iterations"
        iters_done = 0
        for i in range(1, self.T + 1):
            iters_done = i
            self._emit("stage", text=f"进化迭代 {i}/{self.T}")
            top3 = sorted(H, key=lambda h: -h["rate"])[:3]
            with open(analysis_path, encoding="utf-8") as f:
                task_analysis = f.read()
            seed = (best_name, best_code) if best_code is not None else None
            prompt = evo_candidates_prompt(
                spec.instruction, spec.success_criteria, task_analysis,
                [{"name": h["name"], "code": h["code"], "score": h["rate"]} for h in top3],
                skills_text, self.K, i, seed_candidate=seed)
            self._emit("llm_call", purpose=f"propose_iter_{i}")
            try:
                resp = query_model(self.cfg, [
                    text_message("system", EVO_CANDIDATE_SYSTEM.format(
                        api_reference=self.api_reference)),
                    text_message("user", prompt)])
            except Exception as e:
                # LLM 故障不杀死进化 run：带当前最优收口（Stage 2 + findings）
                self._emit("error", text=f"候选生成失败（iter {i}）: {e}")
                stop_reason = "error"
                break
            self._emit("llm_response", content=resp["content"][:4000])
            candidates = self._parse_candidates(resp["content"])

            # E.4：candidate_A = verbatim 种子（精英保留），不信任 LLM 照抄
            if seed is not None:
                candidates["A"] = seed[1]
            if not candidates:
                self._emit("error", text=f"iter {i}: 未解析到候选")
                stop_reason = "error"
                break

            # 写候选文件 + 并行评估（Algorithm 1 第 5-7 行并行化）
            iter_dir = os.path.join(run_dir, f"iter_{i:02d}")
            os.makedirs(iter_dir, exist_ok=True)
            code_paths = {}
            for letter, code in sorted(candidates.items()):
                name = f"candidate_{letter}"
                path = os.path.join(iter_dir, name, "code.py")
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(code)
                code_paths[name] = path
            results = evaluate_population(code_paths, spec.env_task, spec.debug_seeds,
                                          os.path.join(run_dir, "traces", f"iter_{i:02d}"),
                                          spec.extra_engine_args,
                                          spec.time_budget_per_trial, self.max_workers)
            self._emit("eval_done", name=f"iter_{i}",
                       detail=leaderboard(results))

            # Algorithm 1 第 8-11 行：ℋ 更新 + 最优更新
            for name, res in results.items():
                with open(res.code_path, encoding="utf-8") as f:
                    code = f.read()
                H.append({"name": f"{name}@iter{i}", "code": code,
                          "code_path": res.code_path, "rate": res.rate,
                          "iteration": i,
                          "traces": [sr.trace_path for sr in res.seed_results]})
            iter_best = max(results.values(), key=lambda r: r.rate)
            if iter_best.rate > best_rate:
                with open(iter_best.code_path, encoding="utf-8") as f:
                    best_code = f.read()
                best_rate = iter_best.rate
                best_name = f"{iter_best.name}@iter{i}"

            # E.4：UpdateAnalysis 重写 task_analysis.md
            try:
                update_msg = task_analysis_update_prompt(
                    spec.instruction, task_analysis, i, leaderboard(results),
                    self._digest_summaries(results))
                self._emit("llm_call", purpose=f"update_analysis_{i}")
                aresp = query_model(self.cfg, [text_message("user", update_msg)])
                with open(analysis_path, "w", encoding="utf-8") as f:
                    f.write(aresp["content"])
            except Exception as e:
                self._emit("error", text=f"task_analysis 更新失败（继续）: {e}")

            with open(os.path.join(iter_dir, "iter_summary.json"), "w",
                      encoding="utf-8") as f:
                json.dump({"iteration": i, "leaderboard": leaderboard(results),
                           "best": best_name, "best_rate": best_rate},
                          f, ensure_ascii=False, indent=2)

            # Algorithm 1 第 12-14 行：θ 早停
            if best_rate >= self.theta:
                stop_reason = "solved"
                break

        # ---- E.4：最佳候选不超基线 → 回落基线（Stage 2 用最强可用代码）----
        if H and best_rate < H[0]["rate"]:
            best_code, best_rate, best_name = H[0]["code"], H[0]["rate"], H[0]["name"]

        # ---- Algorithm 1 第 16 行：Stage 2 一次性验证（禁调试）----
        stage2_rate = ""
        best_path = os.path.join(run_dir, "evosearch_best_code.py")
        if best_code is not None:
            with open(best_path, "w", encoding="utf-8") as f:
                f.write(best_code)
            self._emit("stage", text="Stage 2: held-out 一次性验证")
            res2 = evaluate_population({"best": best_path}, spec.env_task,
                                       spec.heldout_seeds,
                                       os.path.join(run_dir, "traces", "stage2"),
                                       spec.extra_engine_args,
                                       spec.time_budget_per_trial, self.max_workers)
            r2 = res2["best"]
            stage2_rate = f"{r2.n_success}/{len(r2.seed_results)}"
            self._emit("stage2_done", rate=stage2_rate)

        # ---- Algorithm 1 第 17 行：ExtractValidatedPatterns → findings.md ----
        history_text = "\n".join(
            f"- {h['name']}: debug rate={h['rate']:.3f} ({h['code_path']})" for h in H)
        history_text += (f"\n\nBest: {best_name} (debug {best_rate:.3f}); "
                         f"stop={stop_reason}; iterations={iters_done}")
        findings_path = os.path.join(run_dir, "findings.md")
        try:
            self._emit("llm_call", purpose="findings")
            fresp = query_model(self.cfg, findings_messages(
                spec.name, history_text, stage2_rate or "not run"))
            findings = fresp["content"]
        except Exception as e:
            findings = (f"## Task: {spec.name}\n### Root Cause(s)\n- (findings 生成失败: {e})\n"
                        f"### Stage 2 Success Rate\n{stage2_rate or 'not run'}\n")
        with open(findings_path, "w", encoding="utf-8") as f:
            f.write(findings)
        with open(os.path.join(run_dir, "hall_of_fame.json"), "w", encoding="utf-8") as f:
            json.dump([{k: v for k, v in h.items() if k != "code"} for h in H],
                      f, ensure_ascii=False, indent=2)
        self._emit("evo_done", task=spec.name, status=stop_reason,
                   best=best_name, best_rate=best_rate, stage2_rate=stage2_rate)
        return EvoResult(task=spec.name, status=stop_reason, iterations=iters_done,
                         best_code_path=best_path if best_code else "",
                         best_debug_rate=best_rate, stage2_rate=stage2_rate,
                         findings_path=findings_path, run_dir=run_dir, history=H)
