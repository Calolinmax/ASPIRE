"""Coordinator —— 论文 E.1/E.3 的协调者角色（程序化实现）。

职责（E.1 逐字义务）：
- 管理任务队列与进度（progress tracker），把任务分派给 actor；
- actor 完成后**只读 findings.md**（E.3 Rule 4：不碰 trace/程序等任务级调试产物）；
- 审计 findings（可复用性 + API 合规），**串行**把通用模式写入共享 skill library
  （"serializes skill admission to avoid conflicting library writes"）；
- 绝不重复分派已完成任务（E.3 Rule 5）。

与论文的差异说明：论文的 coordinator/actor 都是 Claude Code（子）会话；
本实现为程序化流水线——actor 在进程内运行（内部评估已并行），任务间默认串行
（单 GPU 现实：并行 actor 的引擎子进程会挤占 EGL/CUDA 服务，得不偿失）。
"""

from __future__ import annotations

import json
import os
import tempfile
import time

from ..skills.library import SkillLibrary
from ..skills.synthesize import synthesize_from_findings
from . import events
from .actor import Actor, ActorResult
from .config import LLMConfig
from .evolve import EvolutionarySearch, EvoResult
from .task_spec import TaskSpec


class Coordinator:
    def __init__(self, llm_cfg: LLMConfig, library_root: str = "skill_library",
                 run_root: str = "agent_runs", mode: str = "actor",
                 max_workers: int = 4, event_cb=None):
        self.cfg = llm_cfg
        self.library = SkillLibrary(library_root)
        self.run_root = run_root
        self.mode = mode                # "actor" | "evosearch"
        self.max_workers = max_workers
        self.event_cb = events.compose(event_cb)   # 文件总线 + 本地回调
        os.makedirs(run_root, exist_ok=True)
        self.progress_path = os.path.join(run_root, "progress.json")

    def _emit(self, etype: str, **payload):
        self.event_cb(etype, payload)

    # ------------------------------------------------------------------
    # progress tracker（E.3：原子更新，并发安全）
    # ------------------------------------------------------------------
    def _load_progress(self) -> dict:
        if not os.path.isfile(self.progress_path):
            return {"tasks": {}}
        with open(self.progress_path, encoding="utf-8") as f:
            return json.load(f)

    def _update_progress(self, task: str, **fields) -> None:
        progress = self._load_progress()
        entry = progress.setdefault("tasks", {}).setdefault(task, {})
        entry.update(fields)
        entry["updated"] = time.strftime("%Y-%m-%d %H:%M:%S")
        fd, tmp = tempfile.mkstemp(dir=self.run_root, suffix=".tmp")
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(progress, f, ensure_ascii=False, indent=2)
        os.replace(tmp, self.progress_path)

    # ------------------------------------------------------------------
    def run(self, specs: list[TaskSpec]) -> dict:
        """主循环：读 progress → 分派 → 收 findings → 审计入库 → 更新 progress。"""
        progress = self._load_progress()
        results: dict[str, dict] = {}
        for spec in specs:
            done = progress.get("tasks", {}).get(spec.name, {})
            if done.get("status") in ("solved",) and done.get("stage2_rate"):
                self._emit("skip", task=spec.name,
                           reason="已完成（E.3 Rule 5: never re-dispatch）")
                continue
            self._emit("dispatch", task=spec.name, mode=self.mode)
            if self.mode == "evosearch":
                agent: Actor | EvolutionarySearch = EvolutionarySearch(
                    self.cfg, self.library, run_root=self.run_root,
                    max_workers=self.max_workers, event_cb=self.event_cb)
            else:
                agent = Actor(self.cfg, self.library, run_root=self.run_root,
                              max_workers=self.max_workers, event_cb=self.event_cb)
            try:
                result: ActorResult | EvoResult = agent.run(spec)
            except Exception as e:
                self._emit("error", task=spec.name, text=str(e))
                self._update_progress(spec.name, status="error", error=str(e))
                results[spec.name] = {"status": "error", "error": str(e)}
                continue

            status = result.status
            stage2 = getattr(result, "stage2_rate", "")
            self._update_progress(
                spec.name, status=status, stage2_rate=stage2,
                stage1_rate=getattr(result, "stage1_rate", getattr(result, "best_debug_rate", 0.0)),
                run_dir=result.run_dir, findings_path=result.findings_path)

            # ---- E.1：审计 findings → 串行入库 ----
            admission = {"admitted": [], "rejected": [], "decision": "skip"}
            if result.findings_path and os.path.isfile(result.findings_path) \
                    and status != "blocked":
                self._emit("audit_start", task=spec.name)
                try:
                    admission = synthesize_from_findings(
                        self.cfg, result.findings_path, self.library, spec.name)
                    self._emit("audit_done", task=spec.name,
                               admitted=[a["name"] for a in admission["admitted"]],
                               rejected=admission["rejected"])
                except Exception as e:
                    self._emit("error", task=spec.name, text=f"审计失败: {e}")
                    admission = {"admitted": [], "rejected": [], "decision": "error",
                                 "rationale": str(e)}
            results[spec.name] = {
                "status": status, "stage2_rate": stage2,
                "run_dir": result.run_dir, "findings_path": result.findings_path,
                "admission": admission,
            }
            progress = self._load_progress()
        return results
