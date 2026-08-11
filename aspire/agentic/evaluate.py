"""并行程序评估 harness（AGENTS.md §2 硬性要求：多进程并行，禁止串行）。

执行通道 = 引擎 CLI 子进程（已封版接口，退出码 0=success/1=fail）：

    python -m aspire.engine.engine_capx --code <candidate.py> --task <场景> \
        --seed <s> --trace-root <run_dir>/traces [extra args]

- 并行 = ThreadPoolExecutor 管理 N 个 subprocess（MuJoCo/EGL 各自独立进程，
  CUDA 推理共享常驻 vision_server/CGN 容器服务）。
- vision_server 由 harness **预检单例拉起**：所有引擎子进程见到健康服务即不接管
  （避免 N 子进程并发拉起同一端口的竞争，也避免子进程退出时误杀共享服务）。
- 结果解析：退出码 + stdout 里 "success        : True" 与 "trace          : <path>"
  两行（trace 目录名同时刻冲突会自动 _2 后缀，必须靠 stdout 精确对回）。
"""

from __future__ import annotations

import concurrent.futures
import os
import re
import subprocess
import threading
import time
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ASPIRE_PY = "/home/stouching/anaconda3/envs/ASPIRE/bin/python"

_SUCCESS_RE = re.compile(r"^success\s+:\s+(\w+)", re.MULTILINE)
_TRACE_RE = re.compile(r"^trace\s+:\s+(\S+)", re.MULTILINE)

# vision_server 所有权（harness 进程级单例）
_VISION_PROC: subprocess.Popen | None = None
_VISION_LOCK = threading.Lock()

# 活跃评估子进程注册表 + 取消信号（web UI 的 stop 按钮用）
_ACTIVE: dict[int, subprocess.Popen] = {}
_ACTIVE_LOCK = threading.Lock()
_CANCEL = threading.Event()

# 逐 seed 结果事件钩子（web UI 实时监管用；actor/evolve 启动时 set）
_EVENT_CB = None


def set_event_cb(cb) -> None:
    global _EVENT_CB
    _EVENT_CB = cb


def _emit_seed(result: "SeedResult", name: str) -> None:
    if _EVENT_CB is not None:
        _EVENT_CB("seed_result", {"name": name, "seed": result.seed,
                                  "success": result.success,
                                  "wall_time": result.wall_time,
                                  "timed_out": result.timed_out,
                                  "trace_path": result.trace_path or "",
                                  "error_line": result.error_line})


class EvaluationCancelled(RuntimeError):
    pass


def cancel_all() -> None:
    """取消全部进行中的评估：终止活跃子进程，后续任务立即失败返回。"""
    _CANCEL.set()
    with _ACTIVE_LOCK:
        procs = list(_ACTIVE.values())
    for p in procs:
        try:
            p.terminate()
        except Exception:
            pass


def reset_cancel() -> None:
    _CANCEL.clear()


def ensure_services(start_vision: bool = True, wait_s: float = 180.0) -> dict:
    """预检常驻服务。vision_server 不健康则拉起（单例）；pyroki/CGN 只报告。

    返回 {"vision": "healthy|started|failed", "pyroki": bool, "cgn": bool}
    """
    global _VISION_PROC
    import requests as _req

    status = {"vision": "unknown", "pyroki": False, "cgn": False}
    with _VISION_LOCK:
        from ..perception import vision_client
        if vision_client.healthy():
            status["vision"] = "healthy"
        elif start_vision:
            print("[evaluate] 拉起 vision_server（harness 单例）...", flush=True)
            _VISION_PROC = subprocess.Popen(
                [ASPIRE_PY, "-m", "aspire.perception.vision_server", "--port", "8123"],
                cwd=REPO_ROOT)
            t0 = time.time()
            while not vision_client.healthy():
                if time.time() - t0 > wait_s:
                    status["vision"] = "failed"
                    break
                time.sleep(2.0)
            else:
                status["vision"] = "started"
        else:
            status["vision"] = "down"
    try:
        r = _req.get("http://127.0.0.1:8116/health", timeout=3)
        status["pyroki"] = r.status_code < 500
    except Exception:
        pass
    try:
        r = _req.get("http://127.0.0.1:8117/health", timeout=3)
        status["cgn"] = '"model_loaded":true' in r.text.replace(" ", "").replace("\n", "") \
            or '"model_loaded": true' in r.text
    except Exception:
        pass
    return status


def shutdown_services() -> None:
    """只回收 harness 自己拉起的 vision_server（子进程接管语义与引擎一致）。"""
    global _VISION_PROC
    with _VISION_LOCK:
        if _VISION_PROC is not None:
            _VISION_PROC.terminate()
            try:
                _VISION_PROC.wait(timeout=5)
            except Exception:
                _VISION_PROC.kill()
            _VISION_PROC = None


@dataclass
class SeedResult:
    seed: int
    success: bool
    exit_code: int
    trace_path: str | None
    wall_time: float
    timed_out: bool = False
    stdout_tail: str = ""
    error_line: str = ""     # stdout 里 "--- error ---" 段的首行（失败一眼定位）


@dataclass
class ProgramResult:
    name: str                       # 候选名（如 candidate_A / baseline / round2）
    code_path: str
    rate: float                     # debug/heldout seeds 成功率（= Algorithm 1 的 r）
    seed_results: list[SeedResult] = field(default_factory=list)

    @property
    def n_success(self) -> int:
        return sum(1 for r in self.seed_results if r.success)

    def leaderboard_line(self) -> str:
        return (f"{self.name}: {self.n_success}/{len(self.seed_results)} "
                f"= {self.rate:.3f}")


def _run_single_seed(code_path: str, env_task: str, seed: int, trace_root: str,
                     extra_args: list[str], timeout: float) -> SeedResult:
    """单 seed 子进程执行（引擎 CLI 封版接口）；可被取消（web stop）。"""
    if _CANCEL.is_set():
        return SeedResult(seed=seed, success=False, exit_code=-1, trace_path=None,
                          wall_time=0.0, error_line="cancelled")
    cmd = [ASPIRE_PY, "-m", "aspire.engine.engine_capx",
           "--code", code_path, "--task", env_task, "--seed", str(seed),
           "--trace-root", trace_root] + list(extra_args)
    t0 = time.time()
    timed_out = cancelled = False
    proc = subprocess.Popen(cmd, cwd=REPO_ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True)
    with _ACTIVE_LOCK:
        _ACTIVE[proc.pid] = proc
    try:
        while True:
            try:
                out, _ = proc.communicate(timeout=1.0)
                break
            except subprocess.TimeoutExpired:
                if _CANCEL.is_set():
                    proc.terminate()
                    cancelled = True
                if time.time() - t0 > timeout:
                    proc.kill()
                    timed_out = True
                if cancelled or timed_out:
                    try:
                        out, _ = proc.communicate(timeout=10)
                    except Exception:
                        out = ""
                    break
        exit_code = proc.returncode if proc.returncode is not None else -9
    finally:
        with _ACTIVE_LOCK:
            _ACTIVE.pop(proc.pid, None)
    wall = time.time() - t0
    out = out or ""

    m = _SUCCESS_RE.search(out)
    success = (m.group(1) == "True") if m else (exit_code == 0)
    if timed_out or cancelled:
        success = False
    m = _TRACE_RE.search(out)
    trace_path = m.group(1) if m else None
    err_line = "cancelled" if cancelled else ""
    if not err_line:
        em = re.search(r"--- error ---\n(?:Traceback.*?)?\n?(\w[^\n]*)", out, re.DOTALL)
        if em:
            err_line = em.group(1).strip()[:200]
    return SeedResult(seed=seed, success=success, exit_code=exit_code,
                      trace_path=trace_path, wall_time=round(wall, 1),
                      timed_out=timed_out, stdout_tail=out[-2000:], error_line=err_line)


def evaluate_program(name: str, code_path: str, env_task: str, seeds: list[int],
                     trace_root: str, extra_args: list[str] | None = None,
                     timeout: float = 900.0, max_workers: int = 4) -> ProgramResult:
    """一个程序 × 多 seeds 的并行评估（ seeds 间并行）。"""
    os.makedirs(trace_root, exist_ok=True)
    results: list[SeedResult] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_run_single_seed, code_path, env_task, s, trace_root,
                            extra_args or [], timeout): s for s in seeds}
        for fut in concurrent.futures.as_completed(futs):
            r = fut.result()
            results.append(r)
            _emit_seed(r, name)
            print(f"[evaluate] {name} seed={r.seed} -> {'OK' if r.success else 'FAIL'}"
                  f" ({r.wall_time}s){' TIMEOUT' if r.timed_out else ''}", flush=True)
    results.sort(key=lambda r: r.seed)
    rate = sum(1 for r in results if r.success) / max(1, len(results))
    return ProgramResult(name=name, code_path=code_path, rate=rate, seed_results=results)


def evaluate_population(candidates: dict[str, str], env_task: str, seeds: list[int],
                        trace_root: str, extra_args: list[str] | None = None,
                        timeout: float = 900.0,
                        max_workers: int = 4) -> dict[str, ProgramResult]:
    """K 候选 × seeds 全并行评估（Algorithm 1 第 5-7 行的并行化——AGENTS.md §2）。

    candidates: {候选名: 代码路径}。所有 (候选, seed) 对进入同一子进程池。
    """
    os.makedirs(trace_root, exist_ok=True)
    jobs = [(name, path, s) for name, path in candidates.items() for s in seeds]
    results: dict[str, list[SeedResult]] = {name: [] for name in candidates}
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futs = {pool.submit(_run_single_seed, path, env_task, s,
                            os.path.join(trace_root, name), extra_args or [], timeout):
                (name, s) for name, path, s in jobs}
        for fut in concurrent.futures.as_completed(futs):
            name, _ = futs[fut]
            r = fut.result()
            results[name].append(r)
            _emit_seed(r, name)
            print(f"[evaluate] {name} seed={r.seed} -> {'OK' if r.success else 'FAIL'}"
                  f" ({r.wall_time}s){' TIMEOUT' if r.timed_out else ''}", flush=True)
    out: dict[str, ProgramResult] = {}
    for name, path in candidates.items():
        rs = sorted(results[name], key=lambda r: r.seed)
        rate = sum(1 for r in rs if r.success) / max(1, len(rs))
        out[name] = ProgramResult(name=name, code_path=path, rate=rate, seed_results=rs)
    return out


def leaderboard(results: dict[str, ProgramResult]) -> str:
    """排行榜文本（task_analysis 与日志用），按 rate 降序。"""
    lines = sorted((r.leaderboard_line() for r in results.values()),
                   key=lambda s: -float(s.split("=")[-1]))
    return "\n".join(lines)
