"""ASPIRE Web UI —— FastAPI 服务端（cap-x web 最小可用版的复刻）。

保留 cap-x `capx/web/server.py` 的核心三件套：
1. POST 启动 run（后台线程跑 actor/evosearch/coordinator）+ POST stop；
2. WebSocket 事件推送（run 状态/LLM 响应/评估进度/skill 入库）；
3. 单活跃会话策略（新 run 启动即拒绝或先停旧 run）。

刻意降级（cap-x 报告的最小可用取舍）：无流式 delta（一次性 LLM 响应事件）、
无 viser 3D iframe（用 trace 图像回传代替）、无刷新重连回放、前端零构建
（单静态页，不用 React/npm）。

事件流：worker 线程 event_cb → 线程安全 broadcaster → 每个 WS 连接的 asyncio.Queue。
"""

from __future__ import annotations

import asyncio
import json
import os
import queue
import threading
import time

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")

# ---------------------------------------------------------------------------
# 事件总线（线程 → asyncio 的安全桥）
# ---------------------------------------------------------------------------


class EventBus:
    def __init__(self) -> None:
        self._subscribers: list[queue.Queue] = []
        self._lock = threading.Lock()
        self.history: list[dict] = []      # 最近 500 条事件（新连接回放概览）

    def publish(self, etype: str, payload: dict) -> None:
        event = {"type": etype, "ts": time.strftime("%H:%M:%S"), **payload}
        self.publish_event(event)

    def publish_event(self, event: dict) -> None:
        with self._lock:
            self.history.append(event)
            self.history = self.history[-500:]
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(event)
            except queue.Full:
                pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=1000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)


BUS = EventBus()

# 文件总线 tailer：agent_runs/events.jsonl → BUS（CLI 与 Web 发起的 run 同源可见）
_tailer_started = False


def _start_tailer() -> None:
    global _tailer_started
    if _tailer_started:
        return
    _tailer_started = True
    from ..agentic import events
    t = threading.Thread(target=events.follow, args=(BUS.publish_event,),
                         kwargs={"start_from_end": True}, daemon=True)
    t.start()


_start_tailer()


# ---------------------------------------------------------------------------
# Run 管理（单活跃会话）
# ---------------------------------------------------------------------------


class RunManager:
    def __init__(self) -> None:
        self.thread: threading.Thread | None = None
        self.cancel_event = threading.Event()
        self.current: dict | None = None
        self._lock = threading.Lock()

    def is_running(self) -> bool:
        return self.thread is not None and self.thread.is_alive()

    def start(self, params: dict) -> tuple[bool, str]:
        with self._lock:
            if self.is_running():
                return False, "已有 run 在进行（单活跃会话）——先 /api/stop"
            from ..agentic.evaluate import reset_cancel
            reset_cancel()
            self.cancel_event.clear()
            self.current = {"params": params, "started": time.strftime("%H:%M:%S"),
                            "status": "running"}
            self.thread = threading.Thread(target=self._run_wrapper, args=(params,),
                                           daemon=True)
            self.thread.start()
            return True, "started"

    def stop(self) -> None:
        from ..agentic.evaluate import cancel_all
        self.cancel_event.set()
        cancel_all()
        _emit_event("run_status", {"status": "stopping"})

    def _event_cb(self, etype: str, payload: dict) -> None:
        if self.cancel_event.is_set():
            from ..agentic.evaluate import EvaluationCancelled
            raise EvaluationCancelled("用户停止")
        from ..agentic import events
        events.emit(etype, payload)   # 统一走文件总线（tailer 线程广播到 WS）

    def _run_wrapper(self, params: dict) -> None:
        from ..agentic.actor import Actor
        from ..agentic.config import LLMConfig
        from ..agentic.coordinator import Coordinator
        from ..agentic.evolve import EvolutionarySearch
        from ..agentic.evaluate import shutdown_services
        from ..agentic.task_spec import BUILTIN_TASKS
        from ..skills.library import SkillLibrary
        try:
            cfg = LLMConfig.from_env()
            cfg.validate()
        except Exception as e:
            _emit_event("run_status", {"status": "error", "error": f"LLM 配置不可用: {e}"})
            return
        library = SkillLibrary(os.path.join(REPO_ROOT, "skill_library"))
        try:
            mode = params.get("mode", "actor")
            task_names = params.get("tasks") or [params.get("task", "Stack")]
            specs = []
            for name in task_names:
                if name not in BUILTIN_TASKS:
                    _emit_event("run_status", {"status": "error",
                                               "error": f"未知任务 {name}"})
                    return
                spec = BUILTIN_TASKS[name]()
                if params.get("no_baseline"):
                    spec.baseline_code = None
                if params.get("seeds"):
                    spec.debug_seeds = [int(s) for s in str(params["seeds"]).split(",")]
                if params.get("heldout"):
                    spec.heldout_seeds = [int(s) for s in str(params["heldout"]).split(",")]
                specs.append(spec)
            _emit_event("run_status", {"status": "running", "mode": mode,
                                       "tasks": task_names})
            if mode == "coordinator":
                results = Coordinator(
                    cfg, library_root=os.path.join(REPO_ROOT, "skill_library"),
                    run_root=os.path.join(REPO_ROOT, "agent_runs"),
                    mode=params.get("agent_mode", "actor"),
                    max_workers=int(params.get("max_workers", 4)),
                    event_cb=self._event_cb).run(specs)
                _emit_event("run_status", {"status": "done",
                                           "results": json.loads(json.dumps(results, default=str))})
            else:
                spec = specs[0]
                if mode == "evosearch":
                    agent = EvolutionarySearch(
                        cfg, library, run_root=os.path.join(REPO_ROOT, "agent_runs"),
                        K=int(params.get("K", 4)), T=int(params.get("T", 5)),
                        theta=float(params.get("theta", 1.0)),
                        max_workers=int(params.get("max_workers", 4)),
                        event_cb=self._event_cb)
                else:
                    agent = Actor(cfg, library,
                                  run_root=os.path.join(REPO_ROOT, "agent_runs"),
                                  max_rounds=int(params.get("rounds", 3)),
                                  theta=float(params.get("theta", 1.0)),
                                  max_workers=int(params.get("max_workers", 4)),
                                  event_cb=self._event_cb)
                result = agent.run(spec)
                _emit_event("run_status", {"status": "done", "task": result.task,
                                           "result_status": result.status,
                                           "run_dir": result.run_dir})
        except Exception as e:
            _emit_event("run_status", {"status": "error", "error": str(e)})
        finally:
            shutdown_services()
            with self._lock:
                if self.current:
                    self.current["status"] = "finished"


def _emit_event(etype: str, payload: dict) -> None:
    """server 侧事件一律走文件总线（tailer 广播），保证 CLI/Web 发起的 run 同源可见。"""
    from ..agentic import events
    events.emit(etype, payload)


MANAGER = RunManager()

# ---------------------------------------------------------------------------
# FastAPI 应用
# ---------------------------------------------------------------------------

app = FastAPI(title="ASPIRE Web UI")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/")
def index():
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/api/status")
def api_status():
    from ..agentic.config import LLMConfig, load_dotenv
    from ..skills.library import SkillLibrary
    cfg = LLMConfig.from_env()
    dotenv = load_dotenv()
    configured = bool(cfg.api_key or cfg.provider in ("file", "mock"))
    lib = SkillLibrary(os.path.join(REPO_ROOT, "skill_library"))
    return {
        "llm": {"provider": cfg.provider, "model": cfg.model,
                "base_url": cfg.base_url, "configured": configured,
                "env_file": bool(dotenv)},
        "skills": {"count": len(lib.list()), "admitted": len(lib.list(status="admitted"))},
        "run": {"active": MANAGER.is_running(), "current": MANAGER.current},
    }


@app.get("/api/services")
def api_services():
    from ..agentic.evaluate import ensure_services
    return ensure_services(start_vision=False)


@app.get("/api/skills")
def api_skills():
    from ..skills.library import SkillLibrary
    return {"skills": SkillLibrary(os.path.join(REPO_ROOT, "skill_library")).list()}


@app.get("/api/skills/{name}")
def api_skill(name: str):
    from ..skills.library import SkillLibrary
    entry = SkillLibrary(os.path.join(REPO_ROOT, "skill_library")).get(name)
    if entry is None:
        return JSONResponse({"error": "not found"}, status_code=404)
    return {"name": name, "markdown": entry.to_markdown()}


@app.get("/api/progress")
def api_progress():
    path = os.path.join(REPO_ROOT, "agent_runs", "progress.json")
    if not os.path.isfile(path):
        return {"tasks": {}}
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/traces")
def api_traces(limit: int = 30):
    """最近 trace 目录（traces/ 与 agent_runs/ 下），读 trace.json 头信息。"""
    out = []
    for root in (os.path.join(REPO_ROOT, "traces"),
                 os.path.join(REPO_ROOT, "agent_runs")):
        if not os.path.isdir(root):
            continue
        for dirpath, _, files in os.walk(root):
            if "trace.json" not in files:
                continue
            path = os.path.join(dirpath, "trace.json")
            try:
                with open(path, encoding="utf-8") as f:
                    head = json.load(f)
                out.append({
                    "trace_path": os.path.relpath(path, REPO_ROOT),
                    "dir": os.path.relpath(dirpath, REPO_ROOT),
                    "task": head.get("task"), "seed": head.get("seed"),
                    "success": head.get("success"),
                    "n_records": len(head.get("records", [])),
                    "code_ref": head.get("code_ref"),
                    "mtime": os.path.getmtime(path),
                })
            except Exception:
                continue
    out.sort(key=lambda d: -d["mtime"])
    return {"traces": out[:limit]}


@app.get("/api/trace_detail")
def api_trace_detail(path: str):
    """trace.json 全文（图像字段保留相对链接，前端经 /api/trace_image 取图）。"""
    abs_path = _safe_path(path)
    with open(abs_path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/api/trace_image")
def api_trace_image(trace: str, rel: str):
    """trace 目录内图像（相对路径防穿越）。"""
    trace_abs = _safe_path(trace)
    img = os.path.normpath(os.path.join(trace_abs, rel))
    if not img.startswith(trace_abs + os.sep):
        return JSONResponse({"error": "path escape"}, status_code=403)
    if not os.path.isfile(img):
        return JSONResponse({"error": "not found"}, status_code=404)
    return FileResponse(img)


def _safe_path(path: str) -> str:
    abs_path = os.path.normpath(os.path.join(REPO_ROOT, path))
    if not abs_path.startswith(REPO_ROOT + os.sep):
        raise ValueError("path escape")
    if not os.path.isfile(abs_path):
        raise FileNotFoundError(path)
    return abs_path


@app.post("/api/run")
def api_run(params: dict):
    ok, msg = MANAGER.start(params)
    return {"ok": ok, "message": msg}


@app.post("/api/stop")
def api_stop():
    MANAGER.stop()
    return {"ok": True}


@app.websocket("/ws")
async def ws_endpoint(websocket: WebSocket):
    await websocket.accept()
    q = BUS.subscribe()
    try:
        # 新连接回放近 50 条事件（从文件总线读，server 重启也不丢历史）
        from ..agentic import events
        for event in events.read_recent(50):
            await websocket.send_json({"type": "replay", "event": event})
        while True:
            try:
                event = await asyncio.get_event_loop().run_in_executor(
                    None, q.get)
                await websocket.send_json(event)
            except WebSocketDisconnect:
                break
    finally:
        BUS.unsubscribe(q)


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8200, log_level="info")


if __name__ == "__main__":
    main()
