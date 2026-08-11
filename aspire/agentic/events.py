"""事件总线（文件桥）——所有 run（CLI 或 Web 发起）统一经 JSONL 文件广播。

动机（2026-08-11 用户指令）：监管可视化——无论 run 从哪里发起，Web UI 都要
能看到事件流。单写者追加、多读者 tail；JSONL 一行一事件，崩溃也只坏最后一行。

- 生产侧：actor/evolve/coordinator/cli 的 event_cb 一律 compose `emit`。
- 消费侧：web server 起线程 follow 本文件 → WebSocket 广播 + 新连接回放最近 N 条。
"""

from __future__ import annotations

import json
import os
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
EVENTS_PATH = os.path.join(REPO_ROOT, "agent_runs", "events.jsonl")


def emit(etype: str, payload: dict) -> None:
    """追加一条事件（best-effort：文件写失败不拖累 run 本体）。"""
    try:
        os.makedirs(os.path.dirname(EVENTS_PATH), exist_ok=True)
        line = json.dumps({"ts": time.strftime("%H:%M:%S"), "type": etype,
                           **{k: v for k, v in payload.items()
                              if isinstance(v, (str, int, float, bool, list, dict, type(None)))}},
                          ensure_ascii=False, default=str)
        with open(EVENTS_PATH, "a", encoding="utf-8") as f:
            f.write(line + "\n")
    except Exception:
        pass


def compose(cb):
    """把本地回调与文件总线组合：emit 先行（持久化），cb 次之（终端/WS）。"""
    def _wrapped(etype: str, payload: dict) -> None:
        emit(etype, payload)
        if cb is not None:
            cb(etype, payload)
    return _wrapped


def read_recent(n: int = 200, path: str | None = None) -> list[dict]:
    """最近 n 条事件（新 WS 连接的回放素材）。"""
    path = path or EVENTS_PATH
    if not os.path.isfile(path):
        return []
    with open(path, encoding="utf-8") as f:
        lines = f.readlines()[-n:]
    out = []
    for line in lines:
        try:
            out.append(json.loads(line))
        except json.JSONDecodeError:
            continue
    return out


def follow(callback, path: str | None = None, poll_s: float = 0.5,
               start_from_end: bool = True) -> None:
    """tail -f 循环（web server 的守护线程用；永不返回）。"""
    path = path or EVENTS_PATH
    os.makedirs(os.path.dirname(path), exist_ok=True)
    open(path, "a", encoding="utf-8").close()
    with open(path, encoding="utf-8") as f:
        if start_from_end:
            f.seek(0, os.SEEK_END)
        while True:
            line = f.readline()
            if not line:
                time.sleep(poll_s)
                continue
            try:
                callback(json.loads(line))
            except json.JSONDecodeError:
                continue
