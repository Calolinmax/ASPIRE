"""SAM3 视觉推理服务进程（与执行引擎隔离的 CUDA 工作进程）。

背景（2026-07-28 实测定位）: 本机环境下, PyTorch CUDA 初始化后, 同进程的
EGL offscreen 渲染会**永久损坏** (之后每次渲染都 wedge, 连新建的 EGL context
也不例外) —— CUDA 与 EGL 在同一 GPU 上冲突。因此 SAM3 推理必须放在独立进程,
引擎进程只做 EGL 渲染, 通过 localhost HTTP 调用视觉服务。

与 cap-x 的 launch_sam3_server 架构一致(HTTP 服务), 但复用本项目 aspire.perception.vision_sam3。

启动:
    /home/stouching/anaconda3/envs/ASPIRE/bin/python -m aspire.perception.vision_server --port 8123

接口 (pickle over HTTP):
    GET  /health      → "ok"
    POST /segment     → {"mode": "text",  "rgb": (H,W,3) u8, "prompt": str}
                        {"mode": "point", "rgb": (H,W,3) u8, "point": (x, y)}
                      ← list[dict] (vision_sam3 原生格式: mask u8/score/area/centroid/...)
"""

# =============================================================================
# 🔒 冻结警示（2026-08-06 用户裁决 · 封版）：本文件属**已测试通过**的 API 层
# （cap-x 契约 15 函数 + 契约外 5 函数/组件，docs/api_asset_map.md 冻结清单）。
# **只能在 scripts 中调用，禁止修改——只有人类（顾问也不行）批准才能更改。**
# 本文件同时被 chmod a-w 机械保护；解冻须人类亲自 chmod +w。
# =============================================================================

from __future__ import annotations

import argparse
import pickle
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def _make_handler():
    # 延迟到服务进程内加载 (CUDA 只存在于本进程)
    from . import vision_sam3

    vision_sam3.warmup()

    class Handler(BaseHTTPRequestHandler):
        def _send_pickle(self, obj, code=200):
            body = pickle.dumps(obj, protocol=4)
            self.send_response(code)
            self.send_header("Content-Type", "application/octet-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            if self.path == "/health":
                self._send_pickle({"status": "ok"})
            else:
                self._send_pickle({"error": "unknown path"}, 404)

        def do_POST(self):
            n = int(self.headers.get("Content-Length", 0))
            req = pickle.loads(self.rfile.read(n))
            try:
                if self.path == "/segment":
                    if req["mode"] == "text":
                        out = vision_sam3.segment_sam3_text_prompt(req["rgb"], req["prompt"])
                    elif req["mode"] == "point":
                        out = vision_sam3.segment_sam3_point_prompt(req["rgb"], tuple(req["point"]))
                    else:
                        raise ValueError(f"unknown mode {req['mode']}")
                    self._send_pickle(out)
                else:
                    self._send_pickle({"error": "unknown path"}, 404)
            except Exception as e:  # 服务永不 500 崩溃, 错误回传
                self._send_pickle({"error": f"{type(e).__name__}: {e}"}, 200)

        def log_message(self, fmt, *args):
            pass  # 静默

    return Handler


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8123)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    server = ThreadingHTTPServer((args.host, args.port), _make_handler())
    print(f"[vision_server] SAM3 推理服务已就绪 http://{args.host}:{args.port}", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()
