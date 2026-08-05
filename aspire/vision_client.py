"""SAM3/CGN 视觉服务的进程内客户端（纯 HTTP, 不引入 torch/CUDA）。

签名与 aspire.vision_sam3 完全一致 —— primitives 可无感切换:
    from .vision_client import segment_sam3_text_prompt, segment_sam3_point_prompt, warmup, grasp_cgn
"""

# =============================================================================
# 🔒 冻结警示（2026-08-05 用户裁决）：本文件属已完成并经验证的 API/组件
# （docs/api_asset_map.md 看板 `- [x]` 项）——
# **此处只有人类（顾问也不行）批准，才能更改。**
# =============================================================================


from __future__ import annotations

import pickle
import urllib.request

_SERVER = "http://127.0.0.1:8123"
_CGN_SERVER = "http://127.0.0.1:8117"


def set_server(url: str):
    global _SERVER
    _SERVER = url.rstrip("/")


def _post(path: str, payload: dict, timeout: float = 120.0):
    body = pickle.dumps(payload, protocol=4)
    req = urllib.request.Request(
        f"{_SERVER}{path}", data=body,
        headers={"Content-Type": "application/octet-stream"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = pickle.loads(resp.read())
    if isinstance(out, dict) and "error" in out:
        raise RuntimeError(f"vision_server: {out['error']}")
    return out


def healthy() -> bool:
    try:
        with urllib.request.urlopen(f"{_SERVER}/health", timeout=3) as resp:
            out = pickle.loads(resp.read())
        return out.get("status") == "ok"
    except Exception:
        return False


def warmup() -> None:
    """等待服务就绪（模型在服务进程内已预热）。"""
    if not healthy():
        raise RuntimeError(f"vision_server 未就绪: {_SERVER} (先启动 python -m aspire.vision_server)")


def segment_sam3_text_prompt(rgb, prompt: str) -> list[dict]:
    return _post("/segment", {"mode": "text", "rgb": rgb, "prompt": prompt})


def segment_sam3_point_prompt(rgb, point) -> list[dict]:
    return _post("/segment", {"mode": "point", "rgb": rgb, "point": point})


# ---------------------------------------------------------------------------
# CGN 服务客户端（端口 8117）
# ---------------------------------------------------------------------------

def grasp_cgn(depth, K, seg, z_range=(0.2, 1.8), forward_passes=1, return_openings=False):
    """调用 CGN 服务，返回 (grasps, scores)（或加 openings）。

    Args:
        depth: (H, W) 深度图，单位米
        K: (3, 3) 相机内参
        seg: (H, W) 分割图
        z_range: 深度范围过滤（默认 (0.2, 1.8)，与官方 inference.py 对齐）
        forward_passes: 前向传播次数（候选数）。【当前仅支持 1】——
            服务端计算图按 batch_size=1 构建，传 >1 会在服务端以
            "Cannot feed value of shape (N, 20000, 3)" 失败，故此处直接拦截。
        return_openings: 为 True 时返回三元组 (grasps, scores, openings)，
            openings 为每个候选的预测夹爪开度 (N,) 米（服务端 2026-07-31 起提供）。

    Returns:
        grasps: (N, 4, 4) numpy 数组，相机系位姿
        scores: (N,) numpy 数组，得分
        openings: (N,) numpy 数组（仅 return_openings=True）
    """
    if forward_passes != 1:
        raise ValueError(
            f"grasp_cgn: forward_passes 当前仅支持 1（服务端计算图 batch_size=1），"
            f"收到 {forward_passes}")
    body = pickle.dumps({
        "depth": depth,
        "K": K,
        "seg": seg,
        "z_range": list(z_range),
        "forward_passes": forward_passes
    }, protocol=4)

    req = urllib.request.Request(
        f"{_CGN_SERVER}/grasp", data=body,
        headers={"Content-Type": "application/octet-stream"}, method="POST",
    )
    with urllib.request.urlopen(req, timeout=120.0) as resp:
        out = pickle.loads(resp.read())

    if isinstance(out, dict) and "error" in out:
        raise RuntimeError(f"cgn_server: {out['error']}")

    import numpy as np
    grasps = np.array(out["grasps"])
    scores = np.array(out["scores"])
    if return_openings:
        openings = np.array(out.get("openings", np.zeros(len(scores), dtype=np.float32)))
        return grasps, scores, openings
    return grasps, scores
