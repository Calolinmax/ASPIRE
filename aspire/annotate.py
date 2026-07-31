"""视觉证据标注：把定位算法的输出画回输入帧，随 trace 落盘（AGENTS.md §3 视觉证据）。

覆盖定位全链路，agentview 与 wrist 通用（标注画在实际被算法处理的那一帧上）：
  segment_sam3_text_prompt  → mask 着色+轮廓+质心，prompt 与每个 mask 的 score/area
  segment_sam3_point_prompt → 同上，另画点提示位置（品红十字）
  point_prompt_molmo        → 预测点位置（品红十字）
  mask_to_world_points      → 深度底图 + mask + 世界系中位中心坐标
标注图按算法分文件夹保存（images/sam3|molmo|mask_to_world/，见 trace.ALGO_FOLDERS），
每次调用都存——含"未检到"的帧，那是失败定位的关键证据。
运动/规划类无算法输出可画（返回 None），其过程帧由帧流（images/top|wrist|depth）覆盖。
伺服调试路径：servo_align_wrist 每轮迭代调用 segment_* + mask_to_world_points，
其腕部帧标注即伺服每步的"算法所见"，配合 stdout 的 [servo] dxy 日志可完整复盘。
"""

from __future__ import annotations

import cv2
import numpy as np

_MASK_COLORS = [(40, 255, 40), (40, 200, 255), (255, 180, 40)]  # RGB，按返回顺序（≈score 降序）
_FONT = cv2.FONT_HERSHEY_SIMPLEX


def _tint(vis, mask, color, alpha=0.45):
    m = mask > 0
    vis[m] = (vis[m].astype(float) * (1 - alpha) + np.asarray(color, float) * alpha).astype(np.uint8)


def _draw_mask(vis, mask, color):
    _tint(vis, mask, color)
    cnts, _ = cv2.findContours((mask > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(vis, cnts, -1, color, 1)


def _cross(vis, u, v, color, size=8):
    cv2.drawMarker(vis, (int(round(u)), int(round(v))), color, cv2.MARKER_CROSS, size, 1)


def _text(vis, lines):
    for i, t in enumerate(lines):
        y = 13 + i * 13
        cv2.putText(vis, t, (4, y), _FONT, 0.34, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, t, (4, y), _FONT, 0.34, (255, 255, 255), 1, cv2.LINE_AA)


def _annotate_masks(rgb, masks, header):
    vis = rgb.copy()
    lines = list(header)
    n = 0
    for m in masks[:3]:
        if not isinstance(m, dict) or m.get("mask") is None:
            continue
        color = _MASK_COLORS[n % len(_MASK_COLORS)]
        _draw_mask(vis, m["mask"], color)
        c = m.get("centroid")
        if c is not None and c[0] is not None:
            _cross(vis, c[0], c[1], color)
        area = int(m.get("area", int(np.sum(m["mask"] > 0))))
        lines.append(f"#{n} s={float(m.get('score', 0.0)):.2f} a={area}")
        n += 1
    if n == 0:
        lines.append("no mask")
    _text(vis, lines)
    return vis


def _project(K, p):
    """OpenCV 相机系点 → 像素；在相机后方/过近返回 None。"""
    z = float(p[2])
    if z <= 0.01:
        return None
    return (K[0, 0] * p[0] / z + K[0, 2], K[1, 1] * p[1] / z + K[1, 2])


def build_annotation(name, args, out):
    """视觉类 primitive 的标注图（RGB uint8）；该 primitive 无可画输出时返回 None。

    任何异常都返回 None——标注是证据增强，绝不能污染任务执行。
    """
    try:
        if name == "segment_sam3_text_prompt":
            return _annotate_masks(np.asarray(args[0]), list(out or []), [f"sam text: {args[1]}"])
        if name == "segment_sam3_point_prompt":
            vis = _annotate_masks(np.asarray(args[0]), list(out or []), [f"sam point: {tuple(args[1])}"])
            _cross(vis, args[1][0], args[1][1], (255, 80, 255), size=10)
            return vis
        if name == "point_prompt_molmo":
            vis = np.asarray(args[0]).copy()
            pts = out or {}
            for p in pts.values():
                if p[0] is not None and p[1] is not None:
                    _cross(vis, p[0], p[1], (255, 80, 255), size=10)
            _text(vis, [f"molmo: {args[1]}", f"pts={len(pts)}"])
            return vis
        if name == "plan_grasp":
            depth, seg = np.asarray(args[0], float), np.asarray(args[2])
            d = np.nan_to_num(depth, nan=0.0)
            dmax = float(d.max())
            bg = (d / dmax * 255 if dmax > 0 else d).astype(np.uint8)
            vis = cv2.cvtColor(cv2.applyColorMap(bg, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
            _draw_mask(vis, np.asarray(seg) > 0, (40, 255, 40))
            grasps, scores = out
            best = float(scores.max()) if len(scores) else 0.0
            _text(vis, [f"plan_grasp: {len(scores)} cand, best={best:.2f}"])
            return vis
        if name == "grasp_cgn":
            # RGB 底图 + top-3 候选 glyph（两指线+逼近线+闭合线+接触点），线旁标 score。
            # 候选在 OpenCV 相机系，直接用 K 投影；0 候选也必须存图（域差/召回证据）。
            rgb, K = np.asarray(args[0]), np.asarray(args[2], float)
            vis = rgb.copy()
            grasps, scores, openings = out if out is not None else ([], [], [])
            if len(grasps) == 0:
                _text(vis, ["CGN: 0 candidates"])
                return vis
            n_shown = 0
            for i in range(min(3, len(grasps))):
                g = np.asarray(grasps[i], float)
                O, R = g[:3, 3], g[:3, :3]
                X, Z = R[:, 0], R[:, 2]
                w = float(openings[i]) if openings is not None and len(openings) > i else 0.05
                C = O + Z * 0.1034  # palm→指尖接触面（CGN 约定, GRIPPER_DEPTH_PANDA）
                segs = [(O, C),
                        (O + X * w / 2, C + X * w / 2),
                        (O - X * w / 2, C - X * w / 2),
                        (C - X * w / 2, C + X * w / 2)]
                proj = [(_project(K, a), _project(K, b)) for a, b in segs]
                if any(pa is None or pb is None for pa, pb in proj):
                    continue
                color = _MASK_COLORS[n_shown % len(_MASK_COLORS)]
                for pa, pb in proj:
                    cv2.line(vis, (int(round(pa[0])), int(round(pa[1]))),
                             (int(round(pb[0])), int(round(pb[1]))), color, 1)
                pc = _project(K, C)
                _cross(vis, pc[0], pc[1], color)
                cv2.putText(vis, f"{float(scores[i]):.2f}",
                            (int(round(pc[0])) + 3, int(round(pc[1])) - 3),
                            _FONT, 0.34, color, 1, cv2.LINE_AA)
                n_shown += 1
            _text(vis, [f"grasp_cgn: {len(grasps)} cand"])
            return vis
        if name == "mask_to_world_points":
            mask, depth = np.asarray(args[0]), np.asarray(args[1], float)
            d = np.nan_to_num(depth, nan=0.0)
            dmax = float(d.max())
            bg = (d / dmax * 255 if dmax > 0 else d).astype(np.uint8)
            vis = cv2.cvtColor(cv2.applyColorMap(bg, cv2.COLORMAP_VIRIDIS), cv2.COLOR_BGR2RGB)
            _draw_mask(vis, mask, (40, 255, 40))
            pts = np.asarray(out).reshape(-1, 3) if out is not None and len(out) else np.zeros((0, 3))
            pts = pts[np.isfinite(pts).all(axis=1)]
            lines = [f"N={len(pts)}"]
            if len(pts):
                c = np.median(pts, axis=0)
                lines.append(f"center=({c[0]:.4f},{c[1]:.4f},{c[2]:.4f})")
            _text(vis, lines)
            return vis
    except Exception:
        return None
    return None
