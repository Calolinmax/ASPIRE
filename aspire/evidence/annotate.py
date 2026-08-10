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

# =============================================================================
# 🔒 冻结警示（2026-08-06 用户裁决 · 封版）：本文件属**已测试通过**的 API 层
# （cap-x 契约 15 函数 + 契约外 5 函数/组件，docs/api_asset_map.md 冻结清单）。
# **只能在 scripts 中调用，禁止修改——只有人类（顾问也不行）批准才能更改。**
# 本文件同时被 chmod a-w 机械保护；解冻须人类亲自 chmod +w。
# =============================================================================

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
            # 定稿画法（2026-07-31 用户肉审定版 F2, 规格见 docs/cgn_container.md §10）:
            #   开口 Π 三线段（palm 横杠+双指, 指尖不封口）, 统一绿 1px,
            #   top-5 按接触面深度 painter 序（远先近后; 倾斜抓时代 12 个
            #   已成"乱签堆", 2026-08-03 裁决收窄保可读）。
            #   宽度固定全开 0.08m（官方 draw_grasps 默认约定, 不随 openings 变）。
            #   指尖端 = 真实接触面 C=O+Z·0.1034（物理, 不动）;
            #   掌心端显示位 = C-Z·0.06（示意指长 6cm——物理 10.34cm 在小物体上
            #   悬空太高, 纯显示压缩, 非物理长度）。
            #   短刺 0.04m 从掌心沿 -Z（臂来方向, 约止于物理 palm O）。
            #   封口矩形/点云剪影均已证伪, 禁止复活。
            #   🔒 冻结（2026-08-05 用户裁决）: 此处只有人类（顾问也不行）批准才能更改。
            rgb, K = np.asarray(args[0]), np.asarray(args[2], float)
            vis = rgb.copy()
            grasps, scores, openings = out if out is not None else ([], [], [])
            if len(grasps) == 0:
                _text(vis, ["CGN: 0 candidates"])
                return vis

            def _px(p):
                return (int(round(p[0])), int(round(p[1])))

            GREEN = _MASK_COLORS[0]
            n_top = min(5, len(grasps))
            order = sorted(
                range(n_top),
                key=lambda i: -float(np.asarray(grasps[i])[2, 3]
                                     + np.asarray(grasps[i])[:3, 2][2] * 0.1034))
            best_vert_i, best_vert_v = -1, -1.0
            for i in order:
                g = np.asarray(grasps[i], float)
                O, R = g[:3, 3], g[:3, :3]
                X, Z = R[:, 0], R[:, 2]
                C = O + Z * 0.1034  # 指尖接触面（物理, GRIPPER_DEPTH_CGN; F2 定稿值,
                                    # 2026-08-05 恢复——Piper 期 0.045 是偏离）
                P = C - Z * 0.06     # 掌心端显示位（示意指长 6cm）
                w = 0.08             # 固定全开（官方约定）
                segs = [
                    (P - X * w / 2, P + X * w / 2),   # palm 横杠
                    (P + X * w / 2, C + X * w / 2),   # 指 +
                    (P - X * w / 2, C - X * w / 2),   # 指 −
                    (P, P - Z * 0.04),                # 短刺: 臂来方向
                ]
                proj = [(_project(K, a), _project(K, b)) for a, b in segs]
                if any(pa is None or pb is None for pa, pb in proj):
                    continue
                for pa, pb in proj:
                    cv2.line(vis, _px(pa), _px(pb), GREEN, 1, cv2.LINE_AA)
                vert = float(abs(R[:, 2][1]))  # 垂直度代理: cam 系 |approach·(0,1,0)|
                if vert > best_vert_v:
                    best_vert_v, best_vert_i = vert, i
            # REC 圈标推荐候选（D3 最垂直）
            if best_vert_i >= 0:
                g = np.asarray(grasps[best_vert_i], float)
                C = g[:3, 3] + g[:3, :3][:, 2] * 0.1034  # GRIPPER_DEPTH_CGN (F2 定稿值)
                pc_c = _project(K, C)
                if pc_c is not None:
                    ctr = _px(pc_c)
                    cv2.circle(vis, ctr, 10, (255, 80, 255), 2, cv2.LINE_AA)
                    cv2.putText(vis, "REC", (ctr[0] + 12, ctr[1] + 4),
                                _FONT, 0.4, (255, 80, 255), 1, cv2.LINE_AA)
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
