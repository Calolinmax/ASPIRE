"""ASPIRE 执行 trace 记录模块。

按 AGENTS.md 第 3 节要求：执行引擎记录每一次 primitive 调用的
【类别：检测/规划/抓取/控制】×【观测、输入、输出、视觉证据】。

图像组织（一次任务一个 trace 目录，trace.json 通过相对路径链接全部图像）：

    images/
      top/    s00000.jpg ...   顶部相机原始帧（帧流：固定帧率+调用边界补帧，文件名=仿真步）
      wrist/  s00000.jpg ...   腕部相机原始帧（同上）
      depth/  s00000.jpg ...   顶部深度 colormap（仅 API 调用边界保存——深度只在
                               感知调用时被消费，不做固定帧率连拍）
      sam3/   005_segment_sam3_text_prompt.jpg ...   算法标注图，仅调用时保存
      <algo>/ ...              每个用到的图像算法一个文件夹（molmo/mask_to_world/grasp...）

帧流按 (流, 仿真步) 去重，零重复；API 调用边界强制补帧，关键帧不丢。
算法标注图每次调用都存（含"未检到"的帧——失败定位的关键证据）。
frame_links 只返回该仿真步真实存在的流文件链接。
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field, asdict
from typing import Any

import numpy as np

# primitive 名 → 类别（AGENTS.md 第 3.2 节的四类）
CATEGORIES = {
    # 检测 (Detection)
    "get_observation": "detection",
    "segment_sam3_text_prompt": "detection",
    "segment_sam3_point_prompt": "detection",
    "point_prompt_molmo": "detection",
    "mask_to_world_points": "detection",
    "get_oriented_bounding_box_from_3d_points": "detection",
    "pixel_to_world_point": "detection",
    "grasp_cgn": "detection",
    # 规划 (Planning)
    "solve_ik": "planning",
    "interpolate_segment": "planning",
    "rotation_matrix_to_quaternion": "planning",
    # 抓取 (Grasping)
    "open_gripper": "grasping",
    "close_gripper": "grasping",
    "plan_grasp": "grasping",
    # 控制调用 (Control)
    "move_to_pose": "control",
    "move_to_joints": "control",
}

# 视觉算法 primitive → 标注图文件夹（未列出的算法以其 primitive 名为文件夹）
ALGO_FOLDERS = {
    "segment_sam3_text_prompt": "sam3",
    "segment_sam3_point_prompt": "sam3",
    "point_prompt_molmo": "molmo",
    "mask_to_world_points": "mask_to_world",
    "plan_grasp": "grasp",
    "grasp_cgn": "cgn",
}

# 帧流子目录（固定帧率保存的原始观测）
STREAM_FOLDERS = ["top", "wrist", "depth"]


def summarize(value: Any, max_len: int = 64) -> Any:
    """把任意值压缩成可 JSON 序列化、人类可读的摘要。"""
    if isinstance(value, np.ndarray):
        if value.size <= 12:
            return {"ndarray": np.round(value.astype(float), 4).tolist()}
        return {
            "ndarray_summary": {
                "shape": list(value.shape),
                "dtype": str(value.dtype),
                "min": round(float(np.nanmin(value)), 4),
                "max": round(float(np.nanmax(value)), 4),
                "mean": round(float(np.nanmean(value)), 4),
            }
        }
    if isinstance(value, (list, tuple)):
        if len(value) > max_len:
            return {"list_summary": {"len": len(value), "head": summarize(value[0])}}
        return [summarize(v) for v in value]
    if isinstance(value, dict):
        return {k: summarize(v) for k, v in value.items()}
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return repr(value)[:200]


@dataclass
class TraceRecord:
    seq: int                    # 调用序号
    name: str                   # primitive 名
    category: str               # detection / planning / grasping / control
    sim_step_before: int        # 调用前仿真步数
    sim_step_after: int         # 调用后仿真步数（运动类 > before）
    wall_time: float            # 相对引擎启动的秒数
    inputs: dict                # 输入参数摘要
    outputs: Any                # 返回值摘要
    observation: dict           # 调用时关键观测（eef/gripper/joints/object）
    visual_evidence: dict | None        # {流名: 调用前帧相对路径}（仅含该步真实存在的流）
    visual_evidence_after: dict | None = None  # 运动类结束帧（同结构）
    annotation: str | None = None       # 算法标注图相对路径（images/<algo>/...）
    collision_events: list[dict] = field(default_factory=list)  # C 阶段：碰撞反馈


@dataclass
class RunTrace:
    task: str
    seed: int
    success: bool = False
    error: str | None = None
    total_sim_steps: int = 0
    frame_interval: int = 0         # top/wrist 帧流固定间隔（仿真步）；边界补帧另计
    code_ref: str | None = None     # 任务代码路径（代码由 agent harness 管理，此处仅引用）
    stdout: str = ""
    records: list[TraceRecord] = field(default_factory=list)


class Tracer:
    """收集一次任务执行的全部 primitive 调用记录与图像证据。"""

    def __init__(self, trace_dir: str, task: str, seed: int, frame_interval: int = 5):
        self.trace_dir = trace_dir
        self.img_root = os.path.join(trace_dir, "images")
        for sub in STREAM_FOLDERS:
            os.makedirs(os.path.join(self.img_root, sub), exist_ok=True)
        self.frame_interval = frame_interval
        self.trace = RunTrace(task=task, seed=seed, frame_interval=frame_interval)
        self._t0 = time.time()
        self._stdout_lines: list[str] = []
        self._captured: dict[str, set[int]] = {s: set() for s in STREAM_FOLDERS}

    # ------------------------------------------------------------------
    # 帧流（top/wrist 固定帧率+边界补帧；depth 仅边界。按 (流, 仿真步) 去重）
    # ------------------------------------------------------------------
    def capture_frame(self, sim_step: int, frames: dict[str, np.ndarray]):
        """把一个仿真步的图像存入帧流；该流该步已存过则跳过（去重）。

        frames: {"top": rgb, "wrist": rgb, "depth": colormap_rgb}，
        缺省键或 None 表示该路本次不存（如固定间隔帧不存 depth）。
        """
        for stream, img in frames.items():
            if img is None or sim_step in self._captured[stream]:
                continue
            self._save_image(img, os.path.join(stream, f"s{sim_step:05d}"))
            self._captured[stream].add(sim_step)

    def frame_links(self, sim_step: int) -> dict[str, str]:
        """某仿真步真实存在的帧流文件链接（depth 仅边界步有）。"""
        return {stream: os.path.join("images", stream, f"s{sim_step:05d}.jpg")
                for stream in STREAM_FOLDERS if sim_step in self._captured[stream]}

    # ------------------------------------------------------------------
    # 调用记录
    # ------------------------------------------------------------------
    def log_stdout(self, text: str):
        self._stdout_lines.append(text)

    def record(
        self,
        name: str,
        inputs: dict,
        outputs: Any,
        observation: dict,
        sim_step_before: int,
        sim_step_after: int,
        annotation: np.ndarray | None = None,
        collision_events: list[dict] | None = None,
    ):
        seq = len(self.trace.records)
        annot_path = None
        if annotation is not None:
            folder = ALGO_FOLDERS.get(name, name)
            annot_path = self._save_image(annotation, os.path.join(folder, f"{seq:03d}_{name}"))
        moved = sim_step_after > sim_step_before
        self.trace.records.append(
            TraceRecord(
                seq=seq,
                name=name,
                category=CATEGORIES.get(name, "unknown"),
                sim_step_before=sim_step_before,
                sim_step_after=sim_step_after,
                wall_time=round(time.time() - self._t0, 3),
                inputs=summarize(inputs),
                outputs=summarize(outputs),
                observation=summarize(observation),
                visual_evidence=self.frame_links(sim_step_before),
                visual_evidence_after=self.frame_links(sim_step_after) if moved else None,
                annotation=annot_path,
                collision_events=collision_events or [],
            )
        )

    def _save_image(self, rgb: np.ndarray, rel_stem: str) -> str:
        """存图到 images/<rel_stem>.jpg（按需建子目录），返回相对 trace 目录路径。"""
        import cv2

        os.makedirs(os.path.join(self.img_root, os.path.dirname(rel_stem)), exist_ok=True)
        path = os.path.join(self.img_root, f"{rel_stem}.jpg")
        cv2.imwrite(path, cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 80])
        return os.path.relpath(path, self.trace_dir)

    def finalize(self, success: bool, error: str | None, total_sim_steps: int) -> str:
        self.trace.success = success
        self.trace.error = error
        self.trace.total_sim_steps = total_sim_steps
        self.trace.stdout = "".join(self._stdout_lines)
        path = os.path.join(self.trace_dir, "trace.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(asdict(self.trace), f, ensure_ascii=False, indent=2)
        return path
