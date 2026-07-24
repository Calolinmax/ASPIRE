"""ASPIRE 执行 trace 记录模块。

按 AGENTS.md 第 3 节要求：执行引擎记录每一次 primitive 调用的
【类别：检测/规划/抓取/控制】×【观测、输入、输出、视觉证据】。
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
    # 规划 (Planning)
    "solve_ik": "planning",
    "interpolate_segment": "planning",
    # 抓取 (Grasping)
    "open_gripper": "grasping",
    "close_gripper": "grasping",
    # 控制调用 (Control)
    "move_to_pose": "control",
    "move_to_joints": "control",
}


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
    visual_evidence: str | None # 调用时图像路径（相对 trace 目录）
    visual_evidence_after: str | None = None  # 运动类结束帧


@dataclass
class RunTrace:
    task: str
    seed: int
    success: bool = False
    error: str | None = None
    total_sim_steps: int = 0
    stdout: str = ""
    records: list[TraceRecord] = field(default_factory=list)


class Tracer:
    """收集一次任务执行的全部 primitive 调用记录。"""

    def __init__(self, trace_dir: str, task: str, seed: int):
        self.trace_dir = trace_dir
        self.img_dir = os.path.join(trace_dir, "images")
        os.makedirs(self.img_dir, exist_ok=True)
        self.trace = RunTrace(task=task, seed=seed)
        self._t0 = time.time()
        self._stdout_lines: list[str] = []

    def log_stdout(self, text: str):
        self._stdout_lines.append(text)

    def record(
        self,
        name: str,
        inputs: dict,
        outputs: Any,
        observation: dict,
        image_before: np.ndarray | None,
        sim_step_before: int,
        sim_step_after: int,
        image_after: np.ndarray | None = None,
    ):
        seq = len(self.trace.records)
        img_path = self._save_image(image_before, f"{seq:03d}_{name}_before") if image_before is not None else None
        img_after = self._save_image(image_after, f"{seq:03d}_{name}_after") if image_after is not None else None
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
                visual_evidence=img_path,
                visual_evidence_after=img_after,
            )
        )

    def _save_image(self, rgb: np.ndarray, stem: str) -> str:
        import cv2

        path = os.path.join(self.img_dir, f"{stem}.jpg")
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
