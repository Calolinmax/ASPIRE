"""ASPIRE 执行引擎（Execution Engine）。

职责（AGENTS.md 第 3 节）：执行 coding agent 生成的任务代码，并记录每一次
primitive 调用的【观测、输入、输出、视觉证据】。

用法：
    engine = ExecutionEngine(task="Lift", seed=0)
    result = engine.run(code_string)

CLI（子进程隔离执行，供 harness 调用）：
    MUJOCO_GL=egl python -m aspire.engine --code task_code.py --task Lift --seed 0
"""

from __future__ import annotations

import argparse
import contextlib
import io
import os
import time
import traceback

import numpy as np
import robosuite as suite

from .primitives import build_namespace
from .trace import Tracer

ENV_DEFAULTS = dict(
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=256,
    camera_widths=256,
    camera_depths=True,
    control_freq=20,
    horizon=1000,  # 足够多次 primitive 调用（move_to_pose 单次可达 160 步）
)


class ExecutionEngine:
    def __init__(self, task: str = "Lift", seed: int = 0, trace_root: str = "traces"):
        self.task = task
        self.seed = seed
        np.random.seed(seed)
        self.env = suite.make(task, **ENV_DEFAULTS)
        self.obs = self.env.reset()
        self.sim_step = 0
        self.gripper_cmd = -1.0  # 初始张开
        self.terminated = False
        ts = time.strftime("%Y%m%d_%H%M%S")
        self.trace_dir = os.path.join(trace_root, f"{ts}_{task}_seed{seed}")
        self.tracer = Tracer(self.trace_dir, task=task, seed=seed)

    def step(self, action: np.ndarray):
        """primitives 的唯一仿真入口：步进并更新观测。"""
        if self.terminated:
            return  # episode 已终止（如 horizon 用尽），静默忽略后续 step
        self.obs, _, done, _ = self.env.step(action)
        self.sim_step += 1
        if done:
            self.terminated = True

    def current_rgb(self) -> np.ndarray | None:
        img = self.obs.get("agentview_image")
        return img.copy() if img is not None else None

    def run(self, code: str) -> dict:
        """执行任务代码，返回结果摘要并落盘 trace。"""
        ns = build_namespace(self)
        buf = io.StringIO()
        error = None
        try:
            with contextlib.redirect_stdout(buf):
                exec(compile(code, "<task_code>", "exec"), ns)
        except Exception:
            error = traceback.format_exc()
        stdout_text = buf.getvalue()
        if error is None:
            try:
                success = bool(self.env._check_success())
            except Exception:
                success = False
        else:
            success = False
        self.tracer.log_stdout(stdout_text)
        trace_path = self.tracer.finalize(success, error, self.sim_step)
        return {
            "success": success,
            "error": error,
            "stdout": stdout_text,
            "sim_steps": self.sim_step,
            "trace_dir": self.trace_dir,
            "trace_path": trace_path,
            "n_primitive_calls": len(self.tracer.trace.records),
        }

    def close(self):
        self.env.close()


def main():
    parser = argparse.ArgumentParser(description="ASPIRE Execution Engine")
    parser.add_argument("--code", required=True, help="任务代码 .py 文件路径")
    parser.add_argument("--task", default="Lift")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-root", default="traces")
    args = parser.parse_args()

    with open(args.code, encoding="utf-8") as f:
        code = f.read()

    engine = ExecutionEngine(task=args.task, seed=args.seed, trace_root=args.trace_root)
    try:
        result = engine.run(code)
    finally:
        engine.close()

    print("=== Execution Result ===")
    print(f"success        : {result['success']}")
    print(f"sim_steps      : {result['sim_steps']}")
    print(f"primitive_calls: {result['n_primitive_calls']}")
    print(f"trace          : {result['trace_path']}")
    if result["stdout"]:
        print("--- stdout ---")
        print(result["stdout"])
    if result["error"]:
        print("--- error ---")
        print(result["error"])
    # 供 harness 解析的退出码
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
