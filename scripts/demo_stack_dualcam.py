#!/usr/bin/env python
"""ASPIRE Primitive API 测试 Demo —— 双相机 SAM3 引导 Stack

用途：验证 docs/primitive_api.md 封装的全部 14 个 Primitive API 是否可用，
重点检验 SAM3 GPU 分割能否正确定位红/绿方块并引导机械臂完成堆叠。

任务代码（scripts/demo_stack_taskcode.py）改编自 ASPIRE 官方公开示例
open_details/primitive_api_cube_reset.py（红/绿方块重堆叠），相机配置与
LIBERO / OpenVLA-OFT 开源惯例一致：agentview(第三人称) + robot0_eye_in_hand(腕部)。

运行（AGENTS.md：必须 ASPIRE 环境（py3.12）；无头服务器用 EGL）：

    MUJOCO_GL=egl /home/stouching/anaconda3/envs/ASPIRE/bin/python scripts/demo_stack_dualcam.py
    MUJOCO_GL=egl ... --seeds 0,1,2     # 多 seed 批量
    MUJOCO_GL=egl ... --render          # 可视化窗口（需显示器）

产物（AGENTS.md §3：每步记录【观测、输入、输出、视觉证据】）：
    traces/<MMDD_HHMM>_Stack/                  一次任务一个目录
      trace.json                               全部 primitive 调用记录（图像以链接关联，
                                               code_ref 记录任务代码路径，代码由 harness 管理）
      images/top|wrist/s*.jpg                  双视角帧流（固定帧率，文件名=仿真步）
      images/depth/s*.jpg                      顶部深度（仅 API 调用边界保存）
      images/sam3|molmo|mask_to_world/*.jpg    各视觉算法标注图（仅调用时保存）
"""

import argparse
import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")  # 无头服务器默认 EGL（已设置则尊重原值）
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from aspire.engine import ExecutionEngine
from aspire.trace import CATEGORIES

# docs/primitive_api.md 注入的全部 API（覆盖统计以它为准）
ALL_APIS = [
    "get_observation",
    "segment_sam3_text_prompt",
    "segment_sam3_point_prompt",
    "point_prompt_molmo",
    "mask_to_world_points",
    "get_oriented_bounding_box_from_3d_points",
    "pixel_to_world_point",
    "solve_ik",
    "move_to_joints",
    "move_to_pose",
    "interpolate_segment",
    "open_gripper",
    "close_gripper",
    "rotation_matrix_to_quaternion",
]

DEFAULT_TASK_CODE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                 "demo_stack_taskcode.py")


def run_one(seed, code, trace_root, code_ref=None, render=False, slow=1.0):
    engine = ExecutionEngine(task="Stack", seed=seed, trace_root=trace_root, render=render,
                             render_slowdown=slow)
    t0 = time.time()
    try:
        result = engine.run(code, code_ref=code_ref)
    finally:
        engine.close()
    result["wall_time"] = round(time.time() - t0, 1)
    return result


def api_coverage(trace_path):
    """从 trace.json 统计每个 API 的实际调用次数（引擎记录即权威证据）。"""
    with open(trace_path, encoding="utf-8") as f:
        trace = json.load(f)
    counts = {}
    for rec in trace["records"]:
        counts[rec["name"]] = counts.get(rec["name"], 0) + 1
    return counts


def print_report(seed, result, counts):
    print("-" * 68)
    print(f"SEED {seed} 结果: success={result['success']}  "
          f"sim_steps={result['sim_steps']}  wall={result['wall_time']}s  "
          f"primitive_calls={result['n_primitive_calls']}")
    print("Primitive API 覆盖（依据 trace 记录）:")
    order = ["detection", "planning", "grasping", "control"]
    for cat in order:
        names = [n for n in ALL_APIS if CATEGORIES.get(n) == cat]
        for n in names:
            c = counts.get(n, 0)
            mark = "OK " if c > 0 else "-- "
            print(f"  [{mark}] {n:42s} {cat:10s} × {c}")
    missing = [n for n in ALL_APIS if counts.get(n, 0) == 0]
    print(f"未覆盖 API: {'无' if not missing else ', '.join(missing)}")
    print(f"trace  : {result['trace_path']}")
    print(f"算法标注: {os.path.join(result['trace_dir'], 'images')}"
          "（sam3/molmo/mask_to_world/，帧流在 top|wrist|depth/）")


def main():
    ap = argparse.ArgumentParser(description="ASPIRE Primitive API 双相机 Stack 测试 Demo")
    ap.add_argument("--seeds", default="0", help="逗号分隔，如 0,1,2")
    ap.add_argument("--task-code", default=DEFAULT_TASK_CODE)
    ap.add_argument("--trace-root", default="traces")
    ap.add_argument("--render", action="store_true", help="打开仿真窗口（需显示器）")
    ap.add_argument("--slow", type=float, default=1.0, help="演示减速倍率（如 2 = 半速）")
    args = ap.parse_args()

    with open(args.task_code, encoding="utf-8") as f:
        code = f.read()

    # 先预热 SAM3（模型缺失/损坏会在这里立刻暴露，而不是污染第一次计时）
    from aspire.vision_sam3 import warmup
    warmup()

    seeds = [int(s) for s in args.seeds.split(",")]
    n_ok = 0
    for seed in seeds:
        print(f"\n{'=' * 68}\n运行 Stack seed={seed}\n{'=' * 68}")
        result = run_one(seed, code, args.trace_root, code_ref=args.task_code,
                         render=args.render, slow=args.slow)
        print(result["stdout"])
        if result["error"]:
            print("--- 任务代码异常 ---")
            print(result["error"])
        counts = api_coverage(result["trace_path"])
        print_report(seed, result, counts)
        n_ok += int(result["success"])

    print(f"\n{'=' * 68}")
    print(f"汇总: {n_ok}/{len(seeds)} 成功")
    print(f"{'=' * 68}")
    raise SystemExit(0 if n_ok == len(seeds) else 1)


if __name__ == "__main__":
    main()
