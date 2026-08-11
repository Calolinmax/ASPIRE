"""Agent 任务规格（actor/evolve 的输入）。

TaskSpec 把"一个可修复/可进化任务"的全部要素固化下来：
场景（引擎 --task 参数）+ 自然语言指令 + 成功判定说明 + debug/held-out seed 划分
（论文 §3.3：两个不相交 seed 集）+ 可选基线程序（E.3 fast path / E.4 candidate_A 种子）。
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


@dataclass
class TaskSpec:
    name: str                          # 展示名，如 "Stack"
    env_task: str                      # 引擎 --task 参数（场景类名），如 "Stack" / "PiperWipeSpill"
    instruction: str                   # 给 actor 的自然语言任务描述
    success_criteria: str              # 成功判定语义（env._check_success 的人类可读说明）
    debug_seeds: list[int] = field(default_factory=lambda: [0, 1, 2])
    heldout_seeds: list[int] = field(default_factory=lambda: [3, 4, 5, 6, 7])
    extra_engine_args: list[str] = field(default_factory=list)  # 如 ["--official-stack"]
    baseline_code: str | None = None   # 基线程序路径（fast path 种子；None=从零生成）
    time_budget_per_trial: int = 900   # 单 seed 执行超时（秒）

    def to_json(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, path: str) -> "TaskSpec":
        with open(path, encoding="utf-8") as f:
            return cls(**json.load(f))


# ---------------------------------------------------------------------------
# 内置任务规格（与 scripts/tasks/ 已验证任务对齐）
# ---------------------------------------------------------------------------

def stack_spec(official: bool = True, **kw) -> TaskSpec:
    return TaskSpec(
        name="Stack",
        env_task="Stack",
        instruction=(
            "Pick up the red cube (cubeA) and place it stably on top of the green cube "
            "(cubeB). The cubes are 4cm cubes on a tabletop. Use SAM3 text prompts to "
            "localize both cubes, plan a CGN grasp for the red cube, transport it, and "
            "release it centered on the green cube's top face."
        ),
        success_criteria=(
            "robosuite 官方 Stack._check_success：红块在绿块上方且接触、夹爪已松开。"
            "引擎 run 末尾自动判定，任务代码无需自验（可打印诊断）。"
        ),
        extra_engine_args=["--official-stack"] if official else [],
        baseline_code=os.path.join(REPO_ROOT, "scripts", "tasks", "stack.py"),
        **kw,
    )


def wipe_spec(**kw) -> TaskSpec:
    return TaskSpec(
        name="PiperWipeSpill",
        env_task="PiperWipeSpill",
        instruction=(
            "Wipe the brown spill on the tabletop with the wiper gripper. Localize the "
            "spill with SAM3 (fallback: molmo point -> point-prompt segmentation), compute "
            "its world-space bounding box, then execute a dense serpentine wiping path "
            "covering the spill area at table contact height."
        ),
        success_criteria=(
            "PiperWipeSpill._check_success：擦板足迹矩形访问覆盖率 >= 50% "
            "（_get_observations 侧效应统计）。"
        ),
        baseline_code=os.path.join(REPO_ROOT, "scripts", "tasks", "wipe.py"),
        **kw,
    )


BUILTIN_TASKS = {"Stack": stack_spec, "PiperWipeSpill": wipe_spec}
