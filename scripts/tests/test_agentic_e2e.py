# agentic coding 端到端冒烟（真实引擎 × mock LLM 重放已验证任务代码）。
# 运行: /home/stouching/anaconda3/envs/ASPIRE/bin/python scripts/tests/test_agentic_e2e.py
#
# 链路：Coordinator(actor 模式) → fast path 评估 baseline(scripts/tasks/stack.py)
# → debug seed 0 真实执行 → Stage 2 held-out seed 1 真实执行 → mock LLM 写 findings
# → mock 审计 → skill 入库真实 skill_library（测试后清理冒烟条目）。
# 前置：CGN(:8117)/pyroki(:8116) 在线；vision_server 由 harness 自动拉起。

import json
import os
import sys
import time

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from aspire.agentic import llm_client  # noqa: E402
from aspire.agentic.config import LLMConfig  # noqa: E402
from aspire.agentic.coordinator import Coordinator  # noqa: E402
from aspire.agentic.task_spec import stack_spec  # noqa: E402
from aspire.skills.library import SkillLibrary  # noqa: E402

SMOKE_SKILL = "e2e-smoke-skill"


def mock_handler(messages):
    text = json.dumps(messages, ensure_ascii=False)
    if "coordinator of the ASPIRE skill library" in text:
        # coordinator 审计：入库一条冒烟 skill（合规代码草图）
        return json.dumps({
            "decision": "admit", "rationale": "e2e smoke",
            "entries": [{"name": SMOKE_SKILL, "category": "debugging",
                         "description": "e2e 冒烟条目",
                         "problem": "p", "when_to_apply": "w", "strategy": "s",
                         "code_sketch": "obs = get_observation()\nprint(len(obs))",
                         "evidence": "e2e", "origin_tasks": ["Stack"]}]})
    if "reporting structured findings" in text:
        return ("## Task: Stack\n### Root Cause(s)\n- e2e smoke (no failure)\n"
                "### What Fixed It\n- baseline verbatim\n"
                "### Generalizable Patterns\n- none\n### Stage 2 Success Rate\n1/1\n")
    # 初始生成/修复：重放已验证的 stack.py
    with open(os.path.join(REPO_ROOT, "scripts", "tasks", "stack.py"), encoding="utf-8") as f:
        return "```python\n" + f.read() + "\n```"


def main():
    llm_client.set_mock_handler(mock_handler)
    cfg = LLMConfig(provider="mock", model="mock-e2e")
    spec = stack_spec(official=True)
    spec.debug_seeds = [0]
    spec.heldout_seeds = [1]
    spec.time_budget_per_trial = 900

    coord = Coordinator(cfg, library_root=os.path.join(REPO_ROOT, "skill_library"),
                        run_root=os.path.join(REPO_ROOT, "agent_runs"), mode="actor",
                        max_workers=2,
                        event_cb=lambda t, p: print(f"[{time.strftime('%H:%M:%S')}] "
                                                    f"[{t}] {p.get('text') or p.get('detail') or p.get('task') or ''}",
                                                    flush=True))
    results = coord.run([spec])
    r = results["Stack"]
    print("\n===== E2E RESULT =====")
    print(json.dumps({k: v for k, v in r.items()}, ensure_ascii=False, indent=2, default=str))

    ok = r["status"] == "solved" and r["stage2_rate"] == "1/1"
    admitted = any(a["name"] == SMOKE_SKILL for a in r["admission"]["admitted"])
    print(f"actor solved+stage2: {ok}; 冒烟 skill 入库: {admitted}")

    # 清理冒烟条目（不污染真实库）
    lib = SkillLibrary(os.path.join(REPO_ROOT, "skill_library"))
    entry = lib.get(SMOKE_SKILL)
    if entry is not None:
        os.remove(os.path.join(lib.root, entry.category, f"{SMOKE_SKILL}.md"))
        index = lib._load_index()
        index.get("skills", {}).pop(SMOKE_SKILL, None)
        lib._save_index(index)
        print("冒烟条目已清理")
    sys.exit(0 if ok and admitted else 1)


if __name__ == "__main__":
    main()
