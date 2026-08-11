"""agentic coding 模块 CLI（python -m aspire.agentic.cli）。

子命令：
- actor      单任务修复闭环（E.3）：--task Stack [--no-baseline] [--rounds 3]
- evosearch  单任务进化搜索（Algorithm 1 + E.4）：--task Stack [-K 4] [-T 5]
- coordinator 多任务分派 + findings 审计入库（E.1）：--tasks Stack,PiperWipeSpill
- skills     技能库管理：list / show <name> / admit <findings.md> --task <name>
- digest     调试工具：trace.json → 摘要文本（看 LLM 实际吃到什么）

LLM 配置走环境变量 / 仓库根 .env（见 config.py）；--provider 等可命令行覆盖。
"""

from __future__ import annotations

import argparse
import json
import os
import sys

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _print_event(etype: str, payload: dict) -> None:
    """默认事件出口：终端单行打印（web 模式由 server 替换为 WS 广播）。"""
    summary = payload.get("text") or payload.get("detail") or payload.get("task") or ""
    print(f"[event:{etype}] {summary}", flush=True)


def _build_cfg(args) -> "LLMConfig":
    from .config import LLMConfig
    overrides = {}
    if getattr(args, "provider", None):
        overrides["provider"] = args.provider
    if getattr(args, "model", None):
        overrides["model"] = args.model
    if getattr(args, "base_url", None):
        overrides["base_url"] = args.base_url
    cfg = LLMConfig.from_env(**overrides)
    cfg.validate()
    return cfg


def _build_spec(args) -> "TaskSpec":
    from .task_spec import BUILTIN_TASKS, TaskSpec
    if args.task in BUILTIN_TASKS:
        spec = BUILTIN_TASKS[args.task]()
    elif args.spec_json:
        spec = TaskSpec.from_json(args.spec_json)
    else:
        raise SystemExit(f"未知任务 {args.task}：用内置名 {list(BUILTIN_TASKS)} 或 --spec-json")
    if getattr(args, "seeds", None):
        spec.debug_seeds = [int(s) for s in args.seeds.split(",")]
    if getattr(args, "heldout", None):
        spec.heldout_seeds = [int(s) for s in args.heldout.split(",")]
    if getattr(args, "no_baseline", False):
        spec.baseline_code = None
    if getattr(args, "baseline_path", None):
        spec.baseline_code = args.baseline_path
    return spec


def _add_common(p: argparse.ArgumentParser) -> None:
    p.add_argument("--provider", choices=["openai", "anthropic", "file", "mock"])
    p.add_argument("--model")
    p.add_argument("--base-url")
    p.add_argument("--skill-root", default=os.path.join(REPO_ROOT, "skill_library"))
    p.add_argument("--run-root", default=os.path.join(REPO_ROOT, "agent_runs"))
    p.add_argument("--max-workers", type=int, default=4,
                   help="并行评估的子进程上限（AGENTS.md §2 并行要求；GPU 共享，默认 4）")


def main() -> None:
    parser = argparse.ArgumentParser(prog="aspire.agentic",
                                   description="ASPIRE agentic coding 模块")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_actor = sub.add_parser("actor", help="单任务修复闭环（E.3）")
    p_actor.add_argument("--task", default="Stack")
    p_actor.add_argument("--spec-json")
    p_actor.add_argument("--no-baseline", action="store_true")
    p_actor.add_argument("--baseline-path", help="自定义基线程序路径（续跑/换种子用）")
    p_actor.add_argument("--rounds", type=int, default=3, help="E.3: 每 seed 最多 3 次 replay")
    p_actor.add_argument("--theta", type=float, default=1.0)
    p_actor.add_argument("--seeds", help="debug seeds，逗号分隔，默认 0,1,2")
    p_actor.add_argument("--heldout", help="held-out seeds，逗号分隔，默认 3-7")
    _add_common(p_actor)

    p_evo = sub.add_parser("evosearch", help="进化搜索（Algorithm 1 + E.4）")
    p_evo.add_argument("--task", default="Stack")
    p_evo.add_argument("--spec-json")
    p_evo.add_argument("--no-baseline", action="store_true")
    p_evo.add_argument("--baseline-path", help="自定义基线程序路径（candidate_A 种子）")
    p_evo.add_argument("-K", type=int, default=4, help="每代候选数")
    p_evo.add_argument("-T", type=int, default=5, help="最大代数")
    p_evo.add_argument("--theta", type=float, default=1.0, help="debug 成功率早停阈值")
    p_evo.add_argument("--seeds")
    p_evo.add_argument("--heldout")
    _add_common(p_evo)

    p_coord = sub.add_parser("coordinator", help="多任务分派 + 入库审计（E.1）")
    p_coord.add_argument("--tasks", default="Stack", help="逗号分隔任务名")
    p_coord.add_argument("--mode", choices=["actor", "evosearch"], default="actor")
    p_coord.add_argument("--no-baseline", action="store_true")
    _add_common(p_coord)

    p_skills = sub.add_parser("skills", help="技能库管理")
    p_skills.add_argument("action", choices=["list", "show", "admit"])
    p_skills.add_argument("arg", nargs="?", help="show: skill 名；admit: findings.md 路径")
    p_skills.add_argument("--task", help="admit 时的来源任务名")
    p_skills.add_argument("--category")
    _add_common(p_skills)

    p_digest = sub.add_parser("digest", help="trace.json → LLM 摘要（调试）")
    p_digest.add_argument("trace_path")
    p_digest.add_argument("--max-images", type=int, default=0)

    args = parser.parse_args()

    if args.cmd == "digest":
        from .trace_digest import build_digest
        d = build_digest(args.trace_path, max_images=args.max_images)
        print(d.summary_text)
        print(f"\n[implicated seqs: {d.implicated_seqs}; content parts: {len(d.content_parts)}]")
        return

    cfg = _build_cfg(args)

    if args.cmd == "skills":
        from ..skills.library import SkillLibrary
        lib = SkillLibrary(args.skill_root)
        if args.action == "list":
            for meta in lib.list(category=args.category):
                print(f"[{meta['status']:9s}] {meta['category']}/{meta['name']}  "
                      f"{meta.get('description', '')[:80]}")
        elif args.action == "show":
            entry = lib.get(args.arg)
            if entry is None:
                raise SystemExit(f"skill 不存在: {args.arg}")
            print(entry.to_markdown())
        elif args.action == "admit":
            from ..skills.synthesize import synthesize_from_findings
            report = synthesize_from_findings(cfg, args.arg, lib, args.task or "(unknown)")
            print(json.dumps(report, ensure_ascii=False, indent=2))
        return

    from ..skills.library import SkillLibrary
    from .evaluate import shutdown_services
    library = SkillLibrary(args.skill_root)
    try:
        if args.cmd == "actor":
            from .actor import Actor
            spec = _build_spec(args)
            result = Actor(cfg, library, run_root=args.run_root, max_rounds=args.rounds,
                           theta=args.theta, max_workers=args.max_workers,
                           event_cb=_print_event).run(spec)
            print(json.dumps({"task": result.task, "status": result.status,
                              "stage1_rate": result.stage1_rate,
                              "stage2_rate": result.stage2_rate,
                              "best_code": result.best_code_path,
                              "findings": result.findings_path,
                              "run_dir": result.run_dir}, ensure_ascii=False, indent=2))
            sys.exit(0 if result.status == "solved" else 1)
        elif args.cmd == "evosearch":
            from .evolve import EvolutionarySearch
            spec = _build_spec(args)
            result = EvolutionarySearch(cfg, library, run_root=args.run_root,
                                        K=args.K, T=args.T, theta=args.theta,
                                        max_workers=args.max_workers,
                                        event_cb=_print_event).run(spec)
            print(json.dumps({"task": result.task, "status": result.status,
                              "iterations": result.iterations,
                              "best_debug_rate": result.best_debug_rate,
                              "stage2_rate": result.stage2_rate,
                              "best_code": result.best_code_path,
                              "findings": result.findings_path,
                              "run_dir": result.run_dir}, ensure_ascii=False, indent=2))
            sys.exit(0 if result.status == "solved" else 1)
        elif args.cmd == "coordinator":
            from .coordinator import Coordinator
            from .task_spec import BUILTIN_TASKS
            specs = []
            for name in args.tasks.split(","):
                name = name.strip()
                if name not in BUILTIN_TASKS:
                    raise SystemExit(f"未知任务 {name}（内置: {list(BUILTIN_TASKS)}）")
                spec = BUILTIN_TASKS[name]()
                if args.no_baseline:
                    spec.baseline_code = None
                specs.append(spec)
            results = Coordinator(cfg, library_root=args.skill_root,
                                  run_root=args.run_root, mode=args.mode,
                                  max_workers=args.max_workers,
                                  event_cb=_print_event).run(specs)
            print(json.dumps(results, ensure_ascii=False, indent=2, default=str))
    finally:
        shutdown_services()


if __name__ == "__main__":
    main()
