# agentic coding + skill library 单元测试（不碰仿真、不调真实 LLM）。
# 运行: /home/stouching/anaconda3/envs/ASPIRE/bin/python scripts/tests/test_agentic.py
#
# 覆盖：schema 往返 / API 合规 / 库 CRUD 与并发串行 / trace_digest 失败信号 /
# llm_client（fence 抽取/collapse/mock/集成综合/anthropic 转换）/ prompts 构造 /
# actor 修复闭环（fake executor）/ evolve Algorithm 1 逐行行为 / coordinator 入库审计。

import json
import os
import sys
import tempfile

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)

from aspire.agentic import llm_client  # noqa: E402
from aspire.agentic.config import LLMConfig  # noqa: E402
from aspire.agentic.task_spec import TaskSpec  # noqa: E402
from aspire.skills.library import SkillLibrary  # noqa: E402
from aspire.skills.schema import SkillEntry, parse_findings  # noqa: E402
from aspire.skills.synthesize import check_api_compliance  # noqa: E402

PASS = []


def check(name, cond, detail=""):
    PASS.append((name, bool(cond)))
    print(f"  [{'PASS' if cond else 'FAIL'}] {name}" + (f"  {detail}" if detail and not cond else ""))
    if not cond:
        raise AssertionError(f"{name}: {detail}")


def mock_cfg():
    return LLMConfig(provider="mock", model="mock")


# ---------------------------------------------------------------------------
def t_schema():
    print("== A. skill schema ==")
    e = SkillEntry(name="a-b", category="grasping", problem="P", when_to_apply="- x",
                   strategy="S", code_sketch="print(np.zeros(2))", evidence="seed 0 OK",
                   origin_tasks=["Stack"], validation_debug="3/3")
    e2 = SkillEntry.from_markdown(e.to_markdown())
    check("A1 往返 name/category", e2.name == "a-b" and e2.category == "grasping")
    check("A2 往返 code_sketch", e2.code_sketch.strip() == "print(np.zeros(2))")
    check("A3 往返 origin/validation", e2.origin_tasks == ["Stack"] and e2.validation_debug == "3/3")
    check("A4 evidence 不被 sketch 污染", "Code Sketch" not in e2.evidence)
    f = parse_findings("## Task: Stack\n### Root Cause(s)\n- rc1\n### What Fixed It\n- fix\n"
                       "### Stage 2 Success Rate\n4/5\n")
    check("A5 findings 解析", f["Root Cause(s)"] == "- rc1" and f["Stage 2 Success Rate"] == "4/5")


def t_compliance():
    print("== B. API 合规静态检查 ==")
    check("B1 import 违规", any("import" in v for v in check_api_compliance("import os\nprint(1)")))
    check("B2 臆造 API 违规", check_api_compliance("goto_pose([0,0,0])") != [])
    ok_code = ("def helper(p):\n    return np.asarray(p) * 2\n"
               "q = solve_ik(helper([0.3, 0, 0.5]), np.array([0., 0., 1., 0.]))\n"
               "execute_legs_rrt([q])\nobs = get_observation()\nprint(len(obs))")
    check("B3 契约+helper 合规", check_api_compliance(ok_code) == [])
    check("B4 语法错误响亮报出", "语法错误" in check_api_compliance("def broken(:\n")[0])


def t_library():
    print("== C. SkillLibrary ==")
    root = tempfile.mkdtemp()
    lib = SkillLibrary(root)
    e = SkillEntry(name="s1", category="c1", problem="P", when_to_apply="W", strategy="S",
                   origin_tasks=["A"])
    lib.admit(e)
    check("C1 admit+get", lib.get("s1").status == "admitted")
    e2 = SkillEntry(name="s1", category="c1", problem="P2", when_to_apply="W", strategy="S",
                    origin_tasks=["B"])
    lib.admit(e2)
    check("C2 同名更新 origin 并集", lib.get("s1").origin_tasks == ["A", "B"])
    check("C3 format_for_prompt 注入", "s1" in lib.format_for_prompt())
    lib.retire("s1")
    check("C4 retire 移出注入", "为空" in lib.format_for_prompt())
    # 并发 admit 串行化（文件锁）
    import threading
    lib2 = SkillLibrary(root)
    def add(i):
        lib2.admit(SkillEntry(name=f"c{i}", category="c", problem="P",
                              when_to_apply="W", strategy="S"))
    threads = [threading.Thread(target=add, args=(i,)) for i in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    check("C5 并发 admit 6 条全在", len(lib2.list(status="admitted")) == 6)
    r = lib2.retrieve("grasp 抓取", k=3)
    check("C6 retrieve 不炸", isinstance(r, list))


def t_digest():
    print("== D. trace_digest ==")
    from aspire.agentic.trace_digest import build_digest
    d = tempfile.mkdtemp()
    os.makedirs(os.path.join(d, "images", "top"), exist_ok=True)
    import cv2
    import numpy as np
    img = np.zeros((8, 8, 3), dtype=np.uint8)
    cv2.imwrite(os.path.join(d, "images", "top", "s00000.jpg"), img)
    trace = {
        "task": "Stack", "seed": 0, "success": False, "error": "RuntimeError: solve_ik 不收敛\n"
        "  File \"<task_code>\", line 9\nRuntimeError: IK failed",
        "total_sim_steps": 100, "code_ref": "x.py", "stdout": "log tail",
        "records": [
            {"seq": 0, "name": "segment_sam3_text_prompt", "category": "detection",
             "sim_step_before": 0, "sim_step_after": 0, "wall_time": 1.0,
             "inputs": {"text_prompt": "red cube"}, "outputs": [], "observation": {},
             "visual_evidence": {"top": "images/top/s00000.jpg"},
             "visual_evidence_after": None, "annotation": None, "collision_events": []},
            {"seq": 1, "name": "move_to_joints", "category": "control",
             "sim_step_before": 0, "sim_step_after": 50, "wall_time": 2.0,
             "inputs": {}, "outputs": None, "observation": {},
             "visual_evidence": {"top": "images/top/s00000.jpg"},
             "visual_evidence_after": {"top": "images/top/s00000.jpg"},
             "annotation": None, "collision_events": [{"pair": "arm-table"}]},
        ],
    }
    path = os.path.join(d, "trace.json")
    with open(path, "w") as f:
        json.dump(trace, f)
    dg = build_digest(path, max_images=4, window=1)
    check("D1 zero_masks 信号", "zero_masks" in dg.summary_text)
    check("D2 collision 信号", "collision" in dg.summary_text)
    check("D3 implicated 覆盖两条", dg.implicated_seqs == [0, 1])
    n_img = sum(1 for p in dg.content_parts if p.get("type") == "image_url")
    check("D4 图像 ≤ max_images", 0 < n_img <= 4, f"n={n_img}")
    dg2 = build_digest(path, max_images=0)
    check("D5 max_images=0 纯文本", all(p.get("type") == "text" for p in dg2.content_parts))


def t_llm():
    print("== E. llm_client ==")
    check("E1 extract_code 取最后一 fence",
          llm_client.extract_code("```python\na=1\n```\ntext\n```python\nb=2\n```") == "b=2\n")
    parts = [{"type": "text", "text": "a"}, {"type": "text", "text": "b"},
             {"type": "image_url", "image_url": {"url": "u"}},
             {"type": "text", "text": "c"}]
    collapsed = llm_client.collapse_text_image_inputs(parts)
    check("E2 collapse 文本合并图像保位",
          collapsed[0]["text"] == "a\nb\n" and collapsed[1]["type"] == "image_url")
    conv, msgs = llm_client._to_anthropic_messages([
        {"role": "system", "content": [{"type": "text", "text": "sys"}]},
        {"role": "user", "content": [{"type": "text", "text": "hi"},
         {"type": "image_url", "image_url": {"url": "data:image/png;base64,QUJD"}}]}])
    check("E3 anthropic 转换 system+image", conv == "sys" and
          msgs[0]["content"][1]["source"]["data"] == "QUJD")
    llm_client.set_mock_handler(lambda m: "```python\nprint(1)\n```")
    r = llm_client.query_model(mock_cfg(), [{"role": "user", "content": "x"}])
    check("E4 mock provider", "print(1)" in r["content"])
    r2 = llm_client.query_single_model_ensemble(mock_cfg(), [{"role": "user", "content": "x"}],
                                                temperatures=[0.1, 0.5, 0.9])
    check("E5 集成综合结构", "ensemble_candidates_txt" in r2 and
          r2["all_responses"][0]["ok"])
    check("E6 streaming 非 openai 降级",
          list(llm_client.query_model_streaming(mock_cfg(), [{"role": "user", "content": "x"}]))[-1]["type"] == "done")


def _fake_result(name, code_path, rate, seeds):
    from aspire.agentic.evaluate import ProgramResult, SeedResult
    return ProgramResult(name=name, code_path=code_path, rate=rate,
                         seed_results=[SeedResult(seed=s, success=rate >= 1.0,
                                                  exit_code=0, trace_path=None,
                                                  wall_time=0.1) for s in seeds])


def t_actor():
    print("== F. actor 修复闭环（fake executor + mock LLM） ==")
    import aspire.agentic.actor as actor_mod

    # F1: fast path——baseline 一轮达标，不进修复循环
    calls = {"eval": 0, "llm": 0}
    def fake_eval(name, code_path, env_task, seeds, trace_root, extra_args, timeout, max_workers):
        calls["eval"] += 1
        return _fake_result(name, code_path, 1.0, seeds)
    actor_mod.evaluate_program = fake_eval
    llm_client.set_mock_handler(lambda m: (calls.__setitem__("llm", calls["llm"] + 1),
                                           "## Task: T\n### Stage 2 Success Rate\n2/2\n")[1])
    lib = SkillLibrary(tempfile.mkdtemp())
    spec = TaskSpec(name="T", env_task="Stack", instruction="i", success_criteria="s",
                    debug_seeds=[0, 1], heldout_seeds=[2, 3],
                    baseline_code=__file__)  # 任意存在的文件当 baseline
    res = actor_mod.Actor(mock_cfg(), lib, run_root=tempfile.mkdtemp(),
                          max_rounds=3).run(spec)
    check("F1 fast path solved", res.status == "solved" and res.stage1_rate == 1.0)
    check("F1 评估两次（debug+heldout），LLM 仅 findings",
          calls["eval"] == 2 and calls["llm"] == 1, str(calls))
    check("F1 findings 落盘", os.path.isfile(res.findings_path))

    # F2: 无 baseline + 修复一轮达标（第 1 版失败 0.5，第 2 版 1.0）
    rates = iter([0.5, 1.0, 1.0])   # v1 debug, v2 debug, v2 heldout
    calls2 = {"llm_gen": 0}
    def fake_eval2(name, code_path, env_task, seeds, trace_root, extra_args, timeout, max_workers):
        return _fake_result(name, code_path, next(rates), seeds)
    actor_mod.evaluate_program = fake_eval2
    def handler2(messages):
        text = json.dumps(messages, ensure_ascii=False)
        if "findings" in text or "Root Cause" in text:
            return "## Task: T2\n### Root Cause(s)\n- x\n### Stage 2 Success Rate\n2/2\n"
        calls2["llm_gen"] += 1
        return "```python\nprint('v')\n```"
    llm_client.set_mock_handler(handler2)
    spec2 = TaskSpec(name="T2", env_task="Stack", instruction="i", success_criteria="s",
                     debug_seeds=[0, 1], heldout_seeds=[2, 3], baseline_code=None)
    res2 = actor_mod.Actor(mock_cfg(), lib, run_root=tempfile.mkdtemp(),
                           max_rounds=3).run(spec2)
    check("F2 修复闭环 solved", res2.status == "solved" and res2.stage2_rate == "2/2")
    check("F2 初始生成+修复+findings = 3 次 LLM", calls2["llm_gen"] == 2, str(calls2))
    v1 = os.path.join(res2.run_dir, "v1.py")
    check("F2 候选代码落盘", os.path.isfile(v1))

    # F3: 永不达标且 0 成功 → blocked（E.3：全 debug 受阻跳过 Stage 2）
    def fake_eval3(name, code_path, env_task, seeds, trace_root, extra_args, timeout, max_workers):
        return _fake_result(name, code_path, 0.0, seeds)
    actor_mod.evaluate_program = fake_eval3
    res3 = actor_mod.Actor(mock_cfg(), lib, run_root=tempfile.mkdtemp(),
                           max_rounds=3).run(spec2)
    check("F3 零成功=blocked 且跳过 Stage 2",
          res3.status == "blocked" and len(res3.history) == 3 and res3.stage2_rate == "")

    # F4: 部分成功但不达标 → max_rounds，仍带最佳版进 Stage 2
    def fake_eval4(name, code_path, env_task, seeds, trace_root, extra_args, timeout, max_workers):
        return _fake_result(name, code_path, 0.5, seeds)
    actor_mod.evaluate_program = fake_eval4
    res4 = actor_mod.Actor(mock_cfg(), lib, run_root=tempfile.mkdtemp(),
                           max_rounds=3).run(spec2)
    check("F4 部分成功=max_rounds+Stage 2 照跑",
          res4.status == "max_rounds" and res4.stage2_rate != "")


def t_evolve():
    print("== G. evolve Algorithm 1（fake executor + mock LLM） ==")
    import aspire.agentic.evolve as evo_mod

    eval_log = []
    def fake_pop(candidates, env_task, seeds, trace_root, extra_args, timeout, max_workers):
        # iter0 baseline 0.0；iter1: A(=baseline verbatim) 0.0, B 0.5, C 0.0；
        # iter2: A(=best verbatim) 0.5, B 1.0 → θ 早停；stage2: 2/2
        nonlocal_iter = len(eval_log)
        eval_log.append(sorted(candidates))
        out = {}
        for name, path in candidates.items():
            letter = name.split("_")[1] if "_" in name else "B"  # stage2 候选名 "best"
            if nonlocal_iter == 0:
                rate = 0.0
            elif nonlocal_iter == 1:
                rate = {"A": 0.0, "B": 0.5}.get(letter, 0.0)
            elif nonlocal_iter == 2:
                rate = {"A": 0.5, "B": 1.0}.get(letter, 0.0)
            else:
                rate = 1.0
            out[name] = _fake_result(name, path, rate, seeds)
        return out
    evo_mod.evaluate_population = fake_pop

    def handler(messages):
        text = json.dumps(messages, ensure_ascii=False)
        if "CANDIDATE" in text and "Output format" in text:
            return ("=== CANDIDATE A ===\n```python\nprint('seed')\n```\n"
                    "=== CANDIDATE B ===\n```python\nprint('b')\n```\n"
                    "=== CANDIDATE C ===\n```python\nprint('c')\n```\n")
        if "task analysis" in text.lower():
            return "# Task Analysis\n\nupdated"
        return "## Task: E\n### Stage 2 Success Rate\n2/2\n"
    llm_client.set_mock_handler(handler)

    lib = SkillLibrary(tempfile.mkdtemp())
    spec = TaskSpec(name="E", env_task="Stack", instruction="i", success_criteria="s",
                    debug_seeds=[0, 1], heldout_seeds=[2, 3], baseline_code=__file__)
    run_root = tempfile.mkdtemp()
    res = evo_mod.EvolutionarySearch(mock_cfg(), lib, run_root=run_root,
                                     K=3, T=5, theta=1.0).run(spec)
    check("G1 θ 早停 solved", res.status == "solved" and res.iterations == 2)
    check("G2 评估 4 轮（P0 + iter1 + iter2 + stage2）", len(eval_log) == 4, str(eval_log))
    check("G3 candidate_A verbatim 种子（iter2 A 含 baseline 内容）",
          "candidate_A" in eval_log[1] and "candidate_A" in eval_log[2])
    check("G4 evosearch_best_code.py 落盘", os.path.isfile(res.best_code_path))
    check("G5 findings + task_analysis + iter_summary 落盘",
          os.path.isfile(res.findings_path)
          and os.path.isfile(os.path.join(res.run_dir, "task_analysis.md"))
          and os.path.isfile(os.path.join(res.run_dir, "iter_01", "iter_summary.json")))
    with open(res.best_code_path) as f:
        check("G6 最优候选 = iter2 B 的代码", "print('b')" in f.read())


def t_coordinator():
    print("== H. coordinator 审计入库（fake actor + mock 审计） ==")
    import aspire.agentic.coordinator as coord_mod

    class FakeActorResult:
        task = "H"; status = "solved"; stage1_rate = 1.0; stage2_rate = "2/2"
        run_dir = ""; findings_path = ""
    findings_file = os.path.join(tempfile.mkdtemp(), "findings.md")
    with open(findings_file, "w") as f:
        f.write("## Task: H\n### Generalizable Patterns\n- tilt grasp\n"
                "### Stage 2 Success Rate\n2/2\n")

    class FakeActor:
        def __init__(self, cfg, library, **kw):
            pass
        def run(self, spec):
            r = FakeActorResult()
            r.findings_path = findings_file
            return r
    coord_mod.Actor = FakeActor

    llm_client.set_mock_handler(lambda m: json.dumps({
        "decision": "admit", "rationale": "transferable",
        "entries": [{"name": "tilt-grasp", "category": "grasping",
                     "description": "d", "problem": "p", "when_to_apply": "w",
                     "strategy": "s", "code_sketch": "q = solve_ik(np.zeros(3), np.array([0.,0.,1.,0.]))\nmove_to_joints(q)",
                     "evidence": "seed 0 OK", "origin_tasks": ["H"]}]}))
    lib_root = tempfile.mkdtemp()
    coord = coord_mod.Coordinator(mock_cfg(), library_root=lib_root,
                                  run_root=tempfile.mkdtemp())
    spec = TaskSpec(name="H", env_task="Stack", instruction="i", success_criteria="s")
    results = coord.run([spec])
    check("H1 actor 结果汇总", results["H"]["status"] == "solved")
    lib = SkillLibrary(lib_root)
    check("H2 skill 入库", lib.get("tilt-grasp") is not None
          and lib.get("tilt-grasp").status == "admitted")
    check("H3 progress.json 更新",
          os.path.isfile(coord.progress_path))
    # H4: 已完成任务不重复分派（E.3 Rule 5）
    results2 = coord.run([spec])
    check("H4 不重复分派", "H" not in results2)
    # H5: 不合规代码草图拒绝入库
    llm_client.set_mock_handler(lambda m: json.dumps({
        "decision": "admit", "rationale": "x",
        "entries": [{"name": "bad-skill", "category": "c", "problem": "p",
                     "when_to_apply": "w", "strategy": "s",
                     "code_sketch": "import os\nos.system('x')", "origin_tasks": ["H2"]}]}))
    coord2 = coord_mod.Coordinator(mock_cfg(), library_root=lib_root,
                                   run_root=tempfile.mkdtemp())
    spec2 = TaskSpec(name="H2", env_task="Stack", instruction="i", success_criteria="s")
    results3 = coord2.run([spec2])
    check("H5 不合规拒绝入库", lib.get("bad-skill") is None
          and results3["H2"]["admission"]["rejected"])


def main():
    for fn in [t_schema, t_compliance, t_library, t_digest, t_llm,
               t_actor, t_evolve, t_coordinator]:
        fn()
    n_ok = sum(1 for _, ok in PASS if ok)
    print(f"\n===== {n_ok}/{len(PASS)} PASS =====")
    sys.exit(0 if n_ok == len(PASS) else 1)


if __name__ == "__main__":
    main()
