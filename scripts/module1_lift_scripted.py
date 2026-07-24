#!/usr/bin/env python
# =============================================================================
# 模块 1 验证脚本：robosuite Lift 任务 + 脚本化策略
# =============================================================================
# 目的：
#   1. 验证 robosuite 环境端到端可用（reset / step / 渲染）
#   2. 用 ground truth 状态写脚本化抓取策略，验证"移动→下降→夹取→抬起"流程
#   3. 验证 robosuite 自带成功判定（_check_success）
#
# 这是后续 Primitive API（模块 2）的物理流程预演：
#   脚本化策略中的每一步对应未来的 move_to / grasp / lift primitive。
#
# 运行方式（遵守 AGENTS.md 第 0 节）：
#   MUJOCO_GL=egl /home/stouching/anaconda3/envs/aspire/bin/python \
#       scripts/module1_lift_scripted.py
# =============================================================================

import numpy as np
import robosuite as suite

# ---------------------------------------------------------------------------
# 环境配置
# ---------------------------------------------------------------------------
# control_freq=20：每步 50ms，脚本化策略足够精细
# horizon=200：10 秒上限，Lift 流程约需 100 步左右
ENV_KWARGS = dict(
    robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=256,
    camera_widths=256,
    control_freq=20,
    horizon=200,
)

# P 控制增益与收敛阈值
KP_POS = 8.0          # 位置比例增益（action 是 delta pose，会被 clip 到 [-1,1]）
POS_TOL = 0.005       # 位置收敛阈值 5mm
GRASP_STEPS = 15      # 夹爪闭合保持步数（等夹爪物理夹紧）


def make_env():
    return suite.make("Lift", **ENV_KWARGS)


def p_control_delta(target, current, kp=KP_POS):
    """位置 P 控制：输出 clip 后的 delta action（xyz 部分）"""
    delta = kp * (np.asarray(target) - np.asarray(current))
    return np.clip(delta, -1.0, 1.0)


def run_episode(seed=None, verbose=True):
    """跑一次脚本化 Lift，返回 (success, steps, info)"""
    if seed is not None:
        np.random.seed(seed)
    env = make_env()
    obs = env.reset()

    # 状态机阶段
    phase = "reach_above"
    grasp_counter = 0
    success = False
    step = 0

    for step in range(ENV_KWARGS["horizon"]):
        eef = obs["robot0_eef_pos"].copy()
        cube = obs["cube_pos"].copy()

        # --- 状态机：确定当前目标点与夹爪指令 ---
        if phase == "reach_above":
            target = cube + np.array([0.0, 0.0, 0.12])   # 方块上方 12cm
            grip = -1.0                                   # 张开
            if np.linalg.norm(target - eef) < POS_TOL:
                phase = "descend"
        elif phase == "descend":
            target = cube + np.array([0.0, 0.0, -0.01])  # 略低于方块中心（夹爪指尖夹住侧面）
            grip = -1.0
            if np.linalg.norm(target - eef) < POS_TOL:
                phase = "grasp"
                grasp_counter = 0
        elif phase == "grasp":
            target = cube + np.array([0.0, 0.0, -0.01])
            grip = 1.0                                    # 闭合
            grasp_counter += 1
            if grasp_counter >= GRASP_STEPS:
                phase = "lift"
        else:  # lift
            target = cube + np.array([0.0, 0.0, 0.25])   # 抬起 25cm
            grip = 1.0                                    # 保持闭合

        # --- 组装 action：[dx,dy,dz, dax,day,daz, gripper]（OSC_POSE 7维）---
        delta = p_control_delta(target, eef)
        action = np.concatenate([delta, np.zeros(3), [grip]])
        obs, reward, done, info = env.step(action)

        if env._check_success():
            success = True
            break

    env.close()
    if verbose:
        print(f"  phase_end={phase}, steps={step+1}, success={success}")
    return success, step + 1, {"final_phase": phase}


def main():
    print("=== 模块 1：Lift 脚本化策略验证 ===")
    n_trials = 5
    results = []
    for i in range(n_trials):
        print(f"[trial {i}]")
        success, steps, info = run_episode(seed=i)
        results.append(success)

    n_success = sum(results)
    print(f"\n成功率: {n_success}/{n_trials} = {n_success/n_trials:.0%}")
    if n_success == n_trials:
        print("✅ 模块 1 验证通过：环境 + 抓取流程 + 成功判定全部正常")
    else:
        print("⚠️ 有失败 trial，需要调参（descend 深度 / 抓取阈值 / P 增益）")


if __name__ == "__main__":
    main()
