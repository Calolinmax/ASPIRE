#!/usr/bin/env python
"""CGN sim 执行抓取（验收门资产）。

流程（单场景）: SAM3 "red cube" → CGN 候选 → 宽度过滤（required_opening 尺子）
→ 按 score 降序最多尝试 3 个候选:
  预抓取(抓取点沿逼近轴后撤 10cm) → 下降到位 → 闭合(contact_dist−2mm)
  → 抬升 5cm → 悬停 2s → 方块不滑落 = 成功。
成功定义: 悬停结束 cubeA 基座系 z > GT + 3cm。

开度策略（2026-07-31 裁决）:
  - 接近/下降开度 = min(cgn_width, PIPER_MAX)（本场景恒为满开 0.045）
  - 闭合目标 = contact_dist − 2mm（joint7 线性模型由 sim 实测拟合:
    inner(q7) ≈ 0.0172 + 0.7914·q7, q7∈[0,0.035]）
  - 风险（文档已记）: 满开 0.045 对 4cm 方块单侧余量 2.5mm, 偏心>2.5mm 时
    下降会蹭方块——sim 可接受, 真机是标定硬指标。

用法:
    MUJOCO_GL=egl python scripts/cgn_execute_grasp.py [--seed 0] [--clutter 4]
输出:
    每候选尝试记录 + 成功时的 trace 目录路径。
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aspire.robots  # noqa: F401 注册 Piper
from aspire.engine_capx import ExecutionEngineCapx
from aspire.primitives_capx import (
    PrimitiveContextCapx,
    build_namespace,
    cgn_to_gripper,
    filter_grasps_by_width,
    PIPER_MAX_WIDTH,
)
from aspire.vision_client import segment_sam3_text_prompt
from robosuite.utils import transform_utils as T

# 夹爪开合线性模型（sim 实测: q7=0 → 内净距 0.0172m; q7=0.035 → 0.0449m）
_GRIP_INNER_0 = 0.0172
_GRIP_INNER_RATE = (0.0449 - 0.0172) / 0.035  # 0.7914 m 净距 / rad
_GRIP_Q7_MAX = 0.035


def fraction_for_inner(inner_target: float) -> float:
    """目标指垫内净距(m) → engine.set_gripper 的 fraction (1=全开)。"""
    q7 = np.clip((inner_target - _GRIP_INNER_0) / _GRIP_INNER_RATE, 0.0, _GRIP_Q7_MAX)
    return float(q7 / _GRIP_Q7_MAX)


def quat_wxyz_from_mat(R: np.ndarray) -> np.ndarray:
    q_xyzw = T.mat2quat(R)  # robosuite xyzw
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def cube_z_base(engine) -> float:
    import mujoco
    m = engine.env.sim.model._model
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "cubeA_main")
    z_world = engine.env.sim.data.xpos[cid][2]
    return float((engine.T_base_world @ np.append(engine.env.sim.data.xpos[cid], 1.0))[2])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--clutter', type=int, default=None)
    parser.add_argument('--max-candidates', type=int, default=3)
    args = parser.parse_args()

    import traceback

    import mujoco

    engine = ExecutionEngineCapx(task="Stack", seed=args.seed, clutter=args.clutter,
                                 trace_root="traces")
    print('trace dir:', engine.trace_dir)
    success = False
    error = None
    try:
        # 经 trace 包装的命名空间调用原语（每步含输入/输出/观测链接入 records）；
        # move_to_joints 单段调用代替 _safely 版, 换取逐步 trace（IK 解已过碰撞检查）
        ns = build_namespace(engine)

        obs = ns['get_observation']()
        cam = obs["robot0_robotview"]
        rgb, depth, K, pose_mat = (cam["images"]["rgb"], cam["images"]["depth"],
                                   cam["intrinsics"], cam["pose_mat"])

        masks = ns['segment_sam3_text_prompt'](rgb, "red cube")
        seg = np.zeros(rgb.shape[:2], dtype=np.int32)
        if masks:
            seg[masks[0]["mask"] > 0] = 1
        else:
            print('WARNING: SAM3 no mask')
        print('mask pixels:', int((seg > 0).sum()))

        grasps, scores, openings = ns['grasp_cgn'](rgb, depth, K, seg)
        print('CGN candidates:', len(grasps),
              'scores:', scores.round(3) if len(scores) else scores)

        # 宽度过滤【改尺子】对比（旧尺 openings<=0.045 / 新尺 required<=0.045）
        CONTACT_DIST = 0.04  # cubeA GT 尺寸（sim 阶段允许, 见 filter 注释）
        n_old = int((openings <= PIPER_MAX_WIDTH).sum()) if len(openings) else 0
        grasps_f, scores_f, openings_f = filter_grasps_by_width(
            grasps, scores, openings, contact_dist=CONTACT_DIST)
        print(f'宽度过滤对比: 旧尺(cgn_width<=0.045) 通过 {n_old}/{len(grasps)} | '
              f'新尺(required={CONTACT_DIST}+0.004<=0.045) 通过 {len(grasps_f)}/{len(grasps)}')

        model_ = engine.env.sim.model._model
        cid = mujoco.mj_name2id(model_, mujoco.mjtObj.mjOBJ_BODY, "cubeA_main")
        gt_base = (engine.T_base_world @ np.append(engine.env.sim.data.xpos[cid], 1.0))[:3]
        print('GT cube (base):', gt_base.round(4))

        # yaw 变体（2026-07-31 批准）：绕抓取点处逼近轴 {0,90,180,270}° 四个旋转位姿。
        # 依据: cubeA 水平截面为正方形(axis-aligned) ⇒ 90° 旋转后闭合轴仍跨 4cm 对面,
        # contact_dist 不变。【仅对方形截面目标启用】位置不动只转姿态。
        YAW_STEPS_DEG = [0, 90, 180, 270]
        n_pre_ok = 0  # pre-grasp IK（任一 yaw）成功的候选数
        for ci in range(min(args.max_candidates, len(grasps_f))):
            g_base_pose = cgn_to_gripper(grasps_f[ci], pose_mat)
            print(f'\n=== 候选 {ci} (score={scores_f[ci]:.3f}, cgn_open={openings_f[ci]:.4f}) ===')
            solved = None
            for yaw_deg in YAW_STEPS_DEG:
                c_, s_ = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
                T_yaw = np.eye(4)
                T_yaw[:3, :3] = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]])
                g_try = g_base_pose @ T_yaw  # 局部 z=逼近轴, 右乘=绕抓取点原地转
                quat_wxyz = quat_wxyz_from_mat(g_try[:3, :3])
                pos_grasp = g_try[:3, 3].copy()
                pos_pre = pos_grasp - g_try[:3, 2] * 0.10
                try:
                    q_pre = ns['solve_ik'](pos_pre, quat_wxyz)
                    solved = (yaw_deg, quat_wxyz, pos_grasp, pos_pre, q_pre)
                    print(f'  yaw={yaw_deg}: pre-grasp IK 收敛')
                    break
                except RuntimeError:
                    print(f'  yaw={yaw_deg}: pre-grasp IK 未收敛')
                    continue
            if solved is None:
                print(f'候选 {ci} 失败: 4 个 yaw 的 pre-grasp IK 均未收敛')
                continue
            n_pre_ok += 1
            yaw_deg, quat_wxyz, pos_grasp, pos_pre, q_pre = solved
            print('grasp pos (base):', pos_grasp.round(4), ' pre:', pos_pre.round(4))
            try:
                # 接近开度 = min(cgn_width, PIPER_MAX) → 本场景恒满开
                open_frac = fraction_for_inner(min(float(openings_f[ci]), PIPER_MAX_WIDTH))
                engine.set_gripper(open_frac)
                ns['move_to_joints'](q_pre)
                print('预抓取到位')
                # 下降（与 pre-grasp 同一 yaw）
                q_grasp = ns['solve_ik'](pos_grasp, quat_wxyz)
                ns['move_to_joints'](q_grasp)
                print('下降到位, site z(base):',
                      float((engine.T_base_world @ np.append(
                          engine.env.sim.data.site_xpos[
                              mujoco.mj_name2id(model_, mujoco.mjtObj.mjOBJ_SITE,
                                                "gripper0_right_grip_site")], 1.0))[2]).__round__(4))
                # 闭合至 contact_dist − 2mm
                engine.set_gripper(fraction_for_inner(CONTACT_DIST - 0.002))
                engine.hold_ticks(15)
                print('已闭合')
                # 抬升 5cm
                pos_lift = pos_grasp + np.array([0, 0, 0.05])
                q_lift = ns['solve_ik'](pos_lift, quat_wxyz)
                ns['move_to_joints'](q_lift)
                print('抬升到位')
                # 悬停 2s
                engine.hold_ticks(40)
                z_end = cube_z_base(engine)
                held = z_end > gt_base[2] + 0.03
                print(f'悬停结束 cube z(base)={z_end:.4f} (GT {gt_base[2]:.4f}) → '
                      + ('夹持成功' if held else '方块滑落'))
                if held:
                    success = True
                    print(f'\nSUCCESS: 候选 {ci} yaw={yaw_deg} 抓取成功')
                    break
                # 失败复位: 松开+撤回
                engine.set_gripper(1.0)
                ns['move_to_joints'](q_pre)
            except RuntimeError as e:
                print(f'候选 {ci} yaw={yaw_deg} 失败: {type(e).__name__}: {e}')
                try:
                    engine.set_gripper(1.0)
                    engine.hold_ticks(10)
                except Exception:
                    pass
                continue
        n_tried = min(args.max_candidates, len(grasps_f))
        if n_tried:
            print(f'\npre-grasp IK（任一 yaw）成功率: {n_pre_ok}/{n_tried}')

        print('\n' + '=' * 56)
        print('RESULT:', 'PASS' if success else 'FAIL')
        print('trace dir:', engine.trace_dir)
    except Exception:
        error = traceback.format_exc()
        raise
    finally:
        engine.mark_trace_result(success, error)
        engine.close()


if __name__ == '__main__':
    main()
