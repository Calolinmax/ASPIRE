#!/usr/bin/env python
"""CGN 位姿对应性检查（2026-08-04 用户指定）: 只把机械臂控制到 CGN 输出
位姿（按 score 序第一个可执行到位的候选）并保持, 不做任何其他动作
（不闭合、不抬升）, 供肉眼核对 glyph 与实体。

品红 glyph = CGN 实际执行到位候选的位姿（Piper 几何锚点, 网格实测:
掌杠贴实体掌面 +23mm、侧线覆盖指尖 -50mm、开度 70mm 满开）。
实体夹爪到位后应与 glyph 完全重合: 路径用 free_approach_roll 到位后,
再做滚转对齐抛光（全 6 自由度 DLS 就地种子）使朝向也与 CGN 输出一致;
滚转不可达的候选跳过（2026-08-05 用户要求: 指尖对指尖、掌面对掌面）。

用法: MUJOCO_GL=glfw python scripts/check_cgn_pose.py [seed] [--show-raw]
  --show-raw: 叠加绿色 glyph = CGN 原始输出的手指段（58.4~112.2mm 双指,
  用户裁决掌体大块不画）——绿色指尖尖与实体指尖尖重合
  （TCP_DEPTH_PIPER=0.0412 指尖对齐, 2026-08-05 用户指令）。
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import aspire.robots  # noqa: F401
from aspire.engine_capx import ExecutionEngineCapx, TUCK_Q
from aspire.motion_planner import plan_joint_path
from aspire.primitives_capx import (PrimitiveContextCapx, cgn_to_gripper,
                                    filter_grasps_by_width, PIPER_MAX_WIDTH)
from aspire.vision_client import segment_sam3_text_prompt, grasp_cgn
from robosuite.utils import transform_utils as T

SEED = int(sys.argv[1]) if len(sys.argv) > 1 and sys.argv[1].isdigit() else 0
SHOW_RAW = '--show-raw' in sys.argv


def add_connector(scn, A, B, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(A, float), np.asarray(B, float))
    scn.ngeom += 1


def main():
    engine = ExecutionEngineCapx(task='Stack', seed=SEED, target_only=True)
    sim = engine.env.sim
    m = sim.model._model

    viewer = mujoco.viewer.launch_passive(m, sim.data._data)
    _orig_tick = engine._tick

    def _tick_with_view(render_frames=True, _f=_orig_tick, _v=viewer):
        _t0 = time.perf_counter()
        _f(False)  # 观看模式: 运动拍不刷观测渲染（640 双相机 ~0.1-0.3s/次, 致卡顿）
        if _v.is_running():
            _v.sync()
            _dt = time.perf_counter() - _t0
            if _dt < 0.025:  # 实时配速: 1 tick = 25ms 仿真（2026-08-05 用户: 提速）
                time.sleep(0.025 - _dt)

    engine._tick = _tick_with_view

    ns = engine._build_namespace()
    ns['move_to_joints'](TUCK_Q)

    obs = ns['get_observation']()
    cam = obs['robot0_robotview']
    rgb, depth, K, pose_mat = (cam['images']['rgb'], cam['images']['depth'],
                               cam['intrinsics'], cam['pose_mat'])
    masks = segment_sam3_text_prompt(rgb, 'red cube')
    seg = np.zeros(rgb.shape[:2], dtype=np.int32)
    if masks:
        seg[masks[0]['mask'] > 0] = 1
    g, s, o = grasp_cgn(depth, K, seg, return_openings=True)
    gf, sf, of = filter_grasps_by_width(g, s, o, contact_dist=0.04)
    if len(gf) == 0:
        print(f'CGN 零候选（seed {SEED}）: 干净单方块场景对 CGN 是双重 OOD '
              '（cgn_container.md §6 已载）, 换个 seed 试试')
        while viewer.is_running():
            engine.hold_ticks(1)
        engine.close()
        return
    print(f'候选 {len(g)} 个, top-1 score={sf[0]:.3f} cgn_open={of[0]*1000:.1f}mm')

    gb = cgn_to_gripper(gf[0], pose_mat)   # 严格: CGN 原姿, 不吸附
    pos = gb[:3, 3].copy()
    qx = T.mat2quat(gb[:3, :3])
    qw = np.array([qx[3], qx[0], qx[1], qx[2]])
    pos_pre = pos + gb[:3, 2] * 0.10
    print(f'CGN 位姿 (base): pos={pos.round(4)}')

    if SHOW_RAW:
        # 【定稿冻结 2026-08-05: 用户验收满意, 未经人类明确允许禁止任何修改】
        # CGN 原始输出叠加显示（绿色）——官方 demo 结构: 尾刺+底杠+双指一体
        # （尾刺 = 原点→指根底杠中点的连接线, 即官方 draw_grasps 的 stem;
        # 只删掌体大块轮廓(原点掌杠+掌体侧线), 2026-08-05 用户裁决）。
        # 双指 = 官方手指段 58.4~112.2mm, 指尖尖对齐实体指尖。
        T_flip = np.diag([1, -1, 1, 1])
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'cubeA_main')
        cube_top = sim.data.xpos[cid][2] + 0.04
        print('--- CGN 原始输出（绿色 = 底杠+双指+尾刺, 手指段 58.4~112.2mm）---')
        for ri in range(min(5, len(gf))):
            Tw_c = engine.T_world_base @ pose_mat @ T_flip @ gf[ri] @ T_flip
            Oc, Rc = Tw_c[:3, 3], Tw_c[:3, :3]
            Xc, Zc = Rc[:, 0], Rc[:, 2]
            B = Oc + Zc * 0.0584                     # 指根(掌体底)
            Cc = Oc + Zc * 0.1034                    # 指垫捏取面
            Ft = Oc + Zc * 0.1122                    # 指尖尖
            wr = 0.08
            rgba = (0.0, 1.0, 0.2, 0.9) if ri == 0 else (0.0, 0.8, 0.2, 0.4)
            for A, Bb in [(Oc, B),                                # 尾刺: 原点→底杠中点
                          (B - Xc * wr / 2, B + Xc * wr / 2),       # 底杠(指根)
                          (B + Xc * wr / 2, Ft + Xc * wr / 2),      # 指 +
                          (B - Xc * wr / 2, Ft - Xc * wr / 2)]:     # 指 −
                add_connector(viewer.user_scn, A, Bb, 0.004, rgba)
            print(f'  cand{ri}: Panda掌心 z={Oc[2]:.4f} (顶面上 {(Oc[2]-cube_top)*1000:+.1f}mm)'
                  f'  捏取面 z={Cc[2]:.4f} (顶面下 {(cube_top-Cc[2])*1000:+.1f}mm)'
                  f'  指尖 z={Ft[2]:.4f}')

    # 两级兜底按 score 序逐个尝试（2026-08-04）:
    #   ① 路径 IK 未收敛 → 跳过（seed 3 实测 top-1 过渡点落在可达流形外）
    #   ② 执行卡死 → 撤回 TUCK 换下一候选。卡死机理: 对角抓取
    #     (cgn_open 56.6 ≈ 40·√2) + 倾斜逼近时, 70mm 全开的指尖侧向余量仅
    #     ~6.7mm, 逼近轴倾斜 ~14° 指尖横移 ~12mm → 扫到方块棱角楔住
    #     （目标构型本身从其它方向可达, diag 实测）——CGN 位姿不动, 只换候选
    chosen = None
    for ci in range(min(8, len(gf))):
        gb_try = cgn_to_gripper(gf[ci], pose_mat)
        pos_try = gb_try[:3, 3].copy()
        qx_try = T.mat2quat(gb_try[:3, :3])
        qw_try = np.array([qx_try[3], qx_try[0], qx_try[1], qx_try[2]])
        # 预抓点以接触面为基准（2026-08-05 高度事故修复）: 指尖对齐锚点把
        # site 抬离接触面 62.2mm, 原 site+10cm 回退把 q_pre 顶到 z≈0.22
        # 的可行性死区(高度-倾角地图实测 z≥0.18 全灭)。site+0.038 ≈ 接触面
        # 上方 10cm——回退的物理意义是离开物体, 应以接触面为参照
        pre_try = pos_try + gb_try[:3, 2] * 0.038
        try:
            # 中途点放宽容差(loose=True), 末步/抛光严格——2026-08-05 平台期
            # 修复: 边际候选 DLS 平台 ~1.5mm/0.57°, 严格容差在 q_pre 就全拒
            q_pre = ns['solve_ik'](pre_try, qw_try, free_approach_roll=True, loose=True)
            descend = []
            q_cur = q_pre
            for k in range(1, 5):
                pos_k = pre_try + (pos_try - pre_try) * (k / 4)
                q_cur = ns['solve_ik'](pos_k, qw_try, free_approach_roll=True,
                                       seed=q_cur, loose=(k < 4))
                descend.append(q_cur)
        except RuntimeError:
            print(f'cand{ci}: 路径 IK 未收敛, 跳过')
            continue
        # 全流程: IK 预算(含抛光) → 逐腿避障规划 → 拼接密化 → 流式连续执行
        # （2026-08-05 用户: 运动要丝滑、插值多些——原逐腿阻塞到位有起停感;
        # 密化到 ~0.03rad/点流式下发, 仅最终点阻塞核验; 撤回沿已流过的密化
        # 路径原路返回(沿途已查碰撞), 不再裸扫）
        engine.set_gripper(1.0)
        dense = []
        abort_i = 0
        try:
            # ③ 滚转对齐抛光改到执行前预算（种子=末步构型, 与到位构型同盆地;
            # 抛光目标本身不变: 全 6 自由度, 使实体朝向与 CGN 输出一致）
            q_roll = ns['solve_ik'](pos_try, qw_try, free_approach_roll=False,
                                    seed=descend[-1])
            # 逐腿: 即时规划（读最新方块位姿）→ 密化流式 → 短沉降;
            # 仅末腿终点阻塞核验。【2026-08-05 剐蹭-移位事故: 全腿一次预算,
            # 早段擦碰移位方块后后续路径过期 → 末步卡死; 改逐腿即时规划】
            q_from = engine.current_arm_qpos()
            dense = []
            for li, q_leg in enumerate([q_pre] + descend + [q_roll]):
                _t0 = time.perf_counter()
                path = plan_joint_path(q_from, q_leg, engine)
                if path is None:
                    raise RuntimeError(f'避障规划失败(腿{li})')
                if len(path) > 2:
                    print(f'  [plan] 腿{li}: RRT 绕行 {len(path)} 点, '
                          f'{time.perf_counter() - _t0:.1f}s')
                waypts = [engine.current_arm_qpos()] + path[1:]
                dense = [waypts[0]]
                for a, b in zip(waypts, waypts[1:]):
                    nseg = max(1, int(np.ceil(np.abs(b - a).max() / 0.05)))
                    for ii in range(1, nseg + 1):
                        dense.append(a + (b - a) * (ii / nseg))
                for di, q_d in enumerate(dense[1:], 1):
                    abort_i = di
                    engine.step_joints(q_d)
                    if di % 10 == 0 and np.linalg.norm(
                            engine.current_arm_qpos() - q_d) > 0.35:
                        raise RuntimeError(f'流式跟踪丢失(腿{li}点{di})')
                # 非末腿短沉降（≤20 拍到 0.05rad 内, 无感）后再规划下一段
                # （含下降末步——抛光腿的规划起点假定已到位）
                if li < len(descend) + 1:
                    for _ in range(20):
                        engine.step_joints(dense[-1])
                        if np.linalg.norm(
                                engine.current_arm_qpos() - dense[-1]) < 0.05:
                            break
                q_from = dense[-1]
            # 末腿终点阻塞核验（到位即真到位）
            ns['move_to_joints'](q_roll, tol=0.02, fk_tol=0.005)
            abort_i = len(dense) - 1
        except RuntimeError as e:
            print(f'cand{ci}: 规划/执行未通过({e}), 撤回 TUCK 换下一候选')
            try:
                if dense:  # 先沿本腿已流路径退回腿起点（沿途构型均已查碰撞）
                    for q_b in reversed(dense[:abort_i + 1]):
                        engine.step_joints(q_b)
                _p = plan_joint_path(engine.current_arm_qpos(), TUCK_Q, engine)
                if _p is not None:  # 规划回 TUCK（读最新方块位姿）
                    _wp = [engine.current_arm_qpos()] + _p[1:]
                    for a, b in zip(_wp, _wp[1:]):
                        _n = max(1, int(np.ceil(np.abs(b - a).max() / 0.05)))
                        for ii in range(1, _n + 1):
                            engine.step_joints(a + (b - a) * (ii / _n))
                ns['move_to_joints'](TUCK_Q, tol=0.10, fk_tol=0.05)
            except RuntimeError:
                print('撤回也未通过——保持现场供查看, 关窗退出')
                while viewer.is_running():
                    engine.hold_ticks(1)
                engine.close()
                return
            continue
        chosen = (ci, gb_try, pos_try)
        if ci > 0:
            print(f'选用 cand{ci}（前 {ci} 个候选未通过已跳过）')
        break
    if chosen is None:
        print('前 8 个候选全部不可执行, 换 seed 试试')
        while viewer.is_running():
            engine.hold_ticks(1)
        engine.close()
        return
    ci, gb, pos = chosen

    # glyph（品红, Piper 几何, 画实际执行到位的候选）
    Tw = engine.T_world_base @ gb
    O, Sz, Sy = Tw[:3, 3], Tw[:3, 2], Tw[:3, 1]
    # 锚点为网格实测（2026-08-04 用户审定版, 2026-08-05 恢复）:
    # CGN 输出(Panda 约定)是基准不许动, glyph 画的是"末端应到的形状"——
    # 掌杠贴实体掌面、侧线覆盖指尖, 与 CGN 网络输出大小无关
    C = O - Sz * 0.050    # 指尖极值（实测 -50mm）
    Pp = O + Sz * 0.023   # 掌心可见表面（实测 +23mm; 掌根原点在壳内穿模）
    w = PIPER_MAX_WIDTH
    for A, B in [(Pp - Sy * w / 2, Pp + Sy * w / 2),
                 (Pp + Sy * w / 2, C + Sy * w / 2),
                 (Pp - Sy * w / 2, C - Sy * w / 2),
                 (Pp, Pp + Sz * 0.05)]:
        add_connector(viewer.user_scn, A, B, 0.004, (1.0, 0.0, 1.0, 0.9))

    sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, 'gripper0_right_grip_site')
    site_b = (engine.T_base_world @ np.append(sim.data.site_xpos[sid], 1.0))[:3]
    Rd = Tw[:3, :3].T @ sim.data.site_xmat[sid].reshape(3, 3)
    ang = np.degrees(np.arccos(np.clip((np.trace(Rd) - 1) / 2, -1, 1)))
    print(f'到位: site(base)={site_b.round(4)} vs CGN 目标 {pos.round(4)} '
          f'位置残差 {np.linalg.norm(site_b - pos)*1000:.1f}mm 姿态残差 {ang:.2f}°')
    if SHOW_RAW:
        print('对照(官方解剖): 实体指尖尖已对齐 Panda 指尖尖(0.1122); '
              'Panda 掌面(掌体底)在捏取面上方 45mm, 实体掌面在其上方 ~64mm——'
              '两手掌面都悬空, 差 19mm 是掌体厚度差')
    print('保持中（不闭合不抬升）。关窗退出。')
    while viewer.is_running():
        engine.hold_ticks(1)
    engine.close()


if __name__ == '__main__':
    main()
