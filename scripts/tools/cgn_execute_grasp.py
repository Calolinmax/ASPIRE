#!/usr/bin/env python
"""CGN sim 执行抓取（验收门资产）。

流程（单场景）: SAM3 "red cube" → CGN 候选 → 宽度过滤（required_opening 尺子）
→ 按 score 降序最多尝试 3 个候选:
  预抓取(抓取点沿逼近轴后撤 10cm) → 下降到位 → 闭合(contact_dist−2mm)
  → 抬升 5cm → 悬停 2s → 方块不滑落 = 成功。
成功定义: 悬停结束 cubeA 基座系 z > GT + 3cm。

开度策略（2026-07-31 裁决; 2026-08-04 数值修正——旧 0.045 系误测, 真值 0.070）:
  - 接近/下降开度 = min(cgn_width, PIPER_MAX)（CGN 对 4cm 方块预测 64.6–72.5mm,
    钳到 70mm 后 ≈ 满开）
  - 闭合目标 = contact_dist − 2mm（joint7 模型: 内净距 = 2·q7, q7∈[0,0.035]）
  - 风险（2026-08-04 修正后放宽）: 满开 70mm 对 4cm 方块单侧余量 15mm。

用法:
    MUJOCO_GL=egl python scripts/tools/cgn_execute_grasp.py [--seed 0] [--clutter 4]
输出:
    每候选尝试记录 + 成功时的 trace 目录路径。
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import aspire.robots  # noqa: F401 注册 Piper
from aspire.engine.engine_capx import ExecutionEngineCapx, TUCK_Q
from aspire.planning.motion_planner import plan_joint_path
from aspire.api.primitives_capx import (
    PrimitiveContextCapx,
    build_namespace,
    cgn_to_gripper,
    filter_grasps_by_width,
    _ik_library,
    PIPER_MAX_WIDTH,
)
from aspire.perception.vision_client import segment_sam3_text_prompt
from robosuite.utils import transform_utils as T


def lib_feasible(pos_base, z_app, dp_max=0.06, da_max=0.35):
    """ik_library 可达性【advisory 哨兵】（2026-08-03 起不拦截任何调用）:
    返回 (库合取是否通过, (最近位置差, 最近方位差)), 仅供日志配对分析。"""
    lib = _ik_library()
    if lib is False:
        return True, (float('nan'), float('nan'))  # 无库不拦
    dp = np.linalg.norm(lib['P'] - np.asarray(pos_base, dtype=float), axis=1)
    da = np.arccos(np.clip(lib['Z'] @ np.asarray(z_app, dtype=float), -1.0, 1.0))
    ok = bool(((dp < dp_max) & (da < da_max)).any())
    return ok, (float(dp.min()), float(da.min()))

# 夹爪开合模型（2026-08-04 修正）: 内净距 = 2·q7（双指镜像滑动: q7=0 两垫贴合,
# q7=0.035 内面 ±35mm → 满开 70mm）。旧拟合 0.0172+0.7914·q7 系坐标系混淆误测
# （指垫沿开合轴半厚度 2.5mm 被误取为局部 y 的 15mm）, 与刚性几何矛盾, 作废。
_GRIP_INNER_0 = 0.0
_GRIP_INNER_RATE = 2.0  # m 净距 / m joint7 行程（slide 关节, 非 rad）
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
    parser.add_argument('--once', action='store_true',
                        help='单发模式: 只试可达性最高的 1 个候选, 不跑 fallback '
                             '安全网 (2026-08-04 用户要求)')
    parser.add_argument('--target-only', action='store_true',
                        help='桌面只留目标物体 cubeA, 去 cubeB 与全部杂物 '
                             '(2026-08-04 用户指令)')
    parser.add_argument('--legacy-adjust', action='store_true',
                        help='启用旧调整逻辑（点云吸附+yaw族+可达性重排）。默认关闭='
                             '严格按 CGN 输出: top-1 原姿直达+直接闭合 (2026-08-04 用户指定)')
    parser.add_argument('--ghost-wrist', action='store_true',
                        help='[已废弃] 腕部碰撞 2026-08-04 幽灵化 → 2026-08-05 已恢复实体; '
                             '此 flag 无效果（保留兼容）')
    parser.add_argument('--wrist-collide', action='store_true',
                        help='[2026-08-05 起为默认状态] 腕部(link6)碰撞已在 robot.xml '
                             '恢复实体（路径规划避障需要）, 此 flag 仅冗余确认')
    parser.add_argument('--view', action='store_true',
                        help='3D 交互窗口实时显示执行 + CGN 候选 glyph '
                             '(需 MUJOCO_GL=glfw; 2026-08-04)')
    args = parser.parse_args()
    if args.once:
        # 2026-08-05 用户: 可以多次尝试抓取——不再锁 max_candidates=1;
        # top-N 按 score 序尝试（IK/规划不可行或夹持滑落 → 顺延下一候选）,
        # 单场景单启动不变; 几何安全网仍跳过（非 CGN 方法, 教义不动）
        pass

    import traceback

    import mujoco

    engine = ExecutionEngineCapx(task="Stack", seed=args.seed, clutter=args.clutter,
                                 trace_root="traces", target_only=args.target_only)
    if args.ghost_wrist:
        print('[ghost-wrist] 注意: 该 flag 已废弃——腕部碰撞 08-04 幽灵化, 08-05 已恢复实体')
    if args.wrist_collide:
        print('[wrist-collide] 2026-08-05 起腕部碰撞已是默认实体状态, 无需此 flag')

    def ramp_gripper(target, ticks):
        """渐进开合（2026-08-05 用户: 夹爪动作太快）——fraction 线性斜坡,
        每 tick 走一步（--view 配速 25ms/tick, 25 ticks ≈ 0.6s）。"""
        start = engine.gripper_fraction
        for i in range(1, ticks + 1):
            engine.set_gripper(start + (target - start) * i / ticks)
            engine.hold_ticks(1)

    viewer = None
    if args.view:
        import time as _time
        import mujoco.viewer as _mjv
        if os.environ.get('MUJOCO_GL', '') != 'glfw':
            print('[view] 警告: MUJOCO_GL 非 glfw, 窗口可能起不来')
        viewer = _mjv.launch_passive(engine.env.sim.model._model,
                                     engine.env.sim.data._data)
        _orig_tick = engine._tick

        def _tick_with_view(render_frames=True, _f=_orig_tick, _v=viewer):
            _t0 = _time.perf_counter()
            _f(False)  # 观看模式: 运动拍不刷观测渲染（640 双相机 ~0.1-0.3s/次,
                       # 是"只有两三帧"卡顿感的来源; 边界帧仍由 trace 包装捕获）
            if _v.is_running():
                _v.sync()
                # 实时配速（2026-08-04 用户反馈"瞬移"）: 1 tick = 25 substeps
                # × 0.002s = 50ms 仿真时间, 睡满 50ms 否则 ~25 倍速播放像瞬移
                _dt = _time.perf_counter() - _t0
                if _dt < 0.025:  # 实时配速 25ms/tick（2026-08-05 用户: 提速）
                    _time.sleep(0.025 - _dt)

        engine._tick = _tick_with_view
        print('[view] 交互窗口已开（品红=CGN top-1, 绿=其余候选；关窗不中断执行）')
        print('[view] 注意: 勿用 viewer 的 Reset 按钮——它是 mj_resetData 物理复位，'
              '不会重摆场景（方块会跳回 XML 默认位）；要看新场景请关窗重跑（换 seed）')

        def _add_connector(scn, A, B, width, rgba):
            if scn.ngeom >= scn.maxgeom:
                return
            g = scn.geoms[scn.ngeom]
            mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                                np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
            mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                                 np.asarray(A, float), np.asarray(B, float))
            scn.ngeom += 1
    print('trace dir:', engine.trace_dir)
    success = False
    error = None
    try:
        # 经 trace 包装的命名空间调用原语（每步含输入/输出/观测链接入 records）；
        # move_to_joints 单段调用代替 _safely 版, 换取逐步 trace（IK 解已过碰撞检查）
        ns = build_namespace(engine)

        # 收臂让拍（2026-08-03 裁决 2）: 感知快照前移到 TUCK_Q（库锚定验证
        # 过的合法位姿）, 让开新相机→工作区走廊; 感知完成后直接进执行,
        # 首段 move_to_joints 从 tuck 出发。
        ns['move_to_joints'](TUCK_Q)
        print('已收臂至 tuck 位姿')

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

        # 宽度过滤【改尺子】对比（旧尺 openings<=MAX / 新尺 required<=MAX）
        CONTACT_DIST = 0.04  # cubeA GT 尺寸（sim 阶段允许, 见 filter 注释）
        n_old = int((openings <= PIPER_MAX_WIDTH).sum()) if len(openings) else 0
        grasps_f, scores_f, openings_f = filter_grasps_by_width(
            grasps, scores, openings, contact_dist=CONTACT_DIST)
        print(f'宽度过滤对比: 旧尺(cgn_width<={PIPER_MAX_WIDTH}) 通过 {n_old}/{len(grasps)} | '
              f'新尺(required={CONTACT_DIST}+0.004<={PIPER_MAX_WIDTH}) 通过 {len(grasps_f)}/{len(grasps)}')

        model_ = engine.env.sim.model._model
        cid = mujoco.mj_name2id(model_, mujoco.mjtObj.mjOBJ_BODY, "cubeA_main")
        gt_base = (engine.T_base_world @ np.append(engine.env.sim.data.xpos[cid], 1.0))[:3]
        print('GT cube (base):', gt_base.round(4))

        # 抓取点吸附（2026-08-03, 挤出事故修复）: CGN 候选位置偏 2-3mm
        # （单侧余量仅 2.5mm）且接触面偏高（z≈0.06 = 8cm 块上 1/3）——
        # 闭合时指腹斜向挤压把块挤出（wrist 帧 s00230 实锤）。
        # 修正: x/y 吸附到 mask 点云 (p5+p95)/2 中心, z 压到块中腰
        # （云顶 - GT 半高 0.04）。CGN 只贡献姿态, 位置以实测点云为准。
        vs_, us_ = np.nonzero(seg > 0)
        z_ = depth[vs_, us_].astype(np.float64).reshape(-1)
        ok_ = np.isfinite(z_) & (z_ > 0.05)
        vs_, us_, z_ = vs_[ok_], us_[ok_], z_[ok_]
        x_ = (us_ - K[0, 2]) * z_ / K[0, 0]
        y_ = -(vs_ - K[1, 2]) * z_ / K[1, 1]
        pts_b = (pose_mat[:3, :3] @ np.stack([x_, y_, z_], 1).T).T + pose_mat[:3, 3]
        ctr_xy = (np.percentile(pts_b, 5, axis=0) + np.percentile(pts_b, 95, axis=0)) / 2
        cube_top_z = float(np.percentile(pts_b[:, 2], 95))
        print(f'点云吸附: 中心=({ctr_xy[0]:.4f},{ctr_xy[1]:.4f}) 云顶 z={cube_top_z:.4f}')

        # 可达性重排（2026-08-04 根因修复）: top-score 候选的近竖直位姿常落在
        # 可达流形薄点外（库实测: 目标 20mm 内最优逼近轴偏差 45.7°, IK 12/12
        # 不收敛）; 按库可达性（grasp 高度联合最近距 dp/0.05+da/34.4°）升序
        # 重排后再按序尝试——最可达候选实测 solve_ik 预抓/抓取双高度收敛。
        # yaw 族只绕逼近轴转, 不改变位置/逼近轴, 不受影响。
        # 【2026-08-04 用户指定: 默认严格按 CGN score 序, 不重排; --legacy-adjust 恢复】
        lib_ = _ik_library()
        if args.legacy_adjust and lib_ is not False and len(grasps_f):
            _P, _Z = lib_['P'], lib_['Z']
            _rk = []
            for _i in range(len(grasps_f)):
                _gb = cgn_to_gripper(grasps_f[_i], pose_mat)
                _p = _gb[:3, 3].copy()
                _p[0], _p[1] = ctr_xy[0], ctr_xy[1]
                _p[2] = min(_p[2], cube_top_z - 0.04)
                _dp = np.linalg.norm(_P - _p, axis=1)
                _da = np.rad2deg(np.arccos(np.clip(_Z @ _gb[:3, 2], -1, 1)))
                _rk.append(float((_dp / 0.05 + _da / 34.4).min()))
            _order = np.argsort(np.asarray(_rk), kind='stable')
            grasps_f = grasps_f[_order]
            scores_f = scores_f[_order]
            openings_f = openings_f[_order]
            print(f'可达性重排: 前 {args.max_candidates} 尝试序 '
                  f'(联合距 {[round(_rk[int(i)], 2) for i in _order[:args.max_candidates]]})')

        # --view: CGN 候选 3D glyph（品红=top-1, 绿=其余; 位姿经点云吸附与执行链
        # 一致——2026-08-04 用户反馈"glyph 还是太高", 原版画的是未吸附原始位姿）
        if viewer is not None:
            for vi in range(min(8, len(grasps_f))):
                gb_v = cgn_to_gripper(grasps_f[vi], pose_mat)
                if args.legacy_adjust:
                    gb_v[:3, 3][0], gb_v[:3, 3][1] = ctr_xy[0], ctr_xy[1]
                    gb_v[2, 3] = min(gb_v[2, 3], cube_top_z - 0.04)
                Tv = engine.T_world_base @ gb_v
                O, Sz, Sy = Tv[:3, 3], Tv[:3, 2], Tv[:3, 1]
                wv = PIPER_MAX_WIDTH  # 与执行一致: 满开 70mm 接近（2026-08-04 修复,
                                      # 原画 cgn_open 预测值, 与实体满开不符）
                C = O - Sz * 0.050  # 指尖极值（实测 -50mm, 用户审定版）
                Pp = O + Sz * 0.023  # 掌心可见表面（实测 +23mm）
                rgba = (1.0, 0.0, 1.0, 0.8) if vi == 0 else (0.0, 1.0, 0.2, 0.45)
                for A, B in [(Pp - Sy * wv / 2, Pp + Sy * wv / 2),
                             (Pp + Sy * wv / 2, C + Sy * wv / 2),
                             (Pp - Sy * wv / 2, C - Sy * wv / 2),
                             (Pp, Pp + Sz * 0.05)]:
                    _add_connector(viewer.user_scn, A, B, 0.004, rgba)
            viewer.sync()

        # yaw 变体（2026-07-31 批准）：绕抓取点处逼近轴 {0,90,180,270}° 四个旋转位姿。
        # 依据: cubeA 水平截面为正方形(axis-aligned) ⇒ 90° 旋转后闭合轴仍跨 4cm 对面,
        # contact_dist 不变。【仅对方形截面目标启用】位置不动只转姿态。
        YAW_STEPS_DEG = [0, 90, 180, 270] if args.legacy_adjust else [0]
        # 【2026-08-04 用户指定】默认严格按 CGN 输出: 不转 yaw、不吸附,
        # 到 CGN 位姿后直接闭合; --legacy-adjust 恢复旧调整逻辑。
        n_pre_ok = 0  # pre-grasp IK（任一 yaw）成功的候选数
        for ci in range(min(args.max_candidates, len(grasps_f))):
            g_base_pose = cgn_to_gripper(grasps_f[ci], pose_mat)
            print(f'\n=== 候选 {ci} (score={scores_f[ci]:.3f}, cgn_open={openings_f[ci]:.4f}) ===')
            # 场景扰动守卫（2026-08-04）: 前次尝试可能已碰跑方块——方块当前 GT
            # 偏离感知快照 >2cm 时, 后续候选全是抓空气, 中止 CGN 线。
            # （sim GT 仅作守卫, 不进感知/规划; 真机阶段改为重感知）
            cube_now = (engine.T_base_world @ np.append(
                engine.env.sim.data.xpos[cid], 1.0))[:3]
            _drift = float(np.linalg.norm(cube_now[:2] - ctr_xy[:2]))
            if _drift > 0.02:
                print(f'方块已被扰动（偏离感知快照 {_drift*1000:.0f}mm），'
                      '后续候选失效，中止 CGN 线')
                break
            solved = None
            for yaw_deg in YAW_STEPS_DEG:
                c_, s_ = np.cos(np.deg2rad(yaw_deg)), np.sin(np.deg2rad(yaw_deg))
                T_yaw = np.eye(4)
                T_yaw[:3, :3] = np.array([[c_, -s_, 0], [s_, c_, 0], [0, 0, 1]])
                g_try = g_base_pose @ T_yaw  # 局部 z=逼近轴, 右乘=绕抓取点原地转
                quat_wxyz = quat_wxyz_from_mat(g_try[:3, :3])
                pos_grasp = g_try[:3, 3].copy()
                if args.legacy_adjust:
                    # 点云吸附: x/y 到云中心, z 压到块中腰（云顶−0.04）, 更低不压
                    pos_grasp[0] = ctr_xy[0]
                    pos_grasp[1] = ctr_xy[1]
                    pos_grasp[2] = min(pos_grasp[2], cube_top_z - 0.04)
                # 修复后约定（2026-08-03）: 目标 z列 = −逼近方向 →
                # 后撤方向 = +z列（沿 −真逼近 = 离开物体）
                # 【2026-08-05】回退改接触面基准: site+0.038 ≈ 接触面上方 10cm
                # （原 site+0.10 在指尖对齐锚点(站高+62mm)后顶进 z≥0.18 的
                # 可达性死区——高度-倾角地图实测全灭; check_cgn_pose 同修复）
                pos_pre = pos_grasp + g_try[:3, 2] * 0.038
                # 预筛拆门留哨（2026-08-03 裁决 1）: 合取硬门在稀库下 100%
                # 误杀（5 seed 15/15 候选全 veto, 而同场景直接调 solve_ik
                # 52% 收敛）——移除一切拦截, 仅留 advisory 配对记录
                # "预筛本会判什么 + 实际结果", 攒数据后再议去留。
                adv_ok, (mdp, mda) = lib_feasible(pos_grasp, g_try[:3, 2])
                adv = (f'[advisory预筛: {"PASS" if adv_ok else "VETO"} '
                       f'dp={mdp:.3f} da={mda:.2f}]')
                if not args.legacy_adjust:
                    # 严格模式: 不做预抓 IK 预检（直达路径由种子锁逐步求解,
                    # 预检解的位姿与实走无关, 检了也是误杀/误导）
                    solved = (yaw_deg, quat_wxyz, pos_grasp, pos_pre, None)
                    break
                try:
                    q_pre = ns['solve_ik'](pos_pre, quat_wxyz, free_approach_roll=True)
                    solved = (yaw_deg, quat_wxyz, pos_grasp, pos_pre, q_pre)
                    print(f'  yaw={yaw_deg}: pre-grasp IK 收敛 {adv}')
                    break
                except RuntimeError:
                    print(f'  yaw={yaw_deg}: pre-grasp IK 未收敛 {adv}')
                    continue
            if solved is None:
                print(f'候选 {ci} 失败: 4 个 yaw 的 pre-grasp IK 均未收敛')
                continue
            n_pre_ok += 1
            yaw_deg, quat_wxyz, pos_grasp, pos_pre, q_pre = solved
            print('grasp pos (base):', pos_grasp.round(4), ' pre:', pos_pre.round(4))
            try:
                # 接近开度 = 满开 70mm（2026-08-04 根因修复: 曾按
                # min(cgn_width, MAX)=56.6mm 接近, 斜逼近时方块楔入双垫
                # 把指垫撬开过位（28.3→36.1mm）、推跑方块、闭合摩擦自锁
                # ——trace 实锤。满开 70mm > 方块对角线 56.6mm, 任意滚转
                # 朝向都能无损放入）
                dense, abort_i = [], 0  # 流式路径记录（异常撤回用, 2026-08-05）
                ramp_gripper(1.0, 10)  # 渐进全开（用户: 夹爪太快）
                if args.legacy_adjust:
                    ns['move_to_joints'](q_pre)
                    print('预抓取到位')
                    # 下降（与 pre-grasp 同一 yaw）: 笛卡尔细分 + 种子锁（2026-08-04
                    # 根因修复——预抓/下降两次独立 IK 跳分支实测 Δq=4.1rad, 关节空间
                    # 直扫画大弧撞块卡死; 1cm 细分 + 上步构型做种子 = 分支锁定直线下压）
                    q_cur = q_pre
                    for k in range(1, 11):
                        pos_k = pos_pre + (pos_grasp - pos_pre) * (k / 10)
                        q_cur = ns['solve_ik'](pos_k, quat_wxyz,
                                               free_approach_roll=True, seed=q_cur)
                        ns['move_to_joints'](q_cur, tol=0.10, fk_tol=0.03)  # 接触段放宽
                else:
                    # 【2026-08-04 用户指定】直达 CGN 位姿 + 直接闭合:
                    # 先算后动（用户反馈频闪: 边算边动 = 每子步 IK 冻结 ~1-2s
                    # + 运动 0.2s, 观感是瞬移+阶梯下压）——全路径（过境终点+
                    # 下降链+抬升链）先一次解算完, 再连续执行, 中途无冻结。
                    # 过境段走关节空间（自由空间可靠）; 末段沿逼近轴种子锁细分。
                    # 【2026-08-05 移植 check_cgn_pose 修复链】中途点 loose
                    # （末步严格）; 预抓+下降段走 RRT 避障规划（含腕部, 手-方块
                    # 净距余量 2mm）+ 密化流式下发（0.03rad/点, 仅末点阻塞）;
                    # 抬升段负载方块在爪上, 不适用静态场景规划, 保持原阻塞执行。
                    q_pre = ns['solve_ik'](pos_pre, quat_wxyz, free_approach_roll=True,
                                           loose=True)
                    descend = []
                    q_cur = q_pre
                    for k in range(1, 5):
                        pos_k = pos_pre + (pos_grasp - pos_pre) * (k / 4)
                        q_cur = ns['solve_ik'](pos_k, quat_wxyz,
                                               free_approach_roll=True, seed=q_cur,
                                               loose=(k < 4))
                        descend.append(q_cur)
                    pos_lift = pos_grasp + np.array([0, 0, 0.05])
                    lift = []
                    for k in range(1, 6):
                        pos_k = pos_grasp + (pos_lift - pos_grasp) * (k / 5)
                        q_cur = ns['solve_ik'](pos_k, quat_wxyz,
                                               free_approach_roll=True, seed=q_cur,
                                               loose=True)
                        lift.append(q_cur)
                    # 逐腿: 即时规划（读最新方块位姿）→ 密化流式 → 短沉降;
                    # 仅末腿终点阻塞核验。【2026-08-05 剐蹭-移位事故: 全腿预算
                    # 在执行前一次做完, 早段亚毫米擦碰移位方块后, 后续路径过期
                    # → 末步卡死 err=0.11rad; 改逐腿即时规划, 每条腿都针对
                    # 方块当前位姿规划】
                    q_from = engine.current_arm_qpos()
                    for li, q_leg in enumerate([q_pre] + descend):
                        path = plan_joint_path(q_from, q_leg, engine)
                        if path is None:
                            raise RuntimeError(f'避障规划失败(腿{li})')
                        if len(path) > 2:
                            print(f'  [plan] 腿{li}: RRT 绕行 {len(path)} 点')
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
                        if li < len(descend):
                            for _ in range(20):
                                engine.step_joints(dense[-1])
                                if np.linalg.norm(
                                        engine.current_arm_qpos() - dense[-1]) < 0.05:
                                    break
                        q_from = dense[-1]
                    # 末腿终点阻塞核验（到位即真到位, 闭合无沉降）
                    ns['move_to_joints'](descend[-1], tol=0.02, fk_tol=0.005)
                print('到位, site z(base):',
                      float((engine.T_base_world @ np.append(
                          engine.env.sim.data.site_xpos[
                              mujoco.mj_name2id(model_, mujoco.mjtObj.mjOBJ_SITE,
                                                "gripper0_right_grip_site")], 1.0))[2]).__round__(4))
                # 闭合: 渐进全闭（2026-08-05 用户: 夹爪动作太快——斜坡 25 ticks
                # 合拢）。全闭目标不变（2026-08-03 实锤 kp=40 在 contact−2mm 目标
                # 下仅 0.08N 夹持力会滑落; 现 kp=120, q7→0 时 kp·err≈3.5N）
                ramp_gripper(0.0, 25)
                engine.hold_ticks(10)
                print('已闭合')
                # 闭合接触自检（2026-08-04）: 双垫是否都吃上方块 + 实际 q7
                _npc = 0
                for _c in range(engine.env.sim.data.ncon):
                    _g1 = mujoco.mj_id2name(model_, mujoco.mjtObj.mjOBJ_GEOM,
                                            engine.env.sim.data.contact.geom1[_c]) or ''
                    _g2 = mujoco.mj_id2name(model_, mujoco.mjtObj.mjOBJ_GEOM,
                                            engine.env.sim.data.contact.geom2[_c]) or ''
                    if ('finger' in _g1 and 'cubeA' in _g2) or ('finger' in _g2 and 'cubeA' in _g1):
                        _npc += 1
                _q7j = mujoco.mj_name2id(model_, mujoco.mjtObj.mjOBJ_JOINT,
                                         'gripper0_right_joint7')
                _q7v = float(engine.env.sim.data.qpos[model_.jnt_qposadr[_q7j]])
                print(f'闭合自检: 指垫-方块接触 {_npc} 处, q7={_q7v*1000:.1f}mm '
                      f'(方块半宽 20mm, 双侧接触期望 ≥2 处)')
                if _npc == 0:
                    # 全接触对转储（摩擦自锁/撬开/抓空 取证用）
                    _pairs = []
                    for _c in range(engine.env.sim.data.ncon):
                        _g1 = mujoco.mj_id2name(model_, mujoco.mjtObj.mjOBJ_GEOM,
                                                engine.env.sim.data.contact.geom1[_c]) or ''
                        _g2 = mujoco.mj_id2name(model_, mujoco.mjtObj.mjOBJ_GEOM,
                                                engine.env.sim.data.contact.geom2[_c]) or ''
                        if 'finger' in _g1 or 'finger' in _g2:
                            _pairs.append((_g1, _g2))
                    print(f'闭合自检: 指部相关全部接触对 = {_pairs if _pairs else "无（抓空）"}')
                # 抬升 5cm（同种子锁, 负载放宽; 严格模式用预先算好的 lift 链,
                # 避免边算边动的频闪）
                if args.legacy_adjust:
                    pos_lift = pos_grasp + np.array([0, 0, 0.05])
                    for k in range(1, 6):
                        pos_k = pos_grasp + (pos_lift - pos_grasp) * (k / 5)
                        q_cur = ns['solve_ik'](pos_k, quat_wxyz,
                                               free_approach_roll=True, seed=q_cur)
                        ns['move_to_joints'](q_cur, tol=0.10, fk_tol=0.03)
                else:
                    for q_k in lift:
                        ns['move_to_joints'](q_k, tol=0.10, fk_tol=0.03)
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
                # 失败复位: 松开+撤回（严格模式回 TUCK; 2026-08-05 改避障
                # 规划——释放后方块留在抓取点, 裸扫可能蹭块）
                engine.set_gripper(1.0) if args.legacy_adjust else ramp_gripper(1.0, 10)
                if args.legacy_adjust:
                    ns['move_to_joints'](q_pre, tol=0.10, fk_tol=0.03)
                else:
                    _p = plan_joint_path(engine.current_arm_qpos(), TUCK_Q, engine)
                    for _q in (_p[1:] if _p is not None else [TUCK_Q]):
                        ns['move_to_joints'](_q, tol=0.10, fk_tol=0.03)  # 接触后回撤放宽
            except RuntimeError as e:
                print(f'候选 {ci} yaw={yaw_deg} 失败: {type(e).__name__}: {e}')
                try:
                    if dense:  # 先沿本腿已流路径退回腿起点（沿途已查碰撞）
                        for q_b in reversed(dense[:abort_i + 1]):
                            engine.step_joints(q_b)
                    _p = plan_joint_path(engine.current_arm_qpos(), TUCK_Q, engine)
                    if _p is not None:  # 规划回 TUCK（读最新方块位姿）
                        _wp = [engine.current_arm_qpos()] + _p[1:]
                        for a, b in zip(_wp, _wp[1:]):
                            _n = max(1, int(np.ceil(np.abs(b - a).max() / 0.05)))
                            for ii in range(1, _n + 1):
                                engine.step_joints(a + (b - a) * (ii / _n))
                    ramp_gripper(1.0, 10)
                    engine.hold_ticks(10)
                    ns['move_to_joints'](TUCK_Q, tol=0.10, fk_tol=0.05)
                except Exception:
                    pass
                continue
        n_tried = min(args.max_candidates, len(grasps_f))
        if n_tried:
            print(f'\npre-grasp IK（任一 yaw）成功率: {n_pre_ok}/{n_tried}')

        # 安全网（2026-08-03 批准, 官方 cube_reset pick_object 蓝本）:
        # CGN 候选全灭时启用几何规划器（重力吸附竖直逼近 + 8-yaw 采样,
        # 姿态天生落在臂可达流形内）。日志记 fallback_used=true, 不静默替换主线。
        # --once 单发模式跳过（2026-08-04 用户要求）。
        if not success and not args.once:
            print('\n=== FALLBACK: plan_grasp 几何安全网 (fallback_used=true) ===')
            ctx_fb = PrimitiveContextCapx(engine)
            fb_grasps, fb_scores = ctx_fb._plan_grasp_geometric(depth, K, seg)
            print('geometric candidates:', len(fb_grasps),
                  'scores:', fb_scores.round(3) if len(fb_scores) else fb_scores)
            for fi in range(min(3, len(fb_grasps))):
                g_fb = fb_grasps[fi]
                approach = g_fb[:3, 2]
                quat_wxyz = quat_wxyz_from_mat(g_fb[:3, :3])
                pos_grasp = g_fb[:3, 3].copy()
                pos_pre = pos_grasp + approach * 0.038  # 同上: z列=−逼近, 后撤取 +;
                # 0.038=接触面上 10cm（2026-08-05 死区修复, 详见主线注释）
                print(f'\n--- fallback 候选 {fi} (score={fb_scores[fi]:.3f}) ---')
                print('grasp pos (base):', pos_grasp.round(4))
                try:
                    ramp_gripper(1.0, 10)
                    q_pre = ns['solve_ik'](pos_pre, quat_wxyz)
                    ns['move_to_joints'](q_pre)
                    print('预抓取到位')
                    q_grasp = ns['solve_ik'](pos_grasp, quat_wxyz)
                    ns['move_to_joints'](q_grasp, tol=0.10, fk_tol=0.03)  # 接触段放宽
                    print('下降到位')
                    ramp_gripper(0.0, 25)  # 渐进全闭（夹持力理由见主线注释）
                    engine.hold_ticks(10)
                    print('已闭合')
                    pos_lift = pos_grasp + np.array([0, 0, 0.05])
                    q_lift = ns['solve_ik'](pos_lift, quat_wxyz)
                    ns['move_to_joints'](q_lift, tol=0.10, fk_tol=0.03)   # 负载抬升放宽
                    print('抬升到位')
                    engine.hold_ticks(40)
                    z_end = cube_z_base(engine)
                    held = z_end > gt_base[2] + 0.03
                    print(f'悬停结束 cube z(base)={z_end:.4f} (GT {gt_base[2]:.4f}) → '
                          + ('夹持成功' if held else '方块滑落'))
                    if held:
                        success = True
                        print(f'\nSUCCESS: fallback 候选 {fi} 抓取成功 (fallback_used=true)')
                        break
                    ramp_gripper(1.0, 10)
                    ns['move_to_joints'](q_pre, tol=0.10, fk_tol=0.03)  # 接触后回撤放宽
                except RuntimeError as e:
                    print(f'fallback 候选 {fi} 失败: {type(e).__name__}: {e}')
                    try:
                        ramp_gripper(1.0, 10)
                        engine.hold_ticks(10)
                    except Exception:
                        pass
                    continue

        print('\n' + '=' * 56)
        print('RESULT:', 'PASS' if success else 'FAIL')
        print('trace dir:', engine.trace_dir)
    except Exception:
        error = traceback.format_exc()
        raise
    finally:
        engine.mark_trace_result(success, error)
        if viewer is not None and viewer.is_running():
            print('[view] 执行结束，结果场景保持显示（关窗退出）')
            import time as _time2
            while viewer.is_running():
                viewer.sync()
                _time2.sleep(0.05)
        engine.close()


if __name__ == '__main__':
    main()
