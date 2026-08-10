"""实时围观脚本（2026-08-05 用户要求）：循环复现"抓-搬-放"失败现场。

窗口常开，每轮 ~1-2 分钟：红块传送回固定点 → 收臂让拍 → SAM3 定位 →
携带兼容姿态族候选链预解 → 慢速执行（3x 慢放）→ 持块停留 6 秒供观察。
直到用户关闭窗口 / Ctrl+C。

用法:
    MUJOCO_GL=glfw /home/stouching/anaconda3/envs/ASPIRE/bin/python scripts/tools/watch_grasp_live.py
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mujoco

from aspire.engine.engine_capx import ExecutionEngineCapx, TUCK_Q

SLOWDOWN = 3.0
HOME_RED = np.array([0.1534, -0.1202, 0.0398])  # 基座系（seed0 clutter0 原始落点）
RED_HEIGHT = 0.08


def teleport_red(engine, pos_base):
    """把 cubeA 传送回指定基座系位置（直立, 零速度）。"""
    model = engine.env.sim.model._model
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "cubeA_joint0")
    adr = int(model.jnt_qposadr[jid])
    dof = int(model.jnt_dofadr[jid])
    p_world = (engine.T_world_base @ np.append(pos_base, 1.0))[:3]
    qpos = engine.env.sim.data.qpos
    qpos[adr:adr + 3] = p_world
    qpos[adr + 3:adr + 7] = [1.0, 0.0, 0.0, 0.0]
    engine.env.sim.data.qvel[dof:dof + 6] = 0.0
    mujoco.mj_forward(model, engine.env.sim.data._data if hasattr(engine.env.sim.data, "_data") else engine.env.sim.data)
    engine.hold_ticks(10)


def seg(p0, p1, n):
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    return [p0 + (p1 - p0) * (k / n) for k in range(1, n + 1)]


def synth_candidates(center, top_z):
    """携带兼容姿态族（tilt 25-45° × az 0/60/300, 接触上 1/3）。"""
    contact = np.array([center[0], center[1], top_z - 0.02])
    out = []
    for tilt_deg in (32, 25, 38, 45):
        for az_deg in (0, 60, 300):
            t, a = np.deg2rad(tilt_deg), np.deg2rad(az_deg)
            approach = np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), -np.cos(t)])
            yaxis = np.array([1.0, 0.0, 0.0]) - approach[0] * approach
            yaxis /= np.linalg.norm(yaxis)
            xaxis = np.cross(yaxis, approach)
            R = np.column_stack([xaxis, yaxis, approach]) @ np.diag([-1.0, 1.0, -1.0])
            g = np.eye(4)
            g[:3, :3] = R
            g[:3, 3] = contact - 0.0412 * approach
            out.append((g, f"t{tilt_deg}/a{az_deg}"))
    return out


def main():
    engine = ExecutionEngineCapx(task="Stack", seed=0, trace_root="traces",
                                 clutter=0, render=True, render_slowdown=SLOWDOWN)
    ns = engine._build_namespace()
    print("窗口已打开。每轮: 传送红块 → 让拍 → 定位 → 预解 → 执行。关窗/Ctrl+C 结束。")

    rnd = 0
    while True:
        rnd += 1
        print(f"\n{'=' * 60}\n第 {rnd} 轮\n{'=' * 60}")
        teleport_red(engine, HOME_RED)
        ns["move_to_joints"](TUCK_Q)

        obs = ns["get_observation"]()
        cam = obs["robot0_robotview"]
        rgb, depth, K, E = cam["images"]["rgb"], cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"]
        masks = ns["segment_sam3_text_prompt"](rgb, "red cube")
        if not masks:
            print("SAM3 未检出红块, 跳过本轮")
            continue
        m = masks[0]["mask"]
        vs, us = np.nonzero(m)
        z = depth[vs, us].astype(float)
        ok = np.isfinite(z) & (z > 0.05)
        x = (us[ok] - K[0, 2]) * z[ok] / K[0, 0]
        y = -(vs[ok] - K[1, 2]) * z[ok] / K[1, 1]
        pts = (E @ np.stack([x, y, z[ok], np.ones_like(z[ok])], 0))[:3].T
        in_ws = (pts[:, 2] > -0.05) & (pts[:, 2] < 0.20)
        pts = pts[in_ws]
        lo, hi = np.percentile(pts, 5, axis=0), np.percentile(pts, 95, axis=0)
        center, top_z = (lo + hi) / 2, float(hi[2])
        print(f"红块: center={center.round(4)} top_z={top_z:.4f}")

        green_masks = ns["segment_sam3_text_prompt"](rgb, "green cube")
        gm = green_masks[0]["mask"]
        vs, us = np.nonzero(gm)
        z = depth[vs, us].astype(float)
        ok = np.isfinite(z) & (z > 0.05)
        x = (us[ok] - K[0, 2]) * z[ok] / K[0, 0]
        y = -(vs[ok] - K[1, 2]) * z[ok] / K[1, 1]
        gpts = (E @ np.stack([x, y, z[ok], np.ones_like(z[ok])], 0))[:3].T
        glo, ghi = np.percentile(gpts, 5, axis=0), np.percentile(gpts, 95, axis=0)
        gcenter, gtop = (glo + ghi) / 2, float(ghi[2])
        print(f"绿块: center={gcenter.round(4)} top_z={gtop:.4f}")

        # 候选链预解（先算后动）
        chosen = None
        for g, tag in synth_candidates(center, top_z):
            quat = ns["rotation_matrix_to_quaternion"](g[:3, :3])
            z_col = g[:3, 2]
            pos_grasp = g[:3, 3].copy()
            pos_pre = pos_grasp + z_col * 0.05
            stb = float(pos_grasp[2] - (top_z - RED_HEIGHT))
            place_z = gtop + stb + 0.008
            place_pos = np.array([gcenter[0], gcenter[1], place_z])
            lift_top = pos_grasp + np.array([0, 0, 0.08])
            hover = np.array([place_pos[0], place_pos[1], max(lift_top[2], place_z + 0.04)])
            pre_place = np.array([place_pos[0], place_pos[1], place_z + 0.025])
            try:
                q_cur = np.asarray(ns["get_observation"]()["robot_joint_pos"][:6], float)
                legs = []
                for name, pts_ in [
                    ("approach", [pos_pre] + seg(pos_pre, pos_grasp, 4)),
                    ("lift", seg(pos_grasp, lift_top, 6)),
                    ("transport", seg(lift_top, hover, 6)),
                    ("place", seg(hover, pre_place, 2) + seg(pre_place, place_pos, 3)),
                ]:
                    leg = []
                    for i, p in enumerate(pts_):
                        q_cur = ns["solve_ik"](p, quat, free_approach_roll=True,
                                               seed=q_cur, loose=(i < len(pts_) - 1))
                        leg.append(q_cur)
                    legs.append((name, leg))
                chosen = (tag, quat, legs)
                print(f"候选 {tag}: 全链预解收敛 (stb={stb:.3f})")
                break
            except RuntimeError as e:
                print(f"候选 {tag}: 预解失败 ({e})")
        if chosen is None:
            print("全部候选预解失败, 下一轮")
            continue

        tag, quat, legs = chosen
        print(f">>> 执行候选 {tag}（3x 慢放）<<<")
        try:
            ns["open_gripper"]()
            for q in legs[0][1][:-1]:
                ns["move_to_joints"](q, tol=0.06, fk_tol=0.05)
            ns["move_to_joints"](legs[0][1][-1], tol=0.02, fk_tol=0.005)
            ns["close_gripper"]()
            frac = ns["get_observation"]()["robot_joint_pos"][-1]
            print(f"闭合后 frac={frac:.3f}")
            for name, leg in legs[1:]:
                print(f"  执行 {name} 腿...")
                for q in leg:
                    ns["move_to_joints"](q, tol=0.10, fk_tol=0.03)
                frac = float(ns["get_observation"]()["robot_joint_pos"][-1])
                print(f"  {name} 后腿角 frac={frac:.3f}" + ("  ← 滑脱!" if frac < 0.5 else ""))
            print("到位, 停留 6 秒供观察（注意爪内是否有块）")
            t0 = time.time()
            while time.time() - t0 < 6.0:
                engine.hold_ticks(20)
            ns["open_gripper"]()
        except RuntimeError as e:
            print(f"执行失败: {e}")
            ns["open_gripper"]()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("用户中断")
