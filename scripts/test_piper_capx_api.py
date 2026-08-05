"""cap-x 标准 API（Piper）自检 + 外参标定验证。

逐项调用 cap-x 函数集，并用 GT（仅测试可用）校验：
  - 观测结构/坐标系（基座系）
  - SAM3 文本/点提示分割（256×256 robotview，EGL 稳定配置）
  - 反投影约定（pose_mat + y 负号规则）vs GT 方块位置
  - plan_grasp 候选位姿合理性（变换到基座系后应落在方块中心附近）
  - solve_ik / move_to_joints / open/close_gripper 基本运动

用法:
    MUJOCO_GL=egl /home/stouching/anaconda3/envs/ASPIRE/bin/python scripts/test_piper_capx_api.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco

from aspire.engine_capx import ExecutionEngineCapx

PASS, FAIL = "[PASS]", "[FAIL]"
results = []


def check(name, ok, detail=""):
    results.append((name, bool(ok)))
    print(f"{PASS if ok else FAIL} {name}  {detail}")


engine = ExecutionEngineCapx(task="Stack", seed=0, trace_root="traces")
ns = engine._build_namespace()
get_observation = ns["get_observation"]

# --- A. 观测结构 ------------------------------------------------------------
obs = get_observation()
cam = obs["robot0_robotview"]
rgb, depth, K, E = cam["images"]["rgb"], cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"]
check("A1 obs robotview rgb (256,256,3) u8", rgb.shape == (256, 256, 3) and rgb.dtype == np.uint8, str(rgb.shape))
check("A2 obs depth (256,256) 米制", depth.shape == (256, 256) and 0.2 < float(np.nanmedian(depth)) < 3.0,
      f"median={float(np.nanmedian(depth)):.3f}m")
check("A3 pose_mat 刚体 (det=+1)", abs(np.linalg.det(E[:3, :3]) - 1.0) < 1e-6, f"det={np.linalg.det(E[:3,:3]):.6f}")
check("A4 robot_joint_pos (7,)", obs["robot_joint_pos"].shape == (7,), str(obs["robot_joint_pos"].shape))
check("A5 robot_cartesian_pos (8,)", obs["robot_cartesian_pos"].shape == (8,), str(obs["robot_cartesian_pos"].shape))

# GT（仅测试用）: 方块世界位置 → 基座系
model = engine.env.sim.model._model
cidA = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cubeA_main")
cidB = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cubeB_main")
gtA_world = engine.env.sim.data.xpos[cidA].copy()
gtB_world = engine.env.sim.data.xpos[cidB].copy()
gtA_base = (engine.T_base_world @ np.append(gtA_world, 1.0))[:3]
gtB_base = (engine.T_base_world @ np.append(gtB_world, 1.0))[:3]
print(f"  GT cubeA base={gtA_base.round(3)}  cubeB base={gtB_base.round(3)}")
tcp_base = obs["robot_cartesian_pos"][:3]
site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site")
gt_tcp = (engine.T_base_world @ np.append(engine.env.sim.data.site_xpos[site_id], 1.0))[:3]
check("A6 cartesian_pos == grip_site (基座系)", np.linalg.norm(tcp_base - gt_tcp) < 1e-9,
      f"tcp={tcp_base.round(3)}")

# --- B. SAM3 分割 ------------------------------------------------------------
masks_r = ns["segment_sam3_text_prompt"](rgb, "red cube")
masks_g = ns["segment_sam3_text_prompt"](rgb, "green cube")
check("B1 sam3 text: red cube", len(masks_r) > 0, f"{len(masks_r)} masks")
check("B2 sam3 text: green cube", len(masks_g) > 0, f"{len(masks_g)} masks")

# --- C. 反投影标定（pose_mat × [x, -y_im, z] 应等于 GT 基座系位置） -------------
def deproject(mask):
    vs, us = np.nonzero(mask)
    z = depth[vs, us].astype(np.float64)
    ok = np.isfinite(z) & (z > 0.05)
    us, vs, z = us[ok], vs[ok], z[ok]
    x = (us - K[0, 2]) * z / K[0, 0]
    y = -(vs - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z, np.ones_like(z)], axis=0)
    return (E @ pts)[:3].T

if masks_r:
    pts_r = deproject(masks_r[0]["mask"])
    estA = np.median(pts_r, axis=0)
    # 可见面偏置: mask 只覆盖顶面+朝向相机的侧面, 中位数应落在方块表面附近
    errA = float(np.linalg.norm(estA - gtA_base))
    z_okA = gtA_base[2] - 0.005 < estA[2] < gtA_base[2] + 0.03
    check("C1 反投影 red cube ↔ GT 表面一致", errA < 0.04 and z_okA,
          f"err={errA*100:.2f}cm est={estA.round(4)} (cube half=2cm)")
if masks_g:
    pts_g = deproject(masks_g[0]["mask"])
    estB = np.median(pts_g, axis=0)
    errB = float(np.linalg.norm(estB - gtB_base))
    z_okB = gtB_base[2] - 0.005 < estB[2] < gtB_base[2] + 0.035
    check("C2 反投影 green cube ↔ GT 表面一致", errB < 0.05 and z_okB,
          f"err={errB*100:.2f}cm est={estB.round(4)} (cube half=2.5cm)")

# --- D. plan_grasp ------------------------------------------------------------
seg = np.zeros(rgb.shape[:2], dtype=np.int32)
if masks_r:
    seg[masks_r[0]["mask"]] = 1
grasps, scores = ns["plan_grasp"](depth, K, seg)
check("D1 plan_grasp 返回 (K,4,4)+(K,)", grasps.ndim == 3 and grasps.shape[1:] == (4, 4) and len(scores) == len(grasps),
      f"K={len(grasps)}")
if len(grasps):
    best_base = E @ grasps[0]
    g_pos = best_base[:3, 3]
    # 抓取 TCP 应落在方块中心附近（顶面下探 2cm → ≈方块中心高度）
    err_g = float(np.linalg.norm(g_pos - gtA_base))
    check("D2 plan_grasp 候选位置 ↔ 方块中心 (<2.5cm)", err_g < 0.025,
          f"err={err_g*100:.2f}cm g_pos={g_pos.round(4)}")
    zaxis = best_base[:3, 2]
    check("D3 plan_grasp 逼近轴朝下", zaxis[2] < -0.5, f"zaxis={zaxis.round(3)}")

# --- E. 点提示/molmo ----------------------------------------------------------
if masks_r:
    ys, xs = np.nonzero(masks_r[0]["mask"])
    pmasks = ns["segment_sam3_point_prompt"](rgb, (int(xs.mean()), int(ys.mean())))
    check("E1 sam3 point prompt", len(pmasks) > 0, f"{len(pmasks)} masks")
mpts = ns["point_prompt_molmo"](rgb, "red cube")
muv = list(mpts.values())[0]
check("E2 point_prompt_molmo", muv[0] is not None, str(mpts))

# --- F. OBB -------------------------------------------------------------------
if masks_r:
    obb = ns["get_oriented_bounding_box_from_3d_points"](pts_r)
    check("F1 OBB extent ≈ 4cm 方块", 0.02 < float(np.median(obb["extent"])) < 0.06,
          f"extent={obb['extent'].round(3)}")

# --- G. 运动 ------------------------------------------------------------------
try:
    q_home = np.asarray(obs["robot_joint_pos"])[:6].copy()
    # 当前 TCP 上方 5cm 的 IK
    target = tcp_base + np.array([0.0, 0.0, 0.05])
    quat = obs["robot_cartesian_pos"][3:7]
    joints = ns["solve_ik"](target, quat)
    check("G1 solve_ik (TCP 上方5cm)", joints.shape == (6,), f"q={joints.round(3)}")
    ns["move_to_joints"](joints)
    obs2 = get_observation()
    err = float(np.linalg.norm(obs2["robot_cartesian_pos"][:3] - target))
    check("G2 move_to_joints 到位 (<8mm)", err < 0.008, f"err={err*1000:.1f}mm")
    ns["move_to_joints"](q_home)
    obs3 = get_observation()
    errh = float(np.linalg.norm(np.asarray(obs3["robot_joint_pos"][:6]) - q_home))
    check("G3 move_to_joints 回家", errh < 0.05, f"err={errh:.4f}rad")
except RuntimeError as e:
    check("G1-G3 运动链", False, str(e))

# --- H. 夹爪 ------------------------------------------------------------------
ns["close_gripper"]()
g_closed = float(get_observation()["robot_joint_pos"][-1])
ns["open_gripper"]()
g_open = float(get_observation()["robot_joint_pos"][-1])
check("H1 close_gripper", g_closed < 0.6, f"frac={g_closed:.3f}")
check("H2 open_gripper", g_open > 0.9, f"frac={g_open:.3f}")

print()
n_ok = sum(1 for _, ok in results if ok)
print(f"=== {n_ok}/{len(results)} PASS ===  trace: {engine.trace_dir}")
engine.close()
sys.exit(0 if n_ok == len(results) else 1)
