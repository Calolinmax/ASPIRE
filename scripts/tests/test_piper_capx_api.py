"""cap-x 标准 API（Piper）自检 + 外参标定验证。

逐项调用 cap-x 函数集，并用 GT（仅测试可用）校验：
  - 观测结构/坐标系（基座系，640×640 robotview）
  - SAM3 文本/点提示分割
  - 反投影约定（pose_mat + y 负号规则）vs GT 方块位置
  - plan_grasp 候选位姿合理性（基座系 site 位姿，对准方块中心 + 逼近轴朝下）
  - solve_ik / move_to_joints / open/close_gripper 基本运动
  - E1 类结构：PiperControlApiReduced.functions() = 契约 15 函数
  - 工具四函数（mask_to_world_points / pixel_to_world_point /
    rotation_matrix_to_quaternion / interpolate_segment）
  - select_top_down_grasp（CGN 约定合成候选三态）

用法:
    MUJOCO_GL=egl /home/stouching/anaconda3/envs/ASPIRE/bin/python scripts/tests/test_piper_capx_api.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import mujoco
from robosuite.utils import transform_utils as T

from aspire.engine.engine_capx import ExecutionEngineCapx
from aspire.api.primitives_capx import (PiperControlApiReduced, PrimitiveContextCapx,
                                    RENDER_H, RENDER_W)

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
check("A1 obs robotview rgb (640,640,3) u8", rgb.shape == (RENDER_H, RENDER_W, 3) and rgb.dtype == np.uint8, str(rgb.shape))
check("A2 obs depth (640,640) 米制", depth.shape == (RENDER_H, RENDER_W) and 0.2 < float(np.nanmedian(depth)) < 3.0,
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
# GT 几何尺寸从模型 geom 读取（2026-08-05 教训：勿硬编码尺寸假设——
# cubeA 为 4×4×8cm 立块（engine_capx.py:266，2026-08-03 裁决 3 细高化）。
# 早前按"4cm 立方"假设把顶面 0.08 当 0.06，一度误判"深度渲染 bias 2.1cm"——
# 经 GT 逐点核实：深度管线与模型一致（亚毫米），无 bias）
gidA = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cubeA_g0")
gidB = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "cubeB_g0")
halfA = model.geom_size[gidA].astype(float)   # [0.02 0.02 0.04]
halfB = model.geom_size[gidB].astype(float)   # [0.025 0.025 0.025]
topA_z = float(gtA_base[2] + halfA[2])
topB_z = float(gtB_base[2] + halfB[2])
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
# 判据（2026-08-05 重写，尺寸读自模型）：可见面点云（顶面+朝向相机的侧面）
# 的 z 向 p95 ≈ GT 顶面高度（顶面水平，亚厘米级标定）；XY 中位落在
# footprint 半径+1cm 容差内（侧面点把中位往相机侧拉，不超过半宽量级）。
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
    z_p95 = float(np.percentile(pts_r[:, 2], 95))
    dxyA = float(np.linalg.norm(np.median(pts_r[:, :2], axis=0) - gtA_base[:2]))
    check("C1 反投影 red cube 顶面高度+footprint ↔ GT",
          abs(z_p95 - topA_z) < 0.005 and dxyA < halfA[0] + 0.01,
          f"z_p95={z_p95:.4f} vs top={topA_z:.4f}  dxy={dxyA*100:.2f}cm")
if masks_g:
    pts_g = deproject(masks_g[0]["mask"])
    z_p95g = float(np.percentile(pts_g[:, 2], 95))
    dxyB = float(np.linalg.norm(np.median(pts_g[:, :2], axis=0) - gtB_base[:2]))
    check("C2 反投影 green cube 顶面高度+footprint ↔ GT",
          abs(z_p95g - topB_z) < 0.005 and dxyB < halfB[0] + 0.01,
          f"z_p95={z_p95g:.4f} vs top={topB_z:.4f}  dxy={dxyB*100:.2f}cm")

# --- D. plan_grasp ------------------------------------------------------------
# plan_grasp 返回**基座系** grip_site 位姿（偏差声明 #3，2026-08-05 注释修正），
# 勿再左乘 pose_mat；site 约定 +z 背向物体（逼近轴 = -z_col）。
seg = np.zeros(rgb.shape[:2], dtype=np.int32)
if masks_r:
    seg[masks_r[0]["mask"]] = 1
grasps, scores = ns["plan_grasp"](depth, K, seg)
check("D1 plan_grasp 返回 (K,4,4)+(K,)", grasps.ndim == 3 and grasps.shape[1:] == (4, 4) and len(scores) == len(grasps),
      f"K={len(grasps)}")
if len(grasps):
    dxy = np.linalg.norm(grasps[:, :2, 3] - gtA_base[:2][None, :], axis=1)
    i_near = int(np.argmin(dxy))
    g_near = grasps[i_near]
    dz = float(g_near[2, 3] - gtA_base[2])
    # 位置判据（双后端兼容）：XY 对准方块中心；z 落在抓取口袋带内
    # （CGN site≈捏取面上方 ~4cm；几何族 site≈中心下探 ~2cm）
    check("D2 plan_grasp 候选对准方块中心 (dxy<2.5cm, |z带|)", dxy[i_near] < 0.025 and -0.03 < dz < 0.09,
          f"dxy={dxy[i_near]*100:.2f}cm dz={dz*100:.1f}cm")
    approach = -g_near[:3, 2]  # site +z 背向物体 → 逼近方向 = -z_col
    check("D3 plan_grasp 逼近轴朝下", approach[2] < -0.5, f"approach_z={approach[2]:.3f}")

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
    # cubeA = 4×4×8cm 立块：extent 长轴 ≈8cm，短轴 ≈4cm（可见面+PCA 有 ±2cm 容差）
    ext = obb["extent"]
    check("F1 OBB extent ↔ 4×4×8 立块",
          0.06 < float(ext.max()) < 0.10 and 0.035 < float(ext.min()) < 0.07,
          f"extent={ext.round(3)}")

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

# --- I. 类结构与契约函数面（验收标准 3，E1 重构） -------------------------------
EXPECTED15 = {"get_observation", "segment_sam3_text_prompt", "segment_sam3_point_prompt",
              "point_prompt_molmo", "plan_grasp", "get_oriented_bounding_box_from_3d_points",
              "solve_ik", "move_to_joints", "open_gripper", "close_gripper",
              "select_top_down_grasp", "mask_to_world_points", "pixel_to_world_point",
              "rotation_matrix_to_quaternion", "interpolate_segment"}
fns = PiperControlApiReduced(engine).functions()
check("I1 functions() = 契约 15 函数", set(fns.keys()) == EXPECTED15,
      f"{len(fns)} keys missing={EXPECTED15 - set(fns)} extra={set(fns) - EXPECTED15}")
check("I2 PrimitiveContextCapx 别名兼容", PrimitiveContextCapx is PiperControlApiReduced)
check("I3 命名空间 = 契约 15 + grasp_cgn", all(k in ns for k in EXPECTED15) and "grasp_cgn" in ns,
      str([k for k in EXPECTED15 if k not in ns]))
check("I4 后端开关常量 (A1/B1 验收结论)",
      PiperControlApiReduced.IK_BACKEND == "pyroki" and PiperControlApiReduced.GRASP_BACKEND == "cgn",
      f"IK={PiperControlApiReduced.IK_BACKEND} GRASP={PiperControlApiReduced.GRASP_BACKEND}")

# --- J. 工具四函数（#11-#14） ---------------------------------------------------
if masks_r:
    pts_w = ns["mask_to_world_points"](masks_r[0]["mask"].astype(np.uint8), depth, K, E)
    # J1a 严格恒等：与冻结参考反投影（C 区 deproject，契约偏差声明 #1 同款数学）
    # 逐点一致 —— 钉死本函数实现本身（与上游深度 bias 无关）
    pts_ref = deproject(masks_r[0]["mask"])
    check("J1a mask_to_world_points ≡ 冻结参考反投影 (1e-9)",
          pts_w.shape == pts_ref.shape and np.allclose(pts_w, pts_ref, atol=1e-9),
          f"N={len(pts_w)}")
    # J1b GT 一致性：与 C1 同判据（顶面 p95 高度 + footprint）
    z_p95w = float(np.percentile(pts_w[:, 2], 95))
    dxy_w = float(np.linalg.norm(np.median(pts_w[:, :2], axis=0) - gtA_base[:2]))
    check("J1b mask_to_world_points ↔ GT (顶面高度+footprint)",
          len(pts_w) > 20 and abs(z_p95w - topA_z) < 0.005 and dxy_w < halfA[0] + 0.01,
          f"z_p95={z_p95w:.4f} vs top={topA_z:.4f}  dxy={dxy_w*100:.2f}cm")
    ys, xs = np.nonzero(masks_r[0]["mask"])
    u_c, v_c = int(xs.mean()), int(ys.mean())
    p_c = ns["pixel_to_world_point"](u_c, v_c, float(depth[v_c, u_c]), K, E)
    # mask 质心像素反投影点应落在方块表面（footprint 内 + z 在 [桌面, 顶面] 带）
    in_fp = (abs(p_c[0] - gtA_base[0]) < halfA[0] + 0.012
             and abs(p_c[1] - gtA_base[1]) < halfA[1] + 0.012)
    z_in = -0.005 < p_c[2] < topA_z + 0.005
    check("J2 pixel_to_world_point 落方块表面", np.isfinite(p_c).all() and in_fp and z_in,
          f"p={p_c.round(3)} top={topA_z:.3f}")
R0 = T.euler2mat([0.3, -0.5, 1.2])
q_out = ns["rotation_matrix_to_quaternion"](R0)
R1 = T.quat2mat(np.array([q_out[1], q_out[2], q_out[3], q_out[0]]))
check("J3 rotation_matrix_to_quaternion 往返一致 (wxyz)",
      abs(float(np.linalg.norm(q_out)) - 1.0) < 1e-9 and float(np.abs(R1 - R0).max()) < 1e-6,
      f"q={q_out.round(4)}")
seg_pts = ns["interpolate_segment"](np.array([0.0, 0.0, 0.0]), np.array([0.11, 0.0, 0.0]), step=0.02)
gaps = [float(np.linalg.norm(seg_pts[i + 1] - seg_pts[i])) for i in range(len(seg_pts) - 1)]
check("J4 interpolate_segment 端点精确 + 间距≤step",
      np.allclose(seg_pts[0], [0, 0, 0]) and np.allclose(seg_pts[-1], [0.11, 0, 0])
      and len(seg_pts) == 7 and max(gaps) <= 0.02 + 1e-9,
      f"n={len(seg_pts)} max_gap={max(gaps):.4f}")

# --- K. select_top_down_grasp（#7，CGN 约定合成候选，cam_to_world=I） ------------
_c = float(np.sqrt(0.5))
g_down = np.eye(4)
g_down[:3, :3] = np.diag([1.0, -1.0, -1.0])   # +z=逼近 竖直朝下 (alignment=1)
g_down[:3, 3] = [0.30, 0.0, 0.80]
g_tilt = np.eye(4)
g_tilt[:3, :3] = np.array([[1.0, 0.0, 0.0], [0.0, -_c, -_c], [0.0, _c, -_c]])  # 逼近轴 45° 倾斜
g_tilt[:3, 3] = [0.30, 0.05, 0.80]
cands = np.stack([g_down, g_tilt])
cscores = np.array([0.5, 0.9])                # 倾斜候选分更高
best1, s1 = ns["select_top_down_grasp"](cands, cscores, np.eye(4))
check("K1 默认阈值选竖直候选（非高分倾斜）", best1 is not None and s1 == 0.5 and np.allclose(best1, g_down),
      f"score={s1}")
best2, s2 = ns["select_top_down_grasp"](cands, cscores, np.eye(4), vertical_threshold=0.5)
check("K2 阈值调低退化为尽量竖直（选高分倾斜）", best2 is not None and s2 == 0.9 and np.allclose(best2, g_tilt),
      f"score={s2}")
best3, s3 = ns["select_top_down_grasp"](cands[1:], cscores[1:], np.eye(4))
check("K3 无竖直候选返回 (None, -inf)", best3 is None and s3 == -np.inf, f"score={s3}")

print()
n_ok = sum(1 for _, ok in results if ok)
print(f"=== {n_ok}/{len(results)} PASS ===  trace: {engine.trace_dir}")
engine.close()
sys.exit(0 if n_ok == len(results) else 1)
