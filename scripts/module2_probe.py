#!/usr/bin/env python
# 探测 robosuite 1.5.2 + mujoco 3.3.7 的关键 API 行为（一次性脚本）
import numpy as np
import mujoco
import robosuite as suite
from robosuite.utils import camera_utils as CU
from robosuite.utils import transform_utils as T

env = suite.make(
    "Lift", robots="Panda", has_renderer=False, has_offscreen_renderer=True,
    use_camera_obs=True, camera_names="agentview",
    camera_heights=256, camera_widths=256, camera_depths=True,
    control_freq=20, horizon=50,
)
obs = env.reset()

print("== 1. obs 键 ==")
print(sorted(obs.keys()))

print("\n== 2. depth 格式 ==")
d = obs["agentview_depth"]
print("shape:", d.shape, "dtype:", d.dtype, "min/max:", d.min(), d.max())

print("\n== 3. 相机内外参 ==")
sim = env.sim
K = CU.get_camera_intrinsic_matrix(sim, "agentview", 256, 256)
print("K:\n", K)
try:
    E = CU.get_camera_extrinsic_matrix(sim, "agentview")
    print("extrinsic (shape %s):\n" % (E.shape,), E)
except Exception as e:
    print("extrinsic error:", e)
print("camera_utils 可用函数:", [f for f in dir(CU) if "camera" in f.lower() or "depth" in f.lower() or "project" in f.lower()])

print("\n== 4. 对象与 body 名 ==")
print("env.objects:", [type(o).__name__ for o in env.objects] if hasattr(env, "objects") else "N/A")
body_names = [n for n in sim.model.body_names if n]
print("body_names:", body_names)
print("cube_pos (GT):", obs.get("cube_pos"))

print("\n== 5. eef site ==")
site_names = [n for n in sim.model.site_names if n]
print("site_names:", [s for s in site_names if "grip" in s or "eef" in s])
print("eef_pos:", obs["robot0_eef_pos"])

print("\n== 6. GT 投影验证（cube 中心 → 像素）==")
cube_pos = obs["cube_pos"]
K_mat = K
try:
    # world → camera: E 是 cam→world 则求逆
    E_mat = E
    cam_from_world = np.linalg.inv(E_mat)
    p_cam = cam_from_world @ np.append(cube_pos, 1.0)
    print("cube in cam frame:", p_cam[:3])
    uvw = K_mat @ p_cam[:3]
    uv = uvw[:2] / uvw[2]
    print("projected uv:", uv, "(图像 256x256)")
except Exception as e:
    print("project error:", e)

print("\n== 7. 四元数约定 ==")
q = obs["robot0_eef_quat"]
print("eef_quat:", q, "norm:", np.linalg.norm(q))
print("quat2mat @ [1,0,0]:", (T.quat2mat(q) @ np.array([1,0,0])).round(3))

print("\n== 8. action 空间 ==")
print("action_dim:", env.action_dim)
print("dof:", env.robots[0].dof)
print("controller:", env.robots[0].composite_controller.name if hasattr(env.robots[0], "composite_controller") else "N/A")

env.close()
print("\nDONE")
