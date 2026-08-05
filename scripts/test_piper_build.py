"""Piper robosuite 集成冒烟测试：建 Stack 环境、渲染、命名/执行器/夹爪自检。

用法:
    MUJOCO_GL=egl /home/stouching/anaconda3/envs/ASPIRE/bin/python scripts/test_piper_build.py
"""

import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)

import aspire.robots  # noqa: F401  注册 Piper
from aspire.robots.piper_robot import _ASSETS

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs", "piper_smoke")
os.makedirs(OUT, exist_ok=True)

ctrl = load_composite_controller_config(controller=os.path.join(_ASSETS, "default_piper.json"))

env = suite.make(
    "Stack",
    robots="Piper",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["robot0_robotview", "robot0_eye_in_hand"],
    camera_heights=512,
    camera_widths=512,
    camera_depths=True,
    control_freq=20,
    horizon=200,
    controller_configs=ctrl,
)
obs = env.reset()
print("[ok] env built. obs keys sample:", sorted(k for k in obs if "robotview" in k or "eye_in" in k))

# --- 命名检查 ---
model = env.sim.model._model
names = {
    "arm joints": [f"robot0_joint{i}" for i in range(1, 7)],
    "eef site": ["gripper0_right_grip_site"],
    "gripper joints": ["gripper0_right_joint7", "gripper0_right_joint8"],
    "cameras": ["robot0_robotview", "robot0_eye_in_hand"],
}
import mujoco
for kind, ns in names.items():
    for n in ns:
        for objtype, label in [(mujoco.mjtObj.mjOBJ_JOINT, "joint"), (mujoco.mjtObj.mjOBJ_SITE, "site"),
                               (mujoco.mjtObj.mjOBJ_CAMERA, "cam"), (mujoco.mjtObj.mjOBJ_ACTUATOR, "act")]:
            idx = mujoco.mj_name2id(model, objtype, n)
            if idx >= 0:
                print(f"[ok] {kind}: {n} -> {label}#{idx}")
                break
        else:
            print(f"[MISS] {kind}: {n} 未找到!")

print("actuators:", [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i) for i in range(model.nu)])
print("arm qpos after reset:", obs["robot0_joint_pos"].round(3))
print("gripper qpos after reset:", obs["robot0_gripper_qpos"].round(4))

# eef site 世界位姿 (home 姿态)
sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "gripper0_right_grip_site")
print("grip_site world pos:", env.sim.data.site_xpos[sid].round(4))

# --- 直接 ctrl 步进测试: 夹爪开/合 + 臂保持 ---
data = env.sim.data
act_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, f"robot0_joint{i}") for i in range(1, 7)]
gact = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, "gripper0_right_gripper")
q_home = obs["robot0_joint_pos"][:6].copy()

def run_ticks(n, grip_ctrl):
    for _ in range(n):
        data.ctrl[act_ids] = q_home
        data.ctrl[gact] = grip_ctrl
        env.sim.step()
    return env._get_observations(force_update=True)

obs = run_ticks(60, 0.035)  # 张开
print("gripper qpos open:", obs["robot0_gripper_qpos"].round(4))
obs = run_ticks(60, 0.0)    # 闭合
print("gripper qpos closed:", obs["robot0_gripper_qpos"].round(4))
obs = run_ticks(60, 0.035)  # 重新张开 (渲染用)

# --- 渲染双相机存图 ---
img_rv = obs["robot0_robotview_image"]
img_wr = obs["robot0_eye_in_hand_image"]
cv2.imwrite(os.path.join(OUT, "robotview.png"), cv2.cvtColor(img_rv[::-1], cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(OUT, "robotview_raw.png"), cv2.cvtColor(img_rv, cv2.COLOR_RGB2BGR))
cv2.imwrite(os.path.join(OUT, "wrist_raw.png"), cv2.cvtColor(img_wr, cv2.COLOR_RGB2BGR))
print("[ok] images saved to", OUT)

# cube 位置 (GT, 仅供摆放校验)
print("cubeA pos:", env.sim.data.body_xpos[env.cubeA.root_body.attrib.get("name") and
      mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cubeA_main")].round(3)
      if mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cubeA_main") >= 0 else "n/a")
base_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_base_link")
print("base_link world pos:", env.sim.data.xpos[base_id].round(4))
print("DONE")
