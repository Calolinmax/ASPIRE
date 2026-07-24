# 模拟 coding agent 生成的任务代码（只用 docs/primitive_api.md 中的 API）
# 任务：Lift —— 找到方块并抓起

print("=== LIFT TASK ===")

QUAT_TOP_DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz

# 1. 感知定位（检测）
obs = get_observation()
masks = segment_sam3_text_prompt(obs["agentview_image"], "cube")
if not masks:
    raise RuntimeError("cannot find cube")

pts = mask_to_world_points(masks[0]["mask"], obs["agentview_depth"],
                           obs["camera_intrinsics"], obs["camera_extrinsics"])
if len(pts) < 10:
    raise RuntimeError(f"too few points: {len(pts)}")

center = np.median(pts, axis=0)
top_z = float(np.percentile(pts[:, 2], 95))
min_z = float(np.percentile(pts[:, 2], 5))
grasp_z = min_z + 0.008  # 夹 cube 下半部（夹上部易滑落，trace seed1 教训）
print(f"cube center={center.round(4)}, top_z={top_z:.4f}, min_z={min_z:.4f}, grasp_z={grasp_z:.4f}")

# 2. IK 可达性检查（规划）
try:
    joints = solve_ik(np.array([center[0], center[1], top_z + 0.10]), QUAT_TOP_DOWN)
    print(f"IK ok, joints[0:3]={joints[:3].round(3)}")
except RuntimeError as e:
    print(f"IK failed: {e}")
    raise

# 3. 抓取流程（控制 + 抓取）
open_gripper()
move_to_pose([center[0], center[1], grasp_z + 0.12], QUAT_TOP_DOWN)
move_to_pose([center[0], center[1], grasp_z], QUAT_TOP_DOWN)
close_gripper()

# 4. 验证夹持
obs2 = get_observation()
gw = float(obs2["robot0_gripper_qpos"][0])
print(f"gripper_qpos={gw:.4f}")

# 5. 抬起
move_to_pose([center[0], center[1], grasp_z + 0.22], QUAT_TOP_DOWN)

# 6. 成功自检
obs3 = get_observation()
masks3 = segment_sam3_text_prompt(obs3["agentview_image"], "cube")
if masks3:
    pts3 = mask_to_world_points(masks3[0]["mask"], obs3["agentview_depth"],
                                obs3["camera_intrinsics"], obs3["camera_extrinsics"])
    lifted_z = float(np.percentile(pts3[:, 2], 95))
    print(f"cube z after lift: {lifted_z:.4f} (was {top_z:.4f})")
    print("LIFT OK" if lifted_z > top_z + 0.05 else "LIFT FAILED")
print("=== DONE ===")
