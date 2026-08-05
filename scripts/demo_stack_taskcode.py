# =============================================================================
# ASPIRE Primitive API 测试任务代码 —— 双相机 SAM3 引导 Stack
# =============================================================================
# 由执行引擎注入 Primitive API 后执行（模拟 coding agent 生成的任务程序）。
# 只允许使用 docs/primitive_api.md 中的 API + np + print（禁止 import）。
#
# 任务：robosuite Stack —— SAM3 定位红/绿方块，把红方块(cubeA)堆到绿方块(cubeB)上。
#
# 单臂双视角配置（与 LIBERO / OpenVLA-OFT 开源惯例一致）：
#   - "agentview" : 第三人称相机 —— 全局定位、抓取后验证
#   - "wrist"     : robot0_eye_in_hand 腕部相机 —— 预抓位近距离横向精化
#
# Phase B 主体改编自 ASPIRE 官方公开示例 open_details/primitive_api_cube_reset.py
# （红/绿方块重堆叠），适配点：
#   1. 观测格式：官方示例为 obs["robot0_robotview"]["images"/"intrinsics"/"pose_mat"]，
#      本复现为扁平键 get_observation(camera)["{cam}_image"/"camera_intrinsics"/...]
#   2. 坐标系：官方示例为机械臂基座系（桌面 z≈0），本仿真为世界系（桌面 z≈0.8）
#   3. 腕部相机精化升级为闭环伺服 servo_align_wrist（双视角扩展，原公开示例为单视角）：
#      观测 → 衰减修正 → 移动到位 → 复测，直至 dxy<eps 或达迭代上限
#   4. 抓取高度修正：agentview 点云只覆盖方块顶面（min_z≈top_z），官方 center_z 公式
#      在此退化为顶面高度导致空抓（实测两次），改为 top_z-6mm 起抓、每次重试再降 2mm
#   5. 放置验收与修复：离手后复测红/绿相对位姿，不合格则重抓重放一次；
#      放置前绿块重定位做 z 跨度合理性校验（mask 混入机械臂时沿用 Step 1 测量）
# 其余 helper（get_best_mask / get_object_info / place_object）逻辑与原示例保持一致。
# =============================================================================

print("=" * 68)
print("PHASE A: Primitive API 自检（感知/规划纯函数 + 一次微运动）")
print("=" * 68)

QUAT_TOP_DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz，绕 x 轴 180°，末端 z 朝下


# === 官方示例 helper（原样逻辑，观测适配为扁平键） ===

def make_topdown_quat():
    R = np.array([
        [1.0,  0.0,  0.0],
        [0.0, -1.0,  0.0],
        [0.0,  0.0, -1.0],
    ], dtype=float)
    return rotation_matrix_to_quaternion(R)

TOP_DOWN_QUAT = make_topdown_quat()

def cam_obs(camera="agentview"):
    """观测适配层：官方示例的 cam dict → 本复现扁平键。"""
    obs = get_observation(camera)
    return (obs, obs[f"{camera}_image"], obs[f"{camera}_depth"],
            obs["camera_intrinsics"], obs["camera_extrinsics"])

def move_to_pose_ik(pos, quat=None):
    """官方示例的 move_to_pose 写法：solve_ik + move_to_joints。"""
    if quat is None:
        quat = TOP_DOWN_QUAT
    joints = solve_ik(np.asarray(pos, dtype=float), np.asarray(quat, dtype=float))
    move_to_joints(joints)

def safe_home():
    """机械臂回安全位，清空相机视野（世界系，桌面 z≈0.8）。"""
    move_to_pose_ik([0.0, 0.0, 1.10])

def get_best_mask(rgb, prompt, min_pixels=50, max_pixels=12000):
    """官方示例原版：面积过滤剔除机械臂/背景 mask，含 molmo→point 兜底链。"""
    masks = segment_sam3_text_prompt(rgb, prompt)
    valid = []
    if masks:
        for m in masks:
            if "mask" not in m or m["mask"] is None:
                continue
            area = int(np.sum(m["mask"] > 0))
            if min_pixels <= area <= max_pixels:
                valid.append(m)

    if not valid:
        # Fallback: Molmo point prompt -> SAM3 point prompt
        pts = point_prompt_molmo(rgb, prompt)
        if pts:
            for _, p in pts.items():
                if p[0] is not None and p[1] is not None:
                    pmasks = segment_sam3_point_prompt(rgb, (int(p[0]), int(p[1])))
                    if pmasks:
                        for m in pmasks:
                            if "mask" not in m or m["mask"] is None:
                                continue
                            area = int(np.sum(m["mask"] > 0))
                            if min_pixels <= area <= max_pixels:
                                valid.append(m)
                    break

    if not valid:
        # Last resort: take best mask without area filter
        if masks:
            candidate = [m for m in masks if "mask" in m and m["mask"] is not None]
            if candidate:
                valid = candidate

    if not valid:
        return None
    return max(valid, key=lambda m: float(m.get("score", 0.0)))["mask"]

def get_object_info(prompt, camera="agentview", retries=2):
    """官方示例原版：定位物体，返回 center/top_z/min_z，带重试。"""
    for attempt in range(retries):
        obs, rgb, depth, K, T = cam_obs(camera)
        mask = get_best_mask(rgb, prompt)
        if mask is None:
            print(f"  [{prompt}] attempt {attempt}: no mask found")
            continue

        pts = mask_to_world_points(mask.astype(np.uint8), depth, K, T)
        pts = pts[np.isfinite(pts).all(axis=1)]
        if pts.shape[0] < 10:
            print(f"  [{prompt}] attempt {attempt}: too few points ({pts.shape[0]})")
            continue

        center = np.median(pts, axis=0)
        top_z = float(np.percentile(pts[:, 2], 95))
        min_z = float(np.percentile(pts[:, 2], 5))
        info = {"center": center, "top_z": top_z, "min_z": min_z, "_prompt": prompt}
        print(f"  [{prompt}] center={center.round(4)}, top_z={top_z:.4f}, min_z={min_z:.4f}")
        return info
    return None


# --- A1/A2: get_observation 双相机 --------------------------------------------------
obs0, rgb0, depth0, K0, E0 = cam_obs("agentview")
print(f"[A1 ] get_observation('agentview')   rgb={rgb0.shape} depth={depth0.shape} "
      f"eef={obs0['robot0_eef_pos'].round(3)}")
obsw, wrgb0, wdepth0, wK0, wE0 = cam_obs("wrist")
print(f"[A2 ] get_observation('wrist')       rgb={wrgb0.shape} depth={wdepth0.shape} "
      f"(eye-in-hand 动态渲染路径)")

# --- A3: SAM3 文本提示分割（核心被测对象：红+绿两个方块） -------------------------
mask_r0 = get_best_mask(rgb0, "red cube")
mask_g0 = get_best_mask(rgb0, "green cube")
if mask_r0 is None or mask_g0 is None:
    raise RuntimeError(f"SAM3 初始检测失败: red={'OK' if mask_r0 is not None else 'MISS'} "
                       f"green={'OK' if mask_g0 is not None else 'MISS'}")
ar, ag = int(np.sum(mask_r0 > 0)), int(np.sum(mask_g0 > 0))
print(f"[A3 ] segment_sam3_text_prompt       red={ar}px green={ag}px (面积过滤后有效)")

# --- A4/A5: 点提示分割 / molmo 风格点定位 ---------------------------------------------
ys, xs = np.nonzero(mask_r0 > 0)
u0, v0 = float(xs.mean()), float(ys.mean())
pmasks = segment_sam3_point_prompt(rgb0, (int(u0), int(v0)))
print(f"[A4 ] segment_sam3_point_prompt      {len(pmasks)} 候选 @({u0:.0f},{v0:.0f})")
mpts = point_prompt_molmo(rgb0, "red cube")
print(f"[A5 ] point_prompt_molmo             {mpts}")

# --- A6: mask + depth → 世界点云 --------------------------------------------------------
pts0 = mask_to_world_points((mask_r0 > 0).astype(np.uint8), depth0, K0, E0)
if len(pts0) < 10:
    raise RuntimeError(f"红方块点云过少: {len(pts0)}")
c0 = np.median(pts0, axis=0)
print(f"[A6 ] mask_to_world_points           N={len(pts0)} center={c0.round(4)} "
      f"top_z={float(np.percentile(pts0[:, 2], 95)):.4f}")

# --- A7: PCA 有向包围盒 -------------------------------------------------------------------
obb0 = get_oriented_bounding_box_from_3d_points(pts0)
print(f"[A7 ] get_oriented_bounding_box      extent={obb0['extent'].round(4)}")

# --- A8: 单像素反投影一致性 ----------------------------------------------------------------
z0 = float(depth0[int(v0), int(u0)])
p_pix = pixel_to_world_point(u0, v0, z0, K0, E0)
print(f"[A8 ] pixel_to_world_point           {p_pix.round(4)} "
      f"(距点云中心 {float(np.linalg.norm(p_pix - c0)) * 100:.1f}cm)")

# --- A9: 注入版 move_to_pose 微运动（闭环阻塞语义 + quat=None 路径） -------------------------
z_before = float(get_observation()["robot0_eef_pos"][2])
move_to_pose([float(obs0["robot0_eef_pos"][0]), float(obs0["robot0_eef_pos"][1]), z_before + 0.05], None)
z_after = float(get_observation()["robot0_eef_pos"][2])
print(f"[A9 ] move_to_pose(微抬5cm)          eef_z {z_before:.4f} → {z_after:.4f}")

# --- A10: 直线插值（纯函数） -----------------------------------------------------------------
wps0 = interpolate_segment(np.array([0.0, 0.0, 1.0]), np.array([0.0, 0.0, 0.9]), step=0.02)
print(f"[A10] interpolate_segment            {len(wps0)} 路点 (step=0.02)")
# rotation_matrix_to_quaternion 已由 make_topdown_quat 覆盖；
# solve_ik / move_to_joints / open/close_gripper 由 Phase B 覆盖（trace 统计）。

print()
print("=" * 68)
print("PHASE B: 官方 cube_reset 流程适配 —— SAM3 引导红上绿堆叠")
print("=" * 68)

# === 官方示例 pick/place（原逻辑 + 腕部伺服扩展 + 世界系坐标） ===

def servo_align_wrist(prompt, cx, cy, top_z, eps=0.006, max_iter=3, gain=0.7):
    """双视角伺服扩展：腕部相机闭环对准（原公开示例为单视角一次性修正）。

    预抓高度上迭代：观测 → dxy<eps 收敛退出 → 否则按 gain 衰减修正目标并移动到位复测。
    只有"移动到位后再测量"，残差才真正可收敛；gain<1 抑制测量噪声引起的振荡。
    两道安全门：跳变 >4cm 拒绝（离谱 mask）、z 偏差 >5cm 拒绝（mask 混入背景/手臂）；
    拒绝后直接沿用当前估计（视角不变重测意义不大）。未收敛则返回最后一次修正值。
    """
    hover_z = top_z + 0.10
    for it in range(max_iter):
        move_to_pose_ik([cx, cy, hover_z])
        wobs, wrgb, wdepth, wK, wE = cam_obs("wrist")
        wmask = get_best_mask(wrgb, prompt, min_pixels=30, max_pixels=30000)
        if wmask is None:
            print(f"  [servo] iter {it}: 腕部未检测到目标，沿用当前估计")
            break
        wpts = mask_to_world_points(wmask.astype(np.uint8), wdepth, wK, wE)
        wpts = wpts[np.isfinite(wpts).all(axis=1)]
        if wpts.shape[0] < 10:
            print(f"  [servo] iter {it}: 点云过少 ({wpts.shape[0]})，沿用当前估计")
            break
        wc = np.median(wpts, axis=0)
        dxy = float(np.linalg.norm(wc[:2] - np.array([cx, cy])))
        dz = abs(float(wc[2]) - top_z)
        if dz > 0.05:
            print(f"  [servo] iter {it}: z 偏差 {dz * 100:.1f}cm 不可信（mask 疑混入背景），沿用当前估计")
            break
        if dxy < eps:
            print(f"  [servo] iter {it}: 收敛 dxy={dxy * 100:.2f}cm < {eps * 100:.1f}cm")
            break
        if dxy > 0.04:
            print(f"  [servo] iter {it}: 跳变 {dxy * 100:.1f}cm 不可信，沿用当前估计")
            break
        cx += gain * float(wc[0] - cx)
        cy += gain * float(wc[1] - cy)
        print(f"  [servo] iter {it}: dxy={dxy * 100:.2f}cm → 衰减修正至 ({cx:.4f},{cy:.4f})")
    return cx, cy

def pick_object(info, max_retries=3):
    """Pick an object with retry logic. Returns True if grasp succeeded."""
    cx, cy = info["center"][0], info["center"][1]
    top_z = info["top_z"]

    for attempt in range(max_retries):
        open_gripper()

        # ---- 双视角伺服扩展：腕部闭环对准横向位置（原公开示例为单视角一次性修正） ----
        cx, cy = servo_align_wrist(info.get("_prompt", "cube"), cx, cy, top_z)

        # 适配点4：点云只覆盖方块顶面（min_z≈top_z），官方 center_z 公式退化为顶面高度，
        # 抓顶面必空抓（实测 attempt 0/1）；改为顶面下 6mm 起抓，每次重试再降 2mm
        grasp_z = top_z - 0.006 - 0.002 * attempt
        move_to_pose_ik([cx, cy, grasp_z])

        close_gripper()

        # Check if we got the object by reading gripper state
        obs = get_observation()
        gripper_qpos = obs.get("robot0_gripper_qpos", None)

        # Lift regardless
        move_to_pose_ik([cx, cy, top_z + 0.12])

        if gripper_qpos is not None:
            gw = float(gripper_qpos[0])
            print(f"  pick attempt {attempt}: gripper_qpos={gw:.4f}")
            if gw > 0.003:  # Object grasped
                return True
            else:
                print(f"  pick attempt {attempt}: air grasp, retrying...")
                open_gripper()
                safe_home()
                new_info = get_object_info(info.get("_prompt", "cube"))
                if new_info is not None:
                    cx, cy = new_info["center"][0], new_info["center"][1]
                    top_z = new_info["top_z"]
        else:
            return True

    return True  # Continue even if uncertain

def place_object(xy, place_z, clearance=0.032):
    """Place the held object at target xy, lowering to place_z + clearance."""
    move_to_pose_ik([xy[0], xy[1], place_z + 0.12])
    move_to_pose_ik([xy[0], xy[1], place_z + clearance])
    open_gripper()
    move_to_pose_ik([xy[0], xy[1], place_z + 0.12])

# === 主流程（对应官方示例 Step 0-6，去掉 Stack 不需要的"绿块移开"步骤） ===
print("=== Stack: SAM3 定位红/绿方块，红堆绿上 ===")

# Step 0: 安全位 + 张爪
open_gripper()
safe_home()

# Step 1: 定位两个方块
print("\n--- Step 1: Initial observation ---")
green = get_object_info("green cube")
red = get_object_info("red cube")

if green is None or red is None:
    print("ERROR: Could not detect both cubes! Trying fallback prompts")
    if green is None:
        green = get_object_info("green block")
    if red is None:
        red = get_object_info("red block")

assert green is not None and red is not None, "Failed to detect cubes after fallback"

# Step 2: 抓取红方块
print("\n--- Step 2: Pick red cube ---")
pick_object(red)

# Step 3: 放置前重新定位绿方块（agentview 全局视角）
print("\n--- Step 3: Re-locate green before placing ---")
green2 = get_object_info("green cube")
if green2 is None:
    print("WARNING: green re-detect failed, fallback prompt")
    green2 = get_object_info("green block")
assert green2 is not None, "Failed to re-detect green cube"

# 适配点5：重定位合理性校验——mask 混入机械臂/背景时点云 z 跨度异常（实测达 9cm，
# 导致放置高度虚高、方块脱手坠落），此时 Step 1 干净场景的测量更可信
if green2["top_z"] - green2["min_z"] > 0.05:
    print(f"WARNING: 绿块重定位 z 跨度 {green2['top_z'] - green2['min_z']:.3f}m 不可信，"
          f"沿用 Step 1 初始测量")
    green2 = green

# Step 4: 红方块放到绿方块上
print("\n--- Step 4: Place red on green ---")
target_xy = green2["center"][:2].copy()
support_top_z = green2["top_z"]
place_object(target_xy, support_top_z, clearance=0.032)

# Step 5: 收尾 + 放置验收（适配点5：验收不过则修复重放一次，原示例只打印结论）
print("\n--- Step 5: Final verification ---")

def check_stacked(red_i, green_i, dxy_thresh=0.025):
    """放置验收：红块底面高于绿块顶面（顶面点云近似），且水平偏移在容差内。"""
    if red_i is None or green_i is None:
        return False
    dz_ok = red_i["min_z"] > green_i["top_z"] - 0.012
    dxy = float(np.linalg.norm(red_i["center"][:2] - green_i["center"][:2]))
    print(f"  [verify] red_min_z={red_i['min_z']:.4f} green_top_z={green_i['top_z']:.4f} "
          f"dz_ok={dz_ok} dxy={dxy * 100:.2f}cm (阈 {dxy_thresh * 100:.1f}cm)")
    return dz_ok and dxy < dxy_thresh

safe_home()
red_f = get_object_info("red cube")
green_f = get_object_info("green cube")

if check_stacked(red_f, green_f):
    print("SUCCESS: Red cube is stacked on green cube!")
elif red_f is not None and green_f is not None:
    print("REPAIR: 验收未通过，重新抓放红块一次")
    pick_object(red_f)
    place_object(green_f["center"][:2].copy(), green_f["top_z"], clearance=0.032)
    safe_home()
    red_f = get_object_info("red cube")
    green_f = get_object_info("green cube")
    if check_stacked(red_f, green_f):
        print("SUCCESS: stacked after repair!")
    else:
        print("FAIL: 修复后仍未堆叠")
else:
    print("NOTE: 验证阶段检测失败，无法验收")

print("=== TASK CODE DONE ===")
