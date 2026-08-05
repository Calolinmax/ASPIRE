# =============================================================================
# cap-x 标准 Stack 任务代码 —— Piper 机械臂（非GT / reduced API）
# =============================================================================
# 由 ExecutionEngineCapx 注入 cap-x 函数集后执行。只允许使用
# docs/primitive_api_capx.md 中的 10 个函数 + np + print（禁止 import）。
#
# 任务：robosuite Stack —— 把红方块(cubeA)堆到绿方块(cubeB)上。
# 流程遵循 cap-x cube_stack prompt 的规则：
#   - 接近一律 z_approach 量级的预备位，不抛物
#   - 抓取后先抬升再横移；放置高度 = 绿块顶面 + 红块半高
#
# --- Piper 本体适配（离线探测标定, 见 docs/primitive_api_capx.md §0）---
# Piper 短臂+腕限位(j5±70°)下, 抓桌面方块的可达姿态族为"远侧 overhead 钩抓":
# 臂从方块上方弓过, 夹爪开口朝后下(偏航≈180° ± 30°, 倾角≈25°)。
# 顶朝下经典姿态在本体上不可达——这是本体差异, 不是 API 差异。
# 坐标：全部在机械臂基座系；四元数 wxyz；TCP = 指尖口袋中心（solve_ik 目标）。
# =============================================================================

print("=" * 68)
print("cap-x/Piper Stack: SAM3 定位 → overhead 钩抓红块 → 堆到绿块上")
print("=" * 68)

Z_APPROACH = 0.10
RED_HALF = 0.02   # 红方块半高 (4cm 方块)


def _quat_from_R(R):
    """(3,3) 旋转矩阵 → wxyz 四元数（Shepperd 法，纯 numpy）。"""
    t = float(np.trace(R))
    if t > 0:
        s = np.sqrt(t + 1.0) * 2
        w, x, y, z = 0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s
    else:
        i = int(np.argmax([R[0, 0], R[1, 1], R[2, 2]]))
        if i == 0:
            s = np.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
            w, x, y, z = (R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s
        elif i == 1:
            s = np.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
            w, x, y, z = (R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s
        else:
            s = np.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
            w, x, y, z = (R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


def topdown_grasp_quat(yaw_deg=0.0):
    """末端竖直朝下姿态 (z轴指向-world_z), 可绕z轴旋转 yaw_deg。"""
    # 竖直向下: z轴 = [0, 0, -1], x/y 在水平面内
    # 默认: x朝前, y朝左, z朝下
    yaw = np.deg2rad(yaw_deg)
    # 旋转矩阵: 先让z朝下 (绕x转180°), 再绕z转yaw
    # 基础姿态: z朝下, x朝前
    # quat for z-down: 绕x轴180度 = [0, 1, 0, 0]
    # 然后绕新z轴旋转yaw
    cz, sz = np.cos(yaw/2), np.sin(yaw/2)
    # z-down quat [0, 1, 0, 0] 绕 z 旋转 yaw
    # 组合: q_new = q_yaw * q_zdown
    # q_zdown = [0, 1, 0, 0]
    # q_yaw = [cz, 0, 0, sz]
    # 组合后: [0*cz - 1*0 - 0*0 - 0*sz, ...] = [0, cz, sz, 0]
    w, x, y, z = 0.0, cz, sz, 0.0
    q = np.array([w, x, y, z])
    return q / np.linalg.norm(q)


# 垂直朝下姿态候选: (yaw_deg) - 尝试不同朝向以避开奇异点
_TOPDOWN_POSE_CANDIDATES = [0, 45, -45, 90, -90, 135, -135, 180]


def move_tcp_topdown(pos, label=""):
    """以竖直朝下姿态移动到 pos，自动尝试不同yaw角度，返回使用的 quat。"""
    pos = np.asarray(pos, dtype=float)
    for yaw in _TOPDOWN_POSE_CANDIDATES:
        quat = topdown_grasp_quat(yaw)
        try:
            joints = solve_ik(pos, quat)
            break
        except RuntimeError:
            continue
    else:
        raise RuntimeError(f"topdown IK 全部候选未收敛 pos={pos.round(3)}")
    move_to_joints(joints)
    if label:
        obs = get_observation()
        err = float(np.linalg.norm(obs["robot_cartesian_pos"][:3] - pos))
        print(f"  [move] {label}: 到位误差 {err * 1000:.1f}mm")
    return quat


def deproject(mask, depth, K, E):
    """mask + depth → 基座系点云。反投影约定 y = -(v-cy)*z/fy（见 API 文档 §0）。

    工作空间门控: 丢弃桌面工作区外的点——SAM3 可能把手臂红色条纹/背景误检为
    "red cube"（实测: 误检 mask 反投影出 [-324, -108, -423] 的鬼点云）。
    注意: 25cm 立柱安装, 桌面在基座系 z≈0.55。
    """
    vs, us = np.nonzero(mask)
    if len(vs) == 0:
        return np.zeros((0, 3))
    z = depth[vs, us].astype(float)
    ok = np.isfinite(z) & (z > 0.05)
    us, vs, z = us[ok], vs[ok], z[ok]
    if len(z) < 5:
        return np.zeros((0, 3))
    x = (us - K[0, 2]) * z / K[0, 0]
    y = -(vs - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z, np.ones_like(z)], axis=0)
    pts = (E @ pts)[:3].T
    in_ws = ((pts[:, 0] > 0.05) & (pts[:, 0] < 0.65)
             & (np.abs(pts[:, 1]) < 0.30)
             & (pts[:, 2] > 0.40) & (pts[:, 2] < 0.80))
    return pts[in_ws]


def locate(prompt, obs_pack=None):
    """SAM3 文本定位（molmo→point 级联兜底），返回 center/top_z/min_z/pts/mask。"""
    if obs_pack is None:
        obs_pack = get_observation()
    cam = obs_pack["robot0_robotview"]
    rgb, depth, K, E = cam["images"]["rgb"], cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"]

    masks = segment_sam3_text_prompt(rgb, prompt)
    if not masks:
        # 兜底链: molmo 点 → 点提示分割
        pts_q = point_prompt_molmo(rgb, prompt)
        uv = list(pts_q.values())[0]
        if uv[0] is not None:
            masks = segment_sam3_point_prompt(rgb, uv)
    if not masks:
        print(f"  [locate] '{prompt}' 未检出")
        return None

    # 逐候选过滤: 工作区门控后点云太少的 mask 视为误检（手臂红纹/背景）
    pts, mask = None, None
    for m in masks[:4]:
        cand = deproject(m["mask"], depth, K, E)
        if len(cand) >= 10:
            pts, mask = cand, m["mask"]
            break
    if pts is None:
        print(f"  [locate] '{prompt}' 候选均过不了工作区门控 (误检?)")
        return None
    obb = get_oriented_bounding_box_from_3d_points(pts)
    # 中值区间中心: 抗"可见面偏置"（mask 只覆盖顶面+朝向相机的侧面时,
    # median 会沿视线偏半个物宽; mid-range 对有界物体基本无偏）
    lo = np.percentile(pts, 5, axis=0)
    hi = np.percentile(pts, 95, axis=0)
    info = {
        "center": (lo + hi) / 2,
        "top_z": float(hi[2]),
        "min_z": float(lo[2]),
        "extent": obb["extent"],
        "pts": pts,
        "mask": mask,
    }
    print(f"  [locate] '{prompt}' center={info['center'].round(4)} top_z={info['top_z']:.4f} "
          f"extent={info['extent'].round(3)}")
    return info


def gripper_frac():
    return float(get_observation()["robot_joint_pos"][-1])


def grasp_pos_of(info):
    """点云信息 → 抓取口袋中心（方块中心高度）。"""
    return np.array([info["center"][0], info["center"][1], info["top_z"] - RED_HALF])


# === Step 0: 初始观测 + 双块定位 ==============================================
print("\n--- Step 0: 观测与定位 ---")
obs0 = get_observation()
cam0 = obs0["robot0_robotview"]
print(f"  TCP(基座系)={obs0['robot_cartesian_pos'][:3].round(3)} "
      f"quat={obs0['robot_cartesian_pos'][3:7].round(3)} frac={obs0['robot_cartesian_pos'][7]:.2f}")

red = locate("red cube", obs0)
green = locate("green cube", obs0)
if red is None:
    red = locate("red block")
if green is None:
    green = locate("green block")
assert red is not None and green is not None, "初始检测失败"

# === Step 1: plan_grasp 规划 + 抓取位姿确定 ====================================
print("\n--- Step 1: plan_grasp (红方块) ---")
seg = np.zeros(cam0["images"]["rgb"].shape[:2], dtype=np.int32)
seg[red["mask"]] = 1
grasps, scores = plan_grasp(cam0["images"]["depth"], cam0["intrinsics"], seg)
print(f"  plan_grasp: {len(grasps)} 候选, best score={float(scores.max()) if len(scores) else 0:.2f}")

# 用点云中心筛选 CGN 候选（cap-x 下游任务语义）
grasp_pose = None
if len(grasps) and len(red["pts"]) >= 10:
    # red["center"] 是 locate 中的点云 5%~95% 中值区间中心
    dists = np.linalg.norm(grasps[:, :3, 3] - red["center"][None, :], axis=1)
    best_idx = np.argmin(dists)
    grasp_pose = grasps[best_idx]
    dev = float(np.linalg.norm(grasp_pose[:3, 3][:2] - red["center"][:2]))
    print(f"  点云中心筛选后候选与中心偏差: {dev * 100:.1f}cm")
else:
    # 候选不足时回落到几何规划
    print("  CGN 候选不足，使用几何规划")
    grasp_pose = None

# 计算抓取口袋位置（竖直朝下姿态用点云中心，或者直接用 CGN 位姿）
if grasp_pose is not None:
    # 用 CGN 位姿的 XY，Z 用点云 top_z - 半高
    grasp_pos = np.array([grasp_pose[0, 3], grasp_pose[1, 3], red["top_z"] - RED_HALF])
    grasp_quat = grasp_pose[:3, :3]  # 用 CGN 姿态
else:
    grasp_pos = grasp_pos_of(red)
print(f"  抓取口袋目标: {grasp_pos.round(4)}")

# === Step 2: 抓取红方块 =======================================================
print("\n--- Step 2: 抓取 ---")
grasped = False
for attempt in range(2):
    open_gripper()
    quat = move_tcp_topdown(grasp_pos + np.array([0, 0, 0.10]), f"预抓位(attempt {attempt})")
    move_tcp_topdown(grasp_pos, "下降到位")
    close_gripper()
    frac = gripper_frac()
    print(f"  闭合后 frac={frac:.3f}")
    # 抬起
    move_tcp_topdown(grasp_pos + np.array([0, 0, 0.10]), "抬起")
    frac_lift = gripper_frac()
    if frac_lift > 0.55:   # 夹爪被 4cm 方块撑开 (~0.8), 空抓会闭到 ~0.4
        grasped = True
        print("  ✓ 夹持成功")
        break
    print("  ✗ 空抓, 重新定位后重试")
    red2 = locate("red cube")
    if red2 is not None:
        red = red2
        grasp_pos = grasp_pos_of(red)

assert grasped, "两次抓取均失败"

# === Step 3: 放置前重定位绿块 ==================================================
print("\n--- Step 3: 重定位绿块 ---")
green2 = locate("green cube")
if green2 is None or abs(green2["top_z"] - green["top_z"]) > 0.03:
    print("  绿块重定位不可信, 沿用初始测量")
    green2 = green
green_top = green2["top_z"]
place_xy = green2["center"][:2]
place_z = green_top + RED_HALF + 0.008   # 绿块顶面 + 红块半高 + 释放余量
print(f"  放置目标: xy={place_xy.round(4)} z={place_z:.4f} (green_top={green_top:.4f})")

# === Step 4: 堆叠 =============================================================
print("\n--- Step 4: 放置 ---")
place_pos = np.array([place_xy[0], place_xy[1], place_z])
move_tcp_topdown(place_pos + np.array([0, 0, 0.10]), "放置预备位")
move_tcp_topdown(place_pos, "下降释放")
open_gripper()
move_tcp_topdown(place_pos + np.array([0, 0, 0.10]), "撤离")

# === Step 5: 验收 + 一次修复 ===================================================
print("\n--- Step 5: 验收 ---")

def check_stacked():
    r = locate("red cube")
    g = locate("green cube")
    if r is None or g is None:
        print("  [verify] 检测失败, 无法验收")
        return None, None, False
    dz_ok = r["min_z"] > g["top_z"] - 0.012
    dxy = float(np.linalg.norm(r["center"][:2] - g["center"][:2]))
    print(f"  [verify] red_min_z={r['min_z']:.4f} green_top={g['top_z']:.4f} "
          f"dz_ok={dz_ok} dxy={dxy * 100:.2f}cm")
    return r, g, dz_ok and dxy < 0.03

r_f, g_f, ok = check_stacked()
if ok:
    print("SUCCESS: 红块已堆在绿块上!")
elif r_f is not None and g_f is not None:
    print("REPAIR: 验收未过, 重抓重放一次")
    grasp_pos2 = grasp_pos_of(r_f)
    open_gripper()
    move_tcp_topdown(grasp_pos2 + np.array([0, 0, 0.10]), "修复预抓")
    move_tcp_topdown(grasp_pos2, "修复抓取")
    close_gripper()
    move_tcp_topdown(grasp_pos2 + np.array([0, 0, 0.10]), "修复抬起")
    place_pos2 = np.array([g_f["center"][0], g_f["center"][1],
                           g_f["top_z"] + RED_HALF + 0.008])
    move_tcp_topdown(place_pos2 + np.array([0, 0, 0.10]), "修复预备")
    move_tcp_topdown(place_pos2, "修复释放")
    open_gripper()
    move_tcp_topdown(place_pos2 + np.array([0, 0, 0.10]), "修复撤离")
    _, _, ok2 = check_stacked()
    print("SUCCESS: 修复后堆叠成功!" if ok2 else "FAIL: 修复后仍未堆叠")

print("=== TASK CODE DONE ===")
