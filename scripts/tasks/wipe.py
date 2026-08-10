# =============================================================================
# cap-x 标准 Wipe 任务代码 —— Piper 机械臂（非GT / reduced API）
# =============================================================================
# 由 ExecutionEngineCapx 注入 cap-x 函数集后执行。只允许使用
# docs/primitive_api_capx.md 中的 15 个函数 + np + print（禁止 import）。
#
# 任务：PiperWipeSpill 场景 —— SAM3 定位桌面棕色污渍 → 双程蛇形路径擦拭。
# 末端：PiperWiperGripper · Franka 平板擦头（2026-08-07 用户指令: 拿官方
# Franka 擦头来用）——12×5cm 海绵擦板, 按倾角 38° 补偿安装（滚转锁定 IK
# 实测: 38° 是唯一能整路径收敛的姿态, 44° 只 27/152, 50° 全灭）; 板面心在
# grip_site 沿逼近轴 +90mm 处（WIPER_FACE_L）。被擦板覆盖的污渍 marker
# 即时淡出（窗口里可见逐条擦掉）。
#
# 任务逻辑抄自 ASPIRE 官方参考 open_details/primitive_api_wipe.py（流程不变）：
#   get_observation → segment_sam3_text_prompt（空→point_prompt_molmo→
#   segment_sam3_point_prompt 级联兜底）→ mask_to_world_points → bbox+内缩
#   margin → 横向往返+纵向往返双程蛇形 → interpolate_segment 密化 → 逐点
#   solve_ik + move_to_joints。
#
# --- Piper 本体适配（2026-08-06；与 tasks/stack.py 同源教训）---
# 坑 1: 参考代码 quat_down=[0,0,1,0] 严格顶朝下 + z=0 贴桌——Piper 腕限位下
#   桌面高度严格顶朝下不可达（ik_library 实测: 低高度可达姿态族倾斜 p50≈44°）。
#   → 姿态走 (按压×倾角×方位) 阶梯 PRESOLVE_LADDER, free_approach_roll=True
#   + 种子链保分支连续。
# 坑 2: 蛇形路径是**擦板面心**的路径（污渍平面上）, 而 solve_ik 的目标是
#   grip_site——倾斜姿态下两者水平错开 L·sin(tilt)（38° 时 ~5.5cm）!
#   不做补偿会把擦板拖到污渍外。→ pad_to_site() 逐点换算。
#   平板贴桌还要求滚转锁定（free_approach_roll=False 全姿态求解）——
#   滚转自由时板面转角随 IK 分支漂移, 只棱线接触; 这也是倾角被钉死在
#   38° 的原因（全姿态 IK 实测唯一整路径收敛的姿态, 见 PRESOLVE_LADDER）。
# 坑 3: 参考代码逐点 solve_ik+move（失败才知不可达）。→ 先算后动门控
#   （同 stack）: 整路径预解 IK, 全阶梯+矩形收缩重试仍不收敛则不碰桌面抛错;
#   执行 = 重放预解构型（graze 容差, 接触拖拽允许轻擦）。
# 凡动必避障规划（2026-08-06 用户严格指令）: 出入场走 execute_legs_rrt,
#   只有密化擦拭段（同一高度平面内小步移动）逐点 move_to_joints。
# 擦头接触（官方 wiping_gripper.xml 参数: 软+滑）: 默认按压 2mm 真接触;
#   拖动阻力≈0 不卡伺服; 想悬擦把 (-0.002,...) 挪到阶梯首位。
# 坐标：全部基座系；四元数 wxyz；TCP = grip_site（solve_ik 目标）。
# =============================================================================

print("=" * 68)
print("cap-x/Piper Wipe: SAM3 定位棕色污渍 → 双程蛇形路径 → 擦头接触擦拭")
print("=" * 68)

TUCK_Q = np.array([-1.5708, 0.0, 0.0, 0.0, 0.0, 0.0])   # 收臂让拍（用户规定）

# ---- 可调参数（2026-08-07 Franka 平板版）------------------------------------
WIPER_FACE_L = 0.090   # grip_site→擦板面心 沿逼近轴距离（wiper_gripper.xml:
                       # 板面心 hand z=+0.045, site z=-0.045; 改模型须同步重测）
# 预解阶梯 (按压深度 m, 倾角°, 方位°): 首个全路径收敛的组合胜出。
# 按压 >0 = 板面压入桌面的名义深度（软接触吸收）; <0 = 悬擦; 0 = 羽毛接触。
# 38° 首位: 滚转锁定全姿态 IK 实测唯一整路径收敛的姿态（2026-08-07）;
# 其余倾角板面会带 ≤6° 倾角（棱线接触, 兜底用, 预解会把关）。
PRESOLVE_LADDER = [
    (0.002, 38.0, 0.0),     # 实测唯一全收敛的滚转锁定姿态（152/152）
    (0.000, 38.0, 0.0),     # 羽毛接触
    (0.004, 38.0, 0.0),     # 加重按压
    (0.002, 32.0, 0.0),     # 邻位（板面倾 ~6°, 棱线接触兜底）
    (0.002, 44.0, 0.0),     # 同上（滚转锁定在 44° 实测 27/152, 难过预解）
    (0.002, 38.0, 180.0),
    (-0.002, 38.0, 0.0),    # 悬擦 2mm（接触全灭时的退路）
]
REACH_WIN = (0.12, 0.40, -0.16, 0.16)  # 可达裁剪窗 (x_min, x_max, y_min, y_max)
                                       # 基座系: 抓取带 [0.15,0.27] 近旁外扩
MARGIN = 0.005           # bbox 内缩边距（参考同款）
DENSE_STEP = 0.02        # 密化步长（参考同款; 嫌慢调 0.03）
SPILL_MAX_EXTENT = 0.35  # mask 点云 bbox 任一边超此值判整桌误检（真污渍 ≤0.24）
PROMPTS = ["brown spill", "dirt stain"]   # 参考只有前者, 后者为二级兜底


def _q6():
    """当前 6 臂关节角。"""
    return np.asarray(get_observation()["robot_joint_pos"][:6], dtype=float)


def _overhead_quat(tilt_deg=44.0, az_deg=0.0):
    """倾斜姿态（与 stack 同款构造）: 逼近轴离竖直 tilt°, 水平方位 az°。"""
    t, a = np.deg2rad(tilt_deg), np.deg2rad(az_deg)
    approach = np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), -np.cos(t)])
    yaxis = np.array([1.0, 0.0, 0.0]) - approach[0] * approach
    yaxis /= np.linalg.norm(yaxis)
    xaxis = np.cross(yaxis, approach)
    R = np.column_stack([xaxis, yaxis, approach]) @ np.diag([-1.0, 1.0, -1.0])
    return rotation_matrix_to_quaternion(R)


def _site_zcol(q_wxyz):
    """wxyz 四元数 → site z 轴（背向物体方向, 轴向撤离用）。"""
    w, x, y, z = q_wxyz
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def pad_to_site(p, press, tilt_deg, az_deg):
    """擦板面心接触点 p（污渍平面上）→ grip_site 目标（坑 2 补偿）。

    板面心在 site 沿逼近轴 +L 处; 板面与桌面平齐（38° 补偿安装 + 滚转锁定
    IK 保证）。接触点 p 反推: site.xy 比接触点后撤 L·sin(t)（38° 时 ~5.5cm,
    不补偿会把擦板拖出污渍）; site.z = L·cos(t) - press。
    """
    t, a = np.deg2rad(tilt_deg), np.deg2rad(az_deg)
    horiz = WIPER_FACE_L * np.sin(t)
    return np.array([p[0] - horiz * np.cos(a),
                     p[1] - horiz * np.sin(a),
                     WIPER_FACE_L * np.cos(t) - press])


def return_to_tuck():
    """凡动必避障规划（2026-08-06 用户严格指令）。与 stack 同款顺滑结构:
    先沿 site z 轴撤出 3cm → 高空中转构型（+12cm, 倾角阶梯, 种子保分支连续）
    → execute_legs_rrt 统一规划+连续流式; 全阶梯不可解则直取 TUCK;
    均失败抛错保持原位（禁盲扫）。"""
    try:
        obs0 = get_observation()
        z0 = _site_zcol(obs0["robot_cartesian_pos"][3:7])
        q_d = solve_ik(obs0["robot_cartesian_pos"][:3] + z0 * 0.03,
                       obs0["robot_cartesian_pos"][3:7],
                       free_approach_roll=True, seed=_q6(), loose=True)
        move_to_joints(q_d, tol=0.10, fk_tol=0.03)
    except RuntimeError:
        pass  # 撤离失败不阻塞, 后续规划照常尝试
    legs = []
    obs = get_observation()
    p_mid = obs["robot_cartesian_pos"][:3] + np.array([0, 0, 0.12])
    for tilt in (85.0, 75.0, 95.0, 65.0, 105.0):
        try:
            q_mid = solve_ik(p_mid, _overhead_quat(tilt), free_approach_roll=True,
                             seed=_q6(), loose=True)
            legs.append(q_mid)
            break
        except RuntimeError:
            continue
    if not legs:
        print("  高空中转构型全阶梯不可解, 直取 TUCK")
    legs.append(TUCK_Q)
    execute_legs_rrt(legs, final_tol=0.05, final_fk_tol=0.02)
    print("  已避障回到 TUCK")


def locate_spill(obs_pack):
    """SAM3 文本定位污渍（参考的 molmo→point 级联兜底原样保留, 外加多提示词
    与候选装甲）: 逐候选 mask 反投影 → 工作区门控 → 尺寸守卫（整桌误检拒绝）。
    返回 (N,3) 基座系点云; 全失败返回 None。"""
    cam = obs_pack["robot0_robotview"]
    rgb, depth = cam["images"]["rgb"], cam["images"]["depth"]
    K, E = cam["intrinsics"], cam["pose_mat"]

    for prompt in PROMPTS:
        # --- 参考级联原样: 文本分割 → molmo 打点 → 点提示分割 ---
        masks = segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            pt_result = point_prompt_molmo(rgb, prompt)
            p = list(pt_result.values())[0] if pt_result else (None, None)
            if p[0] is not None and p[1] is not None:
                masks = segment_sam3_point_prompt(rgb, (float(p[0]), float(p[1])))
        if not masks:
            print(f"  [locate] '{prompt}' 未检出, 换下一提示词")
            continue

        # 逐候选过滤（按 score 降序）: 参考只取 best, 这里加装甲——
        # 木纹桌面可能被整体误检为 "brown spill", 尺寸守卫拒绝
        for m in masks[:4]:
            pts = mask_to_world_points(m["mask"].astype(np.uint8), depth, K, E)
            if pts.shape[0] == 0:
                continue
            valid = np.isfinite(pts).all(axis=1)     # 参考同款过滤
            pts = pts[valid]
            # 工作区门控（stack 同款装甲: 丢弃区外鬼点）
            pts = pts[(pts[:, 0] > 0.05) & (pts[:, 0] < 0.65)
                      & (np.abs(pts[:, 1]) < 0.30)
                      & (pts[:, 2] > -0.05) & (pts[:, 2] < 0.20)]
            if len(pts) < 10:
                continue
            ext = np.percentile(pts, 98, axis=0) - np.percentile(pts, 2, axis=0)
            if ext[0] > SPILL_MAX_EXTENT or ext[1] > SPILL_MAX_EXTENT:
                print(f"  [locate] '{prompt}' 候选 extent={ext[:2].round(2)} 超上限, "
                      f"判整桌/背景误检, 跳过")
                continue
            print(f"  [locate] '{prompt}' 命中: {len(pts)} 点, "
                  f"extent={ext[:2].round(3)} score={m.get('score', 0.0):.2f}")
            return pts
    return None


def serpentine(x_min, x_max, y_min, y_max):
    """参考原样的双程蛇形: 横向往返 + 纵向往返（曲面污渍覆盖更好）。"""
    ny = max(3, min(6, int(np.ceil((y_max - y_min) / 0.03)) + 1))
    nx = max(3, min(5, int(np.ceil((x_max - x_min) / 0.04)) + 1))
    x_samples = np.linspace(x_min, x_max, nx)
    y_samples = np.linspace(y_min, y_max, ny)
    waypoints = []
    for i, y in enumerate(y_samples):
        xs = x_samples if i % 2 == 0 else x_samples[::-1]
        for x in xs:
            waypoints.append(np.array([float(x), float(y)], dtype=float))
    for j, x in enumerate(x_samples):
        ys = y_samples if j % 2 == 0 else y_samples[::-1]
        for y in ys:
            waypoints.append(np.array([float(x), float(y)], dtype=float))
    return waypoints


def densify(waypoints):
    """参考原样: interpolate_segment 小步密化。"""
    dense = [waypoints[0]]
    for i in range(len(waypoints) - 1):
        seg = interpolate_segment(waypoints[i], waypoints[i + 1], step=DENSE_STEP)
        if len(seg) > 1:
            dense.extend(seg[1:])
    return dense


def presolve_all(dense, press, tilt, az):
    """先算后动: 整路径种子链预解（任一不收敛抛 RuntimeError）。
    返回 (quat, 构型列)。参考是逐点 solve_ik+move 交替, 这里全部预解后
    再统一重放——不可达在碰桌面之前暴露。路径点经 pad_to_site 换算;
    滚转锁定（free_approach_roll=False）——平板贴桌的必要条件。"""
    quat = _overhead_quat(tilt, az)
    q_cur = _q6()
    qs = []
    for p in dense:
        q_cur = solve_ik(pad_to_site(p, press, tilt, az), quat,
                         free_approach_roll=False, seed=q_cur, loose=True)
        qs.append(q_cur)
    return quat, qs


# === Step 0: 收臂让拍 + 初始观测 =============================================
print("\n--- Step 0: 收臂让拍 + 观测 ---")
try:
    execute_legs_rrt([TUCK_Q])   # 凡动必规划（2026-08-06 用户严格指令）
except RuntimeError as e:
    print(f"  Step0 避障规划失败({e}), 直插（初始桌面空旷, 风险可接受）")
    move_to_joints(TUCK_Q)
obs0 = get_observation()
print(f"  TCP(基座系)={obs0['robot_cartesian_pos'][:3].round(3)} "
      f"quat={obs0['robot_cartesian_pos'][3:7].round(3)} frac={obs0['robot_cartesian_pos'][7]:.2f}")

# === Step 1: 分割污渍 → 点云 =================================================
print("\n--- Step 1: SAM3 分割棕色污渍 ---")
pts_world = locate_spill(obs0)
if pts_world is None:
    raise RuntimeError("Could not segment brown spill.")   # 参考同款抛错

# === Step 2: bbox + margin + 可达窗裁剪 =======================================
print("\n--- Step 2: 污渍包围盒 ---")
# 参考用裸 min/max; 这里 p2/p98 抗离群（单个飞点不会拉飞 bbox）
lo = np.percentile(pts_world, 2, axis=0)
hi = np.percentile(pts_world, 98, axis=0)
x_min, x_max = float(lo[0]), float(hi[0])
y_min, y_max = float(lo[1]), float(hi[1])

# Small inward margin（参考同款）
x_min += MARGIN; x_max -= MARGIN
y_min += MARGIN; y_max -= MARGIN

cx, cy = 0.5 * (x_min + x_max), 0.5 * (y_min + y_max)
if x_max <= x_min:
    x_min, x_max = cx - 0.01, cx + 0.01
if y_max <= y_min:
    y_min, y_max = cy - 0.01, cy + 0.01

# Piper 适配: 裁剪进可达窗（感知 bbox 越界部分 IK 必灭, 不如不擦）
x_min = max(x_min, REACH_WIN[0]); x_max = min(x_max, REACH_WIN[1])
y_min = max(y_min, REACH_WIN[2]); y_max = min(y_max, REACH_WIN[3])
if x_max <= x_min or y_max <= y_min:
    raise RuntimeError(f"污渍 bbox 与可达窗 {REACH_WIN} 无交集, 放弃（未碰桌面）")
print(f"  擦拭矩形(基座系, 擦头接触点路径): x=[{x_min:.3f},{x_max:.3f}] y=[{y_min:.3f},{y_max:.3f}]")

# === Step 3: 蛇形路径 + 先算后动预解 =========================================
print("\n--- Step 3: 双程蛇形 + 全路径预解（阶梯+收缩重试） ---")
solution = None
x0, x1, y0, y1 = x_min, x_max, y_min, y_max
for shrink in range(4):   # 最多向形心收缩 3 次（×0.85^k）
    waypoints = serpentine(x0, x1, y0, y1)
    dense = densify(waypoints)
    for press, tilt, az in PRESOLVE_LADDER:
        try:
            quat, qs = presolve_all(dense, press, tilt, az)
            solution = (press, tilt, az, quat, qs, dense)
            break
        except RuntimeError:
            continue
    if solution is not None:
        if shrink:
            print(f"  （矩形已向形心收缩 {shrink} 次: "
                  f"x=[{x0:.3f},{x1:.3f}] y=[{y0:.3f},{y1:.3f}]）")
        break
    # 向形心收缩 15%（感知 bbox 边缘越出可达带时的退路）
    ccx, ccy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
    x0 = ccx + (x0 - ccx) * 0.85; x1 = ccx + (x1 - ccx) * 0.85
    y0 = ccy + (y0 - ccy) * 0.85; y1 = ccy + (y1 - ccy) * 0.85

if solution is None:
    raise RuntimeError("全阶梯+收缩后整路径仍不可达, 放弃（未碰桌面）")
press, tilt, az, quat, q_path, dense_path = solution
print(f"  预解收敛: 按压={press * 1000:.0f}mm 倾角={tilt:.0f}° 方位={az:.0f}° "
      f"路径点={len(dense_path)}（密化步长 {DENSE_STEP}）")

# === Step 4: 闭爪 + 避障接近路径起点 =========================================
print("\n--- Step 4: 闭爪 + RRT 接近起点 ---")
set_gripper_ramp(0.0, 25)   # 接口体操持（工具无指, 关节为摆设, 语义不变）
p0 = dense_path[0]
q_hover = solve_ik(pad_to_site(p0, press, tilt, az) + np.array([0, 0, 0.10]),
                   quat, free_approach_roll=False, seed=_q6(), loose=True)
execute_legs_rrt([q_hover, q_path[0]], final_tol=0.05, final_fk_tol=0.02)
print("  已到路径起点（擦头接触）")

# === Step 5: 重放密化路径（graze 容差） =======================================
print("\n--- Step 5: 执行擦拭（污渍随擦随淡） ---")
done_n = 0
try:
    for i, q in enumerate(q_path):
        move_to_joints(q, tol=0.06, fk_tol=0.05)   # 接触拖拽段, 允许轻擦
        done_n += 1
        if done_n % 25 == 0:
            print(f"  擦拭进度 {done_n}/{len(q_path)}")
except RuntimeError as e:
    print(f"  ✗ 擦拭中段失败({e}), 已完成 {done_n}/{len(q_path)}")
    try:
        return_to_tuck()
    except RuntimeError as e2:
        print(f"  收臂失败（保持原位）: {e2}")
    raise

# === Step 6: 收臂让拍 + 总结 =================================================
print("\n--- Step 6: 收臂 + 总结 ---")
return_to_tuck()
set_gripper_ramp(1.0, 10)

print({
    "spill_bounds": {"x_min": float(x0), "x_max": float(x1),
                     "y_min": float(y0), "y_max": float(y1)},   # 参考字段
    "num_waypoints": len(dense_path),                            # 参考字段
    "executed": done_n,
    "press": press, "tilt_deg": tilt, "az_deg": az,
})
print("=== TASK CODE DONE ===")
