# =============================================================================
# cap-x 标准 Stack 任务代码 —— Piper 机械臂（非GT / reduced API）
# =============================================================================
# 由 ExecutionEngineCapx 注入 cap-x 函数集后执行。只允许使用
# docs/primitive_api_capx.md 中的 15 个函数 + np + print（禁止 import）。
#
# 任务：robosuite Stack —— 把红方块(cubeA)堆到绿方块(cubeB)上。
#
# --- Piper 本体适配（2026-08-05 三次迭代定稿）---
# 坑 1: 桌面高度严格顶朝下不可达; 且可达姿态族随高度变化——抓取带(z 0.04-0.10)
# 倾斜 p50≈44°, 搬运带(z 0.16-0.20) p50≈67°（ik_library 实测）。夹爪-方块刚接,
# 换腕姿态=换方块姿态 → 抓取 quat 必须在抓取与放置两个高度同时可达。
# 运动层 = cgn_execute_grasp 验证链 + 先算后动门控：
#   每候选预解全链（预抓→下降→抬升→搬运→放置, 种子锁, 中途 loose 末点严格），
#   任一不收敛直接换下一候选（不碰方块）；执行 = 重放预解构型。
# 坐标：全部在机械臂基座系；四元数 wxyz；TCP = grip_site（solve_ik 目标）。
# =============================================================================

print("=" * 68)
print("cap-x/Piper Stack: SAM3 定位 → CGN 姿态抓红块 → 堆到绿块上")
print("=" * 68)

# 红块尺寸不写死（2026-08-06）: 放置基准用感知 min_z（locate 的 p5 底面），
# 官方 4cm 正方体与裁决3细高立块（4×4×8）均自适应。
# 收臂让拍位姿（2026-08-03 裁决 2, cgn_execute_grasp 同款; = engine_capx.TUCK_Q,
# 任务代码禁止 import 故内联）: j1 -90°, 让开相机→工作区走廊
TUCK_Q = np.array([-1.5708, 0.0, 0.0, 0.0, 0.0, 0.0])


def _q6():
    """当前 6 臂关节角。"""
    return np.asarray(get_observation()["robot_joint_pos"][:6], dtype=float)


def seg(p0, p1, n):
    """p0→p1 等分 n 段的路径点（不含 p0, 含 p1）。"""
    p0, p1 = np.asarray(p0, dtype=float), np.asarray(p1, dtype=float)
    return [p0 + (p1 - p0) * (k / n) for k in range(1, n + 1)]


def deproject(mask, depth, K, E):
    """mask + depth → 基座系点云（反投影约定 y = -(v-cy)*z/fy, 见 API 文档 §0）。

    工作空间门控（基座系: 桌面 z≈0, 物体 0~0.08）丢弃区外点——SAM3 可能把
    手臂红纹/背景误检为 "red cube"（实测误检 mask 反投影出 [-324,-108,-423]
    的鬼点云）。z 带 2026-08-05 修正（旧值系落地安装时代遗留）。
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
             & (pts[:, 2] > -0.05) & (pts[:, 2] < 0.20))
    return pts[in_ws]


def locate(prompt, obs_pack=None, z_cap=None):
    """SAM3 文本定位（molmo→point 级联兜底），返回 center/top_z/min_z/yaw/pts/mask。
    z_cap: 丢弃高于该基座 z 的点（验收/修复时绿块定位专用——红块已堆在绿上,
    SAM3 绿 mask 把红块裹入, p95 锚到红顶 → 绿心算偏 2cm+, 0806_1536_2 实锤）。"""
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
        if z_cap is not None:
            cand = cand[cand[:, 2] <= z_cap]
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
    # 顶面带过滤（2026-08-06 用户反馈: 侧面点混入致 center/top_z 有偏）——
    # 只保留顶面 8mm 带内的点重算 xy 中心与 top_z（侧面点 z 低被剔除）;
    # min_z 仍取全云（侧面点提供块底, stb/验收需要）。带内点太少回退全云。
    top_z0 = float(hi[2])
    band = pts[pts[:, 2] >= top_z0 - 0.008]
    if len(band) >= 30:
        blo = np.percentile(band, 5, axis=0)
        bhi = np.percentile(band, 95, axis=0)
        cx, cy, top_z = float((blo[0] + bhi[0]) / 2), float((blo[1] + bhi[1]) / 2), float(bhi[2])
    else:
        cx, cy, top_z = float((lo[0] + hi[0]) / 2), float((lo[1] + hi[1]) / 2), top_z0
    # 顶面带最小外接 yaw（0806_1501 夹棱事故修复）: 方形 footprint 的 PCA 主轴
    # 简并不稳, 改扫掠法——0~88° 步 2° 旋转投影, 双向量程和最小者即面法线 yaw
    # （方形 90° 简并任取）。供合成族闭合轴对准对面而非对角。
    xy = (band if len(band) >= 30 else pts)[:, :2]
    xy = xy - xy.mean(axis=0)
    yaw, best_r = 0.0, None
    for deg in range(0, 90, 2):
        c, s = np.cos(np.deg2rad(deg)), np.sin(np.deg2rad(deg))
        u = xy[:, 0] * c + xy[:, 1] * s
        v = -xy[:, 0] * s + xy[:, 1] * c
        r = (u.max() - u.min()) + (v.max() - v.min())
        if best_r is None or r < best_r:
            yaw, best_r = np.deg2rad(deg), r
    info = {
        "center": np.array([cx, cy, float((lo[2] + hi[2]) / 2)]),
        "top_z": top_z,
        "min_z": float(lo[2]),
        "yaw": float(yaw),
        "extent": obb["extent"],
        "pts": pts,
        "mask": mask,
    }
    print(f"  [locate] '{prompt}' center={info['center'].round(4)} top_z={info['top_z']:.4f} "
          f"yaw={float(np.rad2deg(yaw)):.0f}° extent={info['extent'].round(3)}")
    return info


def gripper_frac():
    return float(get_observation()["robot_joint_pos"][-1])


def _overhead_quat(tilt_deg=65.0, az_deg=0.0):
    """可达搬运带倾斜姿态（搬运带 z 0.16-0.20 p50≈67°）——抬升脱困用。
    0806_1526 教训: 近竖直抓取 quat 在 z≥0.18 是可达性死区, 带它抬升 IK 全灭。"""
    t, a = np.deg2rad(tilt_deg), np.deg2rad(az_deg)
    approach = np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), -np.cos(t)])
    yaxis = np.array([1.0, 0.0, 0.0]) - approach[0] * approach
    yaxis /= np.linalg.norm(yaxis)
    xaxis = np.cross(yaxis, approach)
    R = np.column_stack([xaxis, yaxis, approach]) @ np.diag([-1.0, 1.0, -1.0])
    return rotation_matrix_to_quaternion(R)


def _site_zcol(q_wxyz):
    """wxyz 四元数 → site z 轴（背向物体方向, 轴向撤离用; 0806_1900）。"""
    w, x, y, z = q_wxyz
    return np.array([2 * (x * z + w * y), 2 * (y * z - w * x), 1 - 2 * (x * x + y * y)])


def return_to_tuck():
    """凡动必避障规划（2026-08-06 用户严格指令）。顺滑结构（用户反馈跳变）:
    高空中转构型（当前正上方 +12cm, 一次 IK, 种子保分支连续）→
    execute_legs_rrt 统一规划+连续流式——中转点使 当前→高空 与 高空→TUCK
    两段都接近直线（直边自由无 zigzag）, 全程零中间阻塞。
    中转姿态走**高度-倾角阶梯**（ik_library 实测: z0.28-0.33 可达带 p50≈83°,
    65° 太平不收敛=0806 晚实锤）; 全阶梯不可解则直取 TUCK（RRT 绕行）;
    均失败抛错保持原位（禁盲扫）。"""
    # 先沿当前 site z 轴撤出 3cm（0806_1900 实测根因: 贴块起点净距 <3mm
    # 会使 RRT 起点非法——指尖-方块 =-1.7mm 实锤; 先脱离再规划）
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


def exec_leg(leg, relax=False, strict_last=False, graze=False):
    """逐构型重放预解腿。
    relax: 接触/负载段放宽 tol=0.10/fk_tol=0.03；
    strict_last: 末点严格 tol=0.02/fk_tol=0.005（抓取点——到位即真到位,
    闭合无沉降; 2026-08-05 空抓教训: 末点松 3cm = 捏偏挤出）；
    graze: 中途点容忍轻擦 tol=0.06/fk_tol=0.05（倾斜吞咽 8cm 立块时
    指腹擦碰属正常, 绝对关节目标会把链拉回来; 末点精度由 strict_last 把守）。
    """
    n = len(leg)
    for i, q in enumerate(leg):
        if strict_last and i == n - 1:
            move_to_joints(q, tol=0.02, fk_tol=0.005)
        elif relax:
            move_to_joints(q, tol=0.10, fk_tol=0.03)
        elif graze:
            move_to_joints(q, tol=0.06, fk_tol=0.05)
        else:
            move_to_joints(q)


def plan_full_chain(g_i, red_info, green_info):
    """先算后动: 预抓→下降→抬升→搬运→放置 全链预解（任一不收敛抛 RuntimeError）。

    返回 (quat, legs, pos_grasp, stb)。legs = [(名字, 构型列, 执行放宽)]。
    stb = 抓取 site→块底竖直距离（放置高度基准）。
    """
    quat = rotation_matrix_to_quaternion(g_i[:3, :3])
    z_col = g_i[:3, 2]            # site +z 背向物体 → 后撤方向 = +z_col
    pos_grasp = g_i[:3, 3].copy()
    pos_pre = pos_grasp + z_col * 0.038     # 预抓后撤（cgn_execute_grasp 严格模式同款:
                                            # site+0.038≈接触面上 10cm; 0.10 会顶进可达性死区）
    stb = float(pos_grasp[2] - red_info["min_z"])   # 抓取 site→感知块底（min_z）竖直距离
    # site→块心水平偏移补偿（0806_1536_2: 倾斜抓取时方块挂在 site 轴外 ~2.5cm,
    # site 对准绿心 = 块落偏 2.4-2.9cm 实锤）——刚接: 同一 quat 下 site→块心
    # 向量在基座系不变, 放置 site = 绿心 − 偏移。
    site_to_cube_xy = red_info["center"][:2] - pos_grasp[:2]
    place_z = green_info["top_z"] + stb + 0.004   # 落距 8→4mm（0806_1549:
        # 松手后 8mm 跌落使块走位, 撤离段拖飞; 指尖仍高于块底 ~12mm, 4mm 安全）
    place_pos = np.array([green_info["center"][0] - site_to_cube_xy[0],
                          green_info["center"][1] - site_to_cube_xy[1], place_z])
    lift_top = pos_grasp + np.array([0, 0, 0.08])   # 抬升够用即可（过高跌出可达带）
    hover = np.array([place_pos[0], place_pos[1], max(lift_top[2], place_z + 0.04)])
    pre_place = np.array([place_pos[0], place_pos[1], place_z + 0.025])
    leg_specs = [
        ("approach", [pos_pre] + seg(pos_pre, pos_grasp, 4), True),
        ("lift", seg(pos_grasp, lift_top, 6), False),
        ("transport", seg(lift_top, hover, 6), False),
        ("place", seg(hover, pre_place, 2) + seg(pre_place, place_pos, 3), True),
    ]
    legs = []
    q_cur = None
    for name, pts, strict_last in leg_specs:
        leg = []
        for i, p in enumerate(pts):
            strict = strict_last and i == len(pts) - 1
            if q_cur is None:
                # 首点（预抓）首选无种子全局解——cgn_execute_grasp 严格模式同款:
                # 全局解落在 glyph 目标滚转附近; 若上种子, 滚转被钉死在
                # 起始臂分支上（0806_1630 实测滚转差 66°=用户"没对应"根因）。
                # 不收敛则上种子兜底保可达（滚转由下降末点抛光兜底对齐）
                try:
                    q_cur = solve_ik(p, quat, free_approach_roll=True, loose=True)
                except RuntimeError:
                    q_cur = solve_ik(p, quat, free_approach_roll=True, seed=_q6(),
                                     loose=True)
            else:
                q_cur = solve_ik(p, quat, free_approach_roll=True, seed=q_cur,
                                 loose=not strict)
            leg.append(q_cur)
        legs.append((name, leg, not strict_last))
    return quat, legs, pos_grasp, stb


def synth_carry_candidates(red_info):
    """合成兜底族（CGN 空候选/全失败时启用）。

    0806_1501 事故修复: 旧版固定方位角 {0,60,300}° 在**旋转**方块上闭合轴
    跨对角 → 夹棱挤出（frac=0 空抓 + 方块飞出 28cm）; 现锚定 locate 顶面带
    扫掠 yaw（面法线方向）, 闭合轴跨对面。方形截面 90° 简并, 取 0/90 两值。
    接触点 = 云中心 xy + 块中腰（top_z-2cm）, site = 接触点 - 41.2mm·approach
    （锚点约定）; 方形截面 roll 自由。返回 4x4 位姿列表（site 约定: z_col 背向逼近）。
    """
    ctr = red_info["center"]
    contact = np.array([ctr[0], ctr[1], red_info["top_z"] - 0.02])
    yaw0 = red_info["yaw"]
    out = []
    for tilt_deg in (32, 25, 38, 45):
        for dyaw in (0.0, np.pi / 2):
            t, a = np.deg2rad(tilt_deg), yaw0 + dyaw
            approach = np.array([np.sin(t) * np.cos(a), np.sin(t) * np.sin(a), -np.cos(t)])
            yaxis = np.array([1.0, 0.0, 0.0]) - approach[0] * approach
            yaxis /= np.linalg.norm(yaxis)
            xaxis = np.cross(yaxis, approach)
            R = np.column_stack([xaxis, yaxis, approach]) @ np.diag([-1.0, 1.0, -1.0])
            g = np.eye(4)
            g[:3, :3] = R
            g[:3, 3] = contact - 0.0412 * approach
            out.append(g)
    return out


def pick_and_place(red_info, green_info, rgb_img, depth_img, K_img, max_candidates=6, _depth=0):
    """抓红块并堆到绿块上。成功返回 True, 失败返回 False（已尽量现场恢复）。
    _depth: 重定位重抓递归深度（防无限循环）。"""
    if _depth >= 2:
        print("  重抓递归深度超限, 放弃")
        return False
    seg_img = np.zeros(depth_img.shape, dtype=np.int32)
    seg_img[red_info["mask"]] = 1
    # CGN 直连: F2 glyph 标注落 trace cgn/ 文件夹（2026-08-05 用户要求证据可见;
    # 契约外后端直连）。执行位姿走 plan_grasp（基座系 site 约定, score 降序——
    # --render 时同一清单画 3D glyph: 品红=当前尝试 top-1, 与执行严格同序）。
    cgn_cam, cgn_scores, _ = grasp_cgn(rgb_img, depth_img, K_img, seg_img)
    print(f"  grasp_cgn: {len(cgn_cam)} 相机系候选（glyph 见 trace cgn/）")
    grasps, scores = plan_grasp(depth_img, K_img, seg_img)
    print(f"  plan_grasp: {len(grasps)} 候选, best score={float(scores.max()) if len(scores) else 0:.2f}")

    # 候选序列 = CGN score 序（前 max_candidates 位）→ 合成族兜底
    # 【2026-08-06 用户指令: 严格对齐 cgn_execute_grasp 严格模式——无 tilt 窗、
    # 无吸附、无 yaw 变体; 方块旋转由 CGN yaw 自适应（0806_1501: tilt 窗杀光
    # 高分 yaw 自适应候选 → 固定方位角合成族夹棱挤出 → 连锁失败）;
    # 抓取/放置双高度可达性由全链预解真实把关, 不预设几何窗口】
    cand_list = [(grasps[ci], f"cgn{ci}(s={scores[ci]:.2f})")
                 for ci in range(min(max_candidates, len(grasps)))]
    cand_list += [(g, f"syn{i}") for i, g in enumerate(synth_carry_candidates(red_info))]

    for ci, (g_i, tag) in enumerate(cand_list):
        print(f"\n=== 候选 {tag} ===")
        # 品红跟踪正在执行的候选（2026-08-06 用户要求）: 前 8 位内高亮对应位,
        # 超出则只画当前候选
        if ci < 8:
            draw_grasp_glyphs([g for g, _ in cand_list[:8]], highlight=ci)
        else:
            draw_grasp_glyphs([g_i], highlight=0)
        # 先算后动: 全链预解, 不可达直接换候选（不碰方块）
        try:
            quat, legs, pos_grasp, stb = plan_full_chain(g_i, red_info, green_info)
        except RuntimeError as e:
            print(f"  候选 {tag} 全链预解失败: {e}")
            continue
        print(f"  全链预解收敛 (stb={stb:.3f}), 开始执行")
        # 执行（cgn_execute_grasp 严格模式原样照抄）: 渐进全开 →
        # [预抓+下降] 逐腿 RRT + 密化流式 + 末点阻塞核验 → 渐进闭合
        try:
            set_gripper_ramp(1.0, 10)
            execute_legs_rrt(legs[0][1])
            set_gripper_ramp(0.0, 25)
        except RuntimeError as e:
            print(f"  候选 {tag} 接近执行失败: {e}")
            open_gripper()
            try:
                return_to_tuck()   # 归位再试（0806_1549: 臂停块旁, 后续候选
            except RuntimeError as e2:  # RRT 起点全脏→三连无解; cgn 脚本同款复位）
                print(f"  候选间归位失败（保持原位继续）: {e2}")
            continue
        frac = gripper_frac()
        print(f"  闭合后 frac={frac:.3f}")
        if frac <= 0.30:   # 空抓（实测: 4cm 方块夹持 frac≈0.55-0.57, 全闭 0.0;
            print("  ✗ 空抓, 下一位候选")  # 0806_1526 实测校准, 旧阈值 0.55 贴脸误判）
            open_gripper()
            try:
                return_to_tuck()   # 归位再定位（TUCK 让开相机走廊, 拍照干净）
            except RuntimeError as e2:
                print(f"  候选间归位失败（保持原位继续）: {e2}")
            # 扰动守卫: 方块若被碰跑, 后续候选全在抓空气——重定位重规划
            r_now = locate("red cube")
            if r_now is not None and np.linalg.norm(
                    r_now["center"][:2] - red_info["center"][:2]) > 0.02:
                print("  方块已被扰动 >2cm, 重定位重规划")
                cam_now = get_observation()["robot0_robotview"]
                return pick_and_place(r_now, green_info, cam_now["images"]["rgb"],
                                      cam_now["images"]["depth"], cam_now["intrinsics"],
                                      max_candidates, _depth + 1)
            continue
        print("  ✓ 夹持成功")
        # 抬升/搬运逐段验持（2026-08-05 空搬运实锤教训: 滑脱无声,
        # 空臂走完搬运/放置全程）; 失败原链反退放回后重定位重抓
        slipped = None
        placed = False
        try:
            for li, lname in ((1, "抬升"), (2, "搬运")):
                exec_leg(legs[li][1], relax=True)
                if gripper_frac() < 0.45:   # 滑脱线（4cm 块夹持≈0.55-0.57, 0806_1526 校准）
                    slipped = lname
                    break
            if slipped is None:
                exec_leg(legs[3][1], relax=True)  # place（末点=释放位）
                placed = True
        except RuntimeError as e:
            print(f"  持块执行失败: {e}")
        if slipped is not None:
            print(f"  ✗ {slipped}段夹持滑脱（frac<0.45）")
        if not placed:
            # 未释放的失败: 仍持有才反退放回（0806_1536_2: 撤离失败误入此支,
            # 空爪被反退回抓取位=用户所见"先回 CGN 位姿"事故）
            try:
                if gripper_frac() > 0.40:
                    for name, leg, _ in reversed(legs[1:]):
                        for q in reversed(leg):
                            move_to_joints(q, tol=0.10, fk_tol=0.03)
            except RuntimeError as e2:
                print(f"  反退失败: {e2}（放弃现场恢复）")
            open_gripper()
            r_now = locate("red cube")
            if r_now is not None:
                cam_now = get_observation()["robot0_robotview"]
                return pick_and_place(r_now, green_info, cam_now["images"]["rgb"],
                                      cam_now["images"]["depth"], cam_now["intrinsics"],
                                      max_candidates, _depth + 1)
            return False
        # 已到位 → 渐进释放 + 轴向撤离（0806_1900 实测根因: 松爪后方块贴
        # 指侧 finger1-cubeA=-1.7mm < 3mm 余量 → RRT 起点非法"腿0 无解";
        # 沿 site +z_col(背向物体=脱离方向) 种子锁撤出 6cm, 指尖净距单调
        # 增长, RRT 起点干净）→ RRT 升悬停, 失败退化走廊回放
        set_gripper_ramp(1.0, 10)
        try:
            z_col = g_i[:3, 2]   # 背向物体方向（与抓取/放置同 quat）
            p_now = get_observation()["robot_cartesian_pos"][:3]
            q_cur = _q6()
            for k in (1, 2):
                q_cur = solve_ik(p_now + z_col * (0.03 * k), quat,
                                 free_approach_roll=True, seed=q_cur, loose=True)
                move_to_joints(q_cur, tol=0.10, fk_tol=0.03)
        except RuntimeError as e:
            print(f"  轴向撤离失败({e}), 方块已释放, 继续尝试规划")
        try:
            execute_legs_rrt([legs[3][1][0]], final_tol=0.10, final_fk_tol=0.03)  # 悬停
        except RuntimeError as e:
            print(f"  悬停避障规划失败({e}), 走廊回放补全程")
            try:
                for q in list(reversed(legs[3][1]))[3:]:
                    move_to_joints(q, tol=0.10, fk_tol=0.03)
            except RuntimeError as e2:
                print(f"  撤离回放失败（方块已释放, 不阻塞）: {e2}")
        return True
    return False


# === Step 0: 收臂让拍 + 初始观测 + 双块定位 =====================================
print("\n--- Step 0: 收臂让拍 + 观测定位 ---")
try:
    execute_legs_rrt([TUCK_Q])   # 凡动必规划（2026-08-06 用户严格指令）
except RuntimeError as e:
    print(f"  Step0 避障规划失败({e}), 直插（初始桌面空旷, 风险可接受）")
    move_to_joints(TUCK_Q)
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

# === Step 1-4: 抓取 + 放置 =======================================================
ok = pick_and_place(red, green, cam0["images"]["rgb"], cam0["images"]["depth"], cam0["intrinsics"])

# === Step 5: 收臂让拍 + 验收 + 一次修复 ==========================================
print("\n--- Step 5: 收臂让拍 + 验收 ---")
try:
    return_to_tuck()   # RRT 避障收臂（原为裸 move_to_joints——扫落已堆红块的事故点）
except RuntimeError as e:
    print(f"  回收让拍位失败（不阻塞验收）: {e}")


def check_stacked():
    r = locate("red cube")
    g = locate("green cube", z_cap=0.065)   # 排除顶上的红块（>绿块身高即非绿本体）
    if r is None or g is None:
        print("  [verify] 检测失败, 无法验收")
        return None, None, False
    dz_ok = r["min_z"] > g["top_z"] - 0.012
    dxy = float(np.linalg.norm(r["center"][:2] - g["center"][:2]))
    print(f"  [verify] red_min_z={r['min_z']:.4f} green_top={g['top_z']:.4f} "
          f"dz_ok={dz_ok} dxy={dxy * 100:.2f}cm")
    return r, g, dz_ok and dxy < 0.03


r_f, g_f, ok_v = check_stacked()
if ok and ok_v:
    print("SUCCESS: 红块已堆在绿块上!")
elif r_f is not None and g_f is not None:
    print("REPAIR: 首次未成, 重抓重放一次")
    obs_r = get_observation()
    ok2 = pick_and_place(r_f, g_f, obs_r["robot0_robotview"]["images"]["rgb"],
                         obs_r["robot0_robotview"]["images"]["depth"],
                         obs_r["robot0_robotview"]["intrinsics"])
    try:
        return_to_tuck()   # RRT 避障（同 Step 5, 防扫落现场）
    except RuntimeError:
        pass
    _, _, ok_v2 = check_stacked()
    print("SUCCESS: 修复后堆叠成功!" if (ok2 and ok_v2) else "FAIL: 修复后仍未堆叠")
else:
    print("FAIL: 抓取/放置失败且无法验收")

print("=== TASK CODE DONE ===")
