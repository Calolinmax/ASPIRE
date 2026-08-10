"""关节空间 RRT-Connect 避障规划（2026-08-05 用户裁决引入）。

架构分工（用户裁决）：CGN 候选筛选只看 IK 可达性（solve_ik 已摘碰撞门）；
驱动机械臂去 CGN 位姿的路径规划带避障（含腕部实体, robot.xml 已恢复）。

碰撞真值 = MuJoCo 模型本身（与执行同模型）——不用球体近似, 指尖跨方块
6.7mm 级余量的贴脸构型判据与执行完全一致（cuRobo 方案因此否决, 2026-08-05）。

碰撞判对规则镜像 primitives_capx._collision_free_q（臂 robot0_g*_col /
robot0_link* vs table/pedestal/floor/cube/riser, 指尖-方块允许），
性能适配：复用单个 MjData + mj_forward（不再每次新建+mj_step）。
⚠ _collision_free_q 的规则若改, 这里必须同步。

🔒 冻结警示（2026-08-05 用户裁决）: 此处只有人类（顾问也不行）批准，才能更改。
"""
# =============================================================================
# 🔒 冻结警示（2026-08-06 用户裁决 · 封版）：本文件属**已测试通过**的 API 层
# （cap-x 契约 15 函数 + 契约外 5 函数/组件，docs/api_asset_map.md 冻结清单）。
# **只能在 scripts 中调用，禁止修改——只有人类（顾问也不行）批准才能更改。**
# 本文件同时被 chmod a-w 机械保护；解冻须人类亲自 chmod +w。
# =============================================================================

import time

import mujoco
import numpy as np

from ..api.primitives_capx import ARM_JOINT_NAMES


def _make_checker(engine, margin=0.003):
    """返回 is_free(q6) —— 复用 MjData 的快速碰撞检查（规则镜像 _collision_free_q）。

    两级判据:
      1. 接触级（同 _collision_free_q）: 臂 vs table/pedestal/floor/cube/riser
         产生接触即不自由;
      2. 余量级（2026-08-05 用户: 到位后仍有轻微剐蹭）: 腕部/手指几何与 cube
         的距离 < margin(默认 2mm) 即不自由——接触判据在 6.7mm 跨指余量下
         会漏掉毫米级擦碰(边离散中点穿模)。mj_geomDistance 逐对测距。
         余量必须小于功能净距(对角抓 6.7mm), 否则合法跨指被误杀。
    """
    model = engine.env.sim.model._model
    data = mujoco.MjData(model)
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINT_NAMES]
    qposadr = [int(model.jnt_qposadr[j]) for j in jids]

    # 余量测距对: 腕/指几何 × 方块几何
    margin_arm, margin_cube = [], []
    cube_pos_id = None
    for gi in range(model.ngeom):
        nm = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gi) or ""
        if nm == "robot0_g6_col" or ("gripper0_right" in nm and "collision" in nm):
            margin_arm.append(gi)
        if "cube" in nm and "_vis" not in nm:
            margin_cube.append(gi)
    cube_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cubeA_main")
    fromto = np.zeros(6)

    def is_free(q):
        data.qpos[:] = engine.env.sim.data.qpos
        data.qvel[:] = 0.0
        for i, adr in enumerate(qposadr):
            data.qpos[adr] = q[i]
        mujoco.mj_forward(model, data)
        for c in range(data.ncon):
            n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact.geom1[c]) or ""
            n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact.geom2[c]) or ""
            pair = {n1, n2}
            arm_hit = [n for n in pair if ("robot0_g" in n and "_col" in n) or "robot0_link" in n]
            if not arm_hit:
                continue
            other = pair - set(arm_hit)
            if any(("table" in o) or ("pedestal" in o) or ("floor" in o)
                   or ("cube" in o) or ("riser" in o) for o in other):
                return False
        # 余量级: 手离方块远(>15cm)时跳过测距省时间
        if margin_arm and margin_cube and cube_body >= 0:
            hand_p = data.geom_xpos[margin_arm[0]]
            cube_p = data.xpos[cube_body]
            if np.linalg.norm(hand_p - cube_p) < 0.15:
                for ga in margin_arm:
                    for gc in margin_cube:
                        if mujoco.mj_geomDistance(model, data, ga, gc, margin, fromto) < margin:
                            return False
        return True

    return is_free


def plan_joint_path(q_start, q_goal, engine, *, max_iters=3000, step=0.05,
                    goal_bias=0.15, smooth_iters=150, time_budget_s=6.0,
                    rng_seed=0, margin=0.003):
    """RRT-Connect 关节空间避障规划。

    直边优先: 起终点直线插值无碰撞则直接返回（绝大多数短腿走这里, ~ms 级）;
    否则双向 RRT-Connect 采样绕行, 最后捷径平滑。

    Args:
        q_start/q_goal: (6,) 起止关节角 (rad)
        engine: ExecutionEngineCapx（碰撞真值来源）
        max_iters: RRT 最大迭代
        step: 扩展步长 / 边离散分辨率 (rad, max 范数)
        goal_bias: 目标偏置采样概率
        smooth_iters: 捷径平滑迭代数
        time_budget_s: 时间预算（超时返回 None）
        rng_seed: 采样种子（可复现）
        margin: 手/腕-方块净距余量 (m, 默认 3mm; 必须 < 跨指功能净距 6.7mm。
            2026-08-05 剐蹭-移位事故: 2mm 余量 + 0.05rad 边分辨率仍有亚毫米
            擦碰, 擦碰移位方块后早先预算的后续路径过期 → 末步卡死 err=0.11rad)

    Returns:
        list of (6,) 路径点（含起终点）; 找不到 / 起终点碰撞 / 超时 → None
    """
    q_start = np.asarray(q_start, dtype=np.float64).reshape(6)
    q_goal = np.asarray(q_goal, dtype=np.float64).reshape(6)
    model = engine.env.sim.model._model
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINT_NAMES]
    q_lo = np.array([model.jnt_range[j][0] for j in jids])
    q_hi = np.array([model.jnt_range[j][1] for j in jids])
    q_start = np.clip(q_start, q_lo, q_hi)
    q_goal = np.clip(q_goal, q_lo, q_hi)

    is_free = _make_checker(engine, margin=margin)

    def edge_free(qa, qb):
        dq = qb - qa
        # 边校验分辨率 = step/2（2026-08-05: 0.05rad 分辨率中点会漏毫米级擦碰）
        n = max(1, int(np.ceil(np.abs(dq).max() / (step / 2))))
        for i in range(1, n + 1):
            if not is_free(qa + dq * (i / n)):
                return False
        return True

    # 直边优先（2026-08-05: 多数腿直边即自由, 省掉 RRT 开销）
    if not is_free(q_start) or not is_free(q_goal):
        return None
    if edge_free(q_start, q_goal):
        return [q_start, q_goal]

    rng = np.random.default_rng(rng_seed)
    Ta_n, Ta_p = [q_start], [-1]   # A 树: 节点 + 父索引
    Tb_n, Tb_p = [q_goal], [-1]

    def nearest(nodes, q):
        return int(np.argmin(np.linalg.norm(np.asarray(nodes) - q, axis=1)))

    def extend(nodes, parents, q_target, idx):
        """从 nodes[idx] 向 q_target 扩展一步（≤step）, 成功返回新索引, 失败 None"""
        q_from = nodes[idx]
        d = q_target - q_from
        dist = np.abs(d).max()
        if dist < 1e-9:
            return idx
        q_new = q_from + d * (step / dist) if dist > step else q_target
        if not edge_free(q_from, q_new):
            return None
        nodes.append(q_new)
        parents.append(idx)
        return len(nodes) - 1

    t0 = time.perf_counter()
    # 可视化心跳（2026-08-06 用户反馈: RRT 规划期窗口冻结像瞬移）:
    # 搜索循环中每 ~0.15s sync 一次 live viewer, 画面保持活动; 无窗口零开销。
    _viewer = getattr(engine, "_live_viewer", None)
    _t_pulse = t0
    path = None
    for _ in range(max_iters):
        if time.perf_counter() - t0 > time_budget_s:
            break
        if _viewer is not None and time.perf_counter() - _t_pulse > 0.15:
            _t_pulse = time.perf_counter()
            if _viewer.is_running():
                _viewer.sync()
        q_rand = Tb_n[0] if rng.random() < goal_bias else rng.uniform(q_lo, q_hi)
        ia = nearest(Ta_n, q_rand)
        ja = extend(Ta_n, Ta_p, q_rand, ia)
        if ja is None:
            continue
        q_new = Ta_n[ja]
        jb = nearest(Tb_n, q_new)
        reached = False
        while True:  # connect: B 树向新节点贪婪扩展直到卡住或到达
            jb2 = extend(Tb_n, Tb_p, q_new, jb)
            if jb2 is None:
                break
            jb = jb2
            if np.abs(Tb_n[jb] - q_new).max() < 1e-9:
                reached = True
                break
        if reached:
            pa, k = [], ja
            while k != -1:
                pa.append(Ta_n[k])
                k = Ta_p[k]
            pa.reverse()
            pb, k = [], jb
            while k != -1:
                pb.append(Tb_n[k])
                k = Tb_p[k]
            path = pa + pb
            # 树每轮互换, 端点顺序不定——按离 start/goal 的距离校正方向
            if np.linalg.norm(path[0] - q_goal) < np.linalg.norm(path[0] - q_start):
                path.reverse()
            break
        Ta_n, Tb_n = Tb_n, Ta_n   # 互换保持双树均衡
        Ta_p, Tb_p = Tb_p, Ta_p

    if path is None:
        return None

    # 捷径平滑: 随机抽两点, 直边自由则短路
    pts = list(path)
    for _ in range(smooth_iters):
        if len(pts) <= 2:
            break
        i, j = sorted(rng.integers(0, len(pts), 2))
        if j - i < 2:
            continue
        if edge_free(pts[i], pts[j]):
            pts = pts[:i + 1] + pts[j:]
    return pts
