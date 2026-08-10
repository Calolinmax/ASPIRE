"""ASPIRE Primitive API —— cap-x 标准（非GT）/ Piper 机械臂。

对应 cap-x 的 FrankaControlApiReduced（external/cap-x/capx/integrations/franka/
control_reduced.py，privileged=false 的 cube_stack 配置所注入的函数集），
在 Piper + 本项目基础设施上的等价实现：

  | cap-x (Franka)                     | 本实现 (Piper)                        |
  |------------------------------------|---------------------------------------|
  | SAM3 server (HTTP)                 | 本地 SAM3 (aspire.perception.vision_sam3, 同权重) |
  | Molmo server (vLLM)                | SAM3 级联 shim (同名同签名)            |
  | Contact-GraspNet server            | CGN docker 容器 (官方 TF 版, 同约定)   |
  | pyroki IK server (panda URDF)      | pyroki server (独立 venv) + DLS 兜底  |
  | JOINT_POSITION 力矩控制            | position 执行器直写 (真机同模式)       |

与 cap-x 契约保持一致：函数名/签名/返回结构、基座坐标系、wxyz 四元数、阻塞语义、
solve_ik 的"目标=夹爪 TCP 位姿"语义
（Piper TCP_OFFSET=0 —— grip_site 即指尖 TCP，见 gripper.xml）。

三处**有意的偏差**（docs/primitive_api_capx.md 均注明）：
  1. 反投影约定：y = -(v-cy)*z/fy（mujoco 相机系 y 向上，本仓库实测标定；
     cap-x 为 y=+(v-cy)*z/fy —— 其 pose_mat 内叠了修正旋转，数学上等价，
     这里保持 pose_mat 为真刚体变换，便于直接复合抓取位姿）。
  2. robot_joint_pos 为 (7,) = 6 臂关节 + 夹爪开合度（Franka 为 7+1=(8,)）。
  3. plan_grasp 返回**基座系** TCP 位姿（cap-x 返回相机系原始候选、由调用方
     自行左乘 pose_mat）。本实现已在内部完成 cam→基座系变换 + cgn_to_gripper
     TCP 对齐（用户肉审+端到端验证 2026-08-04/05），任务代码直接使用返回值，
     **勿再左乘 pose_mat**。需要相机系原始候选时走契约外的 grasp_cgn()。
     【2026-08-05 注释修正：此处原文把"plan_grasp 相机系返回约定"列入上方
     契约一致清单，与已验证行为矛盾——以行为为准】
"""

# =============================================================================
# 🔒 冻结警示（2026-08-05 用户裁决）：本模块中已完成并经行为验证的 API 项
# （docs/api_asset_map.md 看板 `- [x]` 项：cap-x 契约 10 函数、plan_grasp /
# grasp_cgn（CGN 官方 TF 版集成）、cgn_to_gripper retarget 链、solve_ik 链
# （pyroki+DLS+库+滚转扫描+loose）、move_to_joints(+_safely)、碰撞采集
# _collect_collision_events/_collision_free_q 判对规则；
# 2026-08-05 E1 收尾新增：PiperControlApiReduced 类结构 + functions() 15 键、
# select_top_down_grasp、工具四函数 mask_to_world_points /
# pixel_to_world_point / rotation_matrix_to_quaternion / interpolate_segment
# —— 均经 scripts/test_piper_capx_api.py 33/33 行为验证）——
# **此处只有人类（顾问也不行）批准，才能更改。**
# =============================================================================

# =============================================================================
# 🔒 冻结警示（2026-08-06 用户裁决 · 封版）：本文件属**已测试通过**的 API 层
# （cap-x 契约 15 函数 + 契约外 5 函数/组件，docs/api_asset_map.md 冻结清单）。
# **只能在 scripts 中调用，禁止修改——只有人类（顾问也不行）批准才能更改。**
# 本文件同时被 chmod a-w 机械保护；解冻须人类亲自 chmod +w。
# =============================================================================

from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np
from robosuite.utils import camera_utils as CU
from robosuite.utils import transform_utils as T

from ..perception.vision_client import segment_sam3_text_prompt as _sam3_text_raw
from ..perception.vision_client import segment_sam3_point_prompt as _sam3_point_raw
from ..perception.vision_client import grasp_cgn as _grasp_cgn_raw
from ..evidence.annotate import build_annotation

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
ROBOTVIEW = "robot0_robotview"
RENDER_H = RENDER_W = 640   # 2026-07-31 起 640（方块 15-20px→~38px, cube_max 0.41→0.82 实测）
                            # 历史: 512 下 EGL wedge 率显著更高(engine.py:195 旧病);
                            # 640 经 10-seed 门验证后方可转正的档位, 缓冲区由 engine 预置 1280×960

EEF_SITE = "gripper0_right_grip_site"
ARM_JOINT_NAMES = [f"robot0_joint{i}" for i in range(1, 7)]

# move_to_joints 阻塞参数（cap-x robosuite_base.move_to_joints_blocking 同款）
JOINT_TOL = 0.02          # rad (L2)
MAX_TICKS_MOVE = 100      # 100 拍 ≈ 5s @20Hz
GRIPPER_SETTLE_TICKS = 15

# plan_grasp 几何规划器参数
GRASP_N_YAW = 8           # 绕逼近轴的偏航候选数
GRASP_CENTER_DROP = 0.02  # 可见顶面 → 估计中心的下探量 (m)

# CGN 坐标变换参数（两层常量分开，勿合并 —— 2026-07-31 变换链调试结论）
#
# 层 1（CGN 服务端夹爪约定层）: CGN 输出的物理含义。
#   pred_grasps_cam 4x4 直接放置服务端 gripper 模型：原点 = 掌根(hand root)，
#   局部 +z = 逼近方向(腕部→指尖)，指尖垫接触平面在局部 +z GRIPPER_DEPTH_CGN 处。
#   【2026-08-04 曾改 Piper(0.045) → 同日回退】: 用户裁决——CGN 输出是基准,
#   不许动（改的是"末端去够输出"的链, 不是输出）。estimator 已支持 config
#   透传 gripper_depth, config 不写即默认 Panda 0.1034（指尖尖端 0.1122）。
GRIPPER_DEPTH_CGN = 0.1034
#
# 层 2（Piper 实测层）: CGN 指尖接触面 → Piper TCP 的偏移。
#   【2026-08-05 用户指令: 指尖对齐, piper 对齐 cgn, cgn 不动】
#   【同日解剖修正】官方 STL/控制点实测: Panda 掌体 0~58.4mm, 手指
#   58.4~112.2mm, 0.1034 是指垫捏取面而非指尖尖——指尖尖在 0.1122。
#   指尖对齐 = Piper 指尖尖(site 前 50mm, 网格实测) ↔ Panda 指尖尖 0.1122:
#   site 锚在接触面后 0.1034-0.0412=62.2mm 处(0.1122-0.050)。
#   （上一个中间值 0.050 对齐的是捏取面 0.1034; 再早 0.0 = site 锚捏取面）
TCP_DEPTH_PIPER = 0.0412
#
# Piper 最大开度（可行性过滤上限）：joint7 range [0, 0.035]/指（gripper.xml），
# 双指全开嘴内净距 70mm（内净距 = 2·q7：q7=0 两垫贴合，q7=0.035 内面 ±35mm）。
# 【2026-08-04 修正】旧值 45mm 系坐标系混淆误测：指垫盒局部半尺寸 (15,15,2.5)mm
# 经 link body 180° 旋转后 y↔z 互换，沿开合轴的半厚度是 2.5mm 而非 15mm
# （旧账 中心距74.9−2×15=44.9；正确 74.9−2×2.5≈70；mj_forward+mesh 顶点双重验证）。
PIPER_MAX_WIDTH = 0.070

GRIPPER_DEPTH = GRIPPER_DEPTH_CGN  # 兼容旧引用（勿新增使用）


# 🔒 仅人类可改（2026-08-05 用户裁决：API 为只调用面, 顾问无权批准更改）
def cgn_to_gripper(g_cgn, pose_mat, T_base_world=None):
    """CGN 相机系位姿 → 基座系 Piper grip_site 位姿。

    坐标链（2026-07-31 调试定型）：
      Step 1: OpenCV→OpenGL（Y 轴翻转；MuJoCo 相机约定）
      Step 2: cam→base（pose_mat 真刚体）
      Step 3: 手指轴对齐：CGN 局部 x = Panda 开合轴 → Piper grip_site y = 开合轴，
              绕局部 z 转 -90°（robosuite xyzw 序四元数）
      Step 4: palm → 指尖接触平面：沿局部 +z（逼近方向）平移
              GRIPPER_DEPTH_CGN - TCP_DEPTH_PIPER。
              【历史 bug：曾写 -0.1034，把 palm 往后撤了 0.1034，与正确值
              差 2×0.1034≈+0.207m —— 即观测到的 z 系统性 +20cm】
      Step 5: 逼近轴反转（2026-08-03）：site 局部 +z 指向掌心而非物体，
              绕开合轴 y 转 180° 使 IK 目标与真实手指方向一致。

    Args:
        g_cgn: (4,4) CGN 输出，OpenCV 相机系（原点 palm，+z 逼近）
        pose_mat: (4,4) 相机外参（cam→base）
        T_base_world: 未使用（兼容参数）

    Returns:
        (4,4) 基座系 grip_site 位姿（site = 指垫捏取面(0.1034)后 62.2mm 处——
        实体指尖尖对齐 CGN/Panda 指尖尖 0.1122, 2026-08-05 用户指令;
        详见 TCP_DEPTH_PIPER 常量注释）
    """
    # Step 1: OpenCV→OpenGL（Y轴翻转）。
    # 【2026-08-03 根因修复】基变换必须共轭作用: 左乘+右乘 T_flip。
    # 原写法只左乘 —— T_flip 的旋转部分 det=-1(镜像), 使每个候选姿态变成
    # 非正交反射; 下游 mat2quat 静默把反射映射成某个合法但错误的旋转
    # (实测往返误差 90°), solve_ik 收敛到的是被 mangled 的目标。
    # 污染范围: 全部经此函数的 IK 目标(4-yaw 扫描 52% 收敛率、门禁候选、
    # 幻影收敛 134°)。位置精度不受影响(T_flip 无平移, 右乘不改变平移),
    # 与历史 <2cm 位置验证兼容。库 FK 地图未经过此链, 数据干净。
    T_flip = np.diag([1, -1, 1, 1])
    g_gl = T_flip @ g_cgn @ T_flip  # T_flip 自逆; 共轭后 det=+1

    # Step 2: 相机系→基座系（pose_mat 已经是 cam→base）
    g_base = pose_mat @ g_gl

    # Step 3: 手指轴对齐（CGN-X 开合轴→Piper grip_site-Y 开合轴，绕 Z 转 -90°）
    R_z = T.quat2mat([0.0, 0.0, -0.707, 0.707])  # robosuite xyzw 序, 绕 Z 转 -90°
    T_align = np.eye(4)
    T_align[:3, :3] = R_z
    g_grip = g_base @ T_align

    # Step 4: palm → 指尖接触平面(= Piper grip_site)，沿局部 +z（逼近方向）
    T_offset = np.eye(4)
    T_offset[2, 3] = GRIPPER_DEPTH_CGN - TCP_DEPTH_PIPER
    g_final = g_grip @ T_offset

    # Step 5: 【2026-08-03 根因修复——逼近轴方向反转】Piper grip_site 的
    # 局部 +z 从指尖指向掌心（与 CGN/Panda 的 +z=逼近 相反, 用户手动拖拽
    # 实证: site z列朝上时手指朝下包住方块）。绕局部 y(开合轴)转 180°:
    # z→-z, x→-x, y 不变, det 保持 +1, 位置不变(绕自身原点)。
    # 历史上所有"IK 不收敛/下降墙/可达性边界"结论均由此反向目标产生, 作废。
    T_approach_flip = np.diag([-1.0, 1.0, -1.0, 1.0])
    g_final = g_final @ T_approach_flip

    return g_final


def filter_grasps_by_width(grasps, scores, openings, contact_dist=0.04,
                           max_width=PIPER_MAX_WIDTH, margin=0.004):
    """可行性过滤：丢弃【需求开度】超过 Piper 上限的候选。

    尺子（2026-07-31 裁决）：CGN 输出的 opening 是 Panda 几何（含 Panda 指垫
    厚度）下的指尖开度，不是物理需求。物理需求 = 两接触点沿闭合轴间距 + 余量：
        required_opening = contact_dist + margin   (margin 默认 2×2mm)
    丢弃条件: required_opening > max_width。

    Args:
        grasps: (N,4,4) 候选位姿
        scores: (N,) 得分
        openings: (N,) CGN 预测开度（仅记录用，不做丢弃依据）
        contact_dist: 接触间距（米）。sim 阶段允许用方块 GT 尺寸：
            cubeA 为 4×4×8cm 立块（footprint 4cm, axis-aligned；
            engine_capx.py:266, 2026-08-03 裁决 3 细高化），对面夹取接触间距 0.04m。
            【TODO(真机): 从候选接触几何/点云沿闭合轴投影估算，勿用 GT】
        max_width: Piper 最大开度（默认 PIPER_MAX_WIDTH=0.070）
        margin: 余量（默认 0.004 = 2×2mm）

    Returns:
        (grasps, scores, openings) 过滤后的三元组（保持输入顺序）
    """
    if len(grasps) == 0:
        return grasps, scores, openings
    required = contact_dist + margin
    keep = np.ones(len(grasps), dtype=bool) if required <= max_width else np.zeros(len(grasps), dtype=bool)
    n_drop = int((~keep).sum())
    if n_drop:
        print(f"[filter_grasps_by_width] dropped_by_width: {n_drop} 个候选 "
              f"(required_opening {required:.3f}m > {max_width}m)")
    return grasps[keep], scores[keep], openings[keep]


def mask_point_cloud_center(depth: np.ndarray, K: np.ndarray, seg: np.ndarray,
                            pose_mat: np.ndarray | None = None,
                            workspace_z: tuple[float, float] = (0.75, 0.85)) -> np.ndarray | None:
    """mask 点云 → 方块中心估计（基座系，若给 pose_mat；否则 OpenCV 相机系）。

    方法（顶面点法 + 工作区门控）：
      1. 工作区门控：只保留桌面高度附近的点（排除手臂/背景误检）
      2. 取 Z 最高的 20% 点 = 顶面（消除侧面遮挡偏置）
      3. 顶面 XY 质心 = 方块中心 XY（水平面无偏），Z 取顶面均值 - 半高
    """
    vs, us = np.nonzero(seg > 0)
    if len(vs) < 20:
        return None
    z = depth[vs, us].astype(np.float64)
    ok = np.isfinite(z) & (z > 0.05) & (z < 2.0)
    us, vs, z = us[ok], vs[ok], z[ok]
    if len(z) < 20:
        return None
    x = (us - K[0, 2]) * z / K[0, 0]
    y = (vs - K[1, 2]) * z / K[1, 1]  # OpenCV: y 向下
    pts = np.stack([x, y, z, np.ones_like(z)], axis=0)  # (4, N)

    if pose_mat is not None:
        pts_b = (pose_mat @ pts)[:3].T  # 基座系（cap-x pose_mat 标准）
        # 工作区门控：桌面高度 0.75~0.85m（排除误检）
        in_ws = (pts_b[:, 2] >= workspace_z[0]) & (pts_b[:, 2] <= workspace_z[1])
        if in_ws.sum() < 10:
            # 门控后点数不足，用原始数据
            pts_ws = pts_b
        else:
            pts_ws = pts_b[in_ws]
        # 取 Z 最高的 20% = 顶面
        z_thresh = np.percentile(pts_ws[:, 2], 80)
        top_mask = pts_ws[:, 2] >= z_thresh
        if top_mask.sum() < 5:
            # 点数不足时回落到中值区间法
            lo = np.percentile(pts_ws, 5, axis=0)
            hi = np.percentile(pts_ws, 95, axis=0)
            return np.array([(lo[0] + hi[0]) / 2, (lo[1] + hi[1]) / 2,
                             hi[2] - GRASP_CENTER_DROP])
        top_pts = pts_ws[top_mask]
        return np.array([top_pts[:, 0].mean(), top_pts[:, 1].mean(),
                         top_pts[:, 2].mean() - GRASP_CENTER_DROP])

    # 无 pose_mat：OpenCV 相机系
    pts_cv = pts[:3].T
    z_thresh = np.percentile(pts_cv[:, 2], 80)
    top_mask = pts_cv[:, 2] >= z_thresh
    if top_mask.sum() < 5:
        lo = np.percentile(pts_cv, 5, axis=0)
        hi = np.percentile(pts_cv, 95, axis=0)
        return (lo + hi) / 2
    top_pts = pts_cv[top_mask]
    return np.array([top_pts[:, 0].mean(), top_pts[:, 1].mean(), top_pts[:, 2].mean()])


# ---------------------------------------------------------------------------
# IK 种子库（scripts/build_piper_ik_library.py 离线生成）
# 背景: Piper 腕限位紧, 俯身抓桌面的可达流形极薄, 随机重启 DLS 打不中盆地;
# 但盆地内 ±0.8 rad 噪声 4/4 收敛 → 库检索近邻种子 + DLS 精修
# ---------------------------------------------------------------------------
_IK_LIB = None
_IK_LIB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "robots", "assets", "piper", "ik_library.npz")


def _ik_library():
    global _IK_LIB
    if _IK_LIB is None:
        _IK_LIB = np.load(_IK_LIB_PATH) if os.path.exists(_IK_LIB_PATH) else False
    return _IK_LIB


# ---------------------------------------------------------------------------
# 四元数工具（API 边界 wxyz；模块内部 xyzw —— 与 primitives.py 同约定）
# ---------------------------------------------------------------------------
def _q_in(q_wxyz) -> np.ndarray:
    q = np.asarray(q_wxyz, dtype=np.float64).flatten()
    return np.array([q[1], q[2], q[3], q[0]])


def _rot_err_vec(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    q = T.mat2quat(R_target @ R_current.T)
    if q[3] < 0:
        q = -q
    return T.quat2axisangle(q)


def _mask_box(mask: np.ndarray) -> list[float]:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return [0.0, 0.0, 0.0, 0.0]
    return [float(xs.min()), float(ys.min()), float(xs.max()), float(ys.max())]


def _collision_free_q(q: np.ndarray, engine) -> bool:
    """IK 解 / 路径点的物理可实现性检查: 臂几何不得接触桌面 / 方块 / 台座。

    背景 (实测): overhead 钩抓的某些 IK 分支把肘部 (link2/3) 压进桌面,
    位置伺服撞上后完全卡死 (TCP 偏差 >20cm)。指尖 - 方块接触允许。
    2026-08-04 起腕部(link6)碰撞几何已常态化幽灵化（robot.xml, 用户裁决:
    接触任务腕部贴近目标是几何必然）——腕部接触不再产生, 无需在此豁免。
    """
    model = engine.env.sim.model._model
    data2 = mujoco.MjData(model)
    data2.qpos[:] = engine.env.sim.data.qpos
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINT_NAMES]
    qposadr = [int(model.jnt_qposadr[j]) for j in jids]
    for i, adr in enumerate(qposadr):
        data2.qpos[adr] = q[i]
    mujoco.mj_step(model, data2)
    for c in range(data2.ncon):
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data2.contact.geom1[c]) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data2.contact.geom2[c]) or ""
        pair = {n1, n2}
        arm_hit = [n for n in pair if ("robot0_g" in n and "_col" in n) or "robot0_link" in n]
        if not arm_hit:
            continue
        other = pair - set(arm_hit)
        if any(("table" in o) or ("pedestal" in o) or ("floor" in o)
               or ("cube" in o) or ("riser" in o) for o in other):  # riser: 2026-08-03 裁决
            return False
    return True


def _collect_collision_events(engine, seen: set | None = None) -> list[dict]:
    """收集当前 sim 步的臂-环境接触事件。

    规则与 _collision_free_q 一致：臂几何 (robot0_link* / robot0_g*_col*)
    与 table/pedestal/floor/cube 的接触记为碰撞；指尖-方块接触排除。
    返回 [{sim_step, geom1, geom2, pos}], 并用 seen 集合去重 (geom1,geom2) 对。
    """
    model = engine.env.sim.model._model
    data = engine.env.sim.data
    events = []
    for c in range(data.ncon):
        n1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact.geom1[c]) or ""
        n2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, data.contact.geom2[c]) or ""
        pair = {n1, n2}
        arm_hit = [n for n in pair if ("robot0_g" in n and "_col" in n) or "robot0_link" in n]
        if not arm_hit:
            continue
        other = pair - set(arm_hit)
        if not any(("table" in o) or ("pedestal" in o) or ("floor" in o) or ("cube" in o) for o in other):
            continue
        key = (n1, n2) if n1 < n2 else (n2, n1)
        if seen is not None and key in seen:
            continue
        if seen is not None:
            seen.add(key)
        pos = data.contact.pos[c].copy()
        events.append({
            "sim_step": int(engine.sim_step),
            "geom1": n1,
            "geom2": n2,
            "pos": pos.tolist(),
        })
    return events


def _fk_tcp(q: np.ndarray, engine) -> np.ndarray:
    """给定臂关节角，返回 grip_site 世界坐标位置（用于 FK 到位自检）。"""
    model = engine.env.sim.model._model
    data = mujoco.MjData(model)
    data.qpos[:] = engine.env.sim.data.qpos
    jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINT_NAMES]
    qposadr = [int(model.jnt_qposadr[j]) for j in jids]
    for i, adr in enumerate(qposadr):
        data.qpos[adr] = q[i]
    mujoco.mj_forward(model, data)
    site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
    return data.site_xpos[site_id].copy()


class PiperControlApiReduced:
    """Piper 版 cap-x reduced 控制 API（契约函数面 15 个）。

    对应 cap-x FrankaControlApiReduced（control_reduced.py）在 Piper 上的等价物：
    cap-x reduced 10 函数 + select_top_down_grasp（论文 api-reference 列出）
    + 工具 4 函数（open_details 三份官方任务代码实锤调用，cap-x reduced 契约外）。
    完整函数面经 .functions() 导出；build_namespace 消费并注入任务代码命名空间。

    2026-08-05 E1 重构：原 PrimitiveContextCapx 更名（文末保留别名兼容旧脚本），
    函数面 11 → 契约 15（+ 契约外的 grasp_cgn 后端直连）。
    """

    # 后端开关（A1/B1 验收结论 2026-08-05；换后端不改任何函数签名/返回结构 —— D5）
    IK_BACKEND = "pyroki"    # pyroki venv 服务默认 + DLS/IK库/滚转扫描兜底链（内建 solve_ik）；
                             # cuRobo 决策关闭不集成（docs/api_asset_map.md B-c，2026-08-05 用户裁决）
    GRASP_BACKEND = "cgn"    # CGN 官方 TF docker 默认；空候选回落 _plan_grasp_geometric

    def __init__(self, engine):
        self.engine = engine
        self._last_collision_events: list[dict] = []

    @property
    def env(self):
        return self.engine.env

    @property
    def sim(self):
        return self.engine.env.sim

    @property
    def model(self):
        return self.sim.model._model

    # ------------------------------------------------------------------
    # 1. 观测
    # ------------------------------------------------------------------
    # 🔒 仅人类可改（2026-08-05 用户裁决：API 为只调用面, 顾问无权批准更改）
    def get_observation(self) -> dict[str, Any]:
        """获取环境观测（基座坐标系）。

        返回 dict，关键键：
          - ["robot0_robotview"]["images"]["rgb"]: (H,W,3) uint8 第三人称彩色图
          - ["robot0_robotview"]["images"]["depth"]: (H,W) float32 深度（米）
          - ["robot0_robotview"]["intrinsics"]: (3,3) 相机内参 K
          - ["robot0_robotview"]["pose_mat"]: (4,4) 相机外参（相机系→机械臂基座系）
          - ["robot0_robotview"]["pose"]: (7,) [xyz, quat_wxyz] 相机位姿（基座系）
          - ["robot_joint_pos"]: (7,) 6 臂关节角(rad) + 夹爪开合度(0闭1开)
          - ["robot_cartesian_pos"]: (8,) 夹爪 TCP 位姿 [xyz, quat_wxyz, 开合度]（基座系）
        """
        self.engine.refresh_obs()  # API 边界: 保证图像是当前状态而非旧帧
        obs = dict(self.engine.obs)
        img_key = f"{ROBOTVIEW}_image"
        depth_key = f"{ROBOTVIEW}_depth"

        rgb = obs[img_key]
        d = obs[depth_key]
        if d.ndim == 3:
            d = d.squeeze(-1)
        d = np.clip(np.nan_to_num(np.asarray(d, dtype=np.float64), nan=1.0), 0.0, 1.0)
        depth = CU.get_real_depth_map(self.sim, d)

        K = CU.get_camera_intrinsic_matrix(self.sim, ROBOTVIEW, RENDER_H, RENDER_W)
        E_muj = CU.get_camera_extrinsic_matrix(self.sim, ROBOTVIEW)  # cam→world, mujoco 约定
        E_api = self.engine.T_base_world @ E_muj  # cam(mujoco 约定)→基座系，真刚体

        # 相机位姿 (7,): [xyz, quat_wxyz]
        cam_pos = E_api[:3, 3]
        cam_q_xyzw = T.mat2quat(E_api[:3, :3])
        cam_pose = np.concatenate([cam_pos, [cam_q_xyzw[3], cam_q_xyzw[0], cam_q_xyzw[1], cam_q_xyzw[2]]])

        # TCP (grip_site) 基座系位姿
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
        tcp_world = np.eye(4)
        tcp_world[:3, :3] = self.sim.data.site_xmat[site_id].reshape(3, 3)
        tcp_world[:3, 3] = self.sim.data.site_xpos[site_id]
        tcp_base = self.engine.T_base_world @ tcp_world
        tcp_q_xyzw = T.mat2quat(tcp_base[:3, :3])
        tcp_quat_wxyz = np.array([tcp_q_xyzw[3], tcp_q_xyzw[0], tcp_q_xyzw[1], tcp_q_xyzw[2]])

        grip_frac = float(obs["robot0_gripper_qpos"][0]) / 0.035

        obs["robot0_robotview"] = {
            "images": {"rgb": rgb, "depth": depth},
            "intrinsics": K,
            "pose_mat": E_api,
            "pose": cam_pose,
        }
        obs["robot_joint_pos"] = np.concatenate(
            [np.asarray(obs["robot0_joint_pos"], dtype=np.float64)[:6], [grip_frac]]
        )
        obs["robot_cartesian_pos"] = np.concatenate(
            [tcp_base[:3, 3], tcp_quat_wxyz, [grip_frac]]
        )
        return obs

    # ------------------------------------------------------------------
    # 2. 视觉模型（SAM3 本地推理，cap-x 返回格式）
    # ------------------------------------------------------------------
    @staticmethod
    def _to_capx_results(raw: list[dict]) -> list[dict]:
        out = []
        for r in raw:
            mask = np.asarray(r["mask"]) > 0
            if not mask.any():
                continue
            out.append({
                "mask": mask,
                "box": _mask_box(mask),
                "score": float(r.get("score", 0.0)),
            })
        out.sort(key=lambda r: r["score"], reverse=True)
        return out

    # 🔒 仅人类可改（2026-08-05 用户裁决：API 为只调用面, 顾问无权批准更改）
    def segment_sam3_text_prompt(self, rgb: np.ndarray, text_prompt: str) -> list[dict]:
        """SAM3 文本提示分割（GPU 本地推理，开放词汇）。

        Args:
            rgb: (H,W,3) uint8 图像，通常传 obs["robot0_robotview"]["images"]["rgb"]
            text_prompt: 物体描述（英文），如 "red cube"

        Returns:
            list[dict]，按置信度降序: {"mask": (H,W) bool, "box": [x1,y1,x2,y2], "score": float}
            未匹配返回 []
        """
        return self._to_capx_results(_sam3_text_raw(np.asarray(rgb), text_prompt))

    # 🔒 仅人类可改（2026-08-05 用户裁决：API 为只调用面, 顾问无权批准更改）
    def segment_sam3_point_prompt(self, rgb: np.ndarray, point_coords) -> list[dict]:
        """SAM3 点提示分割（Tracker，前景点 → 该实例的候选 mask）。

        Args:
            rgb: (H,W,3) uint8
            point_coords: (x, y) 像素坐标（要点在目标物体上）

        Returns:
            同 segment_sam3_text_prompt 的 list 结构；未命中返回 []
        """
        return self._to_capx_results(_sam3_point_raw(np.asarray(rgb), point_coords))

    # 🔒 仅人类可改（2026-08-05 用户裁决：API 为只调用面, 顾问无权批准更改）
    def grasp_cgn(self, rgb, depth, K, seg, z_range=(0.2, 2.0)):
        """CGN 抓取检测（trace 包装版）。

        rgb 仅用于标注图投影底图，推理只看 depth/K/seg。
        Returns: (grasps (N,4,4) 相机系, scores (N,), openings (N,))
        """
        return _grasp_cgn_raw(depth, K, seg, z_range=z_range, return_openings=True)

    # 🔒 仅人类可改（2026-08-05 用户裁决：API 为只调用面, 顾问无权批准更改）
    def point_prompt_molmo(self, image: np.ndarray, text_prompt: str) -> dict:
        """文本→关键点定位（SAM3 级联实现，接口与 cap-x 的 Molmo 一致）。

        Returns:
            dict: {text_prompt: (x, y)} 像素坐标（int）；未定位到为 (None, None)
        """
        masks = self.segment_sam3_text_prompt(image, text_prompt)
        if not masks:
            return {text_prompt: (None, None)}
        ys, xs = np.nonzero(masks[0]["mask"])
        return {text_prompt: (int(xs.mean()), int(ys.mean()))}

    # ------------------------------------------------------------------
    # 3. 几何感知
    # ------------------------------------------------------------------
    # 🔒 仅人类可改（2026-08-05 用户裁决：API 为只调用面, 顾问无权批准更改）
    def get_oriented_bounding_box_from_3d_points(self, points: np.ndarray) -> dict[str, Any]:
        """对 3D 点云做主轴拟合（PCA），返回有向包围盒。

        Returns:
            {"center": (3,), "R": (3,3) 主轴旋转矩阵, "extent": (3,) 各轴全宽}
        """
        pts = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        pts = pts[np.isfinite(pts).all(axis=1)]
        # 简单离群剔除（替代 cap-x 的 open3d statistical outlier removal）
        if len(pts) > 20:
            c0 = np.median(pts, axis=0)
            dist = np.linalg.norm(pts - c0, axis=1)
            pts = pts[dist <= np.percentile(dist, 95)]
        center = pts.mean(axis=0)
        cov = np.cov((pts - center).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        R = eigvecs[:, order]
        if np.linalg.det(R) < 0:
            R[:, -1] *= -1
        proj = (pts - center) @ R
        extent = proj.max(axis=0) - proj.min(axis=0)
        return {"center": center, "R": R, "extent": extent}

    def _plan_grasp_geometric(self, depth: np.ndarray, intrinsics: np.ndarray,
                              segmentation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """几何抓取规划（Contact-GraspNet 的空候选兜底实现）。"""
        depth = np.asarray(depth, dtype=np.float64)
        if depth.ndim == 3:
            depth = depth[:, :, 0]
        seg = np.asarray(segmentation)
        if seg.ndim == 3:
            seg = seg[:, :, 0]
        K = np.asarray(intrinsics, dtype=np.float64)

        vs, us = np.nonzero(seg > 0)
        if len(vs) < 20:
            return np.zeros((0, 4, 4)), np.zeros(0)
        z = depth[vs, us]
        valid = np.isfinite(z) & (z > 0.05)
        us, vs, z = us[valid], vs[valid], z[valid]
        if len(z) < 20:
            return np.zeros((0, 4, 4)), np.zeros(0)
        # mujoco 相机系反投影（y 向上 → 负号，与 pose_mat 约定一致）
        x = (us - K[0, 2]) * z / K[0, 0]
        y = -(vs - K[1, 2]) * z / K[1, 1]
        pts = np.stack([x, y, z], axis=1)

        # 中值区间中心（而非 median）: mask 含朝向相机的侧面时, median 会沿视线
        # 偏移达半个物宽（实测 1.7cm, 40mm 方块 + 满开 70mm 张嘴足以让指垫压到方块顶面）
        lo = np.percentile(pts, 5, axis=0)
        hi = np.percentile(pts, 95, axis=0)
        centroid = (lo + hi) / 2
        # cam(mujoco 约定)→基座系（与 get_observation 的 E_api 同构）
        E_muj = CU.get_camera_extrinsic_matrix(self.sim, ROBOTVIEW)
        E_api = self.engine.T_base_world @ E_muj
        centroid_base = (E_api @ np.append(centroid, 1.0))[:3]

        # 姿态族（2026-08-03 裁决 2b）: 距竖直 20~26° 倾斜 × 8 方位。
        # 依据 ik_library 实证分布: tilt∈[18,30°] 共 10062 条, 8 个 45°
        # 方位扇区 1080~1347 条全覆盖（scripts/diagnose_fallback_attribution.py）。
        # 取代的 PCA 法线+重力吸附路线有三种实证失效（5-seed 门禁 15/15 全灭）:
        #   正竖直吸附踩"0° tilt 全高度不收敛"死区（seed 5/12/23）;
        #   up_cam 翻转 bug 致逼近轴朝天（seed 16, tilt=180°）;
        #   侧面点污染法线倾斜 38.5° 出舒适带且偏向远离臂侧（seed 18）。
        # 已知边界（如实声明）: 近臂落点（如 base x≈0.25）库中 6cm 内最小
        # tilt=86.7°, 无任何陡降构型——布局死区, 姿态族无法挽救。
        grasps = []
        for k in range(GRASP_N_YAW):
            az = 2 * np.pi * k / GRASP_N_YAW
            tilt = np.deg2rad(20.0 + 6.0 * k / (GRASP_N_YAW - 1))  # 20~26° 分层
            zaxis = np.array([np.sin(tilt) * np.cos(az),
                              np.sin(tilt) * np.sin(az), -np.cos(tilt)])
            # 开合轴 = 世界 x̂ 在 ⊥逼近轴平面上的投影（方块 axis-aligned,
            # 闭合跨 ±x 对面; tilt≤26° 时投影模长 ≥cos26°>0.89, 无退化）
            yaxis = np.array([1.0, 0.0, 0.0]) - zaxis[0] * zaxis
            yaxis /= np.linalg.norm(yaxis)
            xaxis = np.cross(yaxis, zaxis)
            Tg = np.eye(4)
            # site +z 背向逼近（同 cgn_to_gripper Step 5, 2026-08-03）
            Tg[:3, :3] = np.column_stack([xaxis, yaxis, zaxis]) @ np.diag([-1.0, 1.0, -1.0])
            # 目标点: 质心沿逼近轴下探（zaxis 指向物体 → 正号进入物体）
            Tg[:3, 3] = centroid_base + zaxis * GRASP_CENTER_DROP
            grasps.append(Tg)

        # 排序: 库支持度（6cm & 0.25rad 邻域条目数）降序——纯排序不拦截,
        # 无库时退化自然序。score 即支持度, 日志可读。
        lib = _ik_library()
        if lib is False:
            scores = [1.0 - 0.05 * k for k in range(GRASP_N_YAW)]
        else:
            scores = []
            for Tg in grasps:
                dp = np.linalg.norm(lib['P'] - Tg[:3, 3], axis=1)
                da = np.arccos(np.clip(lib['Z'] @ Tg[:3, 2], -1.0, 1.0))
                scores.append(float(((dp < 0.06) & (da < 0.25)).sum()))
        order = sorted(range(GRASP_N_YAW), key=lambda k: -scores[k])  # 稳定序
        return np.stack(grasps)[order], np.asarray(scores)[order]

    # 🔒 仅人类可改（2026-08-05 用户裁决：API 为只调用面, 顾问无权批准更改）
    def plan_grasp(self, depth: np.ndarray, intrinsics: np.ndarray,
                   segmentation: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """抓取规划（优先 CGN，失败回退几何规划器）。

        Args:
            depth: (H,W) 深度图（米）
            intrinsics: (3,3) 相机内参 K
            segmentation: (H,W) 实例分割图，整数 >0 为目标实例

        Returns:
            grasp_poses: (K,4,4) 候选抓取 TCP 位姿，**基座系**
            grasp_scores: (K,) 候选得分，降序排列
        """
        # 1) 优先 CGN
        try:
            from ..perception.vision_client import grasp_cgn

            grasps_cam, scores = grasp_cgn(depth, intrinsics, segmentation)
            if len(grasps_cam) > 0:
                obs = self.get_observation()
                pose_mat = obs["robot0_robotview"]["pose_mat"]
                grasps_base = np.array([
                    cgn_to_gripper(g, pose_mat)
                    for g in grasps_cam
                ])
                # 2026-08-06 用户指令①: 候选 3D glyph 进可视化窗口
                # （仅 --render 且窗口开着时生效, 纯新增不影响契约行为）
                if getattr(self.engine, "_live_viewer", None) is not None:
                    self.engine.draw_grasp_glyphs(grasps_base, scores)
                return grasps_base, scores
        except Exception as e:
            print(f"CGN failed: {e}, falling back to geometric")

        # 2) 几何兜底
        grasps_g, scores_g = self._plan_grasp_geometric(depth, intrinsics, segmentation)
        if getattr(self.engine, "_live_viewer", None) is not None:
            self.engine.draw_grasp_glyphs(grasps_g, scores_g)
        return grasps_g, scores_g

    # ------------------------------------------------------------------
    # 4. 运动（IK + 阻塞关节运动）
    # ------------------------------------------------------------------
    # 🔒 仅人类可改（2026-08-05 用户裁决：API 为只调用面, 顾问无权批准更改）
    def solve_ik(self, position: np.ndarray, quaternion_wxyz: np.ndarray, use_pyroki: bool = True,
                 free_approach_roll: bool = False, seed: np.ndarray | None = None,
                 loose: bool = False) -> np.ndarray:
        """数值 IK（pyroki 优先 + DLS 兜底），目标为夹爪 TCP 位姿（基座系）。

        Args:
            position: (3,) TCP 目标位置（基座系，米）
            quaternion_wxyz: (4,) TCP 目标姿态 [w,x,y,z]（基座系）
            use_pyroki: 是否优先调用 pyroki IK server（默认 True）
            free_approach_roll: 绕逼近轴滚转自由（默认 False）。True 时只收敛
                位置+逼近轴方向——方形截面目标（cubeA）合法，且 2026-08-04 实测
                精确 6 自由度目标落在腕限位薄流形外（pyroki 最优 2.4-7mm、
                DLS 撞 j4 限位卡死），不放宽则 12/12 不收敛。
            seed: 可选 (6,) 种子构型。提供时先就地 DLS——分支连续
                （2026-08-04: 预抓/下降两次独立 IK 会跳到 Δq>4rad 的远分支,
                关节空间直扫画大弧撞块卡死; 笛卡尔细分+种子链锁定分支）。
            loose: 中途点放宽容差（默认 False）。True 时 DLS 收敛判据
                0.2mm/0.17° → 2mm/1.15°。【2026-08-05 q_pre 平台期事故:
                边际候选(近垂直但腕区不佳) DLS 平台 ~1.5mm/0.57°(pyroki 与
                多族种子同平台), 严格容差全拒 → q_pre 12/12 未收敛; 中途点
                (预抓/下降中间步)位姿无关最终精度, 放宽; 末步/滚转抛光仍严格】

        Returns:
            (6,) 臂关节角 (rad)。不收敛抛 RuntimeError（可用 try/except 做可达性检查）
        """
        # 基座系 → 世界系
        T_goal_base = np.eye(4)
        T_goal_base[:3, :3] = T.quat2mat(_q_in(quaternion_wxyz))
        T_goal_base[:3, 3] = np.asarray(position, dtype=np.float64).reshape(3)
        target_p_base = T_goal_base[:3, 3].copy()
        T_goal = self.engine.T_world_base @ T_goal_base
        target_p = T_goal[:3, 3]
        target_q = T.mat2quat(T_goal[:3, :3])  # xyzw

        # 预计算 DLS 用的模型数据与限位（pyroki 回退也需要）
        model = self.model
        jids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in ARM_JOINT_NAMES]
        q_lo = np.array([model.jnt_range[j][0] for j in jids])
        q_hi = np.array([model.jnt_range[j][1] for j in jids])

        # DLS 机制（2026-08-04 上移到 pyroki 之前——pyroki 粗解要用它抛光）
        data = mujoco.MjData(model)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
        dofadr = [int(model.jnt_dofadr[j]) for j in jids]
        qposadr = [int(model.jnt_qposadr[j]) for j in jids]
        q_cur = self.engine.current_arm_qpos()

        def fk(q_in):
            for i, adr in enumerate(qposadr):
                data.qpos[adr] = q_in[i]
            mujoco.mj_forward(model, data)
            return (data.site_xpos[site_id].copy(),
                    data.site_xmat[site_id].reshape(3, 3).copy())

        R_goal_holder = {'R': T.quat2mat(target_q)}  # 可写 holder: 滚转扫描兜底要换目标滚转

        # 可视化心跳（2026-08-06 用户反馈: 规划期窗口冻结像瞬移）: DLS 重计算
        # 循环中每 ~0.15s sync 一次 live viewer, 画面保持活动; 无窗口零开销。
        import time as _t_pulse_mod
        _pulse = {'t': _t_pulse_mod.perf_counter()}

        def _viewer_pulse():
            v = getattr(self.engine, '_live_viewer', None)
            if v is None:
                return
            now = _t_pulse_mod.perf_counter()
            if now - _pulse['t'] > 0.15:
                _pulse['t'] = now
                if v.is_running():
                    v.sync()

        def dls(q0, max_iter=600, tol_p=None, tol_r=None):
            """限位投影 DLS。6 轴无冗余, 收敛域强烈依赖种子 → 多种子策略。"""
            if tol_p is None:
                tol_p = 2e-3 if loose else 2e-4
            if tol_r is None:
                tol_r = 2e-2 if loose else 3e-3
            q = np.clip(q0.copy(), q_lo, q_hi)
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            lam = 0.05
            R_goal = R_goal_holder['R']
            for _ in range(max_iter):
                _viewer_pulse()
                cur_p, cur_R = fk(q)
                err_p = target_p - cur_p
                err_r = _rot_err_vec(R_goal, cur_R)
                if free_approach_roll:
                    # 滚转自由: 投影掉绕逼近轴的误差分量, 只收敛 位置+逼近轴
                    _a = R_goal[:, 2]
                    err_r = err_r - _a * float(err_r @ _a)
                n_p, n_r = np.linalg.norm(err_p), np.linalg.norm(err_r)
                if n_p < tol_p and n_r < tol_r:
                    return q, True
                if n_p > 0.05:
                    err_p *= 0.05 / n_p
                if n_r > 0.30:
                    err_r *= 0.30 / n_r
                mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
                J = np.vstack([jacp[:, dofadr], jacr[:, dofadr]])   # (6, 6)
                err = np.concatenate([err_p, err_r])
                dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), err)
                q = np.clip(q + np.clip(dq, -0.3, 0.3), q_lo, q_hi)
            return q, False

        def dls_clean(q0):
            # 2026-08-05 用户裁决: IK 筛选只看可达性不看碰撞（碰撞规避归路径
            # 规划, 含腕部实体）——本函数不再有 _collision_free_q 门
            q, ok = dls(q0)
            if ok:
                return q, True
            return q, False

        def dls_batch(Q0, R_goals=None, max_iter=600, tol_p=None, tol_r=None):
            """批量限位投影 DLS（2026-08-06 方案 A, 用户裁决）: N 个种子在同一
            迭代环并行, 任一收敛即整体返回（first-ok 早退, 索引序=优先级）。
            数学与 dls() 逐项一致（λ=0.05, 误差钳位 0.05/0.30, dq 钳位 ±0.3,
            限位投影, free_roll 时投影掉绕逼近轴分量）——替代原串行种子瀑布,
            覆盖与速度同步提升。R_goals: 可选 (N,3,3) 逐种子目标矩阵（滚转
            扫描兜底用, 缺省全部取 R_goal_holder）。"""
            if tol_p is None:
                tol_p = 2e-3 if loose else 2e-4
            if tol_r is None:
                tol_r = 2e-2 if loose else 3e-3
            q = np.clip(np.asarray(Q0, dtype=np.float64).reshape(-1, 6).copy(), q_lo, q_hi)
            N = len(q)
            if N == 0:
                return q, False
            if R_goals is None:
                R_goals = np.broadcast_to(R_goal_holder['R'], (N, 3, 3))
            axes = R_goals[:, :, 2] if free_approach_roll else None  # (N,3) 逼近轴
            lam = 0.05
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            best_q, best_tot = q[0].copy(), np.inf
            for _ in range(max_iter):
                _viewer_pulse()
                errs = np.zeros((N, 6))
                Js = np.zeros((N, 6, 6))
                for i in range(N):
                    cur_p, cur_R = fk(q[i])
                    ep = target_p - cur_p
                    er = _rot_err_vec(R_goals[i], cur_R)
                    if free_approach_roll:
                        er = er - axes[i] * float(er @ axes[i])
                    errs[i] = np.concatenate([ep, er])
                    mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
                    Js[i] = np.vstack([jacp[:, dofadr], jacr[:, dofadr]])
                n_p = np.linalg.norm(errs[:, :3], axis=1)
                n_r = np.linalg.norm(errs[:, 3:], axis=1)
                conv = (n_p < tol_p) & (n_r < tol_r)
                if conv.any():
                    return q[int(np.argmax(conv))], True
                tot = n_p / tol_p + n_r / tol_r
                j_best = int(np.argmin(tot))
                if tot[j_best] < best_tot:
                    best_tot, best_q = tot[j_best], q[j_best].copy()
                # 误差钳位（与 dls 一致: pos 0.05 / rot 0.30）
                sc_p = np.where(n_p > 0.05, 0.05 / np.maximum(n_p, 1e-12), 1.0)
                sc_r = np.where(n_r > 0.30, 0.30 / np.maximum(n_r, 1e-12), 1.0)
                E = errs * np.stack([sc_p, sc_p, sc_p, sc_r, sc_r, sc_r], axis=1)
                A = Js @ Js.transpose(0, 2, 1) + lam**2 * np.eye(6)
                dq = np.einsum('nij,nj->ni', Js.transpose(0, 2, 1),
                               np.linalg.solve(A, E))
                q = np.clip(q + np.clip(dq, -0.3, 0.3), q_lo, q_hi)
            return best_q, False

        # 种子锁定通道（2026-08-04）: 给种子则先就地 DLS——分支连续
        if seed is not None:
            q_s, ok_s = dls_clean(np.asarray(seed, dtype=np.float64).reshape(6))
            if ok_s:
                return q_s

        # ------------------------------------------------------------------
        # 1) 优先使用 pyroki IK server（MJCF 真源 URDF），但用 MuJoCo FK 做后验验证
        # ------------------------------------------------------------------
        if use_pyroki:
            try:
                from ..planning.pyroki_client import ik_pyroki
                q_pk = ik_pyroki(target_p_base, _q_in(quaternion_wxyz), prev_cfg=self.engine.current_arm_qpos())
                q_pk = np.asarray(q_pk, dtype=np.float64).reshape(-1)
                if q_pk.size >= 6:
                    q_arm = np.clip(q_pk[:6], q_lo, q_hi)
                    # FK 验证：pyroki 优化器偶尔给出误差 ~15mm/2° 的“成功”解，
                    # 这种解在 move_to_joints 自检中会被拒绝；这里提前用 MuJoCo FK 筛掉。
                    # 2026-08-05 用户裁决: IK 筛选只看可达性, 不查碰撞——
                    # 原 _collision_free_q 门已移除(碰撞规避归路径规划, 含腕部)
                    if True:
                        data_v = mujoco.MjData(model)
                        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
                        qposadr_v = [int(model.jnt_qposadr[j]) for j in jids]
                        for i, adr in enumerate(qposadr_v):
                            data_v.qpos[adr] = q_arm[i]
                        mujoco.mj_forward(model, data_v)
                        fk_p = data_v.site_xpos[site_id]
                        fk_p_base = (self.engine.T_base_world @ np.append(fk_p, 1.0))[:3]
                        fk_R = data_v.site_xmat[site_id].reshape(3, 3)
                        base_R = self.sim.data.xmat[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_base_link")].reshape(3, 3)
                        rel_q_xyzw = T.mat2quat(base_R.T @ fk_R)
                        fk_q_wxyz = np.array([rel_q_xyzw[3], rel_q_xyzw[0], rel_q_xyzw[1], rel_q_xyzw[2]])
                        dpos = float(np.linalg.norm(fk_p_base - target_p_base))
                        qdot = np.clip(np.abs(np.dot(fk_q_wxyz, quaternion_wxyz)), -1.0, 1.0)
                        dquat = float(np.arccos(qdot))
                        if free_approach_roll:
                            # 姿态误差只取逼近轴失准分量（滚转分量不计）
                            _Rg = T_goal_base[:3, :3]
                            _a = _Rg[:, 2]
                            _erv = _rot_err_vec(_Rg, base_R.T @ fk_R)
                            dquat = float(np.linalg.norm(_erv - _a * float(_erv @ _a)))
                        if dpos < 1e-3 and dquat < 0.01:
                            return q_arm
                        # 2026-08-04 根因修复: pyroki 优化解精度为 mm/亚度级
                        # (实测残余 2.4mm/0.49°), 严格门(1mm/0.57°)全拒 → 落到
                        # 远种子 DLS 兜底收敛率极低 → 12/12 未收敛事故。
                        # 粗门(20mm/5.7°)放行进 DLS 抛光: 种子已在解邻域几乎
                        # 必收敛到 0.2mm; phantom 解(>20mm/>5.7°)仍被挡在门外。
                        if dpos < 0.02 and dquat < 0.1:
                            q_p, ok_p = dls_clean(q_arm)
                            if ok_p:
                                return q_p
            except Exception:
                # pyroki 失败时静默回退到 DLS
                pass

        # 种子 1: IK 库检索（离线稠密 FK, 按 位置+开口+开合轴 匹配近邻）
        lib = _ik_library()
        lib_seeds = []
        if lib is not False:
            R_goal_base = T_goal_base[:3, :3]
            z_t = R_goal_base[:, 2]
            y_t = R_goal_base[:, 1]
            dp = np.linalg.norm(lib["P"] - target_p_base, axis=1)
            da = np.arccos(np.clip(lib["Z"] @ z_t, -1.0, 1.0))
            dy = np.arccos(np.clip(np.abs(lib["Y"] @ y_t), -1.0, 1.0))
            score = dp / 0.05 + da / 0.6 + 0.3 * dy / 0.6
            lib_seeds = [np.asarray(lib["Q"][i], dtype=np.float64)
                         for i in np.argsort(score, kind="stable")[:8]]  # 批量后加得起: 5→8

        # 种子 2: 规范构型族 (home / 腕部极值家族)
        q_home = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0])
        fam_seeds = [q_cur, q_home,
                     q_home + np.array([0, 0, 0, 0, 0, np.pi]),
                     q_home + np.array([0, 0, 0, 0, 0, -np.pi])]
        for j1 in (0.0, 0.5, -0.5):
            for j3 in (-0.8, -1.35):
                for j6 in (0.0, np.pi, -np.pi):
                    fam_seeds.append(np.array([j1, 1.57, j3, 0.0, -1.2, j6]))
        for j4 in (1.8, -1.8):
            fam_seeds.append(np.array([0.0, 1.57, -1.0, j4, -1.2, np.pi]))
            fam_seeds.append(np.array([0.0, 1.57, -1.0, j4, 1.2, 0.0]))

        # 批量 DLS（2026-08-06 方案 A, 用户裁决）: 库 top-8 + 规范族一次并行,
        # 任一收敛即返回——替代原串行瀑布（库 5 逐个 + 族 24 逐个）
        q, ok = dls_batch(lib_seeds + fam_seeds)
        if ok:
            return q

        # 滚转扫描兜底（语义同 2026-08-05, 批量版）: 目标绕逼近轴滚 φ 后
        # 位置+轴向不变, 是等价目标——近垂直抓可达性强烈依赖滚转角（腕限位把
        # 可达滚转切成带状, 固定种子滚转常落不可达带 → DLS 平台期轴向误差
        # 17~65°, 解剖实测）。5 变体 × 规范族逐种子目标矩阵一次并行
        # （原为嵌套串行, 只在前面全败后才付 ~5×族 代价; 批量后同一环跑完）。
        if free_approach_roll:
            R_goal0 = R_goal_holder['R']
            Q_roll, R_roll = [], []
            for phi in (np.pi / 4, -np.pi / 4, np.pi / 2, -np.pi / 2, np.pi):
                c, sp = np.cos(phi), np.sin(phi)
                R_phi = R_goal0 @ np.array([[c, -sp, 0], [sp, c, 0], [0, 0, 1]])
                for s in fam_seeds:
                    Q_roll.append(s)
                    R_roll.append(R_phi)
            q, ok = dls_batch(Q_roll, R_goals=np.asarray(R_roll))
            if ok:
                return q
        raise RuntimeError(f"solve_ik 未收敛: pos={position} quat={quaternion_wxyz}")

    # 🔒 仅人类可改（2026-08-05 用户裁决：API 为只调用面, 顾问无权批准更改）
    def move_to_joints(self, joints: np.ndarray, tol: float = None,
                       fk_tol: float = None) -> None:
        """关节空间运动（阻塞）。

        Args:
            joints: (6,) 目标关节角 (rad)。命令后等待 ‖q-target‖₂ < tol，
                最多约 5 秒；未到位抛 RuntimeError（不再静默返回）。
            tol: 到位阈值（默认 JOINT_TOL=0.02）。接触段（下降/抬升）放宽到
                ~0.10——位置伺服顶着接触残余沉降不到严格阈值（2026-08-03
                实测: 下降残余 0.068, 负载抬升残余 0.029, 物理正常）。
            fk_tol: FK 到位自检阈值（默认 0.01m）。接触段放宽——接触使 TCP
                合法偏离自由空间目标位形。
        """
        if tol is None:
            tol = JOINT_TOL
        if fk_tol is None:
            fk_tol = 0.01
        target = np.asarray(joints, dtype=np.float64).reshape(6)
        best_err, stall = np.inf, 0
        self._last_collision_events = []
        seen_pairs = set()
        for _ in range(MAX_TICKS_MOVE):
            self.engine.step_joints(target)
            self._last_collision_events.extend(_collect_collision_events(self.engine, seen_pairs))
            err = np.linalg.norm(self.engine.current_arm_qpos() - target)
            if err < tol:
                break
            # 卡死检测: 30 拍无进展则提前抛错 (如被遮挡物挡住, 不再傻等 5s)
            if err < best_err - 1e-3:
                best_err, stall = err, 0
            else:
                stall += 1
                if stall >= 30:
                    raise RuntimeError(f"move_to_joints 未到位: err={err:.4f}rad target={target}")
        else:
            raise RuntimeError(f"move_to_joints 未到位: err={err:.4f}rad target={target}")

        # FK 到位自检: 目标构型 FK 与当前 sim TCP 偏差 >fk_tol 说明执行器/物理异常
        fk_pos = _fk_tcp(target, self.engine)
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
        cur_pos = self.sim.data.site_xpos[site_id]
        fk_base = (self.engine.T_base_world @ np.append(fk_pos, 1.0))[:3]
        cur_base = (self.engine.T_base_world @ np.append(cur_pos, 1.0))[:3]
        if np.linalg.norm(fk_base - cur_base) > fk_tol:
            raise RuntimeError(
                f"move_to_joints FK 自检失败: 偏差 {np.linalg.norm(fk_base - cur_base) * 1000:.1f}mm"
            )

    # ------------------------------------------------------------------
    # 路径安全包装 (A0)
    # ------------------------------------------------------------------
    MAX_STEP_RAD = 0.15

    def move_to_joints_safely(self, target: np.ndarray) -> None:
        """分段 waypoint 执行 move_to_joints，每段末检查碰撞，避免长距离关节跳转扫碰撞。"""
        q_cur = self.engine.current_arm_qpos()
        dq = np.asarray(target, dtype=np.float64).reshape(6) - q_cur
        n = max(1, int(np.ceil(np.abs(dq).max() / self.MAX_STEP_RAD)))
        for i in range(1, n + 1):
            q_wp = q_cur + dq * i / n
            if not _collision_free_q(q_wp, self.engine):
                raise RuntimeError(f"路径第{i}/{n}段末构型碰撞")
            self.move_to_joints(q_wp)

    # 🔒 仅人类可改（2026-08-05 用户裁决：API 为只调用面, 顾问无权批准更改）
    def open_gripper(self) -> None:
        """张开夹爪（阻塞至到位）。"""
        self.engine.set_gripper(1.0)
        self.engine.hold_ticks(GRIPPER_SETTLE_TICKS)

    # 🔒 仅人类可改（2026-08-05 用户裁决：API 为只调用面, 顾问无权批准更改）
    def close_gripper(self) -> None:
        """闭合夹爪（阻塞）。是否夹住需通过 robot_joint_pos[-1] 观测判断。"""
        self.engine.set_gripper(0.0)
        self.engine.hold_ticks(GRIPPER_SETTLE_TICKS)

    # ------------------------------------------------------------------
    # 5. 抓取启发式（论文 api-reference 补充函数，cap-x skill library vendor）
    # ------------------------------------------------------------------
    # 🔒 仅人类可改（2026-08-05 行为验证：test K1-K3）
    @staticmethod
    def select_top_down_grasp(grasps: np.ndarray, scores: np.ndarray,
                              cam_to_world: np.ndarray,
                              vertical_threshold: float = 0.8) -> tuple:
        """从抓取候选中选出最优 top-down（竖直）抓取。

        【L1 vendor】函数体抄 cap-x franka/control_reduced_skill_library.py:319
        （~30 行原样，MIT）。
        适配 delta（配方 #7）："world" 对 Piper 即**基座系**（cam_to_world 传
        obs["robot0_robotview"]["pose_mat"]，基座系 z 轴竖直向上）；
        vertical_threshold 暴露给任务代码——Piper 腕限位紧、严格 top-down
        常不可达，调低（如 0.5）退化为"尽量竖直"筛选器，与 overhead 姿态族互补。

        【约定注意】输入候选为 **CGN 约定**：局部 +z = 逼近方向（指向物体），
        即 grasp_cgn() 的原始相机系输出。plan_grasp() 返回的是 grip_site
        约定（局部 +z 背向物体，2026-08-03 逼近轴反转）且已在基座系——
        不要直接喂给本函数。

        典型链：grasps_cam, scores, _ = grasp_cgn(rgb, depth, K, seg)
                best_cam, s = select_top_down_grasp(grasps_cam, scores, pose_mat)
                target = cgn_to_gripper(best_cam, pose_mat)  # → 基座系 site 位姿

        Args:
            grasps: (N,4,4) 候选抓取位姿（相机系，CGN 约定 +z=逼近）
            scores: (N,) 候选得分
            cam_to_world: (4,4) 相机外参（cam→基座系）
            vertical_threshold: 逼近轴与竖直向下点积阈值（1.0=完全竖直）

        Returns:
            (best_grasp_base (4,4) 基座系, best_score) 或 (None, -inf)
        """
        best_grasp = None
        best_score = -np.float64("inf")
        world_z = np.array([0, 0, 1])
        for i, g_camera in enumerate(grasps):
            g_world = cam_to_world @ g_camera
            R = g_world[:3, :3]
            gripper_approach = R[:, 2]
            alignment = -np.dot(gripper_approach, world_z)
            if alignment > vertical_threshold:
                if scores[i] > best_score:
                    best_score = scores[i]
                    best_grasp = g_world
        return best_grasp, best_score

    # ------------------------------------------------------------------
    # 6. 工具函数（open_details 三份官方任务代码实锤调用，cap-x reduced 契约外）
    # ------------------------------------------------------------------
    # 🔒 仅人类可改（2026-08-05 行为验证：test J1a/J1b 恒等+GT）
    @staticmethod
    def mask_to_world_points(mask: np.ndarray, depth: np.ndarray,
                             K: np.ndarray, pose_mat: np.ndarray) -> np.ndarray:
        """mask 像素反投影 → 基座系 3D 点云。

        【L1】反投影数学抄 cap-x utils/depth_utils.py:108 depth_to_pointcloud，
        适配 delta（配方 #11）：① y 负号约定（mujoco 相机系 y 向上，与真刚体
        pose_mat 配套 —— 头部偏差声明 #1）；② 左乘 pose_mat 统一输出**基座系**
        点云。与 Panda 线 primitives.py:191 同构（已行为验证）。

        Args:
            mask: (H,W) uint8/bool，>0 的像素被反投影
            depth: (H,W) 深度图（米）
            K: (3,3) 相机内参
            pose_mat: (4,4) 相机外参（cam→基座系）

        Returns:
            (N,3) float64 基座系点云（无效/非有限深度已剔除）；空 mask 返回 (0,3)
        """
        vs, us = np.nonzero(mask)
        if len(vs) == 0:
            return np.zeros((0, 3))
        z = depth[vs, us].astype(np.float64)
        valid = np.isfinite(z) & (z > 0.01)
        us, vs, z = us[valid], vs[valid], z[valid]
        if len(z) == 0:
            return np.zeros((0, 3))
        x = (us - K[0, 2]) * z / K[0, 0]
        y = -(vs - K[1, 2]) * z / K[1, 1]  # mujoco 相机 y 向上 → 负号（偏差声明 #1）
        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)
        return (np.asarray(pose_mat) @ pts_cam)[:3].T

    # 🔒 仅人类可改（2026-08-05 行为验证：test J2）
    @staticmethod
    def pixel_to_world_point(u: float, v: float, z: float,
                             K: np.ndarray, E: np.ndarray) -> np.ndarray:
        """单像素 + 该点深度 → 基座系 3D 点（#11 的单点特化，Molmo 打点落 3D 用）。

        【L1】配方 #12：同 mask_to_world_points 的反投影约定（y 负号 + 左乘 E）。

        Args:
            u, v: 像素坐标；z: 该点深度（米）；K: (3,3) 内参；E: (4,4) cam→基座系

        Returns:
            (3,) float64 基座系 3D 点
        """
        x = (u - K[0, 2]) * z / K[0, 0]
        y = -(v - K[1, 2]) * z / K[1, 1]  # mujoco 相机 y 向上，见 mask_to_world_points
        p = np.asarray(E) @ np.array([x, y, z, 1.0])
        return p[:3]

    # 🔒 仅人类可改（2026-08-05 行为验证：test J3）
    @staticmethod
    def rotation_matrix_to_quaternion(R: np.ndarray) -> np.ndarray:
        """(3,3) 旋转矩阵 → wxyz 四元数（API 边界约定）。

        【自写 5 行，配方 #13】内部 robosuite T.mat2quat（xyzw）再重排；
        与 Panda 线 primitives.py:346/_q_out 同构。
        """
        q_xyzw = T.mat2quat(np.asarray(R, dtype=np.float64))
        return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])

    # 🔒 仅人类可改（2026-08-05 行为验证：test J4）
    @staticmethod
    def interpolate_segment(p1: np.ndarray, p2: np.ndarray,
                            step: float = 0.02) -> list:
        """两点间按 step 密化路径点（wipe 类任务蛇形路径用）。

        【自写，配方 #14】与 Panda 线 primitives.py:322 同构；
        linspace 保证精确含两端点。

        Args:
            p1, p2: (3,) 起止点；step: 相邻路径点间距上限（米）

        Returns:
            list[(3,)] 路径点（含 p1、p2；间距 = dist/ceil(dist/step) ≤ step）
        """
        p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
        n = max(2, int(np.ceil(np.linalg.norm(p2 - p1) / step)) + 1)
        return [p1 + (p2 - p1) * t for t in np.linspace(0, 1, n)]

    # ------------------------------------------------------------------
    # 契约函数面导出（build_namespace 消费）
    # ------------------------------------------------------------------
    def functions(self) -> dict:
        """cap-x 契约函数集（15 个；键名与 cap-x / ASPIRE api-reference 一致）。

        = cap-x reduced 10 函数 + select_top_down_grasp（论文 api-reference）
        + 工具 4（mask_to_world_points / pixel_to_world_point /
        rotation_matrix_to_quaternion / interpolate_segment）。
        契约外的 grasp_cgn（CGN 相机系候选直连）不在此列，由 build_namespace
        单独挂入命名空间。
        """
        return {
            "get_observation": self.get_observation,
            "segment_sam3_text_prompt": self.segment_sam3_text_prompt,
            "segment_sam3_point_prompt": self.segment_sam3_point_prompt,
            "point_prompt_molmo": self.point_prompt_molmo,
            "plan_grasp": self.plan_grasp,
            "get_oriented_bounding_box_from_3d_points": self.get_oriented_bounding_box_from_3d_points,
            "solve_ik": self.solve_ik,
            "move_to_joints": self.move_to_joints,
            "open_gripper": self.open_gripper,
            "close_gripper": self.close_gripper,
            "select_top_down_grasp": self.select_top_down_grasp,
            "mask_to_world_points": self.mask_to_world_points,
            "pixel_to_world_point": self.pixel_to_world_point,
            "rotation_matrix_to_quaternion": self.rotation_matrix_to_quaternion,
            "interpolate_segment": self.interpolate_segment,
        }


# 向后兼容别名：旧脚本 import PrimitiveContextCapx 不变（2026-08-05 E1 重构）
PrimitiveContextCapx = PiperControlApiReduced


# ---------------------------------------------------------------------------
# trace 包装 + 命名空间构建
# ---------------------------------------------------------------------------
_OBS_KEY_STATES = ["robot_joint_pos", "robot_cartesian_pos", "robot0_joint_pos",
                   "robot0_gripper_qpos"]


def _obs_key_states(obs: dict) -> dict:
    return {k: obs[k] for k in _OBS_KEY_STATES if k in obs}


def _obs_outputs_with_links(out: Any, sim_step: int, tracer) -> Any:
    """get_observation 返回的图像/深度替换为帧流链接（JSON 只留指针）。"""
    if not isinstance(out, dict) or "robot0_robotview" not in out:
        return out
    links = tracer.frame_links(sim_step)
    patched = dict(out)
    cam = dict(patched["robot0_robotview"])
    cam["images"] = {
        "rgb": {"ref": links.get("top")},
        "depth": {"ref": links.get("depth")},
    }
    patched["robot0_robotview"] = cam
    return patched


def build_namespace(engine) -> dict:
    """构建注入任务代码的全局命名空间（api.functions() 契约 15 函数 + trace 包装）。

    命名空间 = 契约 15 函数 + grasp_cgn（契约外的 CGN 相机系候选直连，demo/调试用）。
    """
    api = PiperControlApiReduced(engine)
    tracer = engine.tracer
    ns: dict[str, Any] = {"np": np}

    def wrap(name, fn):
        def wrapped(*args, **kwargs):
            step_before = engine.sim_step
            engine.capture_boundary(step_before)
            out = fn(*args, **kwargs)
            engine.capture_boundary(engine.sim_step)
            annot = build_annotation(name, args, out)
            outputs = (_obs_outputs_with_links(out, step_before, tracer)
                       if name == "get_observation" else out)
            collision_events = getattr(api, "_last_collision_events", None) if name == "move_to_joints" else None
            tracer.record(
                name=name,
                inputs={"args": list(args), "kwargs": kwargs},
                outputs=outputs,
                observation=_obs_key_states(engine.obs),
                sim_step_before=step_before,
                sim_step_after=engine.sim_step,
                annotation=annot,
                collision_events=collision_events,
            )
            if name == "move_to_joints":
                api._last_collision_events = []
            return out

        wrapped.__name__ = name
        return wrapped

    fns = dict(api.functions())

    def _move_to_joints_planned(target, tol=0.10, fk_tol=0.03):
        """RRT-Connect 避障关节运动（2026-08-06 用户指令③, 契约外注册同 grasp_cgn 先例）。

        plan_joint_path（含腕部实体, 手-方块净距 3mm, 读最新方块位姿）→
        密化流式执行（0.05 rad 细分）——移植 cgn_execute_grasp.py 已验证模式
        （回 TUCK 同款）。规划失败抛 RuntimeError，调用方决定退化策略。
        """
        from ..planning.motion_planner import plan_joint_path
        q_from = engine.current_arm_qpos()
        path = plan_joint_path(q_from, np.asarray(target, dtype=np.float64).reshape(6),
                               engine, max_iters=8000, time_budget_s=10.0)  # 2026-08-06:
        # 近块/贴脸规划频发 3000 迭代内无解（0806_1549 接近段三连无解）, 加大预算
        if path is None:
            raise RuntimeError("避障规划失败(RRT 无解)")
        waypts = [q_from] + list(path[1:])
        for a, b in zip(waypts, waypts[1:]):
            nseg = max(1, int(np.ceil(np.abs(b - a).max() / 0.05)))
            for ii in range(1, nseg + 1):
                api.move_to_joints(a + (b - a) * (ii / nseg), tol=tol, fk_tol=fk_tol)

    fns["grasp_cgn"] = api.grasp_cgn  # 契约外后端直连（CGN 相机系候选 + openings）
    fns["move_to_joints_planned"] = _move_to_joints_planned  # 契约外 RRT 避障（指令③）

    def _set_gripper_ramp(target, ticks=25):
        """渐进夹爪（2026-08-06 移植 cgn_execute_grasp ramp_gripper；用户反馈
        "夹爪太快"弹块）。target: 1.0=全开 / 0.0=全闭; ticks: 斜坡拍数。
        契约外注册（同 grasp_cgn 先例）。"""
        start = engine.gripper_fraction
        for i in range(1, int(ticks) + 1):
            engine.set_gripper(start + (float(target) - start) * i / int(ticks))
            engine.hold_ticks(1)

    fns["set_gripper_ramp"] = _set_gripper_ramp

    def _execute_legs_rrt(q_legs, final_tol=0.02, final_fk_tol=0.005):
        """先统一规划、再连续执行（2026-08-06 用户指令: 规划与执行分开,
        不要交替——逐腿"规划-执行"交替造成冻结/瞬移观感）。

        规划段: 逐腿 plan_joint_path（对当前场景一次规划全程; RRT 搜索期
        窗口心跳保持活动）; 执行段: 全程 0.05rad 密化 → engine.step_joints
        流式一次走完（中途无规划冻结=丝滑; 跟踪滞后 >0.35rad 中止）→
        末点阻塞核验（到位即真到位）。
        已知取舍（用户知情）: 全程一次预算后, 若早段擦碰移位方块, 后续
        路径可能过期（08-05 剐蹭-移位教训）; 3mm 净距余量下擦碰罕见,
        且用户明确要求规划/执行分离。失败抛 RuntimeError（带腿号）。
        契约外注册（同 grasp_cgn 先例）。
        """
        from ..planning.motion_planner import plan_joint_path
        # ---- 规划段（一次性, 对当前场景） ----
        way_all = [engine.current_arm_qpos()]
        for li, q_leg in enumerate(q_legs):
            path = plan_joint_path(way_all[-1], np.asarray(q_leg, dtype=np.float64).reshape(6),
                                   engine)  # 默认参数（3000 迭代/6s, 与参考一致）
            if path is None:
                raise RuntimeError(f"避障规划失败(腿{li})")
            if len(path) > 2:
                print(f"  [plan] 腿{li}: RRT 绕行 {len(path)} 点")
            way_all += list(path[1:])
        # ---- 执行段（一次流式, 中途零规划） ----
        dense = [way_all[0]]
        for a, b in zip(way_all, way_all[1:]):
            nseg = max(1, int(np.ceil(np.abs(b - a).max() / 0.05)))
            for ii in range(1, nseg + 1):
                dense.append(a + (b - a) * (ii / nseg))
        for di, q_d in enumerate(dense[1:], 1):
            engine.step_joints(q_d)
            if di % 10 == 0 and np.linalg.norm(
                    engine.current_arm_qpos() - q_d) > 0.35:
                raise RuntimeError(f"流式跟踪丢失(点{di})")
        # 末点阻塞核验（到位即真到位, 闭合无沉降）
        api.move_to_joints(np.asarray(q_legs[-1], dtype=np.float64).reshape(6),
                           tol=final_tol, fk_tol=final_fk_tol)

    fns["execute_legs_rrt"] = _execute_legs_rrt

    def _draw_grasp_glyphs(grasps, scores=None, highlight=0):
        """抓取候选 3D glyph 进可视化窗口（品红=highlight 位, 绿=其余）。
        契约外调试可视化（2026-08-06 用户要求: 品红=正在执行的候选）。"""
        engine.draw_grasp_glyphs(np.asarray(grasps, dtype=float).reshape(-1, 4, 4),
                                 scores, highlight=int(highlight))

    fns["draw_grasp_glyphs"] = _draw_grasp_glyphs
    for name, fn in fns.items():
        ns[name] = wrap(name, fn)
    return ns
