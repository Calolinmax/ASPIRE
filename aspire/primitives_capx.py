"""ASPIRE Primitive API —— cap-x 标准（非GT）/ Piper 机械臂。

对应 cap-x 的 FrankaControlApiReduced（external/cap-x/capx/integrations/franka/
control_reduced.py，privileged=false 的 cube_stack 配置所注入的函数集），
在 Piper + 本项目基础设施上的等价实现：

  | cap-x (Franka)                     | 本实现 (Piper)                        |
  |------------------------------------|---------------------------------------|
  | SAM3 server (HTTP)                 | 本地 SAM3 (aspire.vision_sam3, 同权重) |
  | Molmo server (vLLM)                | SAM3 级联 shim (同名同签名)            |
  | Contact-GraspNet server            | 几何抓取规划器 (同签名/同坐标约定)     |
  | pyroki IK server (panda URDF)      | MuJoCo DLS IK (grip_site, 同收敛语义) |
  | JOINT_POSITION 力矩控制            | position 执行器直写 (真机同模式)       |

与 cap-x 契约保持一致：函数名/签名/返回结构、基座坐标系、wxyz 四元数、阻塞语义、
plan_grasp 的相机系返回约定、solve_ik 的"目标=夹爪 TCP 位姿"语义
（Piper TCP_OFFSET=0 —— grip_site 即指尖 TCP，见 gripper.xml）。

两处**有意的偏差**（docs/primitive_api_capx.md 均注明）：
  1. 反投影约定：y = -(v-cy)*z/fy（mujoco 相机系 y 向上，本仓库实测标定；
     cap-x 为 y=+(v-cy)*z/fy —— 其 pose_mat 内叠了修正旋转，数学上等价，
     这里保持 pose_mat 为真刚体变换，便于直接复合抓取位姿）。
  2. robot_joint_pos 为 (7,) = 6 臂关节 + 夹爪开合度（Franka 为 7+1=(8,)）。
"""

from __future__ import annotations

import os
from typing import Any

import mujoco
import numpy as np
from robosuite.utils import camera_utils as CU
from robosuite.utils import transform_utils as T

from .vision_client import segment_sam3_text_prompt as _sam3_text_raw
from .vision_client import segment_sam3_point_prompt as _sam3_point_raw
from .vision_client import grasp_cgn as _grasp_cgn_raw
from .annotate import build_annotation

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
# 层 1（Panda 约定层）: CGN 输出的物理含义。
#   pred_grasps_cam 4x4 直接把 Panda gripper 模型整体放置（官方可视化
#   draw_grasps/plot_mesh 证实）：原点 = hand base(palm)，局部 +z = 逼近方向
#   (腕部→指尖)，指尖垫接触平面在局部 +z GRIPPER_DEPTH_PANDA 处
#   （0.1034 为接触面，物理指尖尖端实测 0.1122 = 0.0584+finger.stl z_max 0.0538）。
GRIPPER_DEPTH_PANDA = 0.1034
#
# 层 2（Piper 实测层）: 抓取点(指尖接触面) → Piper TCP 的偏移。
#   Piper grip_site 由 gripper.xml 的 eef body 定义在「指尖垫接触面中点」
#   (eef pos="0 0 -0.045" 即接触平面，quat 翻转使 site z=逼近方向、y=开合轴)，
#   与 CGN 的指尖接触平面语义同一点 → 偏移为 0。
TCP_DEPTH_PIPER = 0.0
#
# Piper 最大开度（可行性过滤上限）：joint7 range [0, 0.035]/指（gripper.xml），
# 双指全开嘴内净距约 45mm（gripper.xml 注释，原模型实测），取净距为上限。
PIPER_MAX_WIDTH = 0.045

GRIPPER_DEPTH = GRIPPER_DEPTH_PANDA  # 兼容旧引用（勿新增使用）


def cgn_to_gripper(g_cgn, pose_mat, T_base_world=None):
    """CGN 相机系位姿 → 基座系 Piper grip_site 位姿。

    坐标链（2026-07-31 调试定型）：
      Step 1: OpenCV→OpenGL（Y 轴翻转；MuJoCo 相机约定）
      Step 2: cam→base（pose_mat 真刚体）
      Step 3: 手指轴对齐：CGN 局部 x = Panda 开合轴 → Piper grip_site y = 开合轴，
              绕局部 z 转 -90°（robosuite xyzw 序四元数）
      Step 4: palm → 指尖接触平面：沿局部 +z（逼近方向）平移
              GRIPPER_DEPTH_PANDA - TCP_DEPTH_PIPER。
              【历史 bug：曾写 -0.1034，把 palm 往后撤了 0.1034，与正确值
              差 2×0.1034≈+0.207m —— 即观测到的 z 系统性 +20cm】

    Args:
        g_cgn: (4,4) CGN 输出，OpenCV 相机系（原点 palm，+z 逼近）
        pose_mat: (4,4) 相机外参（cam→base）
        T_base_world: 未使用（兼容参数）

    Returns:
        (4,4) 基座系 grip_site 位姿（位置 = 指尖接触面中点 = 抓取点）
    """
    # Step 1: OpenCV→OpenGL（Y轴翻转）
    T_flip = np.diag([1, -1, 1, 1])
    g_gl = T_flip @ g_cgn

    # Step 2: 相机系→基座系（pose_mat 已经是 cam→base）
    g_base = pose_mat @ g_gl

    # Step 3: 手指轴对齐（CGN-X 开合轴→Piper grip_site-Y 开合轴，绕 Z 转 -90°）
    R_z = T.quat2mat([0.0, 0.0, -0.707, 0.707])  # robosuite xyzw 序, 绕 Z 转 -90°
    T_align = np.eye(4)
    T_align[:3, :3] = R_z
    g_grip = g_base @ T_align

    # Step 4: palm → 指尖接触平面(= Piper grip_site)，沿局部 +z（逼近方向）
    T_offset = np.eye(4)
    T_offset[2, 3] = GRIPPER_DEPTH_PANDA - TCP_DEPTH_PIPER
    g_final = g_grip @ T_offset

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
            cubeA 为 4cm 立方（axis-aligned），对面夹取接触间距 0.04m。
            【TODO(真机): 从候选接触几何/点云沿闭合轴投影估算，勿用 GT】
        max_width: Piper 最大开度（默认 PIPER_MAX_WIDTH=0.045）
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
_IK_LIB_PATH = os.path.join(os.path.dirname(__file__), "robots", "assets", "piper", "ik_library.npz")


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
        if any(("table" in o) or ("pedestal" in o) or ("floor" in o) or ("cube" in o) for o in other):
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


class PrimitiveContextCapx:
    """cap-x 标准 API 的运行时上下文（Piper）。"""

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

    def segment_sam3_point_prompt(self, rgb: np.ndarray, point_coords) -> list[dict]:
        """SAM3 点提示分割（Tracker，前景点 → 该实例的候选 mask）。

        Args:
            rgb: (H,W,3) uint8
            point_coords: (x, y) 像素坐标（要点在目标物体上）

        Returns:
            同 segment_sam3_text_prompt 的 list 结构；未命中返回 []
        """
        return self._to_capx_results(_sam3_point_raw(np.asarray(rgb), point_coords))

    def grasp_cgn(self, rgb, depth, K, seg, z_range=(0.2, 2.0)):
        """CGN 抓取检测（trace 包装版）。

        rgb 仅用于标注图投影底图，推理只看 depth/K/seg。
        Returns: (grasps (N,4,4) 相机系, scores (N,), openings (N,))
        """
        return _grasp_cgn_raw(depth, K, seg, z_range=z_range, return_openings=True)

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
        # 偏移达半个物宽（实测 1.7cm, 40mm 方块 + 45mm 张嘴足以让指垫压到方块顶面）
        lo = np.percentile(pts, 5, axis=0)
        hi = np.percentile(pts, 95, axis=0)
        centroid = (lo + hi) / 2
        centered = pts - np.median(pts, axis=0)
        # 顶面法线 = 最小特征向量；orient 使其指向相机一侧
        # （相机在坐标原点, 表面点指向相机的方向是 -centroid）
        cov = centered.T @ centered / len(centered)
        eigvals, eigvecs = np.linalg.eigh(cov)
        normal = eigvecs[:, 0]
        if np.dot(normal, centroid) > 0:
            normal = -normal

        # 重力标定（与 cap-x 的 z_range 工作区过滤同级）: 世界竖直方向已知,
        # 小/斜视角点云的 PCA 最小特征向量容易退化成视线方向（实测候选逼近轴
        # 水平化, IK 不可达）——法线与竖直方向夹角 >60° 时吸附到竖直方向
        E_muj = CU.get_camera_extrinsic_matrix(self.sim, ROBOTVIEW)
        up_cam = E_muj[:3, :3].T @ np.array([0.0, 0.0, 1.0])  # 世界上方向(相机系)
        up_cam /= np.linalg.norm(up_cam)
        if np.dot(normal, up_cam) < 0:
            up_cam = -up_cam
        if np.dot(normal, up_cam) < np.cos(np.deg2rad(60.0)):
            normal = up_cam
        approach = -normal  # 抓取逼近方向（指向物体）

        # TCP 目标点: 可见顶面质心下探 → 估计物体中心高度
        tcp = centroid - normal * GRASP_CENTER_DROP

        # 候选: 绕逼近轴 GRASP_N_YAW 个偏航; z 轴 = 逼近方向
        zaxis = approach / np.linalg.norm(approach)
        # 构造与 zaxis 垂直的基向量
        ref = np.array([0.0, 0.0, 1.0]) if abs(zaxis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
        u0 = np.cross(zaxis, ref)
        u0 /= np.linalg.norm(u0)
        v0 = np.cross(zaxis, u0)

        # 得分: 顶面平面拟合质量（薄片点云 → 小特征值占比低）
        planarity = float(np.clip(1.0 - eigvals[0] / max(eigvals.sum(), 1e-12), 0.0, 1.0))
        grasps, scores = [], []
        for k in range(GRASP_N_YAW):
            yaw = 2 * np.pi * k / GRASP_N_YAW
            yaxis = np.cos(yaw) * u0 + np.sin(yaw) * v0  # 手指开合轴
            xaxis = np.cross(yaxis, zaxis)
            R = np.column_stack([xaxis, yaxis, zaxis])
            Tg = np.eye(4)
            Tg[:3, :3] = R
            Tg[:3, 3] = tcp
            grasps.append(Tg)
            scores.append(planarity * (1.0 - 0.05 * k))
        order = np.argsort(scores)[::-1]
        return np.stack(grasps)[order], np.asarray(scores)[order]

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
            from .vision_client import grasp_cgn

            grasps_cam, scores = grasp_cgn(depth, intrinsics, segmentation)
            if len(grasps_cam) > 0:
                obs = self.get_observation()
                pose_mat = obs["robot0_robotview"]["pose_mat"]
                grasps_base = np.array([
                    cgn_to_gripper(g, pose_mat)
                    for g in grasps_cam
                ])
                return grasps_base, scores
        except Exception as e:
            print(f"CGN failed: {e}, falling back to geometric")

        # 2) 几何兜底
        return self._plan_grasp_geometric(depth, intrinsics, segmentation)

    # ------------------------------------------------------------------
    # 4. 运动（IK + 阻塞关节运动）
    # ------------------------------------------------------------------
    def solve_ik(self, position: np.ndarray, quaternion_wxyz: np.ndarray, use_pyroki: bool = True) -> np.ndarray:
        """数值 IK（pyroki 优先 + DLS 兜底），目标为夹爪 TCP 位姿（基座系）。

        Args:
            position: (3,) TCP 目标位置（基座系，米）
            quaternion_wxyz: (4,) TCP 目标姿态 [w,x,y,z]（基座系）
            use_pyroki: 是否优先调用 pyroki IK server（默认 True）

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

        # ------------------------------------------------------------------
        # 1) 优先使用 pyroki IK server（MJCF 真源 URDF），但用 MuJoCo FK 做后验验证
        # ------------------------------------------------------------------
        if use_pyroki:
            try:
                from .pyroki_client import ik_pyroki
                q_pk = ik_pyroki(target_p_base, _q_in(quaternion_wxyz), prev_cfg=self.engine.current_arm_qpos())
                q_pk = np.asarray(q_pk, dtype=np.float64).reshape(-1)
                if q_pk.size >= 6:
                    q_arm = np.clip(q_pk[:6], q_lo, q_hi)
                    # FK 验证：pyroki 优化器偶尔给出误差 ~15mm/2° 的“成功”解，
                    # 这种解在 move_to_joints 自检中会被拒绝；这里提前用 MuJoCo FK 筛掉。
                    if _collision_free_q(q_arm, self.engine):
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
                        if dpos < 1e-3 and dquat < 0.01:
                            return q_arm
            except Exception:
                # pyroki 失败时静默回退到 DLS
                pass

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

        def dls(q0, max_iter=600, tol_p=2e-4, tol_r=3e-3):
            """限位投影 DLS。6 轴无冗余, 收敛域强烈依赖种子 → 多种子策略。"""
            q = np.clip(q0.copy(), q_lo, q_hi)
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            lam = 0.05
            R_goal = T.quat2mat(target_q)
            for _ in range(max_iter):
                cur_p, cur_R = fk(q)
                err_p = target_p - cur_p
                err_r = _rot_err_vec(R_goal, cur_R)
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
            q, ok = dls(q0)
            if ok and _collision_free_q(q, self.engine):
                return q, True
            return q, False

        # 种子 1: IK 库检索（离线稠密 FK, 按 位置+开口+开合轴 匹配近邻）
        lib = _ik_library()
        if lib is not False:
            R_goal_base = T_goal_base[:3, :3]
            z_t = R_goal_base[:, 2]
            y_t = R_goal_base[:, 1]
            dp = np.linalg.norm(lib["P"] - target_p_base, axis=1)
            da = np.arccos(np.clip(lib["Z"] @ z_t, -1.0, 1.0))
            dy = np.arccos(np.clip(np.abs(lib["Y"] @ y_t), -1.0, 1.0))
            score = dp / 0.05 + da / 0.6 + 0.3 * dy / 0.6
            for i in np.argsort(score, kind="stable")[:5]:
                q, ok = dls_clean(np.asarray(lib["Q"][i], dtype=np.float64))
                if ok:
                    return q

        # 种子 2: 规范构型族 (home / 腕部极值家族)
        q_home = np.array([0.0, 1.57, -1.3485, 0.0, 0.0, 0.0])
        seeds = [q_cur, q_home,
                 q_home + np.array([0, 0, 0, 0, 0, np.pi]),
                 q_home + np.array([0, 0, 0, 0, 0, -np.pi])]
        for j1 in (0.0, 0.5, -0.5):
            for j3 in (-0.8, -1.35):
                for j6 in (0.0, np.pi, -np.pi):
                    seeds.append(np.array([j1, 1.57, j3, 0.0, -1.2, j6]))
        for j4 in (1.8, -1.8):
            seeds.append(np.array([0.0, 1.57, -1.0, j4, -1.2, np.pi]))
            seeds.append(np.array([0.0, 1.57, -1.0, j4, 1.2, 0.0]))
        for s in seeds:
            q, ok = dls_clean(s)
            if ok:
                return q
        raise RuntimeError(f"solve_ik 未收敛: pos={position} quat={quaternion_wxyz}")

    def move_to_joints(self, joints: np.ndarray) -> None:
        """关节空间运动（阻塞）。

        Args:
            joints: (6,) 目标关节角 (rad)。命令后等待 ‖q-target‖₂ < 0.02 rad，
                最多约 5 秒；未到位抛 RuntimeError（不再静默返回）。
        """
        target = np.asarray(joints, dtype=np.float64).reshape(6)
        best_err, stall = np.inf, 0
        self._last_collision_events = []
        seen_pairs = set()
        for _ in range(MAX_TICKS_MOVE):
            self.engine.step_joints(target)
            self._last_collision_events.extend(_collect_collision_events(self.engine, seen_pairs))
            err = np.linalg.norm(self.engine.current_arm_qpos() - target)
            if err < JOINT_TOL:
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

        # FK 到位自检: 目标构型 FK 与当前 sim TCP 偏差 >1cm 说明执行器/物理异常
        fk_pos = _fk_tcp(target, self.engine)
        site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
        cur_pos = self.sim.data.site_xpos[site_id]
        fk_base = (self.engine.T_base_world @ np.append(fk_pos, 1.0))[:3]
        cur_base = (self.engine.T_base_world @ np.append(cur_pos, 1.0))[:3]
        if np.linalg.norm(fk_base - cur_base) > 0.01:
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

    def open_gripper(self) -> None:
        """张开夹爪（阻塞至到位）。"""
        self.engine.set_gripper(1.0)
        self.engine.hold_ticks(GRIPPER_SETTLE_TICKS)

    def close_gripper(self) -> None:
        """闭合夹爪（阻塞）。是否夹住需通过 robot_joint_pos[-1] 观测判断。"""
        self.engine.set_gripper(0.0)
        self.engine.hold_ticks(GRIPPER_SETTLE_TICKS)


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
    """构建注入任务代码的全局命名空间（cap-x 函数集 + trace 包装）。"""
    ctx = PrimitiveContextCapx(engine)
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
            collision_events = getattr(ctx, "_last_collision_events", None) if name == "move_to_joints" else None
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
                ctx._last_collision_events = []
            return out

        wrapped.__name__ = name
        return wrapped

    api = {
        "get_observation": ctx.get_observation,
        "segment_sam3_text_prompt": ctx.segment_sam3_text_prompt,
        "segment_sam3_point_prompt": ctx.segment_sam3_point_prompt,
        "point_prompt_molmo": ctx.point_prompt_molmo,
        "grasp_cgn": ctx.grasp_cgn,
        "plan_grasp": ctx.plan_grasp,
        "get_oriented_bounding_box_from_3d_points": ctx.get_oriented_bounding_box_from_3d_points,
        "solve_ik": ctx.solve_ik,
        "move_to_joints": ctx.move_to_joints,
        "open_gripper": ctx.open_gripper,
        "close_gripper": ctx.close_gripper,
    }
    for name, fn in api.items():
        ns[name] = wrap(name, fn)
    return ns
