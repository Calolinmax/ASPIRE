"""ASPIRE Primitive API 实现（复现版 v0.3 - 双相机+SAM3视觉）。

对应 docs/primitive_api.md 的契约。所有函数由 ExecutionEngine 注入任务代码
的全局命名空间。感知为 SAM3（848M）GPU 分割，原生开放词汇文本提示。
"""

from __future__ import annotations

import time
from typing import Any

import mujoco
import numpy as np
from robosuite.utils import camera_utils as CU
from robosuite.utils import transform_utils as T

# 导入视觉模块（SAM3，transformers 本地权重，GPU）
from .vision_sam3 import segment_sam3_text_prompt, segment_sam3_point_prompt, warmup

# 视觉证据标注（定位算法输出画回输入帧，随 trace 落盘）
from .annotate import build_annotation

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
# 双相机配置
CAMERAS = {
    "agentview": {"name": "agentview", "h": 256, "w": 256},  # 顶部第三人称
    "wrist": {"name": "robot0_eye_in_hand", "h": 256, "w": 256},  # 腕部相机
}

EEF_SITE = "gripper0_right_grip_site"
ARM_JOINTS = [f"robot0_joint{i}" for i in range(1, 8)]

# move_to_pose 闭环参数
KP_POS = 5.0
KP_ROT = 2.0
POS_TOL = 0.005          # 5mm
ROT_TOL = 0.07           # ~4°
MAX_STEPS_MOVE = 160     # 8s @ 20Hz
GRIPPER_SETTLE_STEPS = 12

# ---------------------------------------------------------------------------
# 四元数约定处理（重要！）
# ---------------------------------------------------------------------------
# API 文档对用户暴露的是 **wxyz**（与 open_details 示例一致，如 [0,1,0,0] 朝下）。
# 但 robosuite transform_utils 全套（quat2mat/mat2quat/quat_slerp/quat2axisangle）
# 以及观测 robot0_eef_quat_site 都是 **xyzw**。因此：
#   - API 边界：wxyz ↔ xyzw 转换（_q_in / _q_out）
#   - 模块内部：一律 xyzw


def _q_in(q_wxyz) -> np.ndarray:
    """用户输入 wxyz → 内部 xyzw"""
    q = np.asarray(q_wxyz, dtype=np.float64).flatten()
    return np.array([q[1], q[2], q[3], q[0]])


def _q_out(R: np.ndarray) -> np.ndarray:
    """旋转矩阵 → 用户约定的 wxyz"""
    q_xyzw = T.mat2quat(R)
    return np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]])


def _rot_err_vec(R_target: np.ndarray, R_current: np.ndarray) -> np.ndarray:
    """姿态误差：R_err = R_target @ R_current.T → 指数坐标 (3,) = 轴×角。

    mat2quat 与 quat2axisangle 同为 xyzw，可直接串联。
    必须规范 w≥0（q 与 -q 同旋转），否则轴角会跳到 >π 的等价表示，
    导致 IK/闭环在误差方向间振荡。xyzw 的 w 是第 4 个分量。
    """
    q = T.mat2quat(R_target @ R_current.T)
    if q[3] < 0:
        q = -q
    return T.quat2axisangle(q)


class PrimitiveContext:
    """primitives 的运行时上下文：持有 engine 引用并提供全部 API 实现。"""

    def __init__(self, engine):
        self.engine = engine

    # ------------------------------------------------------------------
    # 内部工具
    # ------------------------------------------------------------------
    @property
    def env(self):
        return self.engine.env

    @property
    def sim(self):
        return self.engine.env.sim

    @property
    def model(self):
        """原生 mujoco.MjModel"""
        return self.sim.model._model

    # ------------------------------------------------------------------
    # 1. 感知类（双相机 + SAM3 GPU 真实视觉）
    # ------------------------------------------------------------------
    def get_observation(self, camera: str = "agentview") -> dict:
        """
        获取指定相机的观测。

        参数:
            camera: "agentview"(顶部) 或 "wrist"(腕部)

        返回包含:
            - robot0_eef_pos/quat: 末端位姿
            - {camera}_image: RGB图像
            - {camera}_depth: 深度图
            - camera_intrinsics: 内参K
            - camera_extrinsics: 外参E (cam→world)
        """
        obs = dict(self.engine.obs)

        # 确保请求的相机图像存在
        if camera not in ["agentview", "wrist"]:
            camera = "agentview"

        img_key = f"{camera}_image"
        depth_key = f"{camera}_depth"

        # 如果环境没有该相机，渲染它
        if img_key not in obs:
            # 动态渲染
            img, depth = self.engine.env.sim.render(
                camera_name=CAMERAS[camera]["name"],
                height=CAMERAS[camera]["h"],
                width=CAMERAS[camera]["w"],
                depth=True
            )
            # EGL context wedge 时动态渲染也会返回垃圾帧：重建后重渲一次
            if self.engine._img_corrupt(img):
                self.engine.recover_renderer()
                img, depth = self.engine.env.sim.render(
                    camera_name=CAMERAS[camera]["name"],
                    height=CAMERAS[camera]["h"],
                    width=CAMERAS[camera]["w"],
                    depth=True
                )
            obs[img_key] = img
            obs[depth_key] = depth

        # 转换深度为真实距离
        if depth_key in obs:
            d = obs[depth_key]
            if d.ndim == 3:
                d = d.squeeze(-1)
            # GL 深度按定义在 [0,1]，但 float32 回读偶有数值噪声越界（含 NaN），
            # robosuite 的 get_real_depth_map 断言过严会直接崩；先归一化再转换。
            d = np.clip(np.nan_to_num(np.asarray(d, dtype=np.float64), nan=1.0), 0.0, 1.0)
            obs[depth_key] = CU.get_real_depth_map(self.sim, d)

        # 相机参数
        K = CU.get_camera_intrinsic_matrix(
            self.sim, CAMERAS[camera]["name"],
            CAMERAS[camera]["h"], CAMERAS[camera]["w"]
        )
        E = CU.get_camera_extrinsic_matrix(self.sim, CAMERAS[camera]["name"])
        obs["camera_intrinsics"] = K
        obs["camera_extrinsics"] = E
        obs["active_camera"] = camera

        return obs

    def segment_text(self, rgb, prompt) -> list[dict]:
        """文本提示分割（SAM3 GPU，原生开放词汇，不再需要颜色 hack）。"""
        return segment_sam3_text_prompt(rgb, prompt)

    def segment_point(self, rgb, point) -> list[dict]:
        """点提示分割（SAM3 Tracker GPU）"""
        return segment_sam3_point_prompt(rgb, point)

    # 兼容旧API
    def segment_sam3_text_prompt(self, rgb, prompt) -> list[dict]:
        return self.segment_text(rgb, prompt)

    def segment_sam3_point_prompt(self, rgb, point) -> list[dict]:
        return self.segment_point(rgb, point)

    def point_prompt_molmo(self, rgb, prompt) -> dict:
        masks = self.segment_sam3_text_prompt(rgb, prompt)
        if not masks:
            return {}
        ys, xs = np.nonzero(masks[0]["mask"])
        return {"point_0": (float(xs.mean()), float(ys.mean()))}

    def mask_to_world_points(self, mask, depth, K, E) -> np.ndarray:
        vs, us = np.nonzero(mask)
        if len(vs) == 0:
            return np.zeros((0, 3))
        z = depth[vs, us].astype(np.float64)
        valid = np.isfinite(z) & (z > 0.01)
        us, vs, z = us[valid], vs[valid], z[valid]
        if len(z) == 0:
            return np.zeros((0, 3))
        # 注意：robosuite 外参 E 的相机系是 mujoco 约定（x右、y上、前方+z），
        # 图像 v 向下 → y_cam 取负号（已实测验证，不翻转会致 z 系统偏高 ~4cm）
        x = (us - K[0, 2]) * z / K[0, 0]
        y = -(vs - K[1, 2]) * z / K[1, 1]
        pts_cam = np.stack([x, y, z, np.ones_like(z)], axis=0)
        return (np.asarray(E) @ pts_cam)[:3].T

    def get_oriented_bounding_box_from_3d_points(self, pts) -> dict:
        pts = np.asarray(pts, dtype=np.float64)
        center = pts.mean(axis=0)
        cov = np.cov((pts - center).T)
        eigvals, eigvecs = np.linalg.eigh(cov)
        order = np.argsort(eigvals)[::-1]
        R = eigvecs[:, order]
        if np.linalg.det(R) < 0:  # 保证右手系
            R[:, -1] *= -1
        proj = (pts - center) @ R
        extent = proj.max(axis=0) - proj.min(axis=0)
        return {"center": center, "R": R, "extent": extent}

    def pixel_to_world_point(self, u, v, z, K, E) -> np.ndarray:
        x = (u - K[0, 2]) * z / K[0, 0]
        y = -(v - K[1, 2]) * z / K[1, 1]  # mujoco 相机 y 向上，见 mask_to_world_points
        p = np.asarray(E) @ np.array([x, y, z, 1.0])
        return p[:3]

    # ------------------------------------------------------------------
    # 2. 运动类
    # ------------------------------------------------------------------
    def move_to_pose(self, pos, quat=None) -> None:
        target_p = np.asarray(pos, dtype=np.float64)
        target_R = T.quat2mat(_q_in(quat)) if quat is not None else None
        for _ in range(MAX_STEPS_MOVE):
            obs = self.engine.obs
            eef_p = obs["robot0_eef_pos"]
            err_p = target_p - eef_p
            if target_R is not None:
                # eef_quat_site 即 grip_site 姿态（xyzw），与 eef_pos 同 frame
                R_c = T.quat2mat(obs["robot0_eef_quat_site"])
                err_r = _rot_err_vec(target_R, R_c)
                done = np.linalg.norm(err_p) < POS_TOL and np.linalg.norm(err_r) < ROT_TOL
            else:
                err_r = np.zeros(3)
                done = np.linalg.norm(err_p) < POS_TOL
            if done:
                break
            dp = np.clip(KP_POS * err_p, -1, 1)
            dr = np.clip(KP_ROT * err_r, -1, 1)
            action = np.concatenate([dp, dr, [self.engine.gripper_cmd]])
            self.engine.step(action)

    def solve_ik(self, pos, quat) -> np.ndarray:
        target_p = np.asarray(pos, dtype=np.float64)
        target_q = _q_in(quat)  # 内部统一 xyzw
        target_q = target_q / np.linalg.norm(target_q)
        model = self.model
        data = mujoco.MjData(model)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
        dofadr = [int(model.jnt_dofadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)]) for j in ARM_JOINTS]
        q = np.asarray(self.engine.obs["robot0_joint_pos"], dtype=np.float64).flatten()[:7].copy()

        def fk(q_in):
            for i, jname in enumerate(ARM_JOINTS):
                jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
                data.qpos[model.jnt_qposadr[jid]] = q_in[i]
            mujoco.mj_forward(model, data)
            return (data.site_xpos[site_id].copy(),
                    data.site_xmat[site_id].reshape(3, 3).copy())

        def dls(q0, R_goal, max_iter, tol_p=1e-4, tol_r=1e-3):
            """阻尼最小二乘内层循环，返回 (q, 是否收敛)"""
            q = q0.copy()
            jacp = np.zeros((3, model.nv))
            jacr = np.zeros((3, model.nv))
            lam = 0.05
            for _ in range(max_iter):
                cur_p, cur_R = fk(q)
                err_p = target_p - cur_p
                err_r = _rot_err_vec(R_goal, cur_R)
                n_p, n_r = np.linalg.norm(err_p), np.linalg.norm(err_r)
                if n_p < tol_p and n_r < tol_r:
                    return q, True
                # 误差限幅：防大步长振荡
                if n_p > 0.05:
                    err_p *= 0.05 / n_p
                if n_r > 0.30:
                    err_r *= 0.30 / n_r
                mujoco.mj_jacSite(model, data, jacp, jacr, site_id)
                J = np.vstack([jacp[:, dofadr], jacr[:, dofadr]])   # (6, 7)
                err = np.concatenate([err_p, err_r])
                dq = J.T @ np.linalg.solve(J @ J.T + lam**2 * np.eye(6), err)
                q = q + np.clip(dq, -0.3, 0.3)
            return q, False

        # 姿态同伦：初始姿态 → 目标姿态分步 slerp，绕开 π 附近 DLS 奇异性
        _, R_init = fk(q)
        q_init = T.mat2quat(R_init)  # xyzw
        if np.dot(q_init, target_q) < 0:  # slerp 最短路径对齐
            target_q = -target_q
        for t in np.linspace(0.15, 1.0, 8):
            q_mid = T.quat_slerp(q_init, target_q, t)
            R_mid = T.quat2mat(q_mid / np.linalg.norm(q_mid))
            q, ok = dls(q, R_mid, max_iter=60)
        # 最终精调
        q, ok = dls(q, T.quat2mat(target_q), max_iter=120)
        if not ok:
            raise RuntimeError(f"solve_ik 未收敛: pos={pos}")
        return q

    def move_to_joints(self, joints) -> None:
        joints = np.asarray(joints, dtype=np.float64).flatten()
        model = self.model
        data = mujoco.MjData(model)
        site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)
        for i, jname in enumerate(ARM_JOINTS):
            jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
            data.qpos[model.jnt_qposadr[jid]] = joints[i]
        mujoco.mj_forward(model, data)
        pos = data.site_xpos[site_id].copy()
        quat = _q_out(data.site_xmat[site_id].reshape(3, 3))  # → wxyz 供 move_to_pose
        self.move_to_pose(pos, quat)

    def interpolate_segment(self, p1, p2, step=0.02) -> list:
        p1, p2 = np.asarray(p1, float), np.asarray(p2, float)
        n = max(2, int(np.ceil(np.linalg.norm(p2 - p1) / step)) + 1)
        return [p1 + (p2 - p1) * t for t in np.linspace(0, 1, n)]

    # ------------------------------------------------------------------
    # 3. 抓取类
    # ------------------------------------------------------------------
    def _gripper_action(self, cmd: float, steps: int):
        for _ in range(steps):
            action = np.concatenate([np.zeros(6), [cmd]])
            self.engine.step(action)

    def open_gripper(self) -> None:
        self.engine.gripper_cmd = -1.0
        self._gripper_action(-1.0, GRIPPER_SETTLE_STEPS)

    def close_gripper(self) -> None:
        self.engine.gripper_cmd = 1.0
        self._gripper_action(1.0, GRIPPER_SETTLE_STEPS)

    # ------------------------------------------------------------------
    # 4. 工具类
    # ------------------------------------------------------------------
    def rotation_matrix_to_quaternion(self, R) -> np.ndarray:
        return _q_out(np.asarray(R, dtype=np.float64))


# ---------------------------------------------------------------------------
# trace 包装 + 命名空间构建
# ---------------------------------------------------------------------------
def _obs_key_states(obs: dict) -> dict:
    keys = ["robot0_eef_pos", "robot0_eef_quat", "robot0_gripper_qpos", "robot0_joint_pos"]
    return {k: obs[k] for k in keys if k in obs}


# get_observation 输出中的图像/深度键 → 帧流别名（不在帧流的键保留统计摘要）
_OBS_STREAM_KEYS = {
    "agentview_image": "top",
    "agentview_depth": "depth",
    "wrist_image": "wrist",
    "robot0_eye_in_hand_image": "wrist",
}


def _obs_outputs_with_links(out: Any, sim_step: int, tracer) -> Any:
    """get_observation 返回的图像/深度替换为帧流链接（JSON 只留指针）。"""
    if not isinstance(out, dict):
        return out
    links = tracer.frame_links(sim_step)
    patched = dict(out)
    for key, stream in _OBS_STREAM_KEYS.items():
        if key in patched:
            patched[key] = {"ref": links[stream]}
    return patched


def build_namespace(engine) -> dict:
    """构建注入任务代码的全局命名空间（含 trace 包装）。"""
    ctx = PrimitiveContext(engine)
    tracer = engine.tracer
    ns: dict[str, Any] = {"np": np}

    def wrap(name, fn):
        def wrapped(*args, **kwargs):
            step_before = engine.sim_step
            engine.capture_boundary(step_before)        # 调用前帧入帧流（按步去重）
            out = fn(*args, **kwargs)
            engine.capture_boundary(engine.sim_step)    # 调用后帧（运动类即结束帧）
            annot = build_annotation(name, args, out)
            outputs = (_obs_outputs_with_links(out, step_before, tracer)
                       if name == "get_observation" else out)
            tracer.record(
                name=name,
                inputs={"args": list(args), "kwargs": kwargs},
                outputs=outputs,
                observation=_obs_key_states(engine.obs),
                sim_step_before=step_before,
                sim_step_after=engine.sim_step,
                annotation=annot,
            )
            return out

        wrapped.__name__ = name
        return wrapped

    api = {
        "get_observation": ctx.get_observation,
        "segment_sam3_text_prompt": ctx.segment_sam3_text_prompt,
        "segment_sam3_point_prompt": ctx.segment_sam3_point_prompt,
        "point_prompt_molmo": ctx.point_prompt_molmo,
        "mask_to_world_points": ctx.mask_to_world_points,
        "get_oriented_bounding_box_from_3d_points": ctx.get_oriented_bounding_box_from_3d_points,
        "pixel_to_world_point": ctx.pixel_to_world_point,
        "move_to_pose": ctx.move_to_pose,
        "solve_ik": ctx.solve_ik,
        "move_to_joints": ctx.move_to_joints,
        "interpolate_segment": ctx.interpolate_segment,
        "open_gripper": ctx.open_gripper,
        "close_gripper": ctx.close_gripper,
        "rotation_matrix_to_quaternion": ctx.rotation_matrix_to_quaternion,
    }
    for name, fn in api.items():
        ns[name] = wrap(name, fn)
    return ns
