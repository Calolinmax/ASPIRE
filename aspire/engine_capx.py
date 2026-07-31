"""ASPIRE 执行引擎 —— cap-x 标准 / Piper 机械臂变体。

与 aspire.engine（Panda + OSC）的差异（全部对应 cap-x 的非GT cube_stack 管线，
见 external/cap-x/capx/envs/simulators/robosuite_{base,cubes}.py）：

  - 机器人: AgileX Piper（aspire.robots 注册，桌面安装于 TableArena 后侧）
  - 相机:   robot0_robotview（第三人称 512×512，cap-x 唯一观测相机）
            + robot0_eye_in_hand（腕部，仅用于 trace 帧流，API 不暴露）
  - 控制:   不用 robosuite 控制器，引擎直接写 position 执行器 ctrl
            （cap-x 为 JOINT_POSITION 力矩控制；Piper 真机即位置伺服，
            语义等价：move_to_joints_blocking = 命令目标角并等待收敛）
  - 坐标系: API 全部在 **机器人基座系**（robot0_base_link），与世界系同向不同原点
  - 步进:   1 tick = 写 ctrl + SUBSTEPS 个 sim.step + 刷新观测（≈20Hz 控制节拍）

CLI:
    MUJOCO_GL=egl python -m aspire.engine_capx --code task.py --task Stack --seed 0
"""

from __future__ import annotations

import argparse
import os

import mujoco
import numpy as np
import robosuite as suite
from robosuite.controllers.composite.composite_controller_factory import (
    load_composite_controller_config,
)
from robosuite.models.arenas import TableArena
from robosuite.models.objects import BallObject, BoxObject, CylinderObject
from robosuite.models.tasks import ManipulationTask
from robosuite.utils import transform_utils as T
from robosuite.utils.mjcf_utils import CustomMaterial
from robosuite.utils.placement_samplers import UniformRandomSampler

import aspire.robots  # noqa: F401  注册 Piper 机器人/夹爪
from .engine import ExecutionEngine, CAMERA_ALIASES, FRAME_INTERVAL
from .robots.piper_robot import _ASSETS

# ---------------------------------------------------------------------------
# 视觉服务进程管理（SAM3 + CGN 共享同一 CUDA 进程，与 EGL 渲染隔离）
# ---------------------------------------------------------------------------
import subprocess
import time as _time

from . import vision_client

_VISION_SERVER_PROC = None


def _ensure_vision_server():
    """如果 vision_server 未运行，自动拉起子进程（SAM3/CGN 共享同一 CUDA 进程）。"""
    global _VISION_SERVER_PROC
    if vision_client.healthy():
        return
    print("[engine_capx] 启动 SAM3/CGN 视觉服务子进程...", flush=True)
    _VISION_SERVER_PROC = subprocess.Popen(
        [os.environ.get("PYTHON", "/home/stouching/anaconda3/envs/ASPIRE/bin/python"),
         "-m", "aspire.vision_server", "--port", "8123"],
        cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
    )
    t0 = _time.time()
    while not vision_client.healthy():
        if _time.time() - t0 > 180:
            raise RuntimeError("vision_server 启动超时")
        _time.sleep(2.0)
    print("[engine_capx] 视觉服务就绪", flush=True)


def _stop_vision_server():
    global _VISION_SERVER_PROC
    if _VISION_SERVER_PROC is not None:
        _VISION_SERVER_PROC.terminate()
        try:
            _VISION_SERVER_PROC.wait(timeout=5)
        except Exception:
            _VISION_SERVER_PROC.kill()
        _VISION_SERVER_PROC = None


# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------
PIPER_CAMERAS = ["robot0_robotview", "robot0_eye_in_hand"]
CAMERA_ALIASES_CAPX = {"robot0_robotview": "top", "robot0_eye_in_hand": "wrist"}

ARM_ACTUATORS = [f"robot0_joint{i}" for i in range(1, 7)]
GRIPPER_ACTUATOR = "gripper0_right_gripper"
ARM_JOINTS = [f"robot0_joint{i}" for i in range(1, 7)]
BASE_BODY = "robot0_base_link"

SUBSTEPS = 25               # 每 tick 的 sim.step 数（0.002s 步长 → 20Hz 控制节拍）
SETTLE_TICKS = 10           # reset 后稳定 tick 数
GRIPPER_TRAVEL = 0.035      # joint7 全行程（0=闭合, 0.035=张开）

# Piper 桌面工作区的方块放置范围（cap-x 的 Franka 范围为 x±0.18/y±0.12；
# 按 25cm 立柱安装的可达带收缩: 基座系 x∈[0.28,0.40] —— 更高处 (悬停/放置
# 预备 z≈0.7) 的可达边界在 x≈0.40-0.45 之间, 再远 IK 不收敛 (实测)）
CUBE_X_RANGE = [-0.02, 0.10]
CUBE_Y_RANGE = [-0.12, 0.12]

# P3: 视线走廊约束的包围球半径表（半对角线, 与 StackClutter 尺寸定义一致）
_CORRIDOR_RADII = {
    "cubeB": float(np.sqrt(3) * 0.025),
    "dist_box1": float(np.linalg.norm([0.02, 0.02, 0.03])),
    "dist_box2": float(np.linalg.norm([0.015, 0.03, 0.02])),
    "dist_cyl": float(np.linalg.norm([0.018, 0.035])),
    "dist_ball": 0.022,
}

# robotview 相机世界坐标（robosuite 编译模型 sim.data.cam_xpos 实测值,
# 与 robot.xml pos="0.68 0 0.46" + 桌柱安装链一致）。 CorridorFreeSampler
# 需要世界系光心; _load_model 阶段模型未编译无法自取, 故为常量——
# 安装/相机改动时必须同步更新（engine 初始化有漂移自检, 见 _post_reset）。
ROBOTVIEW_CAM_WORLD = np.array([0.38, 0.0, 1.26])


class CorridorFreeSampler(UniformRandomSampler):
    """UniformRandomSampler + 视线走廊约束（P3, 2026-07-31）。

    任何非目标物体不得进入「相机→cubeA」视线走廊: 物体包围球中心到
    射线（相机光心→cubeA 中心, 取线段内）的横向距离须 > 半径+1cm。
    rejection sampling 整体重采（rng 消费序列确定 ⇒ seed 复现性保持）。
    """

    def __init__(self, *args, cam_pos=None, corridor_target="cubeA",
                 corridor_radii=None, corridor_margin=0.01, max_attempts=50, **kwargs):
        super().__init__(*args, **kwargs)
        self._cam_pos = None if cam_pos is None else np.asarray(cam_pos, dtype=np.float64)
        self._target = corridor_target
        self._radii = corridor_radii or {}
        self._margin = corridor_margin
        self._max_attempts = max_attempts

    def _corridor_ok(self, placement: dict) -> bool:
        if self._cam_pos is None or self._target not in placement:
            return True
        tgt = np.asarray(placement[self._target][0], dtype=np.float64)
        v = tgt - self._cam_pos
        vv = float(np.dot(v, v))
        if vv <= 0:
            return True
        for name, (pos, quat, obj) in placement.items():
            if name == self._target:
                continue
            r = self._radii.get(name)
            if r is None:
                continue
            w = np.asarray(pos, dtype=np.float64) - self._cam_pos
            t = float(np.dot(w, v) / vv)
            if 0.0 < t < 1.0 and float(np.linalg.norm(w - t * v)) < r + self._margin:
                return False
        return True

    def sample(self, fixtures=None, reference=None, on_top=True):
        placement = super().sample(fixtures, reference, on_top)
        for _ in range(self._max_attempts - 1):
            if self._corridor_ok(placement):
                return placement
            placement = super().sample(fixtures, reference, on_top)
        if not self._corridor_ok(placement):
            print("[CorridorFreeSampler] 50 次未满足走廊约束, 放行末次采样")
        return placement


class StackClutter(suite.environments.manipulation.stack.Stack):
    """Stack + 4 个干扰物体（蓝盒/灰盒/黄柱/紫球）。

    背景: CGN 训练分布 = 多物杂乱场景 + RealSense 噪声, 干净单方块 sim 是
    双重 OOD（方块面 contact score 被压阈值下 → 0 候选）。加杂物后方块面
    分数 0.177→0.265 过线，端到端 ~1/3 出候选（2026-07-31 实测）。
    干扰物颜色避开红色系，SAM3 "red cube" 仍唯一命中 cubeA。
    """

    def _load_model(self):
        super(suite.environments.manipulation.stack.Stack, self)._load_model()

        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        mujoco_arena = TableArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
        )
        mujoco_arena.set_origin([0, 0, 0])

        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(texture="WoodRed", tex_name="redwood",
                                 mat_name="redwood_mat", tex_attrib=tex_attrib, mat_attrib=mat_attrib)
        greenwood = CustomMaterial(texture="WoodGreen", tex_name="greenwood",
                                   mat_name="greenwood_mat", tex_attrib=tex_attrib, mat_attrib=mat_attrib)
        self.cubeA = BoxObject(name="cubeA", size_min=[0.02, 0.02, 0.02], size_max=[0.02, 0.02, 0.02],
                               rgba=[1, 0, 0, 1], material=redwood)
        self.cubeB = BoxObject(name="cubeB", size_min=[0.025, 0.025, 0.025], size_max=[0.025, 0.025, 0.025],
                               rgba=[0, 1, 0, 1], material=greenwood)
        # 杂物尺寸软约束（P4, 面向未来多物体任务）: 新增杂物尽量至少有一个
        # 维度 ≤4cm —— 让杂物本身也可被 Piper 抓取（净开度 4.49cm）。
        # 不强制、不影响现有布局与 seed 复现; 当前 dist_box2(6cm)/dist_cyl
        # 部分维度超限仅作背景, 其抓取由宽度过滤自然淘汰。
        self.distractors = [
            BoxObject(name="dist_box1", size_min=[0.02, 0.02, 0.03], size_max=[0.02, 0.02, 0.03],
                      rgba=[0.1, 0.25, 0.8, 1]),
            BoxObject(name="dist_box2", size_min=[0.015, 0.03, 0.02], size_max=[0.015, 0.03, 0.02],
                      rgba=[0.5, 0.5, 0.5, 1]),
            CylinderObject(name="dist_cyl", size=[0.018, 0.035], rgba=[0.9, 0.8, 0.1, 1]),
            BallObject(name="dist_ball", size=[0.022], rgba=[0.5, 0.1, 0.6, 1]),
        ]
        objects = [self.cubeA, self.cubeB] + self.distractors

        # 采样器必须在这里建（P3 调试结论）: robosuite hard_reset 每次 reset()
        # 都重跑 _load_model, 外部事后替换的 sampler 会被冲掉——
        # engine._make_env 的替换对本类是死代码（CUBE 范围也曾因此失效）。
        # CorridorFreeSampler = CUBE 范围 + 视线走廊约束一体化, 重建也安全。
        self.placement_initializer = CorridorFreeSampler(
            name="ObjectSampler",
            mujoco_objects=objects,
            x_range=list(CUBE_X_RANGE),
            y_range=list(CUBE_Y_RANGE),
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
            rng=self.rng,
            cam_pos=ROBOTVIEW_CAM_WORLD,
            corridor_radii=_CORRIDOR_RADII,
        )
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )


class ExecutionEngineCapx(ExecutionEngine):
    """cap-x 标准（非GT）+ Piper 的执行引擎。

    Args:
        clutter: Stack 场景的干扰物数量。None(默认)=Stack 时 4 个（N2 验证
            配置）, 其他任务 0; 显式传 0 可关闭。seed 传入 robosuite
            (本类已修复基类不传 seed 导致的摆放不可复现)。
    """

    def __init__(self, task: str = "Lift", seed: int = 0, trace_root: str = "traces",
                 render: bool = False, render_slowdown: float = 1.0,
                 clutter: int | None = None):
        self._clutter = (4 if task == "Stack" else 0) if clutter is None else clutter
        _ensure_vision_server()
        super().__init__(task=task, seed=seed, trace_root=trace_root,
                         render=render, render_slowdown=render_slowdown)

    def close(self):
        # "关引擎必落盘" 不变量（2026-07-31 插队修复）:
        # trace.json 原本只在基类 run() 末尾由 tracer.finalize() 写出（engine.py）；
        # 直接驱动模式（cap-x 脚本驱动 engine 后 close, 如 cgn_verify_steps/
        # cgn_execute_grasp）不经 run()，json 永远缺失（traces/0731_16* 那批
        # 只有 images/ 的目录即此因）。此处兜底 finalize。
        # 幂等: run() 路径已写（json 存在）或重复 close 都不重写；基类 run() 不动。
        json_path = os.path.join(self.tracer.trace_dir, "trace.json")
        if not getattr(self, "_trace_finalized", False) and not os.path.exists(json_path):
            self._trace_finalized = True
            try:
                success = getattr(self, "_trace_success", None)
                error = getattr(self, "_trace_error", None)
                if success is None and error is None:
                    error = ("direct-drive: 未经 run() 判定; 调用方可先调 "
                             "mark_trace_result(success, error) 声明结果, "
                             "未声明则 success=null")
                self.tracer.finalize(success, error, getattr(self, "sim_step", 0))
            except Exception as e:
                print(f"[engine_capx.close] tracer.finalize 失败（不阻塞关闭）: {e}")
        super().close()
        # 仅 engine_capx.py main() 显式控制子进程生命周期；
        # 这里不终止，避免多个 engine 实例串行时反复启停。

    def mark_trace_result(self, success, error=None):
        """直接驱动模式下由调用方声明任务结果，close() 落盘 trace.json 时写入。"""
        self._trace_success = success
        self._trace_error = error

    # ------------------------------------------------------------------
    # 环境构造钩子
    # ------------------------------------------------------------------
    def _env_kwargs(self) -> dict:
        return dict(
            robots="Piper",
            has_renderer=False,
            has_offscreen_renderer=True,
            use_camera_obs=True,
            camera_names=list(PIPER_CAMERAS),
            camera_heights=640,
            camera_widths=640,
            camera_depths=True,
            control_freq=20,
            horizon=4000,
            reward_shaping=True,
            seed=self.seed,   # 基类漏传: robosuite env.rng 由此可控, 摆放可复现
            controller_configs=load_composite_controller_config(
                controller=os.path.join(_ASSETS, "default_piper.json")
            ),
        )

    def _make_env(self, kwargs: dict):
        if self.task == "Stack" and self._clutter > 0:
            env = StackClutter(**kwargs)
        else:
            env = suite.make(self.task, **kwargs)
        # 预置离屏缓冲区 1280×960: MJCF <visual> 在 robosuite 合并时被丢弃
        # (实测 readback 640×480), 程序化设置才生效。首个超限 render 由
        # robosuite update_offscreen_size 一次性重建 context 到该尺寸
        # (max(请求, 预置)), 之后任何 ≤1280×960 的渲染不再重建。
        env.sim.model._model.vis.global_.offwidth = 1280
        env.sim.model._model.vis.global_.offheight = 960
        # cap-x 同款做法: 建 env 后替换放置采样器（范围按 Piper 臂展适配）。
        # 注意: 仅 plain Stack 走这里——robosuite Stack._load_model 有
        # "已存在则 reset+add_objects"分支, 替换在 hard_reset 下幸存;
        # StackClutter 的采样器（含走廊约束）在其 _load_model 内一体化构建,
        # 此处跳过（外部替换会被 hard_reset 冲掉, P3 实测）。
        if hasattr(env, "cubeA") and not hasattr(env, "distractors"):
            env.placement_initializer = UniformRandomSampler(
                name="ObjectSampler",
                mujoco_objects=[env.cubeA, env.cubeB],
                x_range=list(CUBE_X_RANGE),
                y_range=list(CUBE_Y_RANGE),
                rotation=None,
                ensure_object_boundary_in_range=False,
                ensure_valid_placement=True,
                reference_pos=env.table_offset,
                z_offset=0.01,
                rng=env.rng,
            )
        return env

    def _post_reset(self):
        """缓存执行器索引/基座变换，写入初始 ctrl 并稳定仿真。"""
        model = self.env.sim.model._model
        # ROBOTVIEW_CAM_WORLD 漂移自检（P3: 常量取自编译模型, 安装改动须同步）
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "robot0_robotview")
        if cam_id >= 0:
            drift = np.linalg.norm(self.env.sim.data.cam_xpos[cam_id] - ROBOTVIEW_CAM_WORLD)
            if drift > 0.01:
                print(f"[engine_capx] WARNING: robotview cam_xpos 漂移 {drift * 100:.1f}cm, "
                      f"ROBOTVIEW_CAM_WORLD 需更新（走廊约束基准失准）")
        self._arm_act_ids = np.array(
            [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, a) for a in ARM_ACTUATORS]
        )
        self._grip_act_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, GRIPPER_ACTUATOR
        )
        base_bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, BASE_BODY)
        self._base_bid = base_bid
        # 基座系固定（焊接在桌面上）: 缓存世界→基座 4x4
        R = np.asarray(self.env.sim.data.xquat[base_bid], dtype=np.float64)
        self._T_world_base = np.eye(4)
        self._T_world_base[:3, :3] = T.quat2mat(T.convert_quat(R, to="xyzw"))
        self._T_world_base[:3, 3] = self.env.sim.data.xpos[base_bid]
        self._T_base_world = np.linalg.inv(self._T_world_base)

        # 控制状态: 当前目标 = 初始关节角, 夹爪张开
        self._q_target = np.asarray(self.obs["robot0_joint_pos"], dtype=np.float64)[:6].copy()
        self.gripper_fraction = 1.0  # 1.0=张开, 0.0=闭合 (cap-x 语义)
        # 稳定: 保持初始位姿 + 张开夹爪若干拍
        for _ in range(SETTLE_TICKS):
            self._tick(render_frames=False)
        self.refresh_obs()

    # ------------------------------------------------------------------
    # 坐标变换（primitives 用）
    # ------------------------------------------------------------------
    @property
    def T_world_base(self) -> np.ndarray:
        return self._T_world_base

    @property
    def T_base_world(self) -> np.ndarray:
        return self._T_base_world

    # ------------------------------------------------------------------
    # 步进（cap-x 语义: 直接写 position 执行器 ctrl）
    # ------------------------------------------------------------------
    def set_gripper(self, fraction: float):
        """设定夹爪开合度: 1.0=全开, 0.0=闭合。"""
        self.gripper_fraction = float(np.clip(fraction, 0.0, 1.0))

    def current_arm_qpos(self) -> np.ndarray:
        """实时 6 臂关节角（直读 sim.data, 不依赖 obs 渲染刷新）。"""
        model = self.env.sim.model._model
        adrs = [int(model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)])
                for j in ARM_JOINTS]
        return np.asarray(self.env.sim.data.qpos[adrs], dtype=np.float64)

    def _write_ctrl(self):
        data = self.env.sim.data
        data.ctrl[self._arm_act_ids] = self._q_target
        data.ctrl[self._grip_act_id] = self.gripper_fraction * GRIPPER_TRAVEL

    def _tick(self, render_frames: bool = True):
        """一拍: 写 ctrl → SUBSTEPS 次 sim.step。

        观测渲染（512×512 双相机）是 EGL 负载大头: 运动拍不刷新图像观测,
        只在帧流边界渲染（实测高频 512 双相机渲染会反复 wedge EGL context）。
        关节状态判断走 current_arm_qpos（sim.data 实时值, 无需渲染）。
        """
        if self.terminated:
            return
        self._write_ctrl()
        for _ in range(SUBSTEPS):
            self.env.sim.step()
        self.sim_step += 1
        if render_frames and self.sim_step % FRAME_INTERVAL == 0:
            self.refresh_obs()
            self.capture_boundary(self.sim_step, include_depth=False)
        if self.render:
            self.env.render()
        if self.sim_step >= self.env.horizon:
            self.terminated = True

    def refresh_obs(self):
        """刷新观测（渲染相机）+ 渲染健康检查。API 边界/帧流捕获前调用。"""
        self.obs = self.env._get_observations(force_update=True)
        self._renderer_healthcheck()

    def capture_boundary(self, sim_step: int, include_depth: bool = True):
        """覆盖: 捕获前先刷新观测（否则拿到的是上一边界的旧帧）。"""
        self.refresh_obs()
        super().capture_boundary(sim_step, include_depth)

    def step_joints(self, q_target) -> None:
        """primitives 的步进入口: 更新臂目标角并进一拍。"""
        self._q_target = np.asarray(q_target, dtype=np.float64).reshape(6)
        self._tick()

    def hold_ticks(self, n: int) -> None:
        """保持当前目标 n 拍（夹爪动作稳定用）。"""
        for _ in range(n):
            self._tick()

    # ------------------------------------------------------------------
    # 覆盖: 相机别名 / 深度图键名（agentview → robotview）
    # ------------------------------------------------------------------
    CAMERA_ALIASES = CAMERA_ALIASES_CAPX

    def _build_namespace(self) -> dict:
        from .primitives_capx import build_namespace

        return build_namespace(self)

    def _depth_colormap(self):
        d = self.obs.get("robot0_robotview_depth")
        if d is None:
            return None
        import cv2
        from robosuite.utils import camera_utils as CU

        if d.ndim == 3:
            d = d.squeeze(-1)
        d = np.clip(np.nan_to_num(np.asarray(d, dtype=np.float64), nan=1.0), 0.0, 1.0)
        real = CU.get_real_depth_map(self.env.sim, d)
        lo, hi = 0.3, 2.0
        norm = (np.clip(real, lo, hi) - lo) / (hi - lo)
        cm = cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_VIRIDIS)
        return cv2.cvtColor(cm, cv2.COLOR_BGR2RGB)


def main():
    parser = argparse.ArgumentParser(description="ASPIRE Execution Engine (cap-x / Piper)")
    parser.add_argument("--code", required=True, help="任务代码 .py 文件路径")
    parser.add_argument("--task", default="Stack")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--trace-root", default="traces")
    parser.add_argument("--render", action="store_true", help="打开可视化仿真器窗口实时演示")
    args = parser.parse_args()

    with open(args.code, encoding="utf-8") as f:
        code = f.read()

    _ensure_vision_server()

    engine = ExecutionEngineCapx(task=args.task, seed=args.seed, trace_root=args.trace_root,
                                 render=args.render)
    try:
        result = engine.run(code, code_ref=args.code)
    finally:
        engine.close()
        _stop_vision_server()

    print("=== Execution Result (cap-x / Piper) ===")
    print(f"success        : {result['success']}")
    print(f"sim_steps      : {result['sim_steps']}")
    print(f"primitive_calls: {result['n_primitive_calls']}")
    print(f"trace          : {result['trace_path']}")
    if result["stdout"]:
        print("--- stdout ---")
        print(result["stdout"])
    if result["error"]:
        print("--- error ---")
        print(result["error"])
    raise SystemExit(0 if result["success"] else 1)


if __name__ == "__main__":
    main()
