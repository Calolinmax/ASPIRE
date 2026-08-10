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
    MUJOCO_GL=egl python -m aspire.engine.engine_capx --code task.py --task Stack --seed 0
"""

# =============================================================================
# 🔒 冻结警示（2026-08-06 用户裁决 · 封版）：本文件属**已测试通过**的 API 层
# （cap-x 契约 15 函数 + 契约外 5 函数/组件，docs/api_asset_map.md 冻结清单）。
# **只能在 scripts 中调用，禁止修改——只有人类（顾问也不行）批准才能更改。**
# 本文件同时被 chmod a-w 机械保护；解冻须人类亲自 chmod +w。
# =============================================================================

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
from ..robots.piper_robot import _ASSETS

# ---------------------------------------------------------------------------
# 视觉服务进程管理（SAM3 + CGN 共享同一 CUDA 进程，与 EGL 渲染隔离）
# ---------------------------------------------------------------------------
import subprocess
import time as _time

from ..perception import vision_client

_VISION_SERVER_PROC = None


def _ensure_vision_server():
    """如果 vision_server 未运行，自动拉起子进程（SAM3/CGN 共享同一 CUDA 进程）。"""
    global _VISION_SERVER_PROC
    if vision_client.healthy():
        return
    print("[engine_capx] 启动 SAM3/CGN 视觉服务子进程...", flush=True)
    _VISION_SERVER_PROC = subprocess.Popen(
        [os.environ.get("PYTHON", "/home/stouching/anaconda3/envs/ASPIRE/bin/python"),
         "-m", "aspire.perception.vision_server", "--port", "8123"],
        cwd=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
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
CUBE_X_RANGE = [-0.15, -0.03]   # 2026-08-03 用户: 较初版移近, 再远离一点点
CUBE_Y_RANGE = [-0.16, 0.16]    # 同日用户: 物品分散一些 (±0.12 → ±0.16)

# 垫高台高度（2026-08-03 方案 1；同日用户终裁: 【物品不需要垫高】→ 0.0,
# 台面不再创建, 物品直放桌面, sampler z_offset 回落 0.01。抬高换可达性的
# 全部杠杆随之搁置, 重启须用户明确同意）。
# 历史依据备查: 库 FK 地图（干净数据）证实抓握点 z≈0.12 高于陡降构型
# z≥0.075 地板 + CGN 召回带台/无台持平; "收敛率平曲线"推导系反射污染期
# 产物, 已作废（docs/cgn_container.md §11.1）。
RISER_H = 0.0
# 台面几何参数（RISER_H=0 时不创建台面, 仅备重启时使用）
_RISER_CENTER_XY = (-0.09, 0.0)
_RISER_HALF = (0.14, 0.20)   # 28×40cm 面板, 覆盖放置区+物体半径余量

# P3: 视线走廊约束的包围球半径表（半对角线, 与 StackClutter 尺寸定义一致）
_CORRIDOR_RADII = {
    "cubeB": float(np.sqrt(3) * 0.025),
    "dist_box1": float(np.linalg.norm([0.02, 0.02, 0.03])),
    "dist_box2": float(np.linalg.norm([0.015, 0.03, 0.02])),
    "dist_cyl": float(np.linalg.norm([0.018, 0.035])),
    "dist_ball": 0.022,
}

# robotview 相机世界坐标（robosuite 编译模型 sim.data.cam_xpos 实测值,
# 与 robot.xml pos="-0.220 0.120 0.420" + 桌柱安装链一致）。 CorridorFreeSampler
# 需要世界系光心; _load_model 阶段模型未编译无法自取, 故为常量——
# 安装/相机改动时必须同步更新（engine 初始化有漂移自检, 见 _post_reset）。
# 2026-08-03 相机搬家并经用户 5 轮肉审定稿: (-0.231,0,1.42), 俯角 70°,
# 横向对正台面中线, 视野几乎全桌面, home 位姿仅小臂入镜。
# 【锁死: 只有人类用户有权修改本常量与 robot.xml 的 robotview pos/quat,
# 顾问提议一律转用户确认】
ROBOTVIEW_CAM_WORLD = np.array([-0.231, 0.0, 1.42])

# 收臂让拍位姿（2026-08-03 用户规定）: 只转 j1 = -90°, 臂从全零 home 直接
# 摆向 -y 侧, 与相机走廊垂直。碰撞检查通过; 真实伺服路径 5 tick 到位
# (err 0.0147 < 0.02)。注: 前两版多关节 tuck（库锚定 entry139633/141568）
# 在台面加宽后夹爪碰撞体蹭 riser 导致伺服卡死, 已弃用。
TUCK_Q = np.array([-1.5708, 0.0, 0.0, 0.0, 0.0, 0.0])


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

    def __init__(self, *args, target_only=False, clutter=4, **kwargs):
        # target_only=True: 桌面只留目标物体 cubeA（2026-08-04 用户指令）。
        # clutter=N: 杂物数量（2026-08-05 用户指令; 0=清台, 默认 4 不变）。
        # 须在 super().__init__ 前落 flag —— _load_model 在其中被调用。
        self._target_only = target_only
        self._clutter = clutter
        super().__init__(*args, **kwargs)

    def _setup_references(self):
        if getattr(self, '_target_only', False):
            # 绕开 Stack 对 cubeB_main 的 body 解析（已移出模型, 新版 robosuite
            # body_name2id 缺失即抛 ValueError）; 跳档调 ManipulationEnv 的。
            super(suite.environments.manipulation.stack.Stack, self)._setup_references()
            self.cubeA_body_id = self.sim.model.body_name2id(self.cubeA.root_body)
            self.cubeB_body_id = -1
        else:
            super()._setup_references()

    def reward(self, action=None):
        if getattr(self, '_target_only', False):
            return 0.0  # 无 cubeB, Stack  shaping 无意义且会解析缺失 body
        return super().reward(action)

    def _check_success(self):
        if getattr(self, '_target_only', False):
            return False
        return super()._check_success()

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

        # 垫高台（方案 1）: arena 静态几何体（无 joint=焊接）, 位于放置区中心。
        # 台面顶 = table_offset.z + RISER_H; 物体经 sampler z_offset 落台面。
        # 2026-08-03 用户终裁 RISER_H=0: 物品不垫高, 不创建台面（零高 box 是
        # 退化几何, 必须整个跳过）。
        if RISER_H > 0:
            import xml.etree.ElementTree as ET
            riser_pos = [self.table_offset[0] + _RISER_CENTER_XY[0],
                         self.table_offset[1] + _RISER_CENTER_XY[1],
                         self.table_offset[2] + RISER_H / 2]
            riser_body = ET.SubElement(mujoco_arena.worldbody, "body",
                                       name="riser_platform",
                                       pos="{} {} {}".format(*riser_pos))
            ET.SubElement(riser_body, "geom", name="riser_top", type="box",
                          size="{} {} {}".format(_RISER_HALF[0], _RISER_HALF[1], RISER_H / 2),
                      rgba="0.82 0.78 0.72 1", condim="4",
                      friction="1 0.005 0.0001")

        tex_attrib = {"type": "cube"}
        mat_attrib = {"texrepeat": "1 1", "specular": "0.4", "shininess": "0.1"}
        redwood = CustomMaterial(texture="WoodRed", tex_name="redwood",
                                 mat_name="redwood_mat", tex_attrib=tex_attrib, mat_attrib=mat_attrib)
        greenwood = CustomMaterial(texture="WoodGreen", tex_name="greenwood",
                                   mat_name="greenwood_mat", tex_attrib=tex_attrib, mat_attrib=mat_attrib)
        # cubeA 细高化（2026-08-03 裁决 3）: 4×4×4 → 4×4×8cm, 接触点上移到
        # 4-6cm 高, 配合侧相机+低台组合拳。x/y 截面不变 → contact_dist 仍 0.04。
        self.cubeA = BoxObject(name="cubeA", size_min=[0.02, 0.02, 0.04], size_max=[0.02, 0.02, 0.04],
                               rgba=[1, 0, 0, 1], material=redwood)
        self.cubeB = BoxObject(name="cubeB", size_min=[0.025, 0.025, 0.025], size_max=[0.025, 0.025, 0.025],
                               rgba=[0, 1, 0, 1], material=greenwood)
        if self._target_only:
            # 2026-08-04 用户指令: 桌面只留目标物体 cubeA（去 cubeB 与全部杂物）。
            # cubeB 不进模型 → robosuite Stack 的 cubeB_body_id=-1, reward 为
            # 垃圾值但本管线不使用; check_contact/观测传感器空转不炸。
            self.distractors = []
            objects = [self.cubeA]
        else:
            # 杂物尺寸软约束（P4, 面向未来多物体任务）: 新增杂物尽量至少有一个
            # 维度 ≤4cm —— 让杂物本身也可被 Piper 抓取（净开度 70mm,
            # 2026-08-04 修正自误测值 4.49cm, 详见 gripper.xml/cgn_container.md）。
            # 不强制、不影响现有布局与 seed 复现; 当前 dist_cyl 长轴(7cm) 超限
            # 仅作背景, 其抓取由宽度过滤自然淘汰。
            self.distractors = [
                BoxObject(name="dist_box1", size_min=[0.02, 0.02, 0.03], size_max=[0.02, 0.02, 0.03],
                          rgba=[0.1, 0.25, 0.8, 1]),
                BoxObject(name="dist_box2", size_min=[0.015, 0.03, 0.02], size_max=[0.015, 0.03, 0.02],
                          rgba=[0.5, 0.5, 0.5, 1]),
                CylinderObject(name="dist_cyl", size=[0.018, 0.035], rgba=[0.9, 0.8, 0.1, 1]),
                BallObject(name="dist_ball", size=[0.022], rgba=[0.5, 0.1, 0.6, 1]),
            ][:self._clutter]   # 2026-08-05: 数量可控（0=清台只留方块, 默认 4 不变）
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
            z_offset=RISER_H + 0.01,   # 落垫高台面（原 0.01=桌面直放）
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
                 clutter: int | None = None, target_only: bool = False,
                 official_stack: bool = False):
        self._clutter = (4 if task == "Stack" else 0) if clutter is None else clutter
        self._target_only = target_only  # 桌面只留 cubeA（2026-08-04 用户指令）
        # 官方 Stack 原样导入（2026-08-06 用户指令: 双 4cm 正方体, 非裁决3细高化）
        self._official_stack = official_stack
        _ensure_vision_server()
        # 须在 super().__init__ 之前: 基类构造内 _post_reset → _tick 会读该属性
        self._live_viewer = None  # --render 时懒启动的交互窗口（_open_live_viewer）
        super().__init__(task=task, seed=seed, trace_root=trace_root,
                         render=render, render_slowdown=render_slowdown)

    def close(self):
        # 可视化窗口清理（2026-08-06 封装）: 先关 viewer 再走落盘流程
        v = getattr(self, "_live_viewer", None)
        if v is not None:
            try:
                v.close()
            except Exception:
                pass
            self._live_viewer = None
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
        # 2026-08-05 用户指令: Stack 一律走 StackClutter（clutter 数量透传,
        # 0=清台 distractors=[]）——旧分支 clutter=0 → plain Stack 会静默退回
        # 4cm cubeA, 与 2026-08-03 裁决 3（细高化 4×4×8）冲突, 已致一次误判。
        if self.task == "Stack" and not self._official_stack:
            env = StackClutter(clutter=self._clutter, target_only=self._target_only, **kwargs)
        else:
            # official_stack=True 时 Stack 也走这里: 官方 suite.make 原样导入,
            # 放置采样器由下方 plain-Stack 替换分支适配 Piper 臂展（仅改落点）。
            env = suite.make(self.task, **kwargs)
        # 预置离屏缓冲区 1280×960: MJCF <visual> 在 robosuite 合并时被丢弃
        # (实测 readback 640×480), 程序化设置才生效。首个超限 render 由
        # robosuite update_offscreen_size 一次性重建 context 到该尺寸
        # (max(请求, 预置)), 之后任何 ≤1280×960 的渲染不再重建。
        env.sim.model._model.vis.global_.offwidth = 1280
        env.sim.model._model.vis.global_.offheight = 960
        # 放置采样器: 2026-08-06 用户指令——official_stack 完全用 robosuite
        # 官方默认策略（Stack._load_model 自建 UniformRandomSampler:
        # 桌心 x/y ±0.08 方形 + 全随机 yaw）, 不做任何外部替换
        # （此前的 CUBE_X/Y 矩形与 DiscTableSampler 圆盘逻辑均已移除）。
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

    def _open_live_viewer(self):
        """launch_passive 直开交互窗口（2026-08-06 用户指令封装, 移植自
        scripts/cgn_execute_grasp.py --view 的 08-04/05 已验证实现）。

        不用 robosuite env.render(): 1.5.2 的 mjviewer renderer render() 是
        no-op（pass）, 真正的开窗/sync 在其 update() 且仅 env.step() 调用——
        本引擎直写 ctrl 走 sim.step(), 故必须自管 viewer。
        MUJOCO_GL 说明: robosuite import 时强制 egl（其 binding_utils）, 离屏
        渲染走 egl（wedge 自愈机制覆盖）; 窗口由 mujoco viewer 独立走 glfw,
        与 MUJOCO_GL 取值无关, 无需也不应为此改 MUJOCO_GL。
        """
        import mujoco.viewer
        self._live_viewer = mujoco.viewer.launch_passive(
            self.env.sim.model._model, self.env.sim.data._data)
        print("[engine_capx] 交互窗口已开（关窗不中断执行, 自动转无窗口跑完）", flush=True)

    def hold_viewer_open(self):
        """执行结束后保持结果场景显示（关窗退出）。窗口未开/已关则直接返回。"""
        v = getattr(self, "_live_viewer", None)
        if v is None or not v.is_running():
            return
        print("[engine_capx] 执行结束, 结果场景保持显示（关窗退出）", flush=True)
        while v.is_running():
            v.sync()
            _time.sleep(0.05)

    @staticmethod
    def _scn_add_connector(scn, A, B, width, rgba):
        """向 viewer.user_scn 追加一根胶囊线段（移植自 cgn_execute_grasp --view）。"""
        if scn.ngeom >= scn.maxgeom:
            return
        g = scn.geoms[scn.ngeom]
        mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                            np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
        mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                             np.asarray(A, float), np.asarray(B, float))
        scn.ngeom += 1

    def draw_grasp_glyphs(self, grasps_base, scores=None, max_show=8, highlight=0):
        """把抓取候选画进可视化窗口（2026-08-06 用户指令①: CGN 输出 3D 可视化）。

        画法移植自 cgn_execute_grasp.py --view 的用户审定版（品红=highlight 位,
        绿=其余; 刺+底杠+双指, 满开 70mm=PIPER_MAX_WIDTH）。输入为 plan_grasp
        输出（基座系 site 约定: +z 背向物体, 指尖朝 -z; user_scn 为世界系,
        内部做基座→世界变换）。仅窗口开着时有效；每次调用先清旧 glyph。
        highlight: 品红高亮的清单下标（越界则全绿; 2026-08-06 用户要求:
        品红跟踪正在执行的候选）。
        """
        v = getattr(self, "_live_viewer", None)
        if v is None or not v.is_running():
            return
        scn = v.user_scn
        scn.ngeom = 0
        for vi in range(min(max_show, len(grasps_base))):
            Tv = self.T_world_base @ np.asarray(grasps_base[vi], dtype=float)
            O, Sz, Sy = Tv[:3, 3], Tv[:3, 2], Tv[:3, 1]
            C = O - Sz * 0.050   # 指尖极值（用户审定 -50mm）
            Pp = O + Sz * 0.023  # 掌心可见表面（用户审定 +23mm）
            rgba = ((1.0, 0.0, 1.0, 0.8) if vi == highlight
                    else (0.0, 1.0, 0.2, 0.45))
            for A, B in [(Pp - Sy * 0.035, Pp + Sy * 0.035),
                         (Pp + Sy * 0.035, C + Sy * 0.035),
                         (Pp - Sy * 0.035, C - Sy * 0.035),
                         (Pp, Pp + Sz * 0.05)]:
                self._scn_add_connector(scn, A, B, 0.004, rgba)
        v.sync()

    def _tick(self, render_frames: bool = True):
        """一拍: 写 ctrl → SUBSTEPS 次 sim.step。

        观测渲染（512×512 双相机）是 EGL 负载大头: 运动拍不刷新图像观测,
        只在帧流边界渲染（实测高频 512 双相机渲染会反复 wedge EGL context）。
        关节状态判断走 current_arm_qpos（sim.data 实时值, 无需渲染）。

        可视化（2026-08-06 用户指令封装, = cgn_execute_grasp --view 已验证模式）:
        --render 时运动拍连帧流渲染也跳过（640 双相机 0.1-0.3s/次是"只有两三帧"
        卡顿感的来源; 边界帧仍由 trace 包装捕获）, 每拍 sync 窗口 + 实时配速
        （render_slowdown=1 → 20Hz 实时; 0.5 → 2 倍速, 08-05 用户定稿手感）;
        窗口被关自动转无窗口继续运行。headless 路径（render=False）行为不变。
        """
        if self.terminated:
            return
        t_tick0 = 0.0
        if self.render:
            render_frames = False
            t_tick0 = _time.perf_counter()
            if self._live_viewer is None:
                self._open_live_viewer()
        self._write_ctrl()
        for _ in range(SUBSTEPS):
            self.env.sim.step()
        self.sim_step += 1
        if render_frames and self.sim_step % FRAME_INTERVAL == 0:
            self.refresh_obs()
            self.capture_boundary(self.sim_step, include_depth=False)
        if self.render:
            v = self._live_viewer
            if v is not None and v.is_running():
                v.sync()
                # 实时配速: 补足到 render_slowdown/control_freq 秒
                budget = self.render_slowdown / getattr(self.env, "control_freq", 20)
                dt = _time.perf_counter() - t_tick0
                if dt < budget:
                    _time.sleep(budget - dt)
            else:
                self.render = False
                print("[engine_capx] 可视化窗口已关闭, 转为无窗口继续运行", flush=True)
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
        from ..api.primitives_capx import build_namespace

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
        # 量程贴合垫高台工作空间（2026-08-03 实测 seed16/18: 方块 mask 区
        # 0.42~0.74m, 全场 p50=0.55m, 远景 2.5m 截断为饱和色）——原 [0.3,2.0]
        # 把工作区压进 colormap 3% 区间, 物体侧面不可分。纯显示, CGN 消费
        # 原始深度不受影响。
        lo, hi = 0.40, 0.85
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
    parser.add_argument("--render-slowdown", type=float, default=1.0,
                        help="实时演示慢放倍率（如 2.0 = 半速, 便于观察抓取瞬间）")
    # 2026-08-05 用户指令（纯新增, 默认行为不变）: Stack 测试清台/留目标开关
    parser.add_argument("--clutter", type=int, default=None,
                        help="杂物数量（默认 None: Stack=4；0=清台只留方块）")
    parser.add_argument("--target-only", action="store_true",
                        help="桌面只留 cubeA（去 cubeB 与全部杂物）")
    # 2026-08-06 用户指令（纯新增, 默认行为不变）: 官方 Stack 原样导入
    parser.add_argument("--official-stack", action="store_true",
                        help="官方 Stack 配置（双 4cm 正方体, 绕开 StackClutter 细高化/杂物）")
    args = parser.parse_args()

    with open(args.code, encoding="utf-8") as f:
        code = f.read()

    _ensure_vision_server()

    engine = ExecutionEngineCapx(task=args.task, seed=args.seed, trace_root=args.trace_root,
                                 render=args.render, render_slowdown=args.render_slowdown,
                                 clutter=args.clutter, target_only=args.target_only,
                                 official_stack=args.official_stack)
    try:
        result = engine.run(code, code_ref=args.code)
        if args.render:
            engine.hold_viewer_open()  # 结果场景保持显示（关窗退出, 同 --view 体验）
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
