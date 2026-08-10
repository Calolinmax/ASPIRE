"""PiperWipeSpill —— 桌面棕色污渍擦拭场景（官方 WipeArena 污渍 + Stack 锁定桌几何）。

背景（2026-08-06）：测试 ASPIRE wipe 任务逻辑（open_details/primitive_api_wipe.py
参考代码）需要桌面上有一摊"棕色污渍"。官方 robosuite Wipe 环境（wipe.py）在
__init__ 断言 gripper_types=="WipingGripper"——会拆掉 Piper 默认夹爪，
grip_site（IK 目标）与 gripper0_right_gripper 执行器（引擎直写 ctrl）全灭，
不可用。本类做法 = 官方 wipe.py::_load_model / _reset_internal 函数体原样抄录
（WipeArena + base_xpos_offset 链，官方参考优先），仅四处有意的偏差：

  1) 走 ManipulationEnv 直建（绕开 Wipe.__init__ 的 WipingGripper 断言），
     Piper 默认夹爪保留——wipe 用闭合夹爪的指腹当"擦子";
  2) 桌几何 = Stack 锁定值 (0.8,0.8,0.05) / (0,0,0.8)——ROBOTVIEW_CAM_WORLD
     相机标定与该几何绑定（锁死, 仅人类可改），官方 Wipe 的 0.9 桌高不能用；
  3) coverage_factor 默认 0.25（官方 0.6 会把污渍撒到世界 ±0.24 =
     基座系 x∈[0.06,0.54]，大半不可达；0.25 → 世界 ±0.10 = 基座
     x∈[0.20,0.40]，落在已验证抓取带 [0.15,0.27] 近旁）；
     基座世界系锚点: PIPER_BASE_XPOS_TABLE=(-0.30,0,0.80)。
  4) 无 reward / 无早停（引擎直写 ctrl + sim.step，不走 env.step/_post_action）；
     成功判定改为**访问覆盖率**（见 _tally_visited）：官方 wiped_markers 依赖
     reward() 内的接触平面检测，本管线 reward 永不被调用，故自力更生。

注册：经 aspire.robots 包 import 时 register_env（engine_capx.py 已冻结，
该包是其模块加载期唯一的注册钩子）。
"""

from __future__ import annotations

import numpy as np
from robosuite.environments.base import register_env
from robosuite.environments.manipulation.manipulation_env import ManipulationEnv
from robosuite.models.arenas import WipeArena
from robosuite.models.tasks import ManipulationTask

# 访问判定几何（2026-08-07 Franka 擦板版）：
#   判定参考 = 擦板足迹矩形（PiperWiperGripper 的 wiper_pad; 12×5cm 官方尺寸
#   + marker 半径外扩）——marker 中心落入板面投影即被擦; 无擦板时回退
#   球形擦头（2.5cm 圆）/ grip_site 悬擦判定。被访问的 marker 即刻淡出
#   （官方 reward 里的 alpha→0 同款, 本管线 reward 不被调用, 故在这里做）
#   ——窗口里能看到污渍被逐条擦掉。
_VISIT_R = 0.025
_VISIT_Z_BALL = (0.805, 0.830)   # 擦头球心世界 z 带（接触时 ≈0.80+r=0.812）
_VISIT_Z_PAD = (0.805, 0.845)    # 擦板中心世界 z 带（板心 = 面 + 1.5cm 板厚）
_VISIT_Z_SITE = (0.80, 0.90)     # 无擦头回退: grip_site 世界 z 带（悬擦 0.845）
_PAD_HALF = (0.06 + 0.02, 0.025 + 0.02)   # 板半长/半宽 + marker 半径 2cm


class PiperWipeSpill(ManipulationEnv):
    """Stack 桌几何 + WipeArena 棕色污渍随机路径；Piper 默认夹爪；无 reward。"""

    # Stack 锁定桌几何（相机标定绑定, 锁死——见 engine_capx.ROBOTVIEW_CAM_WORLD）
    TABLE_FULL_SIZE = (0.8, 0.8, 0.05)
    TABLE_FRICTION = (1.0, 5e-3, 1e-4)
    TABLE_OFFSET = (0.0, 0.0, 0.8)

    def __init__(self, *args, reward_shaping=False, num_markers=100,
                 line_width=0.04, coverage_factor=0.25, two_clusters=False,
                 **kwargs):
        # reward_shaping: 引擎 _env_kwargs 固定传入, 本场景无 reward, 吞掉即可。
        # 擦拭工具末端（2026-08-06 用户指令: 拆掉夹爪换擦头）: 默认换
        # PiperWiperGripper（接口与 PiperGripper 一致, 见 piper_wiper_gripper.py）;
        # 显式传 gripper_types 可覆盖（回退悬擦判定仍可用）。
        kwargs.setdefault("gripper_types", "PiperWiperGripper")
        self.num_markers = int(num_markers)
        self.line_width = float(line_width)
        self.coverage_factor = float(coverage_factor)
        self.two_clusters = bool(two_clusters)
        self.table_full_size = self.TABLE_FULL_SIZE
        self.table_friction = self.TABLE_FRICTION
        self.table_offset = np.array(self.TABLE_OFFSET)
        # 访问累计（_get_observations 侧效应采样, 引擎每次边界捕获都会调观测）
        self._visited: set[int] = set()
        self._marker_pos = np.zeros((0, 3))
        self._marker_gids: list[int] = []
        self._site_id = -1
        self._ball_gid = -1
        self._pad_gid = -1
        super().__init__(*args, **kwargs)

    # ------------------------------------------------------------------
    # 模型（官方 wipe.py::_load_model 抄录；TableArena→WipeArena, 无 delta_height）
    # ------------------------------------------------------------------
    def _load_model(self):
        super()._load_model()

        # Adjust base pose accordingly（与官方同一句）
        xpos = self.robots[0].robot_model.base_xpos_offset["table"](self.table_full_size[0])
        self.robots[0].robot_model.set_base_xpos(xpos)

        # 官方: delta_height 采样叠加桌高——略（桌高 0.8 锁定, 不采样）
        mujoco_arena = WipeArena(
            table_full_size=self.table_full_size,
            table_friction=self.table_friction,
            table_offset=self.table_offset,
            table_friction_std=0,          # 官方采样" peg 摩擦"——markers 纯视觉, 无意义
            coverage_factor=self.coverage_factor,
            num_markers=self.num_markers,
            line_width=self.line_width,
            two_clusters=self.two_clusters,
            rng=self.rng,
        )

        # Arena always gets set to zero origin（与官方同一句）
        mujoco_arena.set_origin([0, 0, 0])

        # task includes arena, robot（无官方 Wipe 之外的物体）
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
        )

    def _reset_internal(self):
        super()._reset_internal()
        # 官方 wipe.py::_reset_internal 同一句: 非确定性 reset 时重撒污渍路径
        if not self.deterministic_reset:
            self.model.mujoco_arena.reset_arena(self.sim)
        self._visited = set()
        # 延迟缓存（2026-08-07 "一下全消失"事故根因）: reset_arena 只写
        # model.body_pos, body_xpos 要等 reset() 后续的 sim.forward() 才刷新
        # （robosuite base.py: _reset_internal → forward 的顺序实锤）——这里
        # 同步读会拿到全部 marker 叠在桌心的旧值, 球一过境全灭。故置 None,
        # 首次 tally（必然在 forward 之后）再取真实坐标。
        self._marker_pos = None
        self._marker_gids = []
        self._site_id = -1
        self._ball_gid = -1

    # ------------------------------------------------------------------
    # 访问覆盖统计（成功判定的数据来源）
    # ------------------------------------------------------------------
    def _safe_ids(self, kind: str, name: str) -> int:
        """robosuite 的 name2id 对缺失名字抛 ValueError（不返回 -1）——可选
        参考点（球/板/site 按所用夹爪并存其一）必须容错。"""
        try:
            fn = getattr(self.sim.model, f"{kind}_name2id")
            return fn(name)
        except (ValueError, KeyError):
            return -1

    def _cache_markers(self):
        """缓存 markers 世界坐标（静态, 每次 reset 后重取）、视觉 geom id
        （淡出用）与判定参考 id（擦板优先, 球头/无擦头回退）。"""
        pos, gids = [], []
        for m in self.model.mujoco_arena.markers:
            bid = self.sim.model.body_name2id(m.root_body)
            pos.append(np.array(self.sim.data.body_xpos[bid], dtype=np.float64))
            gids.append(self.sim.model.geom_name2id(m.visual_geoms[0]))
        self._marker_pos = np.array(pos) if pos else np.zeros((0, 3))
        self._marker_gids = gids
        self._site_id = self._safe_ids("site", "gripper0_right_grip_site")
        self._ball_gid = self._safe_ids("geom", "gripper0_right_wiper_ball")
        self._pad_gid = self._safe_ids("geom", "gripper0_right_wiper_pad")

    def _tally_visited(self):
        """判定参考进入访问带时, 把落在擦板足迹矩形（或擦头球/悬擦半径）内的
        marker 记为已访问并即刻淡出（官方 alpha→0 同款）。首次调用时缓存
        marker 坐标（此刻 body_xpos 已经 sim.forward 刷新, 见 _reset_internal 注释）。"""
        if self._marker_pos is None:
            self._cache_markers()
        if len(self._marker_pos) == 0:
            return
        if self._pad_gid >= 0:
            # Franka 擦板: 足迹 = 板长轴×短轴矩形（活姿态, 从 geom 位姿取轴）
            C = self.sim.data.geom_xpos[self._pad_gid]
            if not (_VISIT_Z_PAD[0] < C[2] <= _VISIT_Z_PAD[1]):
                return
            Rm = self.sim.data.geom_xmat[self._pad_gid].reshape(3, 3)
            a1 = Rm[:2, 0]; a1 = a1 / max(np.linalg.norm(a1), 1e-9)   # 板长轴(水平投影)
            a2 = Rm[:2, 1]; a2 = a2 / max(np.linalg.norm(a2), 1e-9)   # 板短轴
            d = self._marker_pos[:, :2] - C[:2]
            hit = np.nonzero((np.abs(d @ a1) < _PAD_HALF[0])
                             & (np.abs(d @ a2) < _PAD_HALF[1]))[0]
        elif self._ball_gid >= 0:
            p = self.sim.data.geom_xpos[self._ball_gid]
            if not (_VISIT_Z_BALL[0] < p[2] <= _VISIT_Z_BALL[1]):
                return
            d = np.linalg.norm(self._marker_pos[:, :2] - p[:2], axis=1)
            hit = np.nonzero(d < _VISIT_R)[0]
        elif self._site_id >= 0:
            p = self.sim.data.site_xpos[self._site_id]
            if not (_VISIT_Z_SITE[0] < p[2] <= _VISIT_Z_SITE[1]):
                return
            d = np.linalg.norm(self._marker_pos[:, :2] - p[:2], axis=1)
            hit = np.nonzero(d < _VISIT_R)[0]
        else:
            return
        new = [int(i) for i in hit if int(i) not in self._visited]
        if new:
            self._visited.update(new)
            for i in new:   # 官方 reward 同款: 擦掉的 marker 透明化
                self.sim.model.geom_rgba[self._marker_gids[i]][3] = 0.0

    @property
    def visit_fraction(self) -> float:
        return len(self._visited) / max(1, self.num_markers)

    def _get_observations(self, force_update=False):
        obs = super()._get_observations(force_update=force_update)
        self._tally_visited()   # 侧效应采样：引擎每次刷新观测都过这里
        return obs

    # ------------------------------------------------------------------
    # 奖励/成功（引擎 run() 末尾以 _check_success 定 success）
    # ------------------------------------------------------------------
    def reward(self, action=None):
        return 0.0

    def _check_success(self):
        n = len(self._visited)
        frac = self.visit_fraction
        print(f"[PiperWipeSpill] 污渍访问覆盖 {n}/{self.num_markers} = {frac:.0%}"
              f"（成功阈值 ≥50%）", flush=True)
        return frac >= 0.5


register_env(PiperWipeSpill)
