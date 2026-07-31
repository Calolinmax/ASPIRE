#!/usr/bin/env python
"""CGN 探针: Stack 场景加 4 个干扰物体（N2 资产）。

背景: CGN 训练分布 = 多物杂乱场景 + RealSense 噪声。干净单方块 sim 是双重 OOD，
方块面 contact score 被压在阈值下（0.16-0.24 vs first_thres 0.23）→ 0 候选。
加杂物后方块面分数升至 ~0.265，端到端可出候选（姿态相关，约 2/5）。

用法:
    MUJOCO_GL=egl python scripts/cgn_probe_clutter.py
输出:
    服务端候选数 + top5 的 D2/D3（基座系, 对照 GT 方块）+ /tmp/cgn_scene_clutter.npz
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import aspire.robots  # noqa: F401 注册 Piper
import aspire.engine_capx as E
from aspire.engine_capx import ExecutionEngineCapx
from aspire.primitives_capx import PrimitiveContextCapx, cgn_to_gripper
from aspire.vision_client import segment_sam3_text_prompt, grasp_cgn

from robosuite.environments.manipulation.stack import Stack
from robosuite.models.objects import BoxObject, CylinderObject, BallObject
from robosuite.models.arenas import TableArena
from robosuite.models.tasks import ManipulationTask
from robosuite.utils.placement_samplers import UniformRandomSampler
from robosuite.utils.mjcf_utils import CustomMaterial


class StackClutter(Stack):
    """Stack + 4 个干扰物体（蓝盒/灰盒/黄柱/紫球），不与目标重叠（sampler 保证）。"""

    def _load_model(self):
        super(Stack, self)._load_model()  # MujocoEnv 级加载（跳过 Stack 版）

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
        self.distractors = [
            BoxObject(name="dist_box1", size_min=[0.02, 0.02, 0.03], size_max=[0.02, 0.02, 0.03],
                      rgba=[0.1, 0.25, 0.8, 1]),
            BoxObject(name="dist_box2", size_min=[0.015, 0.03, 0.02], size_max=[0.015, 0.03, 0.02],
                      rgba=[0.5, 0.5, 0.5, 1]),
            CylinderObject(name="dist_cyl", size=[0.018, 0.035], rgba=[0.9, 0.8, 0.1, 1]),
            BallObject(name="dist_ball", size=[0.022], rgba=[0.5, 0.1, 0.6, 1]),
        ]
        objects = [self.cubeA, self.cubeB] + self.distractors

        self.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=objects,
            x_range=[-0.08, 0.08],
            y_range=[-0.08, 0.08],
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=self.table_offset,
            z_offset=0.01,
            rng=self.rng,
        )
        self.model = ManipulationTask(
            mujoco_arena=mujoco_arena,
            mujoco_robots=[robot.robot_model for robot in self.robots],
            mujoco_objects=objects,
        )


class ClutterEngine(ExecutionEngineCapx):
    def _make_env(self, kwargs):
        env = StackClutter(**kwargs)
        env.placement_initializer = UniformRandomSampler(
            name="ObjectSampler",
            mujoco_objects=[env.cubeA, env.cubeB] + env.distractors,
            x_range=list(E.CUBE_X_RANGE),
            y_range=list(E.CUBE_Y_RANGE),
            rotation=None,
            ensure_object_boundary_in_range=False,
            ensure_valid_placement=True,
            reference_pos=env.table_offset,
            z_offset=0.01,
            rng=env.rng,
        )
        return env


def main():
    import mujoco

    engine = ClutterEngine(task="Stack", seed=0)
    ctx = PrimitiveContextCapx(engine)
    obs = ctx.get_observation()
    cam = obs["robot0_robotview"]
    rgb = cam["images"]["rgb"]
    depth = cam["images"]["depth"]
    K = cam["intrinsics"]
    pose_mat = cam["pose_mat"]

    masks = segment_sam3_text_prompt(rgb, "red cube")
    seg = np.zeros(rgb.shape[:2], dtype=np.int32)
    if masks:
        seg[masks[0]["mask"] > 0] = 1
        print('SAM3 mask pixels:', int((masks[0]["mask"] > 0).sum()))
    else:
        print('WARNING: SAM3 no mask, fallback seg[:]=1')
        seg[:] = 1

    d = depth[seg == 1]
    print('depth in seg: min %.4f max %.4f mean %.4f' % (d.min(), d.max(), d.mean()))

    grasps, scores = grasp_cgn(depth, K, seg)
    print('service path: grasps', grasps.shape, 'scores',
          scores.round(4) if len(scores) else scores)

    model_ = engine.env.sim.model._model
    cid = mujoco.mj_name2id(model_, mujoco.mjtObj.mjOBJ_BODY, "cubeA_main")
    gt_world = engine.env.sim.data.xpos[cid].copy()
    gt_base = (engine.T_base_world @ np.append(gt_world, 1.0))[:3]
    print('GT cube (base):', gt_base.round(4))

    if len(grasps):
        grasps_final = np.array([cgn_to_gripper(g, pose_mat) for g in grasps])
        for i, g in enumerate(grasps_final[:5]):
            d2i = np.linalg.norm(g[:3, 3] - gt_base)
            print(f'G{i}: pos(base)={g[:3, 3].round(4)}  D2={d2i * 100:.2f} cm  '
                  f'D3={-g[2, 2]:.3f}  score={scores[i]:.4f}')

    np.savez('/tmp/cgn_scene_clutter.npz',
             depth=np.asarray(depth, dtype=np.float32),
             K=np.asarray(K, dtype=np.float64),
             seg=seg.astype(np.int32), rgb=rgb)
    print('saved /tmp/cgn_scene_clutter.npz')
    engine.close()


if __name__ == '__main__':
    main()
