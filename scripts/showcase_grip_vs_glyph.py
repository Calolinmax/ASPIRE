#!/usr/bin/env python
"""同框对比（2026-08-03；2026-08-04 修正开度口径 + 实体开度跟随）: 真臂开到
top-1 CGN 候选位姿, 夹爪开到执行开度 w = min(cgn_open, 70mm), 与品红 glyph
叠加——二者应精确重合, 验证 CGN→glyph→实体夹爪 全链几何一致。

口径: 实体指垫内净距 = 2·q7（旧"满开 45mm"系坐标系混淆误测, 真值 70mm,
见 gripper.xml 注释）; CGN 开度是 Panda 几何逐候选预测值, 非 Piper 最大开度。

用法: MUJOCO_GL=glfw python scripts/showcase_grip_vs_glyph.py [seed]
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mujoco
import mujoco.viewer
import aspire.robots  # noqa: F401
from aspire.engine_capx import ExecutionEngineCapx, TUCK_Q
from aspire.primitives_capx import PrimitiveContextCapx, cgn_to_gripper, PIPER_MAX_WIDTH
from aspire.vision_client import segment_sam3_text_prompt, grasp_cgn
from robosuite.utils import transform_utils as T

SEED = int(sys.argv[1]) if len(sys.argv) > 1 else 5


def add_connector(scn, A, B, width, rgba):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(g, mujoco.mjtGeom.mjGEOM_CAPSULE, np.zeros(3), np.zeros(3),
                        np.eye(3).flatten(), np.array(rgba, dtype=np.float32))
    mujoco.mjv_connector(g, mujoco.mjtGeom.mjGEOM_CAPSULE, width,
                         np.asarray(A, float), np.asarray(B, float))
    scn.ngeom += 1


def main():
    engine = ExecutionEngineCapx(task='Stack', seed=SEED)
    ctx = PrimitiveContextCapx(engine)
    sim = engine.env.sim
    m = sim.model._model
    for i in range(6):
        sim.data.set_joint_qpos(f'robot0_joint{i+1}', TUCK_Q[i])
    mujoco.mj_forward(m, sim.data._data)

    obs = ctx.get_observation()
    cam = obs['robot0_robotview']
    rgb, depth, K, pose_mat = (cam['images']['rgb'], cam['images']['depth'],
                               cam['intrinsics'], cam['pose_mat'])
    masks = segment_sam3_text_prompt(rgb, 'red cube')
    seg = np.zeros(rgb.shape[:2], dtype=np.int32)
    if masks:
        seg[masks[0]['mask'] > 0] = 1
    g, s, o = grasp_cgn(depth, K, seg, return_openings=True)
    gb = cgn_to_gripper(g[0], pose_mat)
    w = min(float(o[0]), PIPER_MAX_WIDTH)   # 执行开度
    q7 = w / 2                              # 指垫内净距 = 2·q7（2026-08-04 实测模型）
    print(f'top-1: score={s[0]:.3f} cgn_open={o[0]*1000:.1f}mm → 执行开度 {w*1000:.1f}mm '
          f'(Piper 上限 {PIPER_MAX_WIDTH*1000:.0f}mm)')
    print('品红 glyph = 执行开度细线; 实体夹爪 = 同开度 q7=w/2 —— 二者应精确重合')

    # IK 解 top-1 候选 grasp 位姿, 真臂直接摆上去, 夹爪开到执行开度 w
    qw_xyzw = T.mat2quat(gb[:3, :3])
    qw = np.array([qw_xyzw[3], qw_xyzw[0], qw_xyzw[1], qw_xyzw[2]])
    q_sol = ctx.solve_ik(gb[:3, 3].copy(), qw)
    for i in range(6):
        sim.data.set_joint_qpos(f'robot0_joint{i+1}', q_sol[i])
    sim.data.set_joint_qpos('gripper0_right_joint7', q7)  # 内净距 = 2·q7 = w
    mujoco.mj_forward(m, sim.data._data)

    # 执行器同步（2026-08-04 根因修复）: viewer 循环持续 mj_step, 位置伺服的
    # ctrl 仍停留在 reset 时的构型——teleport 只改 qpos 不改 ctrl, 臂会在 ~1s
    # 内被伺服拖离 IK 解（实测漂移 173mm/2.46rad, z 抬高 16cm）——即此前
    # "掌心离物理表面很远"观感的来源。
    for a in range(m.nu):
        if m.actuator_trntype[a] == mujoco.mjtTrn.mjTRN_JOINT:
            jid = m.actuator_trnid[a, 0]
            sim.data.ctrl[a] = sim.data.qpos[m.jnt_qposadr[jid]]

    # 候选落点自检（CGN OOD 场景候选逐 run 不稳, 可能偏出方块抓空气）
    cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, 'cubeA_main')
    cube_w = sim.data.xpos[cid]
    Tw = engine.T_world_base @ gb
    print(f'目标 vs cubeA: 水平偏差 {np.linalg.norm(Tw[:3, 3][:2] - cube_w[:2])*1000:.1f}mm  '
          f'目标-顶面 {(Tw[2, 3] - (cube_w[2] + 0.04))*1000:+.1f}mm (>25mm 即偏出方块)')

    # glyph（执行开度细线, 品红; Tw 已在上文计算）
    O, Sz, Sy = Tw[:3, 3], Tw[:3, 2], Tw[:3, 1]
    C = O - Sz * 0.050  # 指尖极值（实测 -50mm, 用户审定版）
    Pp = O + Sz * 0.023  # 掌心可见表面（实测 +23mm）
    segs = [(Pp - Sy * w / 2, Pp + Sy * w / 2),
            (Pp + Sy * w / 2, C + Sy * w / 2),
            (Pp - Sy * w / 2, C - Sy * w / 2)]

    with mujoco.viewer.launch_passive(m, sim.data._data) as v:
        for A, B in segs:
            add_connector(v.user_scn, A, B, 0.004, (1.0, 0.0, 1.0, 0.9))
        while v.is_running():
            sim_obj = v._sim()
            if getattr(sim_obj, 'run', 1):
                mujoco.mj_step(m, sim.data._data)
            else:
                mujoco.mj_forward(m, sim.data._data)
            v.sync()
            time.sleep(0.004)
    engine.close()


if __name__ == '__main__':
    main()
