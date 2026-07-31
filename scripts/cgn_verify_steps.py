#!/usr/bin/env python
"""CGN 分段验证（可视化每步结果）。

用法:
    # 终端 1：启动 CGN 服务
    source external/cgn_venv/bin/activate
    python aspire/cgn_server.py

    # 终端 2：运行验证
    MUJOCO_GL=egl /home/stouching/anaconda3/envs/ASPIRE/bin/python scripts/cgn_verify_steps.py
"""
import os
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from aspire.engine_capx import ExecutionEngineCapx
from aspire.vision_client import grasp_cgn, segment_sam3_text_prompt
from aspire.primitives_capx import cgn_to_gripper, PrimitiveContextCapx


def visualize_step(grasps, title, gt_pos=None, save_path=None):
    """
    grasps: (N,4,4) 位姿数组
    title: 图标题
    gt_pos: (3,) 可选，方块真实位置（绿点）
    """
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # 画每个候选的原点和 Z 轴
    for i, g in enumerate(grasps[:3]):  # 只画 top-3
        origin = g[:3, 3]
        z_axis = g[:3, 2]

        # 原点
        ax.scatter(*origin, c='r', s=100, marker='o', label=f'G{i} origin' if i == 0 else '')
        # Z 轴箭头
        ax.quiver(*origin, *z_axis, length=0.05, color='b', alpha=0.6)

        print(f"{title} G{i}: pos={origin.round(4)}, z={z_axis.round(3)}")

    if gt_pos is not None:
        ax.scatter(*gt_pos, c='g', s=200, marker='*', label='GT cube')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(title)
    ax.legend()

    if save_path:
        plt.savefig(save_path)
        print(f"Saved to {save_path}")
    plt.close()


def main():
    engine = ExecutionEngineCapx(task="Stack", seed=0)
    ctx = PrimitiveContextCapx(engine)
    obs = ctx.get_observation()

    # 获取数据
    cam = obs["robot0_robotview"]
    rgb = cam["images"]["rgb"]
    depth = cam["images"]["depth"]
    K = cam["intrinsics"]
    pose_mat = cam["pose_mat"]

    # 红方块 mask
    masks = segment_sam3_text_prompt(rgb, "red cube")
    seg = np.zeros(rgb.shape[:2], dtype=np.int32)
    if masks:
        # mask 为 uint8(0/255)，必须转 bool——否则 numpy 按整数行索引，seg 变成第 0/255 行
        seg[masks[0]["mask"] > 0] = 1
    else:
        print("WARNING: SAM3 未分割到红方块，尝试使用全图 seg=1")
        seg[:] = 1

    # CGN 推理
    try:
        grasps_cam, scores = grasp_cgn(depth, K, seg)
    except Exception as e:
        print(f"CGN 推理失败: {e}")
        engine.close()
        raise

    print(f"CGN returned {len(grasps_cam)} candidates")
    if len(grasps_cam) == 0:
        print("ERROR: CGN 未返回任何候选")
        engine.close()
        return

    # 获取 GT 方块位置（验证用）
    import mujoco
    model = engine.env.sim.model._model
    cid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "cubeA_main")
    gt_world = engine.env.sim.data.xpos[cid].copy()
    # engine.T_base_world 实际为 world→base（命名历史遗留）
    gt_base = (engine.T_base_world @ np.append(gt_world, 1.0))[:3]
    print(f"GT cube position (base): {gt_base}")

    os.makedirs("outputs/cgn_verify", exist_ok=True)

    # Step 1: CGN 原始输出（相机系）
    visualize_step(grasps_cam, "Step 1: CGN raw (cam frame)",
                   save_path="outputs/cgn_verify/step1_raw.png")

    # Step 2: 乘 pose_mat 后（基座系）
    T_flip = np.diag([1, -1, 1, 1])
    grasps_base = np.array([pose_mat @ T_flip @ g for g in grasps_cam])
    visualize_step(grasps_base, "Step 2: After pose_mat (base frame)",
                   gt_pos=gt_base, save_path="outputs/cgn_verify/step2_base.png")

    # Step 3: 完整变换后（基座系 TCP）
    grasps_final = np.array([cgn_to_gripper(g, pose_mat) for g in grasps_cam])
    visualize_step(grasps_final, "Step 3: Final (base frame, TCP)",
                   gt_pos=gt_base, save_path="outputs/cgn_verify/step3_final.png")

    # 计算 D2/D3
    best = grasps_final[0]
    d2 = np.linalg.norm(best[:3, 3] - gt_base)
    d3 = -best[2, 2]  # Z轴朝下程度
    print(f"\n{'='*50}")
    print(f"D2 (pos error): {d2*100:.2f} cm (target <2.5cm)")
    print(f"D3 (z down):    {d3:.3f} (target >0.5)")
    print(f"{'='*50}")

    if d2 < 0.025 and d3 > 0.5:
        print("✅ CGN 验证通过")
    else:
        print("❌ CGN 验证未通过，请检查坐标变换")

    engine.close()


if __name__ == "__main__":
    main()
