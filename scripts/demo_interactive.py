#!/usr/bin/env python
"""
交互式演示脚本：逐步执行抓取流程，每步暂停等待用户确认。
支持鼠标拖拽旋转视角、滚轮缩放。

操作：
- 左键拖拽：旋转视角
- 右键拖拽：平移视角
- 滚轮：缩放
- 空格键：继续下一步
- ESC：退出
"""

import numpy as np
import robosuite as suite
import mujoco.viewer

# 创建环境
env = suite.make(
    "Lift", robots="Panda",
    has_renderer=False,  # 我们用 mujoco.passive_viewer
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names="agentview",
    camera_heights=256, camera_widths=256, camera_depths=True,
    control_freq=20, horizon=1000,
)

obs = env.reset()
sim = env.sim
model = sim.model._model  # 原生 MjModel
data = sim.data._data      # 原生 MjData

print("=" * 60)
print("ASPIRE 抓取流程交互式演示")
print("=" * 60)
print("\n操作说明：")
print("  左键拖拽：旋转视角")
print("  右键拖拽：平移视角")
print("  滚轮：缩放")
print("  空格键：继续下一步")
print("  ESC：退出")
print("=" * 60)

# 使用 mujoco 的 passive viewer（支持交互）
with mujoco.viewer.launch_passive(sim.model._model, sim.data._data) as viewer:

    def step_and_sync(action, n=1):
        """执行 n 步仿真并同步 viewer"""
        for _ in range(n):
            obs, _, done, _ = env.step(action)
            viewer.sync()
        return obs

    def wait_key(msg="[自动继续...]"):
        """暂停5秒后继续"""
        print(f"\n{msg} (5秒)")
        import time
        for i in range(5, 0, -1):
            print(f"  {i}...", end="", flush=True)
            time.sleep(1)
        print(" 继续!")
        return True

    # 第 0 步：初始状态
    print("\n【步骤 0】初始状态")
    print(f"  末端位置: {obs['robot0_eef_pos'].round(4)}")
    print(f"  夹爪开合: {obs['robot0_gripper_qpos'][0]:.4f}")
    print(f"  Cube 位置: {obs['cube_pos'].round(4)}")
    if not wait_key():
        print("退出")
        exit()

    # 第 1 步：感知定位（get_observation + segment_sam3_text_prompt）
    print("\n" + "=" * 60)
    print("【步骤 1】感知定位 - 调用 get_observation()")
    print("=" * 60)
    print("API: get_observation()")
    print("  → 返回观测字典（包含 RGB、depth、相机内外参、关节状态等）")
    obs = env._get_observations()
    print(f"  RGB 图像 shape: {obs['agentview_image'].shape}")
    print(f"  Depth 图像 shape: {obs['agentview_depth'].shape}")
    print(f"  相机内参 K shape: (3,3)")

    print("\n" + "-" * 50)
    print("【算法详解】segment_sam3_text_prompt(rgb, 'cube')")
    print("-" * 50)
    print("输入: RGB图像(256x256), 文本提示='cube'")
    print("\nGT冒充版实现流程:")
    print("  1) 分割图获取: get_camera_segmentation() → (H,W,2) 数组")
    print("     - channel0: geom类型 (5=mesh, 6=box等)")
    print("     - channel1: geom ID")
    print("  2) 对象匹配: 查找所有属于 'cube_main' body 的 geom IDs")
    print("     - 通过 body_id 关联 geom_bodyid[geom_id]")
    print("  3) 二值掩码: mask = (seg[:,:,1] ∈ cube_geom_ids)")
    print("\n坐标系对齐(关键!):")
    print("  - MuJoCo 3.x render: row0=top (无需翻转)")
    print("  - get_camera_segmentation: 内部多翻了一次 [::-1]")
    print("  - 修复: 调用后再翻转一次 [::-1] 与obs对齐")
    print("\n输出: [{'mask': (256,256) uint8, 'score': 0.99}]")

    # 模拟 mask 生成
    from robosuite.utils import camera_utils as CU
    seg = CU.get_camera_segmentation(sim, "agentview", 256, 256)[::-1]
    import mujoco
    bid = mujoco.mj_name2id(sim.model._model, mujoco.mjtObj.mjOBJ_BODY, "cube_main")
    true_gids = [g for g in range(sim.model._model.ngeom)
                 if sim.model._model.geom_bodyid[g] == bid]
    mask = np.isin(seg[:,:,1].astype(int), true_gids).astype(np.uint8)
    print(f"\n  实际结果: Mask 像素数 = {mask.sum()}")

    print("\n" + "-" * 50)
    print("【算法详解】mask_to_world_points(mask, depth, K, E)")
    print("-" * 50)
    print("输入: mask(2D), depth(H,W), K(3,3内参), E(4,4外参)")
    print("\n反投影公式 (对每个mask像素(u,v)):")
    print("  1) 读取深度: z = depth[v, u] (米制, 已转real depth)")
    print("  2) 相机坐标: x_cam = (u - cx) * z / fx")
    print("              y_cam = -(v - cy) * z / fy  ← 关键负号!")
    print("              z_cam = z")
    print("     (负号原因: 图像v向下, 但MuJoCo相机y向上)")
    print("  3) 世界坐标: [x_w, y_w, z_w, 1] = E @ [x_cam, y_cam, z_cam, 1]")
    print("\n数学推导:")
    print("  - 内参K: [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]")
    print("  - 外参E: cam→world 的4x4齐次变换矩阵")
    print("  - 注意: 不是世界→相机, 是相机→世界!")

    d_real = CU.get_real_depth_map(sim, obs['agentview_depth'].squeeze(-1))
    K = CU.get_camera_intrinsic_matrix(sim, "agentview", 256, 256)
    E = CU.get_camera_extrinsic_matrix(sim, "agentview")
    vs, us = np.nonzero(mask)
    z = d_real[vs, us]
    x = (us - K[0,2]) * z / K[0,0]
    y = -(vs - K[1,2]) * z / K[1,1]
    pts = (np.asarray(E) @ np.stack([x, y, z, np.ones_like(z)]))[:3].T

    print(f"\n  点云统计:")
    print(f"    总点数: {len(pts)}")
    print(f"    x范围: [{pts[:,0].min():.4f}, {pts[:,0].max():.4f}]")
    print(f"    y范围: [{pts[:,1].min():.4f}, {pts[:,1].max():.4f}]")
    print(f"    z范围: [{pts[:,2].min():.4f}, {pts[:,2].max():.4f}]")

    print("\n" + "-" * 50)
    print("【抓取点计算】为什么选择 min_z + 8mm?")
    print("-" * 50)
    print("点云高度分布分析:")
    z_sorted = np.sort(pts[:, 2])
    print(f"  p5 (底面):  {np.percentile(pts[:,2], 5):.4f}")
    print(f"  p50(中心):  {np.percentile(pts[:,2], 50):.4f}")
    print(f"  p95(顶面):  {np.percentile(pts[:,2], 95):.4f}")

    center = np.median(pts, axis=0)
    min_z = float(np.percentile(pts[:, 2], 5))
    grasp_z = min_z + 0.008

    print(f"\n  计算过程:")
    print(f"    center (中位数) = [{center[0]:.4f}, {center[1]:.4f}, {center[2]:.4f}]")
    print(f"    min_z (p5)      = {min_z:.4f}")
    print(f"    grasp_z         = min_z + 0.008 = {grasp_z:.4f}")

    print(f"\n【关键决策】为什么不直接用顶面抓取?")
    print(f"  - 若用 top_z={np.percentile(pts[:,2], 95):.4f}: 夹爪夹持cube上部")
    print(f"  - 问题: 夹持力矩大, cube易滑落(trace seed1的失败案例)")
    print(f"  - 改进: 夹持下部(min_z+8mm), 重心在夹爪下方, 稳定性高")
    print(f"  - 8mm: 经验值, 略高于min_z避免碰撞桌面, 又不会太高")

    if not wait_key():
        print("退出")
        exit()

    # 第 2 步：张开夹爪
    print("\n" + "=" * 60)
    print("【步骤 2】张开夹爪 - 调用 open_gripper()")
    print("=" * 60)
    print("API: open_gripper()")
    print("  → 发送夹爪张开指令，阻塞执行")
    action = np.zeros(7); action[6] = -1.0
    obs = step_and_sync(action, 12)
    print(f"  夹爪状态: {obs['robot0_gripper_qpos'][0]:.4f} (负值=张开)")

    if not wait_key():
        print("退出")
        exit()

    # 第 3 步：移动到预抓位置
    print("\n" + "=" * 60)
    print("【步骤 3】移动至预抓位置 - 调用 move_to_pose()")
    print("=" * 60)
    print("API: move_to_pose(pos, quat)")
    print("  → OSC_POSE 控制器闭环：位置误差 × KP 得 delta 动作")
    print("  → 阻塞式：到达目标 5mm 范围内才返回")
    print(f"  目标: ({center[0]:.3f}, {center[1]:.3f}, {grasp_z + 0.12:.3f})")

    QUAT_TOP_DOWN = np.array([0.0, 1.0, 0.0, 0.0])
    target = np.array([center[0], center[1], grasp_z + 0.12])
    # P 控制移动到目标
    for _ in range(100):
        eef = obs['robot0_eef_pos']
        err = target - eef
        if np.linalg.norm(err) < 0.005:
            break
        action = np.zeros(7)
        action[:3] = np.clip(5.0 * err, -1, 1)
        action[6] = -1.0  # 保持张开
        obs = step_and_sync(action, 1)
    print(f"  到达位置: {obs['robot0_eef_pos'].round(4)}")

    if not wait_key():
        print("退出")
        exit()

    # 第 4 步：下降到位
    print("\n" + "=" * 60)
    print("【步骤 4】下降抓取 - 调用 move_to_pose()")
    print("=" * 60)
    print(f"  目标: ({center[0]:.3f}, {center[1]:.3f}, {grasp_z:.3f})")
    print("  关键：抓取 cube 下半部（min_z + 0.8cm），避免夹上部滑落")

    target = np.array([center[0], center[1], grasp_z])
    for _ in range(80):
        eef = obs['robot0_eef_pos']
        err = target - eef
        if np.linalg.norm(err) < 0.005:
            break
        action = np.zeros(7)
        action[:3] = np.clip(5.0 * err, -1, 1)
        action[6] = -1.0
        obs = step_and_sync(action, 1)
    print(f"  到达位置: {obs['robot0_eef_pos'].round(4)}")
    print(f"  末端与 cube 距离: {np.linalg.norm(obs['robot0_eef_pos'] - obs['cube_pos']):.4f}")

    if not wait_key():
        print("退出")
        exit()

    # 第 5 步：闭合夹爪
    print("\n" + "=" * 60)
    print("【步骤 5】闭合夹爪 - 调用 close_gripper()")
    print("=" * 60)
    print("API: close_gripper()")
    print("  → 发送夹爪闭合指令，阻塞执行")
    print("  → 物理夹紧需要多步仿真（约 12 步）")

    print("\n" + "-" * 50)
    print("【算法详解】夹爪控制与力度感知")
    print("-" * 50)
    print("夹爪模型:")
    print("  - Panda 平行二指夹爪")
    print("  - action[6] ∈ [-1, 1]: -1=张开, +1=闭合")
    print("  - 物理仿真: 位置控制, 有弹簧阻尼模型")
    print("\n力度控制策略:")
    print("  1) 持续发送闭合指令(action[6]=+1) 12步")
    print("  2) 夹爪会尽可能闭合, 遇到物体则挤压")
    print("  3) 通过观测判断夹持状态:")
    print("     gripper_qpos[0]: 夹爪开口宽度")
    print("     - 若 > 0.003 (3mm): 有物体撑开, 夹持成功")
    print("     - 若 ≈ 0: 空抓")

    action = np.zeros(7); action[6] = 1.0
    obs = step_and_sync(action, 12)
    grip_q = obs['robot0_gripper_qpos'][0]
    print(f"\n  夹爪开口: {grip_q:.4f}")
    print(f"  判断: {'✅ 夹持成功' if grip_q > 0.003 else '❌ 空抓!'}")

    print("\n【力度把控的关键】")
    print("  - MuJoCo夹爪没有直接的'力传感器'")
    print("  - 间接指标: gripper_qpos 被撑开的程度")
    print("  - 摩擦依赖: 夹持位置(cube下部) + 摩擦系数")
    print("  - ASPIRE修复策略: 若空抓, 调整 grasp_z 重试")

    if not wait_key():
        print("退出")
        exit()

    # 第 6 步：抬起
    print("\n" + "=" * 60)
    print("【步骤 6】抬起 - 调用 move_to_pose()")
    print("=" * 60)
    print(f"  目标高度: {grasp_z + 0.22:.3f} (抬起 22cm)")

    target = np.array([center[0], center[1], grasp_z + 0.22])
    for _ in range(100):
        eef = obs['robot0_eef_pos']
        err = target - eef
        if np.linalg.norm(err) < 0.005:
            break
        action = np.zeros(7)
        action[:3] = np.clip(5.0 * err, -1, 1)
        action[6] = 1.0  # 保持闭合
        obs = step_and_sync(action, 1)

    print(f"  最终末端位置: {obs['robot0_eef_pos'].round(4)}")
    print(f"  Cube 位置: {obs['cube_pos'].round(4)}")
    print(f"  Cube 高度: {obs['cube_pos'][2]:.4f}")
    lifted = obs['cube_pos'][2] > 0.9
    print(f"\n  {'✅ 抓取成功！' if lifted else '❌ 抓取失败'}")

    print("\n" + "=" * 60)
    print("演示结束。按 ESC 退出或关闭窗口。")
    print("=" * 60)

    # 保持窗口
    import time
    while viewer.is_running():
        time.sleep(0.05)

print("已退出")
