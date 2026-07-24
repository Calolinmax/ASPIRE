#!/usr/bin/env python
"""
双相机 + MobileSAM ONNX GPU 演示
顶部相机全局定位 + 腕部相机局部观察
"""

import numpy as np
import cv2
import robosuite as suite
import sys
import time

sys.path.insert(0, '/home/stouching/Desktop/ASPIRE')

from aspire.vision_sam_onnx import segment_sam3_text_prompt, warmup

print("=" * 60)
print("双相机 + MobileSAM ONNX GPU 演示")
print("=" * 60)

# 创建环境（双相机）
env = suite.make(
    "Lift", robots="Panda",
    has_renderer=False,
    has_offscreen_renderer=True,
    use_camera_obs=True,
    camera_names=["agentview", "robot0_eye_in_hand"],  # 双相机！
    camera_heights=256, camera_widths=256,
    camera_depths=True,
    control_freq=20, horizon=500,
)

obs = env.reset()

# 预热 MobileSAM
print("\n预热 MobileSAM ONNX GPU...")
warmup()

# 获取双相机图像
top_img = obs["agentview_image"]
wrist_img = obs["robot0_eye_in_hand_image"]

print(f"\n顶部相机: {top_img.shape}")
print(f"腕部相机: {wrist_img.shape}")

# 保存初始双视角
combined = np.hstack([top_img, wrist_img])
cv2.imwrite("/tmp/dual_cam_init.jpg", cv2.cvtColor(combined, cv2.COLOR_RGB2BGR))
print("\n已保存初始双视角: /tmp/dual_cam_init.jpg")

# ========== 顶部相机：全局定位 ==========
print("\n" + "=" * 60)
print("【顶部相机】MobileSAM 分割 'cube'")
print("=" * 60)

start = time.time()
results_top = segment_sam3_text_prompt(top_img, "cube")
elapsed_top = time.time() - start

print(f"推理时间: {elapsed_top*1000:.1f} ms (GPU)")
print(f"检测到 {len(results_top)} 个候选")

if results_top:
    best = results_top[0]
    print(f"\n最佳 mask:")
    print(f"  置信度: {best['score']:.3f}")
    print(f"  面积: {best['area']} 像素")
    print(f"  中心: ({best['centroid'][0]:.1f}, {best['centroid'][1]:.1f})")
    print(f"  长宽比: {best['aspect_ratio']:.2f}")

    # 可视化顶部相机分割结果
    vis_top = top_img.copy()
    mask = best["mask"]
    mask_colored = np.zeros_like(vis_top)
    mask_colored[mask > 128] = [0, 255, 0]  # 绿色 mask
    vis_top = cv2.addWeighted(vis_top, 0.7, mask_colored, 0.3, 0)

    cx, cy = int(best['centroid'][0]), int(best['centroid'][1])
    cv2.circle(vis_top, (cx, cy), 5, (255, 0, 0), -1)
    cv2.putText(vis_top, f"cube ({best['score']:.2f})", (cx-40, cy-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite("/tmp/top_cam_sam.jpg", cv2.cvtColor(vis_top, cv2.COLOR_RGB2BGR))
    print("  已保存: /tmp/top_cam_sam.jpg")

# ========== 腕部相机：局部观察 ==========
print("\n" + "=" * 60)
print("【腕部相机】MobileSAM 分割 'cube'")
print("=" * 60)

start = time.time()
results_wrist = segment_sam3_text_prompt(wrist_img, "cube")
elapsed_wrist = time.time() - start

print(f"推理时间: {elapsed_wrist*1000:.1f} ms (GPU)")
print(f"检测到 {len(results_wrist)} 个候选")

if results_wrist:
    best = results_wrist[0]
    print(f"\n最佳 mask:")
    print(f"  置信度: {best['score']:.3f}")
    print(f"  面积: {best['area']} 像素")
    print(f"  中心: ({best['centroid'][0]:.1f}, {best['centroid'][1]:.1f})")

    # 可视化腕部相机分割结果
    vis_wrist = wrist_img.copy()
    mask = best["mask"]
    mask_colored = np.zeros_like(vis_wrist)
    mask_colored[mask > 128] = [0, 255, 0]
    vis_wrist = cv2.addWeighted(vis_wrist, 0.7, mask_colored, 0.3, 0)

    cx, cy = int(best['centroid'][0]), int(best['centroid'][1])
    cv2.circle(vis_wrist, (cx, cy), 5, (255, 0, 0), -1)
    cv2.putText(vis_wrist, f"cube ({best['score']:.2f})", (cx-40, cy-10),
               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

    cv2.imwrite("/tmp/wrist_cam_sam.jpg", cv2.cvtColor(vis_wrist, cv2.COLOR_RGB2BGR))
    print("  已保存: /tmp/wrist_cam_sam.jpg")
else:
    print("  腕部视角未检测到 cube（夹爪距离太远）")

# ========== 双相机对比 ==========
print("\n" + "=" * 60)
print("双相机检测对比")
print("=" * 60)
print(f"顶部相机: {len(results_top)} 个 mask, 推理 {elapsed_top*1000:.1f}ms")
print(f"腕部相机: {len(results_wrist)} 个 mask, 推理 {elapsed_wrist*1000:.1f}ms")

print("\n" + "=" * 60)
print("演示完成！生成的文件：")
print("  - /tmp/dual_cam_init.jpg: 双相机初始视角")
print("  - /tmp/top_cam_sam.jpg: 顶部相机分割结果")
print("  - /tmp/wrist_cam_sam.jpg: 腕部相机分割结果")
print("=" * 60)

env.close()
