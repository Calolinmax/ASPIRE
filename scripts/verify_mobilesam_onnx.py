#!/usr/bin/env python
"""
端到端验证 MobileSAM ONNX：纯 onnxruntime 流程 vs PyTorch SamPredictor

流程对照（两条路径应给出几乎相同的 mask）：
  PyTorch:  SamPredictor.set_image -> predict
  ONNX:     预处理(resize+pad+normalize) -> encoder.onnx -> decoder.onnx -> 后处理(模型内已含)

用法:
    python scripts/verify_mobilesam_onnx.py [图片路径] [x y]
"""

import sys

import cv2
import numpy as np
import onnxruntime as ort
import torch

from mobile_sam import SamPredictor, sam_model_registry
from mobile_sam.utils.transforms import ResizeLongestSide

CHECKPOINT = "models/mobile_sam.pt"
ENCODER_ONNX = "models/mobile_sam_encoder.onnx"
DECODER_ONNX = "models/mobile_sam_decoder.onnx"
IMG_SIZE = 1024

image_path = sys.argv[1] if len(sys.argv) > 1 else "ASPIRE源论文/ASPIRE_files/ken-goldberg.jpg"
point = [float(sys.argv[2]), float(sys.argv[3])] if len(sys.argv) > 3 else None

# ---------- 读图 ----------
img_bgr = cv2.imread(image_path)
assert img_bgr is not None, f"读不到图片: {image_path}"
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]
if point is None:
    point = [W / 2, H / 2]  # 默认点图中心
print(f"图片: {image_path} ({W}x{H}), 提示点: {point}")

# ---------- 路径 A：PyTorch 参考 ----------
sam = sam_model_registry["vit_t"](checkpoint=CHECKPOINT)
sam.eval()
predictor = SamPredictor(sam)
predictor.set_image(img)
masks_pt, scores_pt, _ = predictor.predict(
    point_coords=np.array([point]),
    point_labels=np.array([1]),
    multimask_output=True,
)

# ---------- 路径 B：纯 ONNX ----------
transform = ResizeLongestSide(IMG_SIZE)

# 1. 预处理：resize 长边到 1024 → pad 到 1024x1024 → 归一化（与 Sam.preprocess 一致）
resized = transform.apply_image(img)  # uint8
pad_h, pad_w = IMG_SIZE - resized.shape[0], IMG_SIZE - resized.shape[1]
padded = np.pad(resized, ((0, pad_h), (0, pad_w), (0, 0)))
x = (padded.astype(np.float32) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
x = x.transpose(2, 0, 1)[None].astype(np.float32)  # NCHW, 显式转 float32

# 2. encoder
enc = ort.InferenceSession(ENCODER_ONNX, providers=["CPUExecutionProvider"])
embedding = enc.run(None, {"image": x})[0]

# 3. decoder（点坐标换算到 1024 空间；追加一个 label=-1 的占位点，与官方 notebook 一致）
coords = transform.apply_coords(np.array([point]), (H, W))
coords = np.concatenate([coords, [[0.0, 0.0]]], axis=0)[None].astype(np.float32)
labels = np.array([[1, -1]], dtype=np.float32)
dec = ort.InferenceSession(DECODER_ONNX, providers=["CPUExecutionProvider"])
masks_onnx, scores_onnx, _ = dec.run(
    None,
    {
        "image_embeddings": embedding,
        "point_coords": coords,
        "point_labels": labels,
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.zeros(1, dtype=np.float32),
        "orig_im_size": np.array([H, W], dtype=np.float32),
    },
)

# ---------- 对比 ----------
# 注：ONNX decoder 输出全部 4 个 mask token；PyTorch multimask 模式取的是 [1:] 这 3 个
scores_onnx3 = scores_onnx[0, 1:]
print(f"\nPyTorch  scores: {scores_pt}")
print(f"ONNX     scores: {scores_onnx3}  (最大差 {abs(scores_pt - scores_onnx3).max():.2e})")
for i in range(3):
    a, b = masks_pt[i], masks_onnx[0, i + 1] > 0
    iou = (a & b).sum() / max((a | b).sum(), 1)
    print(f"mask[{i}] IoU(PyTorch vs ONNX): {iou:.6f}")
    assert iou > 0.99, f"mask[{i}] 不一致"

print("\n端到端验证通过 ✓  ONNX 与 PyTorch 输出一致")
