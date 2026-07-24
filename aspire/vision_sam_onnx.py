"""
MobileSAM ONNX 视觉模块（纯 onnxruntime，无需 PyTorch）
双相机支持：顶部 (agentview) + 腕部 (wrist)
"""

import numpy as np
import cv2
import onnxruntime as ort
from typing import List, Dict, Tuple, Optional

# 模型路径
ENCODER_PATH = "/home/stouching/Desktop/ASPIRE/models/mobile_sam_encoder.onnx"
DECODER_PATH = "/home/stouching/Desktop/ASPIRE/models/mobile_sam_decoder.onnx"

IMG_SIZE = 1024

# 全局 session（延迟加载）
_encoder: Optional[ort.InferenceSession] = None
_decoder: Optional[ort.InferenceSession] = None


def _get_providers():
    """检测可用执行提供者（GPU优先）"""
    providers = ort.get_available_providers()
    if "CUDAExecutionProvider" in providers:
        return ["CUDAExecutionProvider", "CPUExecutionProvider"]
    return ["CPUExecutionProvider"]


def _load_models():
    """加载 ONNX 模型（只加载一次）"""
    global _encoder, _decoder

    if _encoder is not None and _decoder is not None:
        return _encoder, _decoder

    providers = _get_providers()
    device = "GPU" if "CUDAExecutionProvider" in providers else "CPU"
    print(f"[MobileSAM-ONNX] 加载模型 ({device})...")

    _encoder = ort.InferenceSession(ENCODER_PATH, providers=providers)
    _decoder = ort.InferenceSession(DECODER_PATH, providers=providers)

    print(f"[MobileSAM-ONNX] 加载完成 (encoder: 27MB, decoder: 16MB)")
    return _encoder, _decoder


def _preprocess(img: np.ndarray) -> Tuple[np.ndarray, float, Tuple[int, int]]:
    """
    预处理：resize 长边到 1024 → pad 到 1024x1024 → 归一化

    返回: (NCHW tensor, scale, (H, W))
    """
    H, W = img.shape[:2]
    scale = IMG_SIZE / max(H, W)
    new_h, new_w = round(H * scale), round(W * scale)

    resized = cv2.resize(img, (new_w, new_h))
    padded = np.pad(resized, ((0, IMG_SIZE - new_h), (0, IMG_SIZE - new_w), (0, 0)))

    # SAM 固定归一化参数
    x = (padded.astype(np.float32) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
    x = x.transpose(2, 0, 1)[None].astype(np.float32)  # NCHW

    return x, scale, (H, W)


def _color_to_point(rgb: np.ndarray, color: str) -> Tuple[int, int]:
    """颜色关键词 → 粗略定位点（用于初始化 SAM）"""
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)

    color_ranges = {
        "red": [(np.array([0, 100, 100]), np.array([10, 255, 255])),
                (np.array([160, 100, 100]), np.array([180, 255, 255]))],
        "green": [(np.array([40, 50, 50]), np.array([80, 255, 255]))],
        "blue": [(np.array([100, 50, 50]), np.array([140, 255, 255]))],
        "yellow": [(np.array([20, 100, 100]), np.array([35, 255, 255]))],
        "brown": [(np.array([10, 100, 50]), np.array([25, 200, 150]))],
        "cube": None,
        "block": None,
    }

    h, w = rgb.shape[:2]

    if color not in color_ranges or color_ranges[color] is None:
        return w // 2, h // 2

    mask = np.zeros(hsv.shape[:2], dtype=np.uint8)
    for lower, upper in color_ranges[color]:
        m = cv2.inRange(hsv, lower, upper)
        mask = cv2.bitwise_or(mask, m)

    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
    if num_labels > 1:
        largest = np.argmax(stats[1:, cv2.CC_STAT_AREA]) + 1
        return int(centroids[largest][0]), int(centroids[largest][1])

    return w // 2, h // 2


def _run_sam(img: np.ndarray, points: List[Tuple[int, int]],
             labels: Optional[List[int]] = None) -> List[Dict]:
    """
    核心 SAM 推理：图像 + 点提示 → masks

    参数:
        img: RGB 图像 (H, W, 3)
        points: [(x, y), ...] 原图坐标
        labels: [1, ...] 1=前景, 0=背景；默认全 1
    """
    enc, dec = _load_models()

    # 预处理
    x, scale, (H, W) = _preprocess(img)

    # Encoder（每张图一次）
    embedding = enc.run(None, {"image": x})[0]  # (1, 256, 64, 64)

    # Decoder（每次提示一次）
    if labels is None:
        labels = [1] * len(points)

    # 坐标变换到 1024 空间 + 追加占位点
    coords = np.array(points, dtype=np.float32) * scale
    coords = np.concatenate([coords, [[0.0, 0.0]]], axis=0)[None].astype(np.float32)
    labels_arr = np.array(labels + [-1], dtype=np.float32)[None]

    masks, scores, _ = dec.run(None, {
        "image_embeddings": embedding,
        "point_coords": coords,
        "point_labels": labels_arr,
        "mask_input": np.zeros((1, 1, 256, 256), dtype=np.float32),
        "has_mask_input": np.zeros(1, dtype=np.float32),
        "orig_im_size": np.array([H, W], dtype=np.float32),
    })

    # 取索引 [1:] 的 3 个 mask（与 PyTorch multimask 对齐）
    masks3, scores3 = masks[0, 1:], scores[0, 1:]

    results = []
    for mask, score in zip(masks3, scores3):
        mask_bool = mask > 0
        mask_uint8 = (mask_bool * 255).astype(np.uint8)
        area = int(mask_bool.sum())

        ys, xs = np.nonzero(mask_bool)
        centroid = (float(xs.mean()), float(ys.mean())) if len(xs) > 0 else (0, 0)

        # 长宽比
        if len(xs) > 0 and len(ys) > 0:
            x_span = xs.max() - xs.min()
            y_span = ys.max() - ys.min()
            aspect = max(x_span, y_span) / (min(x_span, y_span) + 1e-5)
        else:
            aspect = 1.0

        results.append({
            "mask": mask_uint8,
            "score": float(score),
            "area": area,
            "centroid": centroid,
            "aspect_ratio": float(aspect),
        })

    results.sort(key=lambda x: x["score"], reverse=True)
    return results


def segment_sam3_text_prompt(rgb: np.ndarray, prompt: str) -> List[Dict]:
    """
    MobileSAM text prompt 分割（ONNX GPU 加速）

    流程: 颜色关键词 → 粗略定位点 → SAM 精修
    """
    prompt_lower = prompt.lower()
    color = None
    for c in ["red", "green", "blue", "yellow", "brown", "cube", "block"]:
        if c in prompt_lower:
            color = c
            break

    cx, cy = _color_to_point(rgb, color or "cube")
    return _run_sam(rgb, [(cx, cy)], [1])


def segment_sam3_point_prompt(rgb: np.ndarray, point: Tuple[float, float]) -> List[Dict]:
    """MobileSAM point prompt 分割"""
    u, v = int(point[0]), int(point[1])
    return _run_sam(rgb, [(u, v)], [1])


def warmup():
    """预热模型"""
    print("[MobileSAM-ONNX] 预热...")
    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    segment_sam3_text_prompt(dummy, "cube")
    print("[MobileSAM-ONNX] 预热完成")
