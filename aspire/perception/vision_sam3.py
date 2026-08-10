"""
SAM3 视觉模块（transformers 本地权重，GPU 推理）
双相机支持：顶部 (agentview) + 腕部 (wrist)

与旧 MobileSAM 模块（vision_sam_onnx.py，已删除）接口契约完全一致：
    segment_sam3_text_prompt(rgb, prompt)  -> List[Dict]
    segment_sam3_point_prompt(rgb, point)  -> List[Dict]
    Dict = {mask(uint8 0/255), score, area, centroid, aspect_ratio}，按 score 降序

差异：SAM3 原生理解文本（开放词汇概念分割，返回所有匹配实例），
不再需要 MobileSAM 的颜色关键词 hack；点提示走 Sam3Tracker（SAM2 式交互分割）。
"""

import numpy as np
import torch
from typing import List, Dict, Tuple, Optional

# 本地权重目录（facebook/sam3 的 HF 仓库快照，含 model.safetensors + 处理器配置）
SAM3_PATH = "/home/stouching/Desktop/ASPIRE/SAM3"

# 全局模型（延迟加载；text 与 point 各用一套，首次调用时才加载对应模型）
_model_text = None          # Sam3Model：文本/概念提示 → 所有实例
_processor_text = None      # Sam3Processor
_model_point = None         # Sam3TrackerModel：点/框提示 → 单实例多候选
_processor_point = None     # Sam3TrackerProcessor

TEXT_THRESHOLD = 0.5  # 概念检测置信度阈值


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_text_model():
    """加载文本分割模型（只加载一次）"""
    global _model_text, _processor_text
    if _model_text is not None:
        return _model_text, _processor_text

    from transformers import Sam3Model, Sam3Processor

    print(f"[SAM3] 加载文本分割模型 ({_device()})...")
    _model_text = Sam3Model.from_pretrained(SAM3_PATH).to(_device()).eval()
    _processor_text = Sam3Processor.from_pretrained(SAM3_PATH)
    print("[SAM3] 文本模型加载完成")
    return _model_text, _processor_text


def _load_point_model():
    """加载点提示分割模型（只加载一次）"""
    global _model_point, _processor_point
    if _model_point is not None:
        return _model_point, _processor_point

    from transformers import Sam3TrackerModel, Sam3TrackerProcessor

    print(f"[SAM3] 加载点提示模型 ({_device()})...")
    _model_point = Sam3TrackerModel.from_pretrained(SAM3_PATH).to(_device()).eval()
    _processor_point = Sam3TrackerProcessor.from_pretrained(SAM3_PATH)
    print("[SAM3] 点提示模型加载完成")
    return _model_point, _processor_point


def _masks_to_results(masks: np.ndarray, scores: np.ndarray) -> List[Dict]:
    """(N, H, W) bool/float masks + (N,) scores → 接口契约的 List[Dict]（按 score 降序）"""
    results = []
    for mask, score in zip(masks, scores):
        mask_bool = np.asarray(mask) > 0
        mask_uint8 = (mask_bool * 255).astype(np.uint8)
        area = int(mask_bool.sum())
        if area == 0:
            continue

        ys, xs = np.nonzero(mask_bool)
        centroid = (float(xs.mean()), float(ys.mean()))

        x_span = xs.max() - xs.min()
        y_span = ys.max() - ys.min()
        aspect = max(x_span, y_span) / (min(x_span, y_span) + 1e-5)

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
    SAM3 文本提示分割：开放词汇概念 → 所有匹配实例的 mask

    参数:
        rgb: RGB 图像 (H, W, 3) uint8
        prompt: 文本描述，如 "red cube"、"green block"、"mug"
    """
    from PIL import Image

    model, processor = _load_text_model()

    image = Image.fromarray(rgb)
    inputs = processor(images=image, text=prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    results = processor.post_process_instance_segmentation(
        outputs,
        threshold=TEXT_THRESHOLD,
        mask_threshold=0.5,
        target_sizes=inputs["original_sizes"].tolist(),
    )[0]

    masks = results["masks"].cpu().numpy()
    scores = results["scores"].cpu().numpy()
    return _masks_to_results(masks, scores)


def segment_sam3_point_prompt(rgb: np.ndarray, point: Tuple[float, float]) -> List[Dict]:
    """
    SAM3 点提示分割（Sam3Tracker，SAM2 式）：单前景点 → 该实例的多个候选 mask

    参数:
        rgb: RGB 图像 (H, W, 3) uint8
        point: (x, y) 原图像素坐标（前景点）
    """
    from PIL import Image

    model, processor = _load_point_model()

    u, v = float(point[0]), float(point[1])
    image = Image.fromarray(rgb)
    # 4 维: (image, object, point_per_object, xy)；label 1=前景点
    input_points = [[[[u, v]]]]
    input_labels = [[[1]]]

    inputs = processor(images=image, input_points=input_points,
                       input_labels=input_labels, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model(**inputs)

    # post_process_masks: List[image] → (n_objects, n_masks, H, W)
    masks = processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]
    masks = masks.reshape(-1, *masks.shape[-2:]).numpy()

    # 质量分（SAM2 式 iou_scores: (B, n_objects, n_masks)）；没有则按输出顺序降序赋分
    if hasattr(outputs, "iou_scores") and outputs.iou_scores is not None:
        scores = outputs.iou_scores.cpu().numpy().reshape(-1)[: len(masks)]
    else:
        scores = np.linspace(1.0, 0.5, len(masks))

    return _masks_to_results(masks, scores)


def warmup():
    """预热两个模型"""
    print("[SAM3] 预热...")
    dummy = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    segment_sam3_text_prompt(dummy, "cube")
    segment_sam3_point_prompt(dummy, (128, 128))
    print("[SAM3] 预热完成")
