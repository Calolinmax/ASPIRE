# MobileSAM ONNX 使用文档

> **状态**：✅ 已转换、已端到端验证（2026-07-24）
> 验证结果：ONNX 与 PyTorch 输出 mask IoU = 1.000000，IoU 分数最大差 2.4e-07
> 验证脚本：[scripts/verify_mobilesam_onnx.py](../scripts/verify_mobilesam_onnx.py)

MobileSAM 是 Meta SAM 的轻量蒸馏版（TinyViT 编码器，9.66M 参数，[官方仓库](https://github.com/ChaoningZhang/MobileSAM)，Apache-2.0 许可）。本项目已将其转为 ONNX 格式，**推理时无需 PyTorch**，只需 onnxruntime 即可部署。

---

## 1. 文件清单

### 模型文件（`models/`）

| 文件 | 大小 | 说明 |
|------|------|------|
| `mobile_sam_encoder.onnx` | 27 MB | TinyViT 图像编码器：图像 → embedding。**每张图只跑一次** |
| `mobile_sam_decoder.onnx` | 16 MB | prompt encoder + mask decoder：embedding + 点/框提示 → mask。**每次提示跑一次** |
| `mobile_sam.pt` | 39 MB | 原始 PyTorch 权重（仅重新导出/对照验证时需要） |

### 脚本（`scripts/`）

| 脚本 | 用途 |
|------|------|
| `verify_mobilesam_onnx.py` | **使用示例 + 正确性验证**，照抄即可 |
| `export_mobile_sam_encoder.py` | 重新导出 encoder（官方脚本不含这部分） |
| `export_onnx_model.py` | 官方 decoder 导出脚本（已打 `dynamo=False` 补丁，见 §5） |
| `convert_mobilesam_onnx.py` | ⚠️ 早期手写版，已被上面两个脚本取代，**不建议用** |

---

## 2. 环境依赖

**推理（最低要求）**：

```bash
pip install onnxruntime numpy opencv-python
# 有 NVIDIA GPU 时装 onnxruntime-gpu 可加速
```

**重新导出/验证时额外需要**：`torch`、`mobile_sam`、`timm`、`onnx`、`onnxscript`（aspire conda 环境已全部装好）。

---

## 3. 快速开始（纯 ONNX 推理）

完整可运行示例见 [scripts/verify_mobilesam_onnx.py](../scripts/verify_mobilesam_onnx.py)。核心流程：

```python
import cv2
import numpy as np
import onnxruntime as ort

IMG_SIZE = 1024

# ---------- 1. 加载两个 ONNX 模型 ----------
enc = ort.InferenceSession("models/mobile_sam_encoder.onnx",
                           providers=["CPUExecutionProvider"])  # GPU: "CUDAExecutionProvider"
dec = ort.InferenceSession("models/mobile_sam_decoder.onnx",
                           providers=["CPUExecutionProvider"])

# ---------- 2. 预处理：resize 长边到 1024 → 右/下 pad 到 1024x1024 → 归一化 ----------
img = cv2.cvtColor(cv2.imread("your_image.jpg"), cv2.COLOR_BGR2RGB)
H, W = img.shape[:2]

scale = IMG_SIZE / max(H, W)
new_h, new_w = round(H * scale), round(W * scale)
resized = cv2.resize(img, (new_w, new_h))
padded = np.pad(resized, ((0, IMG_SIZE - new_h), (0, IMG_SIZE - new_w), (0, 0)))

# SAM 固定归一化参数，不要改
x = (padded.astype(np.float32) - [123.675, 116.28, 103.53]) / [58.395, 57.12, 57.375]
x = x.transpose(2, 0, 1)[None].astype(np.float32)  # NCHW

# ---------- 3. Encoder：每张图一次 ----------
embedding = enc.run(None, {"image": x})[0]  # (1, 256, 64, 64)

# ---------- 4. Decoder：每次提示一次 ----------
# ⚠️ 点坐标必须乘 scale 换算到 1024 空间，并追加一个 label=-1 的占位点
point = np.array([[320.0, 240.0]])            # 原图坐标 (x, y)
coords = point * scale
coords = np.concatenate([coords, [[0.0, 0.0]]], axis=0)[None].astype(np.float32)
labels = np.array([[1, -1]], dtype=np.float32)  # 1=前景点, 0=背景点, -1=占位

masks, scores, _ = dec.run(None, {
    "image_embeddings": embedding,
    "point_coords": coords,
    "point_labels": labels,
    "mask_input":   np.zeros((1, 1, 256, 256), dtype=np.float32),
    "has_mask_input": np.zeros(1, dtype=np.float32),
    "orig_im_size": np.array([H, W], dtype=np.float32),
})

# ---------- 5. 取最优 mask ----------
# ⚠️ decoder 输出 4 个 mask，与 PyTorch multimask 对齐的是索引 [1:]
masks3, scores3 = masks[0, 1:], scores[0, 1:]
best = masks3[np.argmax(scores3)] > 0  # (H, W) bool，已自动裁剪回原图尺寸
```

**多点提示**：`point_coords` 支持任意 N 个点（追加占位点后再加 1），label 含义：`1`=前景点，`0`=背景点，`-1`=占位点。

---

## 4. 模型接口参考

### encoder（`mobile_sam_encoder.onnx`）

| 输入 | 形状 | 说明 |
|------|------|------|
| `image` | (1, 3, 1024, 1024) float32 | 预处理后的图像（§3 步骤 2） |

| 输出 | 形状 | 说明 |
|------|------|------|
| `image_embeddings` | (1, 256, 64, 64) float32 | 图像特征 |

### decoder（`mobile_sam_decoder.onnx`）

| 输入 | 形状 | 说明 |
|------|------|------|
| `image_embeddings` | (1, 256, 64, 64) float32 | encoder 输出 |
| `point_coords` | (1, N, 2) float32 | **1024 空间**坐标（原图坐标 × scale），末尾需 1 个占位点 |
| `point_labels` | (1, N) float32 | 1=前景 / 0=背景 / -1=占位 |
| `mask_input` | (1, 1, 256, 256) float32 | 迭代细化的上一帧 mask；无则填 0 |
| `has_mask_input` | (1,) float32 | 0=不使用 mask_input |
| `orig_im_size` | (2,) float32 | 原图 `[H, W]`，用于后处理裁剪/缩放 |

| 输出 | 形状 | 说明 |
|------|------|------|
| `masks` | (1, 4, H, W) float32 | logits，>0 为前景；**索引用 [1:]** |
| `iou_predictions` | (1, 4) float32 | 各 mask 质量分，同样取 [1:] |
| `low_res_masks` | (1, 4, 256, 256) float32 | 低分辨率 mask，可作下次迭代的 mask_input |

---

## 5. 注意事项（踩过的坑）

1. **坐标必须变换**：`point_coords` 不是原图像素坐标，要乘 `scale = 1024 / max(H, W)` 换算到 1024 空间，且**必须追加一个 label=-1 的 (0,0) 占位点**（与官方 notebook 一致，否则点数变化会影响 mask 选择逻辑）。
2. **mask 索引对齐**：decoder 输出 4 个 mask（token 0 是单 mask 模式用的），PyTorch `SamPredictor.predict(multimask_output=True)` 返回的是索引 **[1:] 的 3 个**。对照或选最优时务必对齐。
3. **dtype 必须是 float32**：NumPy 1.x 下 `float32数组 - 列表` 会被提升成 float64，onnxruntime 会报 `Unexpected input data type`。所有输入显式 `.astype(np.float32)`。
4. **归一化参数固定**：mean=[123.675, 116.28, 103.53]，std=[58.395, 57.12, 57.375]（输入为 0-255 的 RGB），改了就对不上。
5. **重新导出需加 `dynamo=False`**：torch ≥ 2.9 的 `torch.onnx.export` 默认走 dynamo 新导出器，与官方脚本的 `dynamic_axes` 写法冲突。`scripts/export_onnx_model.py` 已打补丁；自己写导出代码时同样要加。
6. **环境变动提醒**：2026-07-24 安装 timm 时 aspire 环境的 torch 被从 `2.7.0.dev+cu124` 升到 `2.13.0+cu130`（GPU RTX 5090 已验证正常）。若环境内其他旧脚本异常，优先排查此项。
7. **encoder 是性能大头**：27MB 的 encoder 每张图跑一次；decoder 仅 16MB 且很轻。同一张图多次交互（点不同位置）时**缓存 embedding 复用**，不要重复跑 encoder。
8. **许可**：MobileSAM 为 Apache-2.0，可商用；底层架构衍生自 Meta SAM（同为 Apache-2.0）。
9. **cv2 与 PyTorch 预处理的细微差异**：§3 示例用 `cv2.resize` 做预处理（无需 torch），与官方 `ResizeLongestSide`（PIL 双线性）存在亚像素级插值差异。实测 mask IoU ≈ 0.995~0.999、score 差 ≈ 1e-3，实际使用无感；若需与 PyTorch 逐位对齐（如调试对照），改用 `mobile_sam.utils.transforms.ResizeLongestSide`（见 `verify_mobilesam_onnx.py`）。

### 重新生成 ONNX（如需）

```bash
conda activate aspire

# 1. decoder（官方脚本，已打补丁）
python scripts/export_onnx_model.py \
    --checkpoint models/mobile_sam.pt --model-type vit_t \
    --output models/mobile_sam_decoder.onnx

# 2. encoder（自写脚本，含数值校验）
python scripts/export_mobile_sam_encoder.py

# 3. 端到端验证（应输出 IoU=1.000000）
python scripts/verify_mobilesam_onnx.py [图片路径] [x y]
```

---

## 6. 清理记录（2026-07-24）

已删除的旧方案（MobileSAM 部署完成后不再需要）：

- `models/sam_vit_h.pth`（2.4GB 完整版 SAM 权重，如需可从 [Meta 官方](https://github.com/facebookresearch/segment-anything)重新下载）
- `models/yolov8n-seg.onnx`（空文件，YOLO 分割方案废弃）
- `aspire/vision_sam.py`、`scripts/demo_sam_vit_h_gpu.py`、`scripts/demo_dual_camera.py`（旧视觉模块与演示）

同时修复了两处遗留问题：

- `aspire/vision_sam_gpu.py` 的 import 从 `segment_anything` 改为 `mobile_sam`（vit_t 只存在于 mobile_sam 的 registry，原写法会 KeyError）
- `aspire/primitives.py` 的 `segment_text`/`segment_point` 原惰性引用已删除的 `vision_sam.py`，改为使用模块顶部导入的 `vision_sam_gpu`

## 参考

- 官方仓库：https://github.com/ChaoningZhang/MobileSAM （Apache-2.0）
- 论文：*Faster Segment Anything* (arXiv:2306.14289)
- 权重镜像：https://huggingface.co/dhkim2810/MobileSAM
