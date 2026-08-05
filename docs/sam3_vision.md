# SAM3 视觉模块使用文档

> **状态**：✅ 已上线（2026-07-27），替代 MobileSAM ONNX 方案
> 验证：Stack demo 双相机流程 12/12 + 20/20 成功（ASPIRE 环境，py3.12）

SAM3（Segment Anything Model 3，Meta，848M 参数）原生支持**开放词汇文本提示分割**，
不再需要 MobileSAM 时代的"颜色关键词 → HSV 阈值 → 点提示"hack。
本模块通过 **transformers 库**（≥4.57）加载本地权重，PyTorch GPU 推理。

---

## 1. 文件清单

| 文件 | 说明 |
|------|------|
| `aspire/vision_sam3.py` | 视觉模块（唯一入口） |
| `SAM3/` | facebook/sam3 的 HF 仓库快照（model.safetensors 3.4GB + sam3.pt + tokenizer/processor 配置） |

> 旧 MobileSAM ONNX 方案（`vision_sam_onnx.py`、`models/mobile_sam_*`、导出/验证脚本）
> 已于 2026-07-27 随本方案上线后删除，历史文档见 git 记录。

## 2. 环境

- conda 环境 **ASPIRE**（Python 3.12），包版本与旧 aspire（py3.10）完全一致 + transformers 5.14.1
- 安装复现：`requirements_aspire312.txt` + `torch==2.13.0 torchvision==0.28.0`（pytorch cu130 index）+ `transformers`
- 推理 GPU 占用 ~8GB（文本模型 + Tracker 模型 + robosuite EGL 渲染共存，24GB 充裕）

### 迁移中踩过的三个坑（py3.12 新环境）

1. **NCCL cu12/cu13 互覆**：`nvidia-nccl-cu12` 与 `nvidia-nccl-cu13` 都写 `nvidia/nccl/lib/libnccl.so.2`，
   后装的覆盖先装的；若 cu12 覆盖 cu13，torch 报 `undefined symbol: ncclCommResume`。
   修复：`pip install --force-reinstall --no-deps nvidia-nccl-cu13==2.29.7`。
2. **onnxruntime 双包遮蔽**：`onnxruntime` 与 `onnxruntime-gpu` 同装时后装的覆盖前者的
   CUDA 二进制。只能装 `onnxruntime-gpu` 一个（否则 CUDA EP 静默消失）。
3. **EGL offscreen 渲染偶发 wedge**：见 §5。

## 3. 接口契约（与旧模块完全一致）

```python
from aspire.vision_sam3 import segment_sam3_text_prompt, segment_sam3_point_prompt, warmup

masks = segment_sam3_text_prompt(rgb, "red cube")   # rgb: (H,W,3) uint8 RGB
# → List[Dict]，按 score 降序：
#   {"mask": uint8(0/255), "score": float, "area": int,
#    "centroid": (x, y), "aspect_ratio": float}

cands = segment_sam3_point_prompt(rgb, (142, 139))  # 点前景点 → 该实例的多个候选 mask
```

- **text**：`Sam3Model` 概念分割，一次返回**所有**匹配实例（可多目标），检测阈值 0.5。
- **point**：`Sam3TrackerModel`（SAM2 式交互分割），单前景点 → 多个候选 mask 带质量分。
- 无匹配时返回空列表（调用方需判空，`point_prompt_molmo` 已处理）。

## 4. 性能（RTX 5090）

| 指标 | SAM3 | MobileSAM ONNX |
|------|------|----------------|
| 文本推理 | ~0.17s/次 | ~0.03s/次（颜色 hack+点提示） |
| 模型加载 | ~4s（双模型 ~8s） | ~1s |
| Stack demo 全程 | ~6s/run | ~2.5s/run |

per-step 检测场景无感；非颜色 prompt（"mug"、"handle"）现在可直接用。

## 5. EGL 渲染 wedge 与自愈机制（重要）

新环境（py3.12）中 robosuite 的 EGL offscreen context **偶发进入 wedge 状态**：
之后的所有 readback 返回固定垃圾帧（竖条纹/大面积死黑），不自愈，流入视觉会导致检测全灭
（旧 py3.10 环境 16 次未观察到，机制未明，与 CUDA 推理共存的时序竞争相关）。

**防护**（已实现于 [engine.py](../aspire/engine.py)）：

1. **检测**：`ExecutionEngine._img_corrupt()` —— 水平相邻像素差 p95 > 25 或近零像素 > 30%
   （正常帧 ≤10，损坏帧 ≥60，间隔充足）。每个 `engine.step()` 后自动 healthcheck。
2. **恢复**：`recover_renderer()` 逐级升级——
   - L1：`update_offscreen_size` 抖动强制 MjrContext free+重建（buffer 对齐）；
   - L2：整个 `MjRenderContextOffscreen` 重建（需拦截 `add_render_context` 避免
     init 中途 del 旧 context 导致 GL current 解绑；重建后必须 `make_current` + 尺寸对齐，
     否则读出噪声帧）。
3. 两级后仍损坏 → 抛 `RuntimeError`（不让垃圾帧静默流入下游）。
4. `primitives.get_observation` 的动态腕部渲染路径同样带检测+重建重试。
5. 深度通道另有浮点过冲防护：`get_real_depth_map` 前 `nan_to_num + clip[0,1]`
   （robosuite 的 [0,1] 断言对 float32 回读噪声过严）。

trace 中可见 `renderer_recoveries` 计数与恢复日志。

## 6. 回退方案

MobileSAM 相关文件（模块/权重/脚本）已删除，如需回退请从 git 历史恢复。
视觉接口契约（函数签名与返回结构）自 MobileSAM 时代起保持一致，
替换视觉实现只需改 `aspire/primitives.py` 顶部的一行 import。

## 参考

- SAM3 官方仓库：https://github.com/facebookresearch/sam3 （图像/视频统一分割，SAM License）
- 权重：https://huggingface.co/facebook/sam3 （gated，需申请；本项目已存本地 `SAM3/`）
- transformers 集成：`Sam3Model`（文本/概念）、`Sam3TrackerModel`（点/框交互）
- 旧方案（MobileSAM ONNX）文档与代码已从仓库删除，见 git 历史
