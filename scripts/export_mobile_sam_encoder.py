#!/usr/bin/env python
"""
导出 MobileSAM Image Encoder 为 ONNX

官方 scripts/export_onnx_model.py 只导出 prompt_encoder + mask_decoder，
图像编码器（TinyViT，模型的大头）需要单独导出，本脚本补齐这一部分。

用法:
    python scripts/export_mobile_sam_encoder.py
输出:
    models/mobile_sam_encoder.onnx   输入 image(1,3,1024,1024) → 输出 image_embeddings(1,256,64,64)
"""

import torch
from mobile_sam import sam_model_registry

CHECKPOINT = "models/mobile_sam.pt"
OUTPUT = "models/mobile_sam_encoder.onnx"

print(f"加载模型: {CHECKPOINT}")
sam = sam_model_registry["vit_t"](checkpoint=CHECKPOINT)
sam.eval()

dummy_input = torch.randn(1, 3, 1024, 1024)  # SAM 标准输入：resize+pad 到 1024x1024

with torch.no_grad():
    # 先用 PyTorch 跑一遍，记录参考输出供后续校验
    ref = sam.image_encoder(dummy_input)
    print(f"PyTorch encoder 输出形状: {tuple(ref.shape)}")

    torch.onnx.export(
        sam.image_encoder,
        dummy_input,
        OUTPUT,
        export_params=True,
        opset_version=17,
        do_constant_folding=True,
        input_names=["image"],
        output_names=["image_embeddings"],
        dynamo=False,  # 与官方 decoder 导出保持一致，走 TorchScript 旧导出器
    )
print(f"已保存: {OUTPUT}")

# 用 onnxruntime 校验数值一致性
import onnxruntime as ort

sess = ort.InferenceSession(OUTPUT, providers=["CPUExecutionProvider"])
out = sess.run(None, {"image": dummy_input.numpy()})[0]
diff = abs(out - ref.numpy()).max()
print(f"onnxruntime 输出形状: {out.shape}, 与 PyTorch 最大绝对误差: {diff:.2e}")
assert diff < 1e-4, "误差过大，请检查导出"
print("Encoder ONNX 校验通过 ✓")
