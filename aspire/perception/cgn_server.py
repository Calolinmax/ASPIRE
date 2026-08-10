#!/usr/bin/env python
"""CGN 官方 TF 版服务（端口 8117）

用法:
    source external/cgn_venv/bin/activate
    python aspire/cgn_server.py
"""
import os
os.environ['TF_USE_LEGACY_KERAS'] = '1'  # Force Keras 2 for TF 2.16 compat
import sys
import pickle
import numpy as np
from fastapi import FastAPI, Request
from fastapi.responses import Response

# 导入官方 CGN
sys.path.insert(0, '/home/stouching/Desktop/ASPIRE/external/contact_graspnet')
sys.path.insert(0, '/home/stouching/Desktop/ASPIRE/external/contact_graspnet/contact_graspnet')
sys.path.insert(0, '/home/stouching/Desktop/ASPIRE/external/contact_graspnet/pointnet2')
sys.path.insert(0, '/home/stouching/Desktop/ASPIRE/external/contact_graspnet/pointnet2/utils')
from wrapper import ContactGraspnet

app = FastAPI(title="Contact-GraspNet Service")
model = None

# 模型权重路径（用户下载位置）
# 可用模型：
#   - scene_test_2048_bs3_hor_sigma_001  (默认，推荐)
#   - scene_test_2048_bs3_hor_sigma_0025 (噪声较强场景)
#   - scene_2048_bs3_rad2_32             (干净深度数据)
#   - contact_graspnet_train_and_test-20260730T045030Z-1-001
CKPT_BASE_DIR = '/home/stouching/Desktop/ASPIRE/external/cgn_models/cgn_models'
DEFAULT_MODEL = 'scene_test_2048_bs3_hor_sigma_001-20260730T045031Z-1-001/scene_test_2048_bs3_hor_sigma_001'
DEFAULT_CKPT_ITER = 144144


def find_checkpoint():
    """自动查找可用的 checkpoint。"""
    # 首选默认模型
    ckpt_path = f'{CKPT_BASE_DIR}/{DEFAULT_MODEL}/model.ckpt-{DEFAULT_CKPT_ITER}'
    if os.path.exists(f'{ckpt_path}.index'):
        return ckpt_path

    # 尝试找该目录下其他 checkpoint
    import glob
    model_dir = f'{CKPT_BASE_DIR}/{DEFAULT_MODEL}'
    if os.path.exists(model_dir):
        index_files = sorted(glob.glob(f'{model_dir}/*.index'))
        if index_files:
            return index_files[-1].replace('.index', '')  # 用最新的

    # 遍历所有可用模型
    all_index = glob.glob(f'{CKPT_BASE_DIR}/*/*.index')
    if all_index:
        return sorted(all_index)[-1].replace('.index', '')

    raise RuntimeError(f"No checkpoint found in {CKPT_BASE_DIR}")


@app.on_event("startup")
def load_model():
    global model
    ckpt_path = find_checkpoint()
    model = ContactGraspnet(ckpt_path)
    print(f"[CGN] Model loaded from {ckpt_path}")


@app.post("/grasp")
async def grasp(request: Request):
    """接收 pickle 序列化的请求，返回 pickle 序列化的结果。"""
    body = await request.body()
    req = pickle.loads(body)

    depth = np.array(req["depth"], dtype=np.float32)
    K = np.array(req["K"], dtype=np.float64)
    seg = np.array(req["seg"], dtype=np.int32)
    z_range = req.get("z_range", [0.2, 2.0])
    forward_passes = req.get("forward_passes", 1)

    grasps, scores = model.predict(
        depth, K, seg,
        z_range=z_range,
        forward_passes=forward_passes
    )

    out = pickle.dumps({
        "grasps": grasps,
        "scores": scores,
    }, protocol=4)
    return Response(content=out, media_type="application/octet-stream")


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8117)
