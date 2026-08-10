#!/usr/bin/env python
"""容器内 CGN HTTP 服务器 - 供主机 ASPIRE 调用"""
import os
import sys
import pickle
import numpy as np

os.environ['TF_USE_LEGACY_KERAS'] = '1'

# 延迟导入 FastAPI，先加载模型
sys.path.insert(0, '/workspace')
sys.path.insert(0, '/workspace/contact_graspnet')
sys.path.insert(0, '/workspace/pointnet2')
sys.path.insert(0, '/workspace/pointnet2/utils')

from wrapper import ContactGraspnet

# 加载模型（启动时一次）
CKPT = '/workspace/checkpoint/model.ckpt-144144'
print(f'[CGN Server] Loading model from {CKPT}...', flush=True)
model = ContactGraspnet(CKPT)
print(f'[CGN Server] Model loaded. Ready for requests.', flush=True)

# 现在导入 FastAPI
from fastapi import FastAPI, Request
from fastapi.responses import Response
import uvicorn

app = FastAPI(title="Contact-GraspNet Container Service")

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

    grasps, scores, openings = model.predict(
        depth, K, seg,
        z_range=z_range,
        forward_passes=forward_passes,
        return_openings=True,
    )

    out = pickle.dumps({
        "grasps": grasps,
        "scores": scores,
        "openings": openings,
    }, protocol=4)
    return Response(content=out, media_type="application/octet-stream")

@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8117)
