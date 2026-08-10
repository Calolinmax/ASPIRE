# aspire/perception/ —— 感知服务层 🔒

SAM3 分割与 CGN 抓取的服务化封装。**进程隔离是本层的存在理由**
（2026-07-28 实测：PyTorch CUDA 初始化后同进程 EGL 离屏渲染永久损坏）——
引擎进程只做 EGL，CUDA 推理全在本层服务里。

| 文件 | 说明 |
|---|---|
| `vision_server.py` | SAM3 推理服务进程（:8123，pickle over HTTP）：/health + /segment(text/point)。引擎自动拉起，无需手动启动。 |
| `vision_client.py` | 引擎侧纯 HTTP 客户端（不引 torch/CUDA）：SAM3 两分函数 + `grasp_cgn`（:8117）。 |
| `vision_sam3.py` | SAM3 本体（transformers 本地权重，只在 vision_server 进程内跑）：text=Sam3Model 开放词汇、point=Sam3TrackerModel；输出 `{mask,score,area,centroid,aspect_ratio}` 按 score 降序。 |
| `cgn_server.py` | CGN 官方 TF 版 FastAPI 服务壳（:8117）。**只能跑在 docker 容器**（cgn-tf:25.02；宿主机直跑触发非确定性 CUDA_ERROR_ILLEGAL_ADDRESS，留档禁用）。 |

⚠️ CGN 容器化部署/运维见 [../../docs/cgn_container.md](../../docs/cgn_container.md)；
SAM3 迁移三坑（NCCL/onnxruntime/EGL）见 [../../docs/sam3_vision.md](../../docs/sam3_vision.md)。

🔒 2026-08-07 用户指令加锁，只调用不修改。
