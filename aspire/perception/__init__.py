"""感知服务层：vision_server.py（SAM3 进程隔离服务，🔒）+ vision_client.py（🔒）
+ vision_sam3.py（SAM3 本体）+ cgn_server.py（CGN 容器内服务壳）。
CUDA 与 EGL 同进程互毁（2026-07-28 实测）是本层进程隔离的根因。"""
