# ASPIRE 复现项目

复现论文 **ASPIRE: Agentic Skills Discovery for Robotics**（NVIDIA GEAR，2026，
[项目页](https://research.nvidia.com/labs/gear/aspire/)），基底框架为
**CaP-X**（[项目页](https://capgym.github.io/)，MIT，本地镜像见 `external/cap-x/`）。

在 MuJoCo 仿真中用 **AgileX Piper** 机械臂执行操作任务：SAM3 开放词汇分割做感知，
Contact-GraspNet 做 6-DoF 抓取位姿估计，pyroki 做 IK，RRT-Connect 做运动规划，
上层对接 ASPIRE 论文的 Primitive API / Skills 体系（官方任务代码见 `open_details/`）。

> **当前状态**：Phase 0 完成——`PiperControlApiReduced` 15 函数 API 全齐并通过自检
> （33/33），2026-08-06 封版；Stack 与 Wipe 两条任务线端到端 PASS。
> 详细进度看板见 [docs/api_asset_map.md](docs/api_asset_map.md)，
> 后续路线见 [docs/roadmap.md](docs/roadmap.md)。

---

## 1. 系统组成

```
┌─────────────────────────────── 宿主机（conda 环境 ASPIRE, py3.12）──────────────────────────────┐
│  任务层   scripts/stack.sh + tasks/stack.py · wipe.sh + tasks/wipe.py（统一 sh 入口）           │
│  API 层   aspire/api/primitives_capx.py      （PiperControlApiReduced, 15 函数, 🔒封版）        │
│  执行层   aspire/engine/engine_capx.py + engine.py（执行引擎, 🔒）+ aspire/evidence/（trace）   │
│  感知     aspire/perception/vision_sam3.py   → SAM3 本地权重（transformers, GPU ~8GB）          │
│  抓取     aspire/perception/vision_client.py → HTTP → CGN Docker 容器（见下）                   │
│  控制     aspire/planning/pyroki_client.py   → pyroki IK 服务 ／ motion_planner.py (RRT)        │
│  仿真     robosuite 1.5.2 + MuJoCo 3.3.7 + aspire/robots/（Piper MJCF + 201k 条 IK 种子库）     │
└───────────────────────────────────────────────────────────────────────────────────────────────┘
┌────────────── Docker 容器（cgn-tf:25.02）──────────────┐
│  Contact-GraspNet（官方 TF1 版）FastAPI 服务 :8117       │
│  NGC TensorFlow 2.17 镜像，pointnet2 ops 已编译          │
│  详见 docs/cgn_container.md                              │
└──────────────────────────────────────────────────────────┘
```

> ⚠️ CGN **只能**在 Docker 容器里跑（RTX 5090 sm_120 与 TF pip wheel 系统性不兼容，
> 宿主机直跑会触发非确定性 `CUDA_ERROR_ILLEGAL_ADDRESS`）。

## 2. 仓库结构

| 路径 | 说明 |
|---|---|
| `aspire/` | 核心包，按层分目录：`engine/`（执行引擎）`api/`（Primitive API）`evidence/`（trace+标注）`planning/`（RRT+pyroki 客户端）`perception/`（SAM3/CGN 服务）`envs/`（场景）`robots/`（Piper 模型资产） |
| `scripts/` | 任务层（统一 sh 入口）：`stack.sh`/`wipe.sh` 一键可视化 + `tasks/`（任务代码）`tests/`（自检）`tools/`（资产生产线/查看器/常驻服务）`archive/`（历史诊断） |
| `docs/` | 设计与运维文档（API 契约、CGN 容器、SAM3、文件地图、路线图） |
| `docker/cgn/` | CGN 容器构建资产存档（Dockerfile + 服务脚本 + 官方代码补丁），重建步骤见 [docs/env_rebuild.md](docs/env_rebuild.md) |
| `open_details/` | ASPIRE 官方公布的任务代码 3 份 + skill 样例（复现基准） |
| `external/` | ❗**不在仓库中**，第三方资产，需按 §4.4 准备 |
| `SAM3/` | ❗**不在仓库中**，SAM3 模型权重，需按 §4.3 下载 |
| `requirements_core_sim.txt` | 核心依赖清单（仿真线，98 条，PyPI 全部可装）✅ 装环境用这个 |
| `requirements_ASPIRE312.txt` | 旧机器完整 freeze（含 136 个 ROS 2 包，PyPI 无源，仅真机/ROS 联调需要） |

`.gitignore` 已排除：`SAM3/`、`external/`、`outputs/`、`traces/`、`.env`、`__pycache__/` 等
本地产物——**环境与大文件不进仓库，按下文重建**。

## 3. 环境要求

| 项 | 要求 | 本机实测 |
|---|---|---|
| OS | Linux（Ubuntu 22.04/24.04） | Ubuntu，内核 6.x |
| GPU | NVIDIA，≥16GB 显存推荐 | RTX 5090 D v2（sm_120，24GB） |
| CUDA | 13.0（torch cu130） | CUDA 13.0 |
| Docker | 需 nvidia-container-toolkit（CGN 用） | 离线 deb 见 `external/NVIDIA_deb/` |
| Python | 3.12（conda 环境名 **ASPIRE**，大写） | anaconda3 |

## 4. 安装

### 4.1 conda 环境

```bash
conda create -n ASPIRE python=3.12 -y
conda activate ASPIRE
```

### 4.2 PyTorch（cu130，必须单独先装）

```bash
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
```

> 原因：`pytorch-triton` 只在 PyTorch 官方 index；默认 PyPI 源装不到匹配的 CUDA 13 构建。

### 4.3 其余依赖 + SAM3 权重

```bash
pip install -r requirements_core_sim.txt

# SAM3 权重（Meta gated repo，需先在 HuggingFace 网页登录并同意模型协议）
huggingface-cli download facebook/sam3 --local-dir SAM3/
```

`requirements_core_sim.txt` 为仿真线核心依赖（robosuite、mujoco、transformers 5.14.1 等，
已逐条 PyPI 验证）；torch 已在 4.2 装好会自动跳过。
完整 freeze `requirements_ASPIRE312.txt` 另含 136 个 ROS 2 Jazzy 包（PyPI 无源，
仅真机/ROS 联调需要，仿真线不装）。
**换新机器完整恢复流程见 [docs/env_rebuild.md](docs/env_rebuild.md)。**

### 4.4 external/ 第三方资产

以下目录体积过大不入库，按需自行准备并放到 `external/` 下。
**逐项精确对齐（文件名/大小/摆放路径/钉版号/注意事项）见 [docs/external_assets.md](docs/external_assets.md)。**
（代码类 2026-08-10 已逐个审计：除 contact_graspnet 外**均无本地改动**，直接重新克隆官方即可）：

| 目录 | 内容 | 用途 |
|---|---|---|
| `cap-x/` | CaP-X 框架源码（[capgym/cap-x](https://github.com/capgym/cap-x) @ `53e9966`，MIT） | Primitive API 契约参照——**运行时不 import**，克隆仅为对照阅读 |
| `contact_graspnet/` | Contact-GraspNet 官方 TF1 代码（[NVlabs/contact_graspnet](https://github.com/NVlabs/contact_graspnet)）| 可从仓库内 `docker/cgn/contact_graspnet.bundle` 直接恢复（含本地修复），权重需另备 |
| `cgn_models/` | CGN 模型权重（[官方 Google Drive](https://drive.google.com/drive/folders/1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl)，用 `scene_test_2048_bs3_hor_sigma_001` 档，109M） | CGN 抓取，**必需** |
| `piper_description/` | AgileX Piper 官方 URDF（[agilexrobotics/agx_arm_urdf](https://github.com/agilexrobotics/agx_arm_urdf) @ `f6642ce`） | **pyroki IK 服务运行时加载** `piper/urdf/piper_description.urdf`；URDF↔MJCF 标定来源 |
| `agilex_arm_mujoco/` | AgileX 官方 MuJoCo 模型（[yanyuze1/agilex_arm_mujoco](https://github.com/yanyuze1/agilex_arm_mujoco) @ `4cd52b0`） | Piper MJCF 来源参照（只读） |
| `NVIDIA_deb/` | nvidia-container-toolkit 离线安装包（NVIDIA 官方 apt 源可重下） | 无网环境装 Docker GPU 支持 |
| `cgn_venv/` | 宿主机 TF fallback 环境（6.9G，venv 不可移植） | 留档调试用，**换机不用带，正常任务不要用** |

> 克隆统一 `git clone <上表地址> external/<目录名>`，钉版本再 `git checkout <commit>`；
> HF 遇 403/401 多半是 gated repo 未授权，不要误判为项目不存在。

### 4.5 CGN Docker 容器

```bash
cd external/contact_graspnet
sudo docker build -f Dockerfile.cgn -t cgn-tf:25.02 .
```

构建资产（Dockerfile、服务脚本、官方代码补丁）已在仓库 `docker/cgn/` 存档，
从官方克隆重建的完整步骤见 [docs/env_rebuild.md](docs/env_rebuild.md) §5。
启动与联调细节（容器名 `cgn`、端口 8117、健康检查）见
[docs/cgn_container.md](docs/cgn_container.md)。

## 5. 验证安装

```bash
# API 全量自检（15 函数契约，33 项）
python scripts/tests/test_piper_capx_api.py

# CGN 容器健康检查
curl http://localhost:8117/health
```

## 6. 运行入口

| 脚本 | 说明 |
|---|---|
| `scripts/stack.sh [seed] [slow]` | **Stack 任务一键可视化**（任务代码 `scripts/tasks/stack.py`） |
| `scripts/wipe.sh [seed] [slow]` | **Wipe 任务一键可视化**（任务代码 `scripts/tasks/wipe.py`，Franka 擦板） |
| `scripts/tools/cgn_execute_grasp.py` | CGN 抓取端到端验收门（CLI 直驱，`--view` 开窗） |

更多标定/诊断/可视化脚本见 `scripts/tools/` 与 `scripts/archive/`（逐文件说明
见 [docs/project_files.md](docs/project_files.md)）。

## 7. 文档索引

- [docs/env_rebuild.md](docs/env_rebuild.md) — **换新机器完整恢复指南**（含旧机器备份清单）
- [docs/external_assets.md](docs/external_assets.md) — **外部资产清单**（SAM3+external 逐项对齐：文件名/来源/摆放/注意事项）
- [docs/project_files.md](docs/project_files.md) — 全项目逐文件说明（仓库地图）
- [docs/roadmap.md](docs/roadmap.md) — Phase 1 起复现路线图（Skill Library / 进化搜索 / 真机）
- [docs/api_asset_map.md](docs/api_asset_map.md) — API 资产对照表 + 进度看板（含 🔒 冻结声明）
- [docs/sam3_vision.md](docs/sam3_vision.md) — SAM3 视觉模块
- [docs/cgn_container.md](docs/cgn_container.md) — CGN 容器化部署
- [AGENTS.md](AGENTS.md) — 给 AI agent 的仓库工作约定（环境路由表等）

## 8. 常见问题（踩坑记录）

1. **`undefined symbol: ncclCommResume`**：`nvidia-nccl-cu12` 与 `cu13` 互覆
   `libnccl.so.2`。修复：
   `pip install --force-reinstall --no-deps nvidia-nccl-cu13==2.29.7 nvidia-cudnn-cu13`
2. **渲染读出固定条纹/黑帧**：robosuite EGL offscreen context 偶发 wedge。
   `engine.py` 已内置每步健康检查 + 逐级自愈；若仍异常，重启进程即可。
3. **onnxruntime 装不出 GPU 版**：`onnxruntime` 与 `onnxruntime-gpu` 同装会互相遮蔽
   CUDA EP。本项目视觉已迁 SAM3，onnxruntime 已整体移除，无需再装。
4. **CGN 在宿主机起服务**：禁止。一律走 Docker 容器（见 §1 警告）。
5. **HuggingFace 下载 403/401**：SAM3 是 gated repo，先在网页登录并同意协议。

## 9. 许可与致谢

- 本仓库代码为学习复现用途，许可证暂未指定。
- `external/` 下第三方项目各随其许可：CaP-X（MIT）、Contact-GraspNet、
  AgileX Piper 描述文件、Meta SAM3（HF 协议）。
- 感谢 NVIDIA GEAR（ASPIRE）、CaP-X 团队、Contact-GraspNet 作者、AgileX 与 Meta 的开源工作。
