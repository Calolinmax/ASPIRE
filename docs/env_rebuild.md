# 新机器环境恢复指南（换机/重建专用）

> 2026-08-07 编制。目标：在一台全新 Linux 机器上把 ASPIRE 仿真线跑起来，
> 按顺序执行，每步过了验证门再往下。全程约 1–2 小时（含下载）。
> 日常安装说明见 [README.md](../README.md) §4；本文档侧重**换机恢复的完整闭环**。

---

## 0. 从旧机器带走的东西（离开前必做！）

| 物品 | 大小 | 必要性 | 说明 |
|---|---|---|---|
| `external/` 整目录 | 8.5G | ⭐ 必需 | CGN 权重 + 官方克隆 + 离线 deb 包。官方模型来自 Google Drive，**链接可能失效，务必离线带走** |
| CGN docker 镜像 | ~10G 级 | ⭐ 强烈推荐 | `sudo docker save cgn-tf:25.02 \| gzip > cgn-tf-25.02.tar.gz`——免去 NGC 拉镜像 + 容器内编译 pointnet2（历史痛点） |
| `SAM3/` 权重 | 6.5G | 推荐 | 可从 HuggingFace 重下（gated 要授权），直接带走更省事 |
| `traces/` `outputs/` | 按需 | 可选 | 执行 trace 与调试图，研究产物，不进 git |
| `笔记.pdf` | 6M | 可选 | 个人笔记，不进 git |

代码本体在 GitHub 私有库 `Calolinmax/ASPIRE`，随时可 clone，不用拷。

## 1. 系统层准备

- Ubuntu 22.04/24.04 + NVIDIA 驱动 + Docker + nvidia-container-toolkit
  （离线安装包在备份的 `external/NVIDIA_deb/`）
- **GPU 建议 ≥16GB 显存**（SAM3 双模型 ~8GB + MuJoCo EGL 渲染共存）。
  无 NVIDIA 显卡的机器：仿真/视觉/CGN 都跑不起来，只能读代码。
- anaconda 或 miniconda。

## 2. 拉代码

新机器生成 SSH 密钥并把公钥加到 GitHub **Calolinmax** 账号
（Settings → SSH and GPG keys → New SSH key），然后：

```bash
git clone git@github.com:Calolinmax/ASPIRE.git
cd ASPIRE
```

## 3. conda 主环境（ASPIRE，py3.12）

```bash
conda create -n ASPIRE python=3.12 -y
conda activate ASPIRE

# torch 必须先用官方 cu130 源单独装（PyPI 默认源装不到匹配的 CUDA 13 构建）
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130

# 核心依赖（仿真线）：98 条，2026-08-07 已逐条 PyPI 干跑验证（含钉版号）
pip install -r requirements_core_sim.txt
```

> `requirements_ASPIRE312.txt` 是旧机器完整 freeze，另外含 **136 个 ROS 2 Jazzy 包，
> PyPI 无源装不上**（当时来自特定渠道）。仿真线不 import 它们，**不用装**；
> 只有真机/ROS 联调时才需要，届时走 ROS 2 官方渠道。

## 4. 大文件归位

```bash
# 从备份拷回（或重新收集，见 README §4.4）：
#   external/  → 仓库根/external/
#   SAM3/      → 仓库根/SAM3/
# SAM3 重新下载（gated repo，需先在 HF 网页登录+同意协议，且已 hf auth login）：
huggingface-cli download facebook/sam3 --local-dir SAM3/
```

## 5. CGN 服务（仅抓取任务需要）

**有镜像 tar（推荐）**：

```bash
sudo docker load < cgn-tf-25.02.tar.gz
```

**无 tar 重建**（构建资产已存档在仓库 `docker/cgn/`，含对官方代码的补丁）：

```bash
git clone https://github.com/NVlabs/contact_graspnet.git external/contact_graspnet
cd external/contact_graspnet
git checkout $(cat ../../docker/cgn/UPSTREAM_COMMIT.txt)
cp ../../docker/cgn/{Dockerfile.cgn,.dockerignore,cgn_server_container.py,cgn_test_container.py,compile_pointnet_tfops.sh,compile_pointnet_tfops_container.sh} .
git apply ../../docker/cgn/contact_grasp_estimator.patch
# 权重从备份拷入：checkpoints/scene_test_2048_bs3_hor_sigma_001（109M，容器运行时挂载用）
sudo docker build -f Dockerfile.cgn -t cgn-tf:25.02 .
```

容器启动与联调（容器名 `cgn`、端口 8117）见 [cgn_container.md](cgn_container.md)。

## 6. pyroki IK 服务（独立 venv，CPU 版 jax，不占显卡）

```bash
python3.12 -m venv ~/venvs/pyroki
~/venvs/pyroki/bin/pip install fastapi uvicorn jax jaxlie viser yourdfpy trimesh
# pyroki 不在 PyPI，按旧机器记录的来源+commit 安装：
~/venvs/pyroki/bin/pip install "git+https://github.com/chungmin99/pyroki.git@388e43e1fc0d0ee382968d3dd72970fd62a0450c"
```

启动命令见 `scripts/tools/pyroki_server_minimal.py` 头部 docstring
（**注意把其中的 `/home/stouching/...` 绝对路径换成新机器的实际路径**）。

## 7. 验证门（每步过了再往下）

| 顺序 | 命令 | 通过标准 |
|---|---|---|
| 1 | `python scripts/tests/test_piper_build.py` | 场景构建无异常 |
| 2 | `python scripts/tests/test_piper_capx_api.py` | 33/33 PASS |
| 3 | 启动 pyroki 服务 | :8116 响应 IK 请求 |
| 4 | 启动 CGN 容器后 `curl localhost:8117/health` | 返回 JSON |
| 5 | `scripts/stack.sh` | 任务跑通（可视化） |

## 8. 已知坑速查

1. **`undefined symbol: ncclCommResume`** → `pip install --force-reinstall --no-deps nvidia-nccl-cu13==2.29.7 nvidia-cudnn-cu13`（NCCL cu12/cu13 互覆）
2. **渲染条纹/黑帧** → EGL wedge，`engine` 已内置自愈；仍异常就重启进程
3. **HF 下载 403/401** → SAM3 是 gated repo，网页授权 + 本机 `hf auth login`
4. **绝对路径**：`AGENTS.md` 与少数脚本/docstring 里的 `/home/stouching/...`
   是旧机器路径，新机器按需替换
5. **CGN 别在宿主机直跑** → 只走 Docker（TF wheel 与新显卡不兼容，详见 cgn_container.md）
