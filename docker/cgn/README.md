# CGN 容器构建资产存档

`external/contact_graspnet/` 是从官方仓库克隆后**带本地修改**的工作副本。
本目录存档了在一台新机器上从零重建 CGN 服务所需的全部项目自有资产。

## 来源（均已验证）

| 项 | 来源 |
|---|---|
| 官方代码 | https://github.com/NVlabs/contact_graspnet （上游 HEAD = `7217060`） |
| CGN 权重 | 官方 Google Drive：https://drive.google.com/drive/folders/1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl （本项目用 `scene_test_2048_bs3_hor_sigma_001`，109M，官方默认档，适合干净 sim 深度） |
| 官方 test_data | https://drive.google.com/drive/folders/1TqpM2wHAAo0j3i1neu3Xeru3_WnsYQnx （仅回归测试用，可选） |
| 基础镜像 | `nvcr.io/nvidia/tensorflow:25.02-tf2-py3`（NGC 公共目录，免登录拉取） |

## 文件清单

| 文件 | 说明 |
|---|---|
| `contact_graspnet.bundle` | ⭐ 嵌套 git 仓库完整快照（`git clone` 它即可）。含上游全部历史 + **2 个本地提交**：`bc5c1bb`（TF2 stream 竞态修复，4 个 pointnet2 源文件）+ `f7e7772`（openings 贯通，wrapper/server 返回预测开度）。这两个提交官方仓库没有，丢了就得重做 |
| `contact_grasp_estimator.patch` | 克隆里**未提交**的本地补丁（+2/-1），bundle 之外需额外 `git apply` |
| `Dockerfile.cgn` / `.dockerignore` | 镜像构建文件，放到克隆根目录使用 |
| `cgn_server_container.py` | FastAPI 抓取服务本体（:8117），容器 CMD |
| `cgn_test_container.py` | 容器内自检脚本 |
| `compile_pointnet_tfops.sh` / `compile_pointnet_tfops_container.sh` | pointnet2 TF ops 编译脚本（宿主机版/容器内版，容器内版为 sm_120） |

## 从零重建流程

```bash
# 1. 从 bundle 恢复克隆（自带 2 个本地提交）
git clone docker/cgn/contact_graspnet.bundle external/contact_graspnet
cd external/contact_graspnet

# 2. 应用未提交补丁 + 拷入构建文件
git apply ../../docker/cgn/contact_grasp_estimator.patch
cp ../../docker/cgn/{Dockerfile.cgn,.dockerignore,cgn_server_container.py,cgn_test_container.py,compile_pointnet_tfops.sh,compile_pointnet_tfops_container.sh} .

# 3. 放入权重（Google Drive 下载或旧机器备份）：checkpoints/scene_test_2048_bs3_hor_sigma_001/

# 4. 构建镜像（pointnet2 *.so 已随 bundle 内的代码快照…若缺 .so 或改了 .cu/.cpp，先编译，见下）
sudo docker build -f Dockerfile.cgn -t cgn-tf:25.02 .
```

pointnet2 ops 重编译（首次重建/修改 .cu/.cpp 后必须）：

```bash
sudo docker run --rm --gpus all \
  -v $PWD/external/contact_graspnet:/workspace \
  cgn-tf:25.02 bash /workspace/compile_pointnet_tfops_container.sh
# 编译产出 .so 落在克隆目录，再重建镜像把 .so bake 进去
```

> 注意鸡生蛋问题：首次重建时基础镜像里没有编译产物——先用基础镜像
> `nvcr.io/nvidia/tensorflow:25.02-tf2-py3` 跑上面的编译命令（把镜像名换成基础镜像），
> 拿到 .so 后再 build。

启动与联调见 [../docs/cgn_container.md](../docs/cgn_container.md)。
