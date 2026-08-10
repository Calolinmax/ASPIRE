# 外部资产清单（SAM3/ + external/ 逐项对齐）

> 2026-08-10 编制。**原则：代码全部进 git，大文件一律不进**——本清单把每个不进库的
> 资产对齐到"代码真正期待的样子"：保持原名、写明来源、标注部署位置与注意事项。
> 换机/重装时按本清单逐项摆位即可，无需再猜任何路径。
> 安装顺序主线见 [env_rebuild.md](env_rebuild.md)；本文档是资产维度的总账。

---

## 0. 对齐基准：代码期待的路径树

```
ASPIRE/                              （仓库根）
├── SAM3/                            ← aspire/perception/vision_sam3.py:19 SAM3_PATH
│   └── （HF 快照 12 个文件，见 §1，两模型分别用 safetensors 与 .pt）
└── external/
    ├── contact_graspnet/
    │   │  （代码：从仓库内 bundle 恢复，见 §2）
    │   └── checkpoints/scene_test_2048_bs3_hor_sigma_001/
    │       ← docker run -v 挂载为容器 /workspace/checkpoint（cgn_container.md §4）
    ├── cgn_models/                  ← 权重下载原件存档（可选，见 §6）
    ├── cap-x/                       ← 纯契约参照，运行时零 import（§3）
    ├── piper_description/
    │   └── piper/urdf/piper_description.urdf
    │       ← pyroki IK 服务 --urdf 运行时加载（§4）
    ├── agilex_arm_mujoco/           ← MJCF 血缘参照，只读（§5）
    ├── NVIDIA_deb/                  ← 离线 deb，可选（§7）
    └── cgn_venv/                    ← 废弃，换机不带（§8）
```

---

## 1. `SAM3/` — 视觉模型权重（6.6G）⭐必需

**来源**：HuggingFace [`facebook/sam3`](https://huggingface.co/facebook/sam3)（**gated**，需网页登录+同意协议）

**获取**：

```bash
huggingface-cli download facebook/sam3 --local-dir SAM3/   # 需先 hf auth login
```

**对齐要求**：HF 快照**整目录原样**放仓库根，文件名一个都不能少/不能改：

| 文件 | 大小 | 说明 |
|---|---|---|
| `model.safetensors` | 3.3G | 文本提示模型（Sam3Model）权重 |
| `sam3.pt` | 3.3G | 点提示 Tracker 模型（Sam3TrackerModel）权重 |
| `config.json` / `processor_config.json` | 26K / 1.7K | 模型与处理器配置 |
| `tokenizer.json` / `tokenizer_config.json` / `vocab.json` / `merges.txt` / `special_tokens_map.json` | ~4.4M | tokenizer 全家 |
| `README.md` / `LICENSE` / `gitattributes` | — | HF 附带 |

**消费方**：`aspire/perception/vision_sam3.py`（双模型 GPU 占用 ~8GB，推理 ~0.17s/次）

**注意事项**：
1. ⚠️ `vision_sam3.py:19` 的 `SAM3_PATH` 是**写死的绝对路径**（`/home/stouching/...`），
   换机必须改成新机器的实际路径
2. HF 403/401 = gated 未授权，先网页同意协议再下载，不要误判为项目不存在
3. 只要权重文件齐，SAM3 服务由引擎自动拉起，无需手动启动

## 2. `external/contact_graspnet/` — CGN 代码+权重（283M）⭐抓取必需

**代码来源**（二选一，**强烈推荐前者**）：

| 方式 | 命令 | 说明 |
|---|---|---|
| ✅ 仓库内 bundle | `git clone docker/cgn/contact_graspnet.bundle external/contact_graspnet` | 含官方历史 + **2 个本地提交**（stream 竞态修复 / openings 贯通）+ 编译好的 pointnet2 `.so` + 未提交补丁 `contact_grasp_estimator.patch` 需另行 `git apply` |
| 官方克隆（会丢修复） | `git clone https://github.com/NVlabs/contact_graspnet.git` | 上游 @ `7217060`，**不含本地修复，CGN 会随机崩显存**，仅作血缘参考 |

**权重来源**：官方 [Google Drive 模型库](https://drive.google.com/drive/folders/1tBHKf60K8DLM5arm-Chyf7jxkzOr5zGl)
→ 取 `scene_test_2048_bs3_hor_sigma_001`（官方默认档，适合干净 sim 深度）

**权重摆放**：整目录放成 `external/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001/`，
关键内容（109M 总计）：

```
config.yaml                              ← 训练配置（服务启动读）
model.ckpt-144144.data-00000-of-00001    ← 实际加载的 ckpt（Dockerfile CGN_CKPT 指向）
model.ckpt-144144.index
（另有 ckpt-45045/54054/72072 历史快照、log_train.txt、训练脚本，随档附赠）
```

**部署**：`docker build` + `docker run -v ...:/workspace/checkpoint:ro`，
全流程见 [cgn_container.md](cgn_container.md) §2/§4 与 [../docker/cgn/README.md](../docker/cgn/README.md)。

**注意事项**：
1. 容器是唯一服务端，**禁止宿主机任何 Python 直跑 CGN**（TF wheel × sm_120 系统性不兼容）
2. bundle 内 `.so` 是 sm_120 编译产物；换新架构显卡才需重编译（流程见 docker/cgn/README.md）
3. 官方 `test_data`（[GDrive](https://drive.google.com/drive/folders/1TqpM2wHAAo0j3i1neu3Xeru3_WnsYQnx)）仅回归测试用，可选

## 3. `external/cap-x/` — CaP-X 框架源码（513M，只读参照）

- **来源**：[`capgym/cap-x`](https://github.com/capgym/cap-x) @ `53e9966`，MIT 许可
- **获取**：`git clone https://github.com/capgym/cap-x.git external/cap-x && cd external/cap-x && git checkout 53e9966`
- **定位**：Primitive API 契约的**对照阅读材料**。已实测运行时**零 import**——没有它任务照跑，
  但对照契约/排错时需要
- **注意**：`capx/third_party/contact_graspnet_pytorch` 子模块的构建残留无需对齐；
  pyroki 启动命令里的 `PYTHONPATH=external/cap-x` 是历史残留，加不加都能跑

## 4. `external/piper_description/` — Piper 官方 URDF（253M）⭐pyroki 运行时必需

- **来源**：[`agilexrobotics/agx_arm_urdf`](https://github.com/agilexrobotics/agx_arm_urdf) @ `f6642ce`
- **获取**：`git clone https://github.com/agilexrobotics/agx_arm_urdf.git external/piper_description && cd external/piper_description && git checkout f6642ce`
- **代码消费路径**：`piper/urdf/piper_description.urdf`——pyroki IK 服务 `--urdf` 参数
  **运行时加载**（`scripts/tools/pyroki_server_minimal.py` 启动命令）
- **注意**：目录下的 `piper_mjcf/` 是本项目工具生成的中间产物（生成器
  `scripts/tools/generate_piper_urdf_from_mjcf.py` 在库内），**不在官方克隆里，无需对齐**

## 5. `external/agilex_arm_mujoco/` — AgileX 官方 MJCF（81M，只读参照）

- **来源**：[`yanyuze1/agilex_arm_mujoco`](https://github.com/yanyuze1/agilex_arm_mujoco) @ `4cd52b0`
- **定位**：`aspire/robots/` 的 Piper MJCF 血缘参照（`piper_robot.py` 头注释），
  运行时不直接读，克隆仅为溯源

## 6. `external/cgn_models/` — CGN 权重下载原件（523M，可选存档）

- **来源**：同 §2 的官方 Google Drive，含全部档位与工具包：
  `scene_test_2048_bs3_hor_sigma_001`（默认/干净）、`scene_test_2048_bs3_hor_sigma_0025`（强噪）、
  `scene_2048_bs3_rad2_32`（无噪）、`contact_graspnet_train_and_test`（训练测试脚本包），
  zip 原件 + 解压目录各一份
- **定位**：纯存档。部署只需把 sigma_001 档放进 §2 的 `checkpoints/`；换机时噪声档用不上可不下载

## 7. `external/NVIDIA_deb/` — 离线 deb（14M，可选）

- **内容**：`nvidia-container-toolkit_1.16.2` 的 amd64/arm64 deb 包
- **来源**：NVIDIA 官方 apt 源（在线机器直接 `apt install nvidia-container-toolkit` 即可）
- **定位**：无网环境装 Docker GPU 支持的离线包

## 8. 不带清单（这些**不要**带去新机器）

| 项 | 原因 |
|---|---|
| `external/cgn_venv/`（6.9G） | 宿主机 TF fallback 遗留 venv。venv 不可移植；对应代码 `aspire/perception/cgn_server.py` 已标注"留档不用" |
| `external/checkpoints/` | 空目录 |
| `external/piper_description/piper_mjcf/` | 本地生成产物，可由库内工具再生成 |
| `SAM3/`、`outputs/`、`traces/` 等 | 纯本地产物（traces/outputs 是研究产出，想要才拷） |

## 9. 通用注意事项

1. **不要试图 `git add` 本清单内任何目录**——`.gitignore` 已拦截（SAM3/external/outputs/traces），
   强塞会撞 GitHub 单文件 100MB 与仓库体积上限
2. **版本钉住**：§3/§4/§5 的 commit hash 是从本机克隆 `git remote`+`git log` 核实钉上的，
   重克隆官方 main 可能漂移，务必 checkout 到钉版
3. **绝对路径地雷**（换机必改）：
   - `aspire/perception/vision_sam3.py:19` → `SAM3_PATH`
   - `scripts/stack.sh:28` / `scripts/wipe.sh:25` → `PY=.../envs/ASPIRE/bin/python`
   - `scripts/tools/pyroki_server_minimal.py` docstring 启动示例
4. **许可各随其主**：CaP-X（MIT）、SAM3（HF gated 协议）、CGN（官方克隆内 License.pdf）、
   AgileX 资产（各仓库自带许可）。本仓库仅作学习复现引用
5. **网络**：HF/GDrive 下载走 HTTPS 一般不受代理影响；但 Clash TUN 模式会拦
   SSH 协议的 git 操作（clone/push 报"仓库不存在"先查这个）
