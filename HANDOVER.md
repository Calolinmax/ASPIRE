# ASPIRE 项目交接文档

> 编制：2026-08-11。读者：项目接手人。
> 本文只讲"在哪、是什么、怎么用"三件事；逐文件细节以各专题文档为准（链接在文中）。

---

## 一、项目位置

### 1.1 代码与文档（GitHub）

| 项 | 位置 |
|---|---|
| 仓库 | https://github.com/Calolinmax/ASPIRE （**私有**，master 分支） |
| 首页说明 | `README.md`（架构图、安装、验证、FAQ） |
| 专题文档 | `docs/`（8 份，导引见 `docs/README.md`） |
| 逐文件地图 | `docs/project_files.md`；每个目录另有自己的 `README.md` |
| 本机工作副本 | `/home/stouching/Desktop/ASPIRE`（旧开发机） |

### 1.2 大文件（不在 GitHub，需单独获取）

| 资产 | 位置/来源 | 大小 |
|---|---|---|
| SAM3 视觉权重 | 本机 `SAM3/`；重下：HuggingFace `facebook/sam3`（gated） | 6.6G |
| external/ 第三方资产 | 本机 `external/`；来源逐项见 **`docs/external_assets.md`** | 8.5G |
| CGN docker 镜像 | 本机 `sudo docker images` 的 `cgn-tf:25.02`；重建方法见 `docker/cgn/README.md` | ~10G 级 |

> 精确到文件名的对齐说明、钉版 commit、注意事项全部在
> [docs/external_assets.md](docs/external_assets.md)——**恢复环境先看它**。

### 1.3 运行产物与配置（本机，不进库）

- `traces/`（执行 trace）、`outputs/`（调试图）、`agent_runs/`（agentic 生成物）
- `.env`：LLM key（模板见 `.env.example`，真 key 只在本机，已 gitignore）
- conda 环境 `ASPIRE`（py3.12）、pyroki 独立 venv（`~/venvs/pyroki`）

---

## 二、项目详细介绍

### 2.1 这是什么

复现论文 **ASPIRE: Agentic Skills Discovery for Robotics**（NVIDIA GEAR，2026，
[项目页](https://research.nvidia.com/labs/gear/aspire/)），基底框架
[CaP-X](https://capgym.github.io/)。在 MuJoCo 仿真中让 AgileX Piper 机械臂完成
操作任务，并让 LLM agent 基于执行 trace 自动修复/进化任务代码、沉淀技能库。

技术栈一句话：**SAM3 开放词汇分割**（感知）→ **Contact-GraspNet**（6-DoF 抓取位姿，
Docker 容器服务）→ **pyroki**（IK）+ **RRT-Connect**（运动规划）→ **cap-x 契约
Primitive API**（15 函数执行层）→ **agentic 三层**（actor 生成 / coordinator 编排 /
evolve 技能进化 + web 面板）。

架构图与组件职责见 [README.md](README.md) §1；设计细节见
[docs/agentic_design.md](docs/agentic_design.md)。

### 2.2 当前进展（截至 2026-08-11）

| 里程碑 | 状态 | 证据/入口 |
|---|---|---|
| Phase 0：API 层封版 | ✅ 15 函数契约 + 自检 **33/33**（2026-08-06 封版） | `aspire/api/primitives_capx.py`、`docs/api_asset_map.md` |
| Stack 任务线 | ✅ 端到端 PASS | `scripts/stack.sh` |
| Wipe 任务线 | ✅ 端到端 PASS（PiperWipeSpill 场景） | `scripts/wipe.sh` |
| 组件 2+3：agentic + 技能库 + web | ✅ 单测 44/44 + E2E + API 回归全过（2026-08-10/11） | `aspire/agentic|skills|web`、`SESSION_REPORT_2026-08-11.md` |
| 技能库 A/B 实证 | ✅ 无技能全灭 ↔ 有技能 wipe 一击 3/3 | `skill_library/manipulation/wipe_serpentine_coverage.md` |

### 2.3 关键决策与硬踩过的坑（为什么长这样）

1. **视觉换 SAM3**（MobileSAM 已删）：原生开放词汇文本提示，干掉颜色 hack。
   详见 [docs/sam3_vision.md](docs/sam3_vision.md)
2. **CGN 必须 Docker**：RTX 5090（sm_120）与 TF pip wheel 系统性不兼容，宿主机直跑
   随机崩显存（`CUDA_ERROR_ILLEGAL_ADDRESS`）。容器内另有 pointnet2 stream 竞态修复
   （2 个本地提交，**官方仓库没有**，已 bundle 入库）。详见 [docs/cgn_container.md](docs/cgn_container.md)
3. **运动规划选 RRT-Connect 弃 cuRobo**：球体包络在 6.7mm 贴脸跨指场景不可用 +
   安装风险（裁决记录见 `docs/api_asset_map.md` §0.6）
4. **渲染深度偏短 2.1cm@59cm**：已表征、测试阈值已标定，根因修复未定（渲染侧冻结区）
5. **EGL 渲染偶发 wedge**：引擎已内置每步健康检查 + 逐级自愈

### 2.4 未尽事项（接手后的活）

- 主线路线：[docs/roadmap.md](docs/roadmap.md)（Phase 1：技能库扩充 / 进化搜索 / 真机迁移）
- wipe held-out 2/5 残留：seed 5/6/7 的 SAM3 污渍定位失败（**感知鲁棒性问题**，非运动几何）
- 深度 bias 根因修复（待裁决）
- 进化搜索后续轮次（见 `SESSION_REPORT_2026-08-11.md` 的续跑清单）

### 2.5 项目纪律（接手必读，违反即返工）

- **🔒 冻结制度**：`docs/api_asset_map.md` §0.5——所有 `[x]` 已验收项
  （API/CGN/IK 链/碰撞/RRT/感知/机器人资产层）**只有人类批准才能改**，AI 顾问也不行
- **工作约定**：[AGENTS.md](AGENTS.md) 全文（环境路由表、开源参考优先纪律、HF 下载规则）
- **官方参考优先**：凡有开源实现，先读完官方函数体再动手（CGN glyph 三轮返工的教训）

---

## 三、项目操作指南

### 3.1 环境搭建（新机器从零开始）

**照 [docs/env_rebuild.md](docs/env_rebuild.md) 执行**，主线：

```bash
conda create -n ASPIRE python=3.12 -y && conda activate ASPIRE
pip install torch==2.13.0 torchvision==0.28.0 --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements_core_sim.txt          # 98 条已验证，别用完整 freeze（含 PyPI 无源的 ROS 包）
huggingface-cli download facebook/sam3 --local-dir SAM3/   # gated，先网页授权
# external/ 按 docs/external_assets.md 逐项摆放
```

**验证门**（每步过了再往下，详见 env_rebuild §7）：
`test_piper_capx_api.py` 33/33 → pyroki :8116 应答 → CGN :8117 health → `stack.sh` 跑通

### 3.2 日常运行

```bash
# 1) 常驻服务（任务前起好；stack.sh/wipe.sh 的 preflight 会检查并提示）
sudo docker start cgn                      # CGN 抓取服务 :8117（wipe 不需要）
PYTHONPATH=$PWD/external/cap-x ~/venvs/pyroki/bin/python \
    scripts/tools/pyroki_server_minimal.py \
    --urdf external/piper_description/piper/urdf/piper_description.urdf \
    --target-link link6 --port 8116 &      # pyroki IK 服务（SAM3 服务引擎自动拉起，不用管）

# 2) 任务（一键可视化，seed 可选）
scripts/stack.sh 0        # Stack 验收基准
scripts/wipe.sh           # Wipe 随机污渍

# 3) agentic 生成/进化（组件 2+3）
scripts/agentic.sh actor --task PiperWipeSpill --rounds 3      # 例：actor 生成
scripts/web.sh                                                 # web 面板

# 4) 测试
python scripts/tests/test_piper_capx_api.py   # API 契约 33 项
python scripts/tests/test_agentic.py          # agentic 单测
```

LLM 配置：复制 `.env.example` 为 `.env` 填入 key（当前用 Kimi coding 订阅；
配额耗尽时 file provider 可当备胎桥接，见 `aspire/agentic/llm_client.py`）。

### 3.3 踩坑速查（详细版在 README §8 与 external_assets §9）

| 症状 | 处置 |
|---|---|
| `ncclCommResume` 报错 | `pip install --force-reinstall --no-deps nvidia-nccl-cu13==2.29.7 nvidia-cudnn-cu13` |
| 渲染条纹/黑帧 | EGL wedge 已自愈；仍异常重启进程 |
| HF 下载 403 | gated repo，网页授权 + `hf auth login` |
| git push 报"仓库不存在" | Clash Verge **TUN 模式会拦 SSH**——关掉 TUN 再推 |
| 换机后路径报错 | 三处写死路径要改：`vision_sam3.py:19`、`stack.sh:28`、`wipe.sh:25` |
| CGN 想宿主机直跑 | **禁止**，只走 Docker（见 §2.3-2） |

### 3.4 Git 操作

- 推送：本机已配 `github-calolinmax` SSH 别名，`git push` 直用；
  新机器把新公钥加到 Calolinmax 账号即可
- 边界：`SAM3/ external/ outputs/ traces/ agent_runs/ .env 笔记.pdf` 均被
  gitignore——**不要强塞大文件进库**（GitHub 单文件 100MB 上限）

---

## 附：交接待办（原负责人填写）

- [ ] 接手人 / 联系方式：
- [ ] 仓库权限安排（加协作者 / 转移 / 保持现状）：
- [ ] `traces/`、`outputs/`、`笔记.pdf` 等本地产物归属：
- [ ] 未尽事项优先级共识（roadmap 哪条先做）：
