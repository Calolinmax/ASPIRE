# CGN (Contact-GraspNet) 容器化部署文档

> **状态**：✅ 服务上线（2026-07-31）。容器方案已消除宿主机 TF 直跑时代的
> 非确定性 `CUDA_ERROR_ILLEGAL_ADDRESS`（根因已修复，验收 20/20）。
> 抓取质量：对干净 sim 单方块场景 0 候选为**已知模型域差现象**（非部署故障），
> 见 §6「背景共识与 sim 输入注意事项」。

Contact-GraspNet（官方 TF1 版）为 6-DoF 抓取位姿估计模型。宿主机直跑因
RTX 5090（sm_120）与 TF pip wheel 系统性不兼容而放弃，转为 **NGC 容器方案**：
容器是唯一 CGN 服务端，宿主机通过 HTTP 客户端调用。

---

## 1. 架构

```
宿主机                                    容器 (cgn-tf:25.02)
┌─────────────────────────┐              ┌──────────────────────────────┐
│ aspire/vision_client.py │  pickle/HTTP │ cgn_server_container.py      │
│   grasp_cgn()  ─────────┼─────────────►│   FastAPI, 0.0.0.0:8117      │
│        (N,4,4),(N,) ◄───┼──────────────│   wrapper.ContactGraspnet    │
└─────────────────────────┘              │   TF 2.17 + pointnet2 ops    │
                                         │   (sm_120, stream 已修复)    │
                                         └──────────────────────────────┘
```

- **契约**：POST `/grasp`，pickle `{depth(H,W米), K(3,3), seg(H,W), z_range, forward_passes}`
  → 返回 pickle `{grasps(N,4,4) 相机系OpenCV, scores(N,)}`。GET `/health` 返回纯 JSON。
- `aspire/cgn_server.py`（宿主机直跑时代服务端）**留档不用**，不要启动。

## 2. 镜像构建

```bash
cd external/contact_graspnet
sudo docker build -f Dockerfile.cgn -t cgn-tf:25.02 .
```

- 基础镜像 `nvcr.io/nvidia/tensorflow:25.02-tf2-py3`（TF 2.17.0, py3.12, CUDA 12.8 支持 sm_120）
- `pip install fastapi uvicorn`（注意：不是 Flask——旧镜像装错导致健康检查失败）
- `tf_keras` 版本动态锁定 `==tf.__version__`（`TF_USE_LEGACY_KERAS=1` 必须，
  否则老 checkpoint restore 失败；ENV 已在 Dockerfile 设置，server 代码顶部双保险）
- 代码 bake 进镜像（`.dockerignore` 排除 checkpoints/.git/examples），
  checkpoint 运行时挂载

### pointnet2 ops 重编译（仅当修改 .cu/.cpp 后）

```bash
sudo docker run --rm --gpus all \
  -v $PWD/external/contact_graspnet:/workspace \
  cgn-tf:25.02 bash /workspace/compile_pointnet_tfops_container.sh
# 然后重建镜像把新 .so bake 进去
```

## 3. 关键修复：default-stream 竞态（已修复，勿回退）

宿主机时代 `scripts/cgn_repro_wrapper_synthetic.py` 报非确定性
`CUDA_ERROR_ILLEGAL_ADDRESS`（两次运行一崩一过、崩溃点漂移、sanitizer 下消失）。
NGC 容器内同症状复现 → 排除工具链，定性为代码 bug。

**根因**：pointnet2 是 2019 年代码，全部 10 处 kernel 用裸 `<<<g,b>>>` 在
default stream 启动；现代 TF 的执行/显存回收在非阻塞流上异步进行 →
自定义 op 读到未写完/已回收显存。TF 2.2 时代流语义碰巧兼容，TF 2.17 随机崩。

**修复**（4 个源文件，已提交于嵌套 repo `bc5c1bb`）：
- 全部 kernel launch 改为 `<<<g,b,0,stream>>>`，`stream = context->eigen_gpu_device().stream()`
- 2 处 `cudaMemset` → `cudaMemsetAsync`（同流）
- 两个 .cpp 补 `EIGEN_USE_GPU` + Eigen Tensor 头（`Eigen::GpuDevice` 完整类型）

**验证**（判别 + 验收）：`CUDA_LAUNCH_BLOCKING=1` 连跑 20/20 过（判别实验）；
修复后不加任何同步、全新容器连跑 **20/20** 过（验收门）。回归资产：
`scripts/cgn_repro_*.py`（容器内经 `CGN_REPO`/`CGN_CKPT` 环境变量自适应）。

## 4. 启动与验证

```bash
# 启动（checkpoint 挂载为只读）
sudo docker run -d --name cgn --gpus all -p 8117:8117 \
  -v $PWD/external/contact_graspnet/checkpoints/scene_test_2048_bs3_hor_sigma_001:/workspace/checkpoint:ro \
  cgn-tf:25.02

# 验证三段
curl http://127.0.0.1:8117/health          # {"status":"ok","model_loaded":true}
python -c "from aspire.vision_client import grasp_cgn; ..."   # 见 scripts/cgn_verify_steps.py
MUJOCO_GL=egl python scripts/cgn_verify_steps.py              # 端到端（需 SAM3 服务 8123 在线）
```

## 5. 已知现象（非故障）

- **`'+ptx85' is not a recognized feature for this target (ignoring feature)`**：
  kernel_gen JIT 的 LLVM 目标特性提示，良性，刷日志但不影响结果，不修。
- **`forward_passes` 仅支持 1**：wrapper 计算图按 `batch_size=1` 构建，
  `vision_client.grasp_cgn` 传 >1 会直接 `ValueError`（有意拦截，勿绕过）。
- **gripper 模型为 Franka Panda**（`mesh_utils.py` 硬编码 `gripper_models/panda_gripper`，
  `config.yaml DATA.gripper_width` 亦为 Panda 尺寸）。对当前「接触点置信度」阶段无影响；
  **上线真机执行抓取前必须替换为 Piper 夹爪几何**（已知缺口）。
- 宿主机时代偶发一次未捕获原文的瞬时崩溃（~1/35），复测未再现；如遇 CUDA 报错
  请保留**完整日志原文**再排查（禁止 grep 过滤后覆盖保存）。

## 6. 背景共识与 sim 输入注意事项（重要）

**CGN 训练分布 = 合成多物杂乱场景 + 仿真 RealSense 噪声。**
ASPIRE 的 sim 输入（单方块 + 空桌面 + 零噪声深度）相对训练分布是**双重 OOD**。

实测（2026-07-31，容器内探针）：

| 输入 | contact score max（方块面） | 端到端候选 |
|------|---------------------------|-----------|
| 官方 test_data 14 场景 | 0.70–0.90 | 291–601/场景 ✅ |
| sim 单方块+空桌面（干净深度） | 0.16–0.24（阈值 0.23 线下） | **0** |
| 上者 + 推理时深度加噪（σ≤0.01z 各档） | 0.14–0.18（不升反降） | 0（N1 不成立） |
| sim + 桌面 4 个干扰物体 | **0.265**（过线） | 非零（N2 成立，姿态相关 ~2/6） |

- 输入管线已排除嫌疑四项：深度语义（平面 z/米制）✓、OpenGL/OpenCV 方向 ✓、
  K 与 fovy=60 一致 ✓、mask 索引（`seg[mask > 0]`，uint8 不能直接当索引）✓。
- **结论**：干净单方块场景 0 候选是模型域差，不是部署/输入错误。
  深度加噪无效；**场景加杂物有效**（方块面分数 0.177→0.265，端到端出候选，
  但随方块姿态仍有概率性失败）。sim 场景复杂化已纳入后续计划
  （对 SAM3 和未来数据采集同为增益）。探针资产：`scripts/cgn_probe_clutter.py`。
- 调阈值旋钮：`--arg_configs TEST.first_thres:0.19 TEST.second_thres:0.15`
  （当前 0.23/0.18），服务端 `wrapper.py` 未暴露，需要时再议。

### 已知缺口：cgn_to_gripper 坐标变换链（首次出候选后暴露）

变换链在 0 候选时代从未被真实数据执行过。杂物场景首次出候选后实测
（`scripts/cgn_probe_clutter.py` 5 连跑，2 次出候选）：

- **姿态 D3 = 0.94–0.996**（top-down 正确）✓
- **位置系统性偏差**：z 高约 +20cm（≈2×`GRIPPER_DEPTH` 0.1034 的特征），
  x 偏 3–5cm，y 基本吻合。怀疑点在 `cgn_to_gripper` 的 TCP 偏移方向/次数，
  待专项调试（未归因，原始数据见任务记录）。

## 7. 文件清单

| 位置 | 说明 |
|------|------|
| `external/contact_graspnet/Dockerfile.cgn` | 镜像定义（含 tf_keras 动态锁定） |
| `external/contact_graspnet/cgn_server_container.py` | 容器服务端（唯一在线形态） |
| `external/contact_graspnet/cgn_test_container.py` | 容器内合成数据冒烟 |
| `external/contact_graspnet/compile_pointnet_tfops_container.sh` | 容器内 ops 编译（sm_120） |
| `external/contact_graspnet/contact_graspnet/wrapper.py` | GraspEstimator 高层封装 |
| `external/contact_graspnet/pointnet2/tf_ops/{sampling,grouping}/*` | stream 修复现场 |
| `external/contact_graspnet/test_data/*.npy` | 官方测试数据（P0 判决用，勿提交） |
| `aspire/vision_client.py` | 客户端（`grasp_cgn`，8117） |
| `aspire/cgn_server.py` | 宿主机直跑服务端（留档，勿启动） |
| `scripts/cgn_repro_*.py` | CUDA 崩溃回归资产（容器/宿主机双环境） |
| `scripts/cgn_verify_steps.py` | 端到端分段验证 + D2/D3 指标 |
