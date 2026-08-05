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

- **CGN 推理未固定随机种子（2026-08-04 实测）**：同 seed 同场景多次调用，
  top-1 候选逐 run 不同（开度 48.7/56.6mm 波动、落点水平偏差 6.2~67.4mm 波动，
  差候选会偏出方块抓空气）。验收门数据均为单次采样，复现实验需按分布理解。
- **`'+ptx85' is not a recognized feature for this target (ignoring feature)`**：
  kernel_gen JIT 的 LLVM 目标特性提示，良性，刷日志但不影响结果，不修。
- **`forward_passes` 仅支持 1**：wrapper 计算图按 `batch_size=1` 构建，
  `vision_client.grasp_cgn` 传 >1 会直接 `ValueError`（有意拦截，勿绕过）。
- **gripper 几何维持 Franka Panda（2026-08-04 曾改 Piper → 同日用户裁决回退）**：
  CGN 输出是基准不许动；改的是"末端去够输出"的链（retarget/IK/路径），
  不是输出。推理图中唯一入图的夹爪参数是 `build_6d_grasp` 的
  `gripper_depth`（默认 0.1034）；`contact_grasp_estimator.py` 已支持从
  config 透传（容器内为 docker cp 注入版本），config 不写即 Panda 默认。
  物理事实：Panda 掌心在接触面后 103.4mm，Piper 实体掌心在 45mm——
  两者不可兼得，由 retarget（contact plane 对齐）消化。
- 宿主机时代偶发一次未捕获原文的瞬时崩溃（~1/35），复测未再现；如遇 CUDA 报错
  请保留**完整日志原文**再排查（禁止 grep 过滤后覆盖保存）。

### 尺寸核查与设计约束（2026-07-31 P4）

- **cubeA = 4cm ≤ Piper 净开度 70mm** ✓ 可抓，单侧余量 **15mm**（(70−40)/2）。
  【2026-08-04 修正：净开度旧值 44.9mm 系坐标系混淆误测（指垫沿开合轴半厚度
  2.5mm 被误取为局部 y 的 15mm，详见 §6）；真值 70mm。原"单侧余量 2.5mm →
  真机标定精度硬指标（误差预算 ≲2.5mm）"的推导随之作废——标定预算大幅放宽】
- **cubeB = 5cm**：需求开度 54mm（50+2×2）< 70mm，**几何上可抓**
  【2026-08-04 修正，旧判"5cm > 开度不可抓"基于误测值】——任务上仍是
  堆叠底座，不需抓取 ✓
- **杂物**：dist_cyl 长轴（7cm，需求 74mm > 70mm）仍超开度，由
  `filter_grasps_by_width` 自然淘汰；6cm 边（需求 64mm）在 70mm 开度内
  不再淘汰【2026-08-04 修正】。
- 软约束（sampler 注释）：新增杂物尽量至少一维 ≤4cm，为将来多物体
  抓取任务留可能性；不强制、不影响 seed 复现。

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

### cgn_to_gripper 坐标变换链（2026-07-31 位置链修复并验收；
**2026-08-03 反射 bug 修复——姿态链见 §11.1，本节 Step 1 旧描述已作废**）

变换链在 0 候选时代从未被真实数据执行，首次出候选时暴露系统性 z 偏高
+20cm（≈2×0.1034）。调试结论：

- **CGN 输出约定**（官方 draw_grasps/plot_mesh 反推）：`pred_grasps_cam` 4×4
  直接放置服务端夹爪模型——原点 = 掌根，局部 +z = 逼近方向，指尖接触平面
  在局部 +z **gripper_depth** 处。【2026-08-04 曾改 0.045（Piper）→ 同日
  用户裁决回退：输出是基准不许动，维持 Panda **0.1034**（指尖尖端 0.1122
  = 0.0584+finger.stl z_max 0.0538）】
- **Piper 实测**（gripper.xml + sim 实测）：`grip_site` 由 `eef` body 定义在
  指尖垫接触面中点（site z=逼近方向、y=开合轴）→ `TCP_DEPTH_PIPER = 0`。
  最大开度：joint7 range [0,0.035]/指，两指垫中心距 0.0749m（满量程）、
  **内净距 0.070m** → `PIPER_MAX_WIDTH = 0.070`。
  【2026-08-04 修正：旧值 0.0449 = 0.0749 − 2×0.015，误把指垫盒局部 y 半尺寸
  15mm 当作开合轴半厚度；link7/8 body 的 180° 旋转使局部 y↔z 互换，真实半厚度
  是局部 z 的 2.5mm → 0.0749 − 2×0.0025 ≈ 0.070。mj_forward 实测 q7=0 两垫
  贴合、q7=0.035 内面 ±35mm（内净距 = 2·q7，q7∈[0,0.035]），可视化 mesh 内缘
  同在 ±35mm，双重验证。原始 AgileX 模型几何相同，"原模型实测 45mm"同源错误。】
- **根因**：`cgn_to_gripper` step 4 把「palm→指尖接触面」的 +z 偏移误写为
  −0.1034（后撤到 palm 后方），与正确值差 2×0.1034≈+0.207m —— 即 +20cm 特征。
  残余 x 3–5cm 亦为同一错误沿倾斜逼近轴的投影，随 z 修复一并消失。
- **修复**：两层常量分离 `GRIPPER_DEPTH_PANDA`(0.1034) / `TCP_DEPTH_PIPER`(0)，
  step 4 平移量 = 二者之差；`filter_grasps_by_width`（开度 > 0.070 丢弃并记
  `dropped_by_width` 日志）；openings 贯通 服务端→wrapper→`vision_client`
  （`return_openings=True`，旧二元组调用向后兼容）。
- **验收门**（StackClutter 场景 12 连跑，4 次出候选）：top-1 距 GT 中心
  0.75–4.67cm、距表面 0–1.46cm（门限 <2cm）、D3 0.935–0.997（门限 >0.9）、
  零崩溃 —— **4/4 出候选运行全 PASS**。
- **已知现象**：网络对 4cm 方块预测开度 0.0646–0.0725m（Panda 几何含指垫
  厚度的指尖开度，非物理需求）。**开度策略（2026-07-31 裁决；
  2026-08-04 数值修正：上限 0.045→0.070、闭合模型改为 内净距=2·q7，
  误测根源见本节上文）**：解耦
  "CGN 宽度"与"Piper 需求宽度"——
  1. 宽度过滤尺子：`required_opening = contact_dist + 2×2mm > 0.070` 才丢弃
     （contact_dist sim 阶段用方块 GT 0.04，真机改点云投影估算）；
  2. 执行开度：接近/下降 = `min(cgn_width, 0.070)`（对 4cm 方块 ≈ 满开），
     到位后闭合至 `contact_dist − 2mm`（joint7 模型：内净距 = 2·q7，
     即 q7 = 目标净距/2）；
  3. **风险提示（2026-08-04 修正后放宽）**：满开 0.070 对 4cm 方块单侧余量
     15mm，抓取点偏心 <15mm 时下降过程指垫不蹭方块；原"2.5mm 余量 /
     真机标定硬指标"基于误测值，作废。

## 7. 感知算法接入规范（长期规则）

今后任何新接入的检测/感知算法，**入 trace 的标注图是接入定义的一部分，
没有标注图不算接入完成**（含"未检出"帧——那是失败定位的关键证据）。
照 SAM3/CGN 现有模式：primitive 注册进 `trace.py` CATEGORIES/ALGO_FOLDERS +
`annotate.py` 对应分支 + `PrimitiveContextCapx` 包装 + `build_namespace` 挂载，
`record.annotation` 由 `tracer.record` 自动链接。

## 8. 相机分辨率 640（2026-07-31 转正）

- 分辨率 256→640，方块 ~15-20px → ~38px。K 自动缩放已复核
  （f = W/2/tan(30°)，640 下 554.26）。
- **离屏缓冲区**：`<visual><global offwidth offheight>` 在 robosuite
  ManipulationTask 合并 MJCF 时会被丢弃（实测 readback 仍 640×480）——
  由 `engine_capx._make_env` 程序化预置 1280×960；首个超限 render 时
  robosuite `update_offscreen_size` 一次性重建 context（已验证干净），
  之后 ≤1280×960 渲染不再重建。**不要**在 XML 里重复配置。
- 质量增益（seed 5 同场景）：cube_max 0.41→0.82（逼近官方场景水平），
  cube≥0.23 点数 11→25，召回基线 ~15-25% → 10/10 全出候选。
- EGL 稳定性：640×10 seeds 全过、零渲染异常、每 run 48–65s。
  512px 高频渲染 context 错乱旧病（engine.py:195）在 640 未复现，
  但若复现按"立即停止并汇报"处置，不硬扛不静默重试。

## 10. CGN 标注图画法（2026-07-31 用户肉审定版 F2，禁止回退）

三轮迭代结论：**封口矩形线框**（top-3 时代）与 **Panda 点云剪影**均被
用户证伪（糊团/误读），禁止复活。定稿为官方 `draw_grasps` 对齐的
开口 Π 风格（参照 outputs/cgn_official_demo/glyph_hug.png F2 面板）：

- **开口 Π 三线段**：palm 横杠 + 双指，**指尖不封口**（无指尖横杠）
- **宽度固定全开 0.08m**：官方 draw_grasps 默认约定，不随 `openings` 变
- **指尖端 = 真实接触面** `C = O + Z·0.1034`（物理，不动）；
  **掌心端显示位 = `C - Z·0.06`**（示意指长 6cm——物理 10.34cm 在
  小物体上悬空太高，纯显示压缩，非物理长度）
- **短刺 0.04m**：从掌心沿 −Z（指示臂来方向，约止于物理 palm O）
- **top-5、1px、统一绿色**（`_MASK_COLORS[0]`），按接触面深度
  painter 序（远先近后）（2026-08-03 裁决：倾斜抓时代 top-12 已成
  "乱签堆"，收窄保可读）
- REC 圈标推荐（D3 最垂直）候选，**圈 2px** 保持主角可读（2026-08-03）；0 候选存 "CGN: 0 candidates" 图；
  标题计数；`_project` 与 C 的坐标定义一律不动

历史教训：改标注分支后必须做一次端到端验证（`build_annotation` 的
try/except 会静默吞错降级为无图——putText 浮点 org 事故，2026-07-31）。

## 11. 运动学约束终审与「反射污染期」声明（2026-08-03）

### 11.1 cgn_to_gripper 反射 bug（已修复，勿回退）

`cgn_to_gripper` Step 1 原写法 `g_gl = T_flip @ g_cgn`（T_flip=diag(1,-1,1,1)
**det=-1，镜像**）。基变换必须共轭：`T_flip @ g_cgn @ T_flip`。原写法使每个
候选姿态成为非正交反射，下游 `mat2quat` 静默把反射映射为某个合法但错误的
旋转（实测往返误差 90°）。**位置链从未坏**（T_flip 无平移，右乘不改变平移，
历史 <2cm 位置验收仍然有效）；坏的是姿态——z 列（逼近）在 gb 里其实是对的，
mangled 发生在 mat2quat 转 IK 目标四元数的那一刻。

**反射污染期声明**：凡经过「cgn_to_gripper → solve_ik」链的收敛率/收敛姿态
数据一律作废，包括：4-yaw 扫描的 52% 收敛率平曲线（RISER_H 原推导）、
pedestal 时代的候选收敛观测、幻影收敛（pos 0.0000/ori 134°= 对 mangled
目标的合法收敛，solve_ik 无罪）。**干净数据只有两类**：库 FK 地图
（`ik_library.npz`，逐条 FK 复核 z列·Z=+1.000）与不经过该链的合成目标
实测。修复验证：`scripts/verify_flip_fix.py` 三层（det=+1 / 位置逐位一致 /
逼近方向物理 sane）。

### 11.2 可达性终审（库 FK 地图 + 干净合成目标互证）

- **陡降死区**：当前安装（桌面基座 x=-0.30, 6 轴 Piper 腕 ±70°）下，全部
  5 个方块位 10cm 内最小可达 tilt = 87-90°；陡降（tilt<50°）构型全库仅
  33-46 条，**全部位于基座立柱正上方/背后（x_base∈[-0.10,0.10]）——桌面
  工作区不存在"放近就可达"的位置**（杠杆 b 处决，2026-08-03 数据：
  x=0.08,z=0.12 处 tilt=23°/48° 各 8 方位 0/8 收敛）。约束本质是腕部限位
  vs 桌面夹角，与水平距离无关。
- **高度-可达性单调关系**（各方块位 6cm 球最小可达 tilt）：z=0.12→
  92-101°；z=0.24→74-83°；z=0.30→63-73°；z=0.36→50-63°；z=0.42→43-50°。
  每 +12cm 换 ~15-20°。
- **solve_ik 健康度**：对在流形上的精确库位姿复解 5/6 成功，FK 误差
  0.0000m/0.00°；失败的全部是任意合成的精确 6D 目标（6 轴零冗余臂的
  可达边界，非求解器故障）。
- **杠杆 c 实测死刑**（2026-08-03 双高度扫描 `scripts/riser_height_sweep.py`）：
  RISER_H=0.28 重叠 0/26（分布仍 16-60° vs 需 ≥67-69°）；RISER_H=0.34
  **cube 被 home 位姿的臂完全遮挡**（mask 0px，trace 帧
  traces/0803_1629_Stack_4/top/s00000.jpg）。单纯抬高在相机-臂几何下
  走不通——方块升进臂在 home 位姿的剪影里。

### 11.3 相机搬家 + 收臂让拍 + 细高块（2026-08-03 裁决，sim2real 设计约束）

**结构性矛盾**：旧场景相机在臂对面（world x=+0.38 vs 臂 x=-0.30），看到的
是方块远侧面，而臂只能从自己一侧够——可见性与可达性在相对的两面上；
且方块抬高后钻进 home 位姿臂的剪影（§11.2 杠杆 c 死因）。**真机教训：
相机应装臂同侧**——相机看到的面 = 臂将要接近的面，CGN 候选方向与
可达族天然同侧；臂在相机下方，剪影遮挡同时消失。

落地三件套（robot.xml robotview 的 pos/quat、engine_capx
ROBOTVIEW_CAM_WORLD、 CorridorFreeSampler 基准必须同步，漏一个即幽灵
bug；engine 初始化有 cam_xpos 漂移自检）：
- robotview 迁至臂同侧后上方 world (-0.52, 0.12, 1.22)，俯角 26.9°
- **收臂让拍**：感知快照前移 TUCK_Q（engine_capx 常量，库锚定+IK 复解+
  碰撞检查+FK 四重验证），观测后进执行
- **cubeA 细高化** 4×4×8cm：接触点上移 4-6cm，x/y 截面不变
  （contact_dist 仍 0.04）；垫高台维持 0.10

**C 方案登记**（库锚定几何侧抓，绕开 CGN 候选生成）：CGN 彻底出局时的
最终兜底，现在不动。依据：库内 tilt 80-100° 精确位姿 solve_ik 复解
5/6 成功 FK 0.0000/0.00°（2026-08-03 实测）。

## 9. 文件清单

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
