# 项目文件总览（逐文件说明）

> v2.0，2026-08-07 编制（同日目录重组后修订）。范围 = 仓库一方文件（git 已跟踪
> + 未跟踪的一方新增），逐文件说明用途/关键内容/状态/坑。
> **不逐文件展开**：`external/`（第三方资产，不入库，见 §8）、`SAM3/`（权重，不入库）、
> `traces/` `outputs/`（运行产物）、`__pycache__/`。
> 图例：🔒 = 封版冻结（`chmod a-w` 机械保护，解冻仅人类亲自 `chmod +w`，
> 覆盖引擎/API/证据/规划/**感知服务/机器人资产**各层——后两层 2026-08-07 用户指令加锁）;
> 🔏 = 注释条款锁定（相机位姿/home 位姿/场景约束/CGN glyph 等，仅用户可改）。
> **每个文件夹都有自己的 README.md 导引**（2026-08-10 补齐）；本文档是全局详表。

---

## 0. 快速导航（分层 → 文件）

| 层 | 文件 |
|---|---|
| 任务层（可自由写） | `scripts/stack.sh` + `scripts/tasks/stack.py`、`scripts/wipe.sh` + `scripts/tasks/wipe.py`（统一 sh 入口标准，见 §4） |
| API 层 🔒 | `aspire/api/primitives_capx.py`（契约 15 + 契约外 5）、`docs/primitive_api_capx.md`（权威参考） |
| 执行引擎层 🔒 | `aspire/engine/engine.py`（基类）、`aspire/engine/engine_capx.py`（Piper/cap-x 主线） |
| 证据层 🔒 | `aspire/evidence/trace.py`、`aspire/evidence/annotate.py` → `traces/` |
| 规划层 🔒 | `aspire/planning/motion_planner.py`（RRT-Connect）、`aspire/planning/pyroki_client.py` + `scripts/tools/pyroki_server_minimal.py`（IK 服务） |
| 感知服务层 🔒 | `aspire/perception/`：`vision_server.py` + `vision_sam3.py` + `vision_client.py`（SAM3，:8123）、`cgn_server.py`（CGN 容器内，:8117） |
| 机器人资产层 🔒 | `aspire/robots/`（注册入口 + 模型类 + MJCF/网格/IK 库） |
| 场景层（不锁） | `aspire/engine/engine_capx.py` 内 StackClutter、`aspire/envs/wipe_spill.py`（PiperWipeSpill）；新场景放 `aspire/envs/` 并经 `aspire/robots/__init__.py` 注册（需解锁） |
| **Agentic 层**（2026-08-10 新建，不锁） | `aspire/agentic/`（LLM client/trace 摘要/prompts/并行评估/actor/coordinator/进化搜索/CLI）、`aspire/skills/`（技能库 schema/库/审计）、`aspire/web/`（Web UI）——论文组件 2+3，见 `docs/agentic_design.md` |
| **技能库数据**（git 跟踪） | `skill_library/<category>/<name>.md` + `index.json`（5 条种子条目，E.5 格式）；运行产物 `agent_runs/`（gitignore） |
| 文档层 | `README.md`、`AGENTS.md`、`docs/` |
| 复现基准 | `open_details/`（官方任务代码 3 份 + skill 样例）、`ASPIRE源论文/` |

运行时的外部常驻依赖：CGN docker 容器（:8117，仅 Stack 需要）、pyroki 服务（:8116，
两条任务线都需要）、vision_server（:8123，引擎自动拉起，无需手动预检）。

---

## 1. 根目录文件

| 文件 | 说明 |
|---|---|
| `README.md` | 项目门面：简介+架构图+环境重建说明书（安装五步/验证/入口/FAQ 踩坑/许可）。新环境重建**从它开始**。 |
| `AGENTS.md` | 给 AI agent 的仓库工作约定：环境路由表（主代码=conda ASPIRE py3.12、CGN 一律 docker、禁系统 python）、**开源参考纪律（先读官方实现完整函数体，禁止凭签名手搓）**、trace 四要素硬性要求、进化搜索并行要求。每次执行前必读。 |
| `requirements_ASPIRE312.txt` | conda ASPIRE（py3.12）pip freeze 全量快照（含 ROS2 Jazzy 包/robosuite 1.5.2/mujoco 3.3.7/transformers 5.14.1/numpy 1.26.4 钉版等）。**torch 不在此装**——须按 README §4.2 用 PyTorch 官方 cu130 源单独先装。pyroki 不住本环境（独立 venv）。 |
| `.gitignore` | 排除 `__pycache__`/`.env`/`traces/`/`outputs/`/`*.log`/`SAM3/`/`external/`/`models/`/`.claude/`/IDE 配置/个人笔记。 |
| ~~`ASPIRE_REPRO_PLAN.md`~~ | **已删除（2026-08-07 用户指令）**：Phase 0 已完成，前瞻内容提炼进 `docs/roadmap.md`。 |
| ~~`handoff_piper_control_api.md`~~ | **已删除（2026-08-07 用户指令）**：Phase 0 交接 prompt，使命已完成（33/33 PASS、封版）。 |

## 2. `aspire/` 核心包（按层分目录，2026-08-07 重组）

```
aspire/
├── __init__.py          # 刻意空包（__all__=[]）：子模块按需显式导入，避免把
│                        #   vision_server 等独立进程拖入 robosuite/mujoco/EGL 初始化
├── engine/              # 执行引擎层 🔒
│   ├── engine.py        #   基类（363 行）
│   └── engine_capx.py   #   cap-x/Piper 主线（700 行）
├── api/                 # Primitive API 层 🔒
│   ├── primitives_capx.py  # cap-x 契约 15+契约外 5（1364 行）
│   └── primitives.py    #   Panda 旧线存档（426 行）
├── evidence/            # 证据层 🔒
│   ├── trace.py         #   调用记录/帧流落盘（230 行）
│   └── annotate.py      #   视觉证据标注（193 行）
├── planning/            # 规划层 🔒
│   ├── motion_planner.py   # RRT-Connect（221 行）
│   ├── pyroki_client.py    # IK 服务客户端（61 行）
│   └── pyroki_joint_map.json # URDF↔MJCF 映射标定存档（joint3 残差大不可信，无代码消费）
├── perception/          # 感知服务层 🔒（2026-08-07 加锁）
│   ├── vision_server.py #   SAM3 推理服务进程 :8123（89 行）
│   ├── vision_client.py #   引擎侧纯 HTTP 客户端（116 行）
│   ├── vision_sam3.py   #   SAM3 本体（transformers 本地权重，166 行）
│   └── cgn_server.py    #   CGN FastAPI 服务壳 :8117（跑在 docker 容器内，100 行）
├── envs/                # 场景层（不锁，新场景放这里）
│   └── wipe_spill.py    #   PiperWipeSpill 污渍场景（215 行）
├── agentic/             # Agentic coding 层（2026-08-10 新建，不锁）
│   ├── config.py / llm_client.py / trace_digest.py / prompts.py / task_spec.py
│   ├── evaluate.py / actor.py / coordinator.py / evolve.py / cli.py
├── skills/              # Skill Library 层（2026-08-10 新建，不锁）
│   ├── schema.py / library.py / synthesize.py
├── web/                 # Web UI 层（2026-08-10 新建，不锁）
│   ├── server.py / static/index.html
└── robots/              # 机器人资产层 🔒（2026-08-07 加锁，见 §3）
```

### `agentic/`（Agentic coding 层，2026-08-10 新建，不锁）

> 论文组件 3（+组件 1 的 LLM 消费侧）。**导引见 `aspire/agentic/README.md`，
> 设计对照表见 `docs/agentic_design.md`**。测试 44/44 + 真实引擎 E2E PASS。

| 文件 | 说明 |
|---|---|
| `config.py` | LLM 配置（env/.env，`ASPIRE_LLM_*`）：provider = openai/anthropic/file/mock；file=与 AGENTS.md §4 交互式工作流同构的文件桥接。 |
| `llm_client.py` | vendor cap-x `capx/llm/client.py`：单次/并行集成（并发候选+LLM 综合）/流式/多模态 data URL/fence 抽取；delta=provider 分派、重试封顶、删 OpenRouter 改道。 |
| `trace_digest.py` | 论文 §2.1 LLM 消费侧：trace.json → 调用摘要+失败信号扫描+失败邻近前后帧/标注图（max_images 上限）。 |
| `prompts.py` | ACTOR_SYSTEM（任务代码规则+E.2 FORBIDDEN+API 参考）、E.3 findings 逐字模板、E.1 审计、E.4 候选生成（反过拟合条款）与 task_analysis 更新。 |
| `task_spec.py` | TaskSpec（场景/指令/成功判定/seed 划分/基线路径）；内置 Stack、PiperWipeSpill。 |
| `evaluate.py` | **并行评估 harness**（AGENTS.md §2）：(候选,seed) 子进程池跑引擎 CLI；vision_server 单例预拉；cancel_all 支持 web 停止。 |
| `actor.py` | E.3 修复闭环：fast path→debug loop（≤3 轮）→Stage 2 一次性→findings.md。 |
| `coordinator.py` | E.1：progress.json 队列、分派、只读 findings、串行入库、不重复分派。 |
| `evolve.py` | **Algorithm 1 逐行** + E.4：task_analysis.md 跨代、candidate_A verbatim 精英种子、K 并行评估、θ 早停、基线回落。 |
| `cli.py` | `python -m aspire.agentic.cli {actor|evosearch|coordinator|skills|digest}`。 |

### `skills/`（Skill Library 层，2026-08-10 新建，不锁）

> 论文组件 2。格式标杆 `open_details/skill_grasp.md`；导引见 `aspire/skills/README.md`。

| 文件 | 说明 |
|---|---|
| `schema.py` | `SkillEntry`（四要素+Evidence+Code Sketch）markdown 往返；`parse_findings`（E.3 schema）。 |
| `library.py` | 文件系统库（`skill_library/`）+ index.json 原子更新；`admit()` 文件锁串行；`format_for_prompt()` 全量注入；`retrieve()` 关键词检索。 |
| `synthesize.py` | findings→SKILL 审计管线：LLM 可复用性审计 + `check_api_compliance` AST 静态卡（契约 15+契约外 5+np+helper 之外禁调用）。 |

### `web/`（Web UI 层，2026-08-10 新建，不锁）

> cap-x `capx/web` 最小可用版复刻；导引见 `aspire/web/README.md`。

| 文件 | 说明 |
|---|---|
| `server.py` | FastAPI：REST（status/skills/traces/run/stop）+ WS 事件流 + 静态托管；单活跃会话；线程→asyncio 事件总线。 |
| `static/index.html` | 单页 SPA（零构建）：运行控制台/技能库/Traces 三栏。 |

### `skill_library/`（仓库根，技能库数据，git 跟踪）

> 5 条种子条目（2026-08-10，E.5 格式 + 本项目实测 Evidence）：grasping/piper_reach_and_grasp、
> localization/sam3_prompt_cascade、motion/rrt_move_discipline、manipulation/wipe_serpentine_coverage、
> debugging/trace_driven_repair。`index.json` 由库自动维护，勿手改。

### `agent_runs/`（运行产物，gitignore）

> agentic 运行产物：候选代码/leaderboard/task_analysis.md/findings.md/progress.json/各 run 的 traces/。

### `engine/`（执行引擎层 🔒）

| 文件 | 说明 |
|---|---|
| `engine.py` 🔒（363 行） | 执行引擎**基类**：`run(code)` 用 `exec` 注入 API 命名空间执行任务代码并全程记录 trace；四个子类钩子（`_env_kwargs/_make_env/_post_reset/_build_namespace`）供 engine_capx 覆盖。核心资产是 **EGL 自愈**：`_img_corrupt` 双指标（p95 梯度+死黑占比，含"双相机读到同一 buffer"的跨相机一致性检查）+ `recover_renderer` 两级恢复（L1 尺寸抖动/L2 重建离屏 context）。render 模式 stdout 用 _Tee 实时回显。Panda 旧线也用它（ENV_DEFAULTS 是 Panda 双相机 256px）。 |
| `engine_capx.py` 🔒（700 行） | **cap-x/Piper 执行引擎（现役主线）**：直写 position 执行器 ctrl（=Piper 真机位置伺服语义），1 tick = ctrl + 25 个 sim.step（20Hz）；`StackClutter` 场景类（杂物数量/target-only 开关/走廊约束采样器）；`--render` 自管 `launch_passive` 交互窗口 + 实时配速 + 关窗不中断；`hold_viewer_open` 结果保持；`draw_grasp_glyphs` CGN 候选 3D glyph（品红=当前执行位）；vision_server 自动拉起/回收；"关引擎必落盘"不变量。CLI：`python -m aspire.engine.engine_capx --code task.py --task Stack --seed 0 [--render --render-slowdown 0.5 --official-stack --clutter N --target-only]`。 |

### `api/`（Primitive API 层 🔒）

| 文件 | 说明 |
|---|---|
| `primitives_capx.py` 🔒（1364 行） | **API 层本体**：`PiperControlApiReduced`——cap-x 契约 15 函数（感知 4/几何抓取 3/运动 2/控制 2/工具 4）+ 契约外 5 个（grasp_cgn/move_to_joints_planned/execute_legs_rrt/set_gripper_ramp/draw_grasp_glyphs）。后端开关类常量 `IK_BACKEND="pyroki"`/`GRASP_BACKEND="cgn"`。内含 solve_ik 兜底链（pyroki→DLS+IK 库近邻种子→滚转扫描）、`cgn_to_gripper` 变换链（🔏 CGN 输出约定 0.1034 冻结）、`_plan_grasp_geometric`（CGN 空候选回落）、碰撞事件收集。测试 33/33。 |
| `primitives.py`（426 行） | **存档（Panda 旧线）**：`PrimitiveContext` 14 个 API（OSC 闭环 move_to_pose、手写 DLS solve_ik 姿态同伦绕 π 奇异性等）。注意它**进程内** import vision_sam3（torch/CUDA 与 EGL 同进程），与 cap-x 线的进程隔离路线是两条视觉路径的分叉点。（旧线验收 harness demo_stack_dualcam/taskcode 已于 2026-08-10 删除，git 历史可查。） |

### `evidence/`（证据层 🔒）

| 文件 | 说明 |
|---|---|
| `trace.py` 🔒（230 行） | trace 落盘：`Tracer`（帧流按 (流,仿真步) 去重、`record` 记调用、`finalize` 写 trace.json）；`CATEGORIES`（primitive→检测/规划/抓取/控制四类映射）；`ALGO_FOLDERS`（算法→标注图文件夹）；`summarize()` 大数组压成 shape/min/max/mean。目录约定：`traces/MMDD_HHMM_任务名/`，images/top\|wrist/ 固定帧率+调用边界补帧，算法标注图每次调用都存（含"未检到"帧——失败定位关键证据）。 |
| `annotate.py` 🔒（193 行） | 视觉证据标注：`build_annotation` 按 primitive 分派（SAM3 mask 着色+质心+score、molmo 品红十字、深度 VIRIDIS 底图、CGN 3D glyph 线框投影）。CGN glyph 为 🔏 F2 定稿（开口 Π 三线段+短刺、统一绿 1px、指尖端 O+Z·0.1034、封口矩形/点云剪影已证伪禁止复活）。任何异常返回 None——标注绝不污染任务执行。 |

### `planning/`（规划层 🔒）

| 文件 | 说明 |
|---|---|
| `motion_planner.py` 🔒（221 行） | RRT-Connect 关节空间规划器：接触级判对（臂体 `robot0_g*_col`/`robot0_link*` vs table/pedestal/floor/cube/riser，腕部 `robot0_g6_col` 对目标豁免）+ 净距级判对（手-方块 3mm，仅在 cubeA_main 在场时启用——wipe 场景无方块故天然免疫）；读最新方块位姿。cuRobo 按 D6 决策关闭后这是唯一规划器。 |
| `pyroki_client.py` 🔒（61 行） | pyroki IK 服务 HTTP 客户端（:8116，JSON 协议）：`ik_pyroki(pos, quat_xyzw, prev_cfg=None)`→(6,) 关节角；prev_cfg 热启动；server 失败抛 RuntimeError 由 solve_ik 兜底链接管。 |
| `pyroki_joint_map.json`（31 行） | **存档标定记录**：URDF↔MJCF 关节映射拟合表。⚠️ joint3 err≈238mm 不可信；当前无代码直接读它（URDF 改由 MJCF 直接生成）。 |

### `perception/`（感知服务层 🔒，2026-08-07 加锁）

| 文件 | 说明 |
|---|---|
| `vision_server.py` 🔒（89 行） | SAM3 推理服务进程（:8123，pickle over HTTP）：GET /health、POST /segment（text/point）。**存在理由（2026-07-28 实测）：同进程内 PyTorch CUDA 初始化后 EGL 离屏渲染永久损坏**，故引擎只做 EGL、CUDA 全隔离到本进程。错误以 `{"error":...}` 200 回传不 500。 |
| `vision_client.py` 🔒（116 行） | 引擎侧纯 HTTP 客户端（不引 torch/CUDA）：SAM3 两分函数签名与 vision_sam3 完全一致 + `grasp_cgn` 走 CGN 服务。端口约定 SAM3=8123/CGN=8117；`forward_passes` 仅支持 1。 |
| `vision_sam3.py` 🔒（166 行） | SAM3 视觉本体（transformers 本地权重，只在 vision_server 进程内用）：text 提示走 Sam3Model 开放词汇概念分割、point 提示走 Sam3TrackerModel；输出契约 `{mask u8, score, area, centroid, aspect_ratio}` 按 score 降序。 |
| `cgn_server.py` 🔒（100 行） | CGN 官方 TF 版 FastAPI 服务壳（:8117）：sys.path 插 external/contact_graspnet 后用官方 `wrapper.ContactGraspnet`；`find_checkpoint()` 自动选权重。**必须跑在独立环境**（docker 容器 cgn-tf:25.02；宿主机直跑留档禁用——RTX 5090 sm_120 触发非确定性 CUDA_ERROR_ILLEGAL_ADDRESS）。 |

### `envs/`（场景层，不锁）

| 文件 | 说明 |
|---|---|
| `wipe_spill.py`（215 行） | **Wipe 场景**（2026-08-06 新建）：`PiperWipeSpill` = 官方 wipe.py `_load_model/_reset_internal` 抄录换 ManipulationEnv 直建（绕开 WipingGripper 断言——它会拆掉 Piper 夹爪致 grip_site/执行器全灭）；桌几何锁 Stack 值 🔏；WipeArena coverage 0.25（污渍落可达带旁）；默认 `gripper_types="PiperWiperGripper"`。成功判定=**访问覆盖率**（`_get_observations` 侧效应统计擦板足迹矩形覆盖的 marker，即时 alpha→0 淡出，≥50% 判 success）。marker 坐标**延迟缓存**（reset_arena 只写 body_pos，body_xpos 要等 sim.forward——"一下全消失"事故根因）。新场景照此办理：放本目录 + 挂 `aspire/robots/__init__.py` 注册（需解锁）。 |

## 3. `aspire/robots/` 机器人资产层 🔒（2026-08-07 用户指令加锁）

| 文件 | 说明 |
|---|---|
| `__init__.py`（22 行） | **注册入口**：import 即完成 Piper 机器人 + 两种夹爪 + PiperPedestal 台架 + PiperWipeSpill 场景的全部注册。**engine_capx 冻结后，本包是其模块加载期唯一的注册钩子**——新场景/新夹爪挂这里（本层已锁，改动须用户解锁）。 |
| `piper_robot.py`（93 行） | Piper ManipulatorModel（MJCF 拆自松灵官方 agilex_arm_mujoco）：`PIPER_BASE_XPOS_TABLE=(-0.30,0,0.55+PIPER_BASE_RAISE)`（+pedestal 0.25 = 基座 z=0.80 落桌面）；`init_qpos=zeros(6)` 🔏（全零 home 用户指定）；默认 base/gripper/controller 指定。踩坑史：25cm 立柱落地安装在 0.80 桌面下整臂埋桌下（相机实锤）→ 改桌面安装。 |
| `piper_mount.py`（25 行） | PiperPedestal 台架包装类（top_offset 0、半径 0.12）。docstring 自称"25cm 落地立柱"，现实际当桌面增高台用（语义差，几何没变）。 |
| `piper_gripper.py`（59 行） | 平行夹爪 GripperModel：单 dof（joint7 slide 0..0.035，joint8 equality 镜像）；format_action 与 PandaGripper 同语义。注释里"净距 45mm"是旧值——实为 **70mm**（2026-08-04 修正）。Stack 默认夹爪。 |
| `piper_wiper_gripper.py`（60 行） | **擦拭工具 GripperModel**（2026-08-07）：与 PiperGripper 同一 XML 骨架（grip_site/joint7/8/equality/执行器逐字一致——冻结引擎依赖面零差异），指体几何全拆；仅 wipe 场景选用。 |
| `assets/piper/robot.xml` | 臂体 MJCF：joint1-6（j5 腕限位 ±70°——大量可达性坑的物理根源）；**robotview 相机 pos/quat 🔏 锁死**（2026-08-03 用户肉审定稿，同步项 engine_capx.ROBOTVIEW_CAM_WORLD 同锁）；eye_in_hand 腕部相机；6 个 position 执行器（j1-j3 kp80/kv5、j4 kp40、j5-j6 kp10/kv1.5）；碰撞 geom 无名由 robosuite 自动命名 `robot0_g{i}_col`（全管线碰撞判对命名面的来源）；link6 腕部碰撞实体（08-05 裁决恢复，幽灵化方案作废）。 |
| `assets/piper/pedestal.xml` | 25cm 圆柱立柱（带底盘）。现作桌面增高台。 |
| `assets/piper/gripper.xml` | 夹爪 MJCF：`piper_hand`→`eef`（z=-0.045, 绕 y 180° 翻转——**不可去**，实测去掉后 top-down 指令手指朝上掌心撞桌）承载 grip_site；link7/8 带 box 碰撞指垫；执行器 kp=120（08-04 用户批准增强）。**注意 hand 系 +z = 逼近轴方向**（FK 实测，指垫在 -0.045 抓取时朝后上方）——头注释"手指向 root -z 伸展"有误导。 |
| `assets/piper/wiper_gripper.xml` | **Franka 平板擦头（现役，wipe 用）**：12×5×3cm 海绵板 + 颈，几何/接触参数移植自官方 wiping_gripper.xml，**38° 补偿安装**（滚转锁定全姿态 IK 实测 38° 唯一整路径收敛）+ **solref 0.05→0.20 软化**（官方参数系 Franka 力矩臂设定，Piper 位置伺服会被百牛法向力顶漂）；site→板面心沿逼近轴 Δ=0.090（=任务代码 WIPER_FACE_L，改模型须同步重测）。 |
| `assets/piper/wiper_gripper_ball.xml` | **存档（首版球形擦头）**：r=12mm 球（L=0.074），08-06 首版 08-07 被 Franka 平板取代，回退备用。 |
| `assets/piper/default_piper.json` | composite 控制器配置（JOINT_POSITION kp200/kd40/kv20 + 夹爪 GRIP）。注意这是**控制器层**增益，与 robot.xml **执行器层** kp/kv 是两套参数，勿混。 |
| `assets/piper/ik_library.npz`（~11MB） | **IK 种子库**（2026-08-03 重建，rng=42 可复现）：Q/P/Z/Y 四键各 **201,182 条** float32。筛选：开口距竖直向下 <55° 且 site 落工作盒 +1/25 全域兜底。存在理由：Piper 短臂+腕限位使俯身可达流形极薄，随机种子 DLS 打不中；库检索近邻种子+DLS 精修是 solve_ik 兜底链的一环。几何/安装变更后须用 `scripts/tools/build_piper_ik_library.py` 重建。 |
| `assets/piper/` 网格（84 个：72 OBJ + 12 STL） | **双轨制**：OBJ=纯视觉件（group=1，整杆外观按颜色拆零件）；STL=碰撞体为主（group=0，robot0_g*_col 的网格来源；link7/8.stl 在 gripper.xml 仅作视觉）。link2_gray/link2_red/linke2_dark_gray 为官方导出残留未引用件。 |

## 4. `scripts/`（任务层，统一 .sh 入口标准，2026-08-07 重组）

> **规范文档：[../scripts/README.md](../scripts/README.md)**——任务层格式标准、
> 新任务编写指南（任务代码规则/场景/启动器模板/验证流程）、服务依赖与常见坑。
> **任务层格式标准（用户指令）**：每个任务 = `scripts/tasks/<task>.py`（引擎注入的
> 任务代码，禁止 import，只用契约 15 函数+契约外+np）+ `scripts/<task>.sh`
> （一键启动器：preflight 服务检查 → exec 引擎 `--render`）。
> 分组：**tasks 2 / tests 1 / tools 8 / archive 17**（历史诊断=一次性攻坚产物，
> 结论已落进文档/记忆，删前请先读文件头结论）。
> 2026-08-10 用户批准精简 7 个（git 历史可恢复）：test_piper_build.py（被 33 项覆盖）、
> demo_stack_dualcam.py + demo_stack_taskcode.py（Panda 旧线验收资产）、
> cgn_repro_gather_point / cgn_repro_pointnet_msg / cgn_repro_wrapper_synthetic
> （CUDA 崩溃最小复现，容器化结论已固化）、calc_cam_rot.py（一次性计算稿）。

### 根目录：任务入口（统一 .sh）

| 文件 | 说明 |
|---|---|
| `README.md` | **任务层规范**（2026-08-10 定稿）：格式标准 + 新任务编写指南 + 服务依赖 + 常见坑。新任务先读它。 |
| `stack.sh`（55 行） | Stack 可视化一键入口：`scripts/stack.sh [seed] [slowdown]`（seed 默认 $RANDOM，slow 默认 0.5=2 倍速）。预检 CGN :8117（须 model_loaded:true）+ pyroki :8116（任何 HTTP 应答即算活）；exec `python -m aspire.engine.engine_capx --code scripts/tasks/stack.py --task Stack --official-stack --render`。 |
| `wipe.sh`（45 行） | Wipe 可视化一键入口（镜像 stack.sh）：**预检只查 pyroki**（无抓取规划故 CGN 不需要）；`--code scripts/tasks/wipe.py --task PiperWipeSpill`。 |
| `agentic.sh` | **agentic coding 入口**（2026-08-10）：`scripts/agentic.sh actor|evosearch|coordinator <任务> [参数]`；预检 pyroki/CGN + LLM 配置提示（无 .env 时引导 file 桥接）。 |
| `web.sh` | **Web UI 入口**（2026-08-10）：FastAPI+静态 SPA，`http://127.0.0.1:8200`。 |

### `tasks/`（任务代码，引擎注入执行）

| 文件 | 说明 |
|---|---|
| `stack.py`（500 行） | **Stack 任务代码**：SAM3 定位双块 → CGN 姿态抓红 → 堆绿。运动层 = cgn_execute_grasp 验证链 + 先算后动门控（全链预解任一不收敛换候选）+ 种子锁重放 + 夹持滑脱检测 + 扰动重定位 + 验收一次修复。头部记三次迭代定稿的坑（顶朝下死区、抓取带 p50≈44°/搬运带 p50≈67° 等）。 |
| `wipe.py`（337 行） | **Wipe 任务代码**：抄官方 `open_details/primitive_api_wipe.py` 流程 + Piper 适配三坑（(按压×倾角×方位) PRESOLVE_LADDER 预解阶梯，38° 滚转锁定首发；pad_to_site 补偿板面心 vs grip_site 的 5.5cm 水平错位；全路径预解+矩形收缩重试）。 |

### `tests/`（自检）

| 文件 | 说明 |
|---|---|
| `test_piper_capx_api.py`（258 行） | **契约 15 函数全量自检，33 项**（A 观测结构 6 / B SAM3 2 / C 反投影标定 2 / D plan_grasp 3 / E 点提示 2 / F OBB 1 / G 运动 3 / H 夹爪 2 / I 类结构 4 / J 工具 5 / K select_top_down 3），全过 exit 0。GT 仅测试可用；C 区注释载"GT 尺寸从 model.geom_size 读勿硬编码"教训。 |
| `test_agentic.py` | **agentic/skills 单元测试 44 项**（2026-08-10）：schema 往返/API 合规/库 CRUD 与并发串行/trace_digest 失败信号/llm_client（fence/collapse/mock/集成/anthropic 转换）/actor 修复闭环/evolve Algorithm 1 行为/coordinator 入库审计——全 mock LLM + fake executor，不碰仿真。 |
| `test_agentic_e2e.py` | **agentic 端到端冒烟**（2026-08-10）：真实引擎 × mock LLM 重放 stack.py——fast path（seed 0）→ Stage 2（seed 1）→ findings → 审计入库冒烟条目（测后清理）。前置 CGN/pyroki 在线。 |

### `tools/`（资产生产线 / 查看器 / 常驻服务）

| 文件 | 说明 |
|---|---|
| `cgn_execute_grasp.py`（584 行） | **CGN 抓取验收门**（自建引擎 CLI 直驱）：SAM3→CGN→宽度过滤→最多试 3 候选（预抓后撤→下降→闭合→抬升→悬停 2s 不滑落=成功）。`--view` 交互窗口。tasks/stack.py 的运动层即本脚本验证链。 |
| `pyroki_server_minimal.py`（167 行） | **pyroki IK 常驻服务**（独立 venv `~/venvs/pyroki`，FastAPI :8116）：`POST /ik`（prev_cfg 热启动）+ `/fk`；无 /health 路由。启动：`PYTHONPATH=external/cap-x ~/venvs/pyroki/bin/python scripts/tools/pyroki_server_minimal.py --urdf external/piper_description/piper_mjcf/urdf/piper_mjcf.urdf --target-link link6 --port 8116`。关键修复注释：rest_cost weight 10→0.01（原值把解拉离目标 140-284mm）。 |
| `build_piper_ik_library.py`（95 行） | IK 种子库生产线（FK 稠密采样+俯身筛选+全域兜底）。**几何/安装变更后必须重建**。 |
| `generate_piper_urdf_from_mjcf.py`（174 行） | 从 MJCF 直接生成 pyroki 配套简化 URDF（frame+限位，无 mesh）。模型变更时重跑。 |
| `check_cgn_pose.py`（271 行） | CGN 位姿对应性肉审工具：臂控到 CGN top-1 候选保持，品红 glyph=执行候选，`--show-raw` 叠绿 glyph=CGN 原始候选（glfw 交互窗口）。 |
| `view_cgn_poses.py`（100 行） | CGN top-N 候选 3D 可视化查看器（幽灵 glyph，top-1 品红）。 |
| `view_scene_interactive.py`（40 行） | 通用场景查看器（鼠标交互 + Joint 滑块拖关节 + 空格暂停）。需显示器。 |
| `watch_grasp_live.py`（179 行） | 实时围观"抓-搬-放"循环（红块传送回固定点→收臂让拍→SAM3→预解→3 倍慢放执行→持块停留 6s，循环到关窗）。调试围观用。 |

### `archive/`（历史诊断，21 个）

| 文件 | 一句话（攻坚对象 → 结论） |
|---|---|
| `calibrate_pyroki_tcp.py` | pyroki 接入标定：FK(link6) vs grip_site 偏差 <1mm → TCP_OFFSET=0 可用。 |
| `calibrate_urdf_mjcf_map.py` | URDF↔MJCF 关节映射拟合（杆长一致，偏差来自零位/轴系约定）→ 产物 pyroki_joint_map.json（joint3 不可信）。 |
| `verify_mjcf_urdf_fk.py` | MJCF 派生 URDF 的 FK 一致性验收（URDF 已定型）。 |
| `calc_cam_rot.py` | ~~相机绕光轴旋转 quat 的一次性计算稿~~（2026-08-10 已删，git 历史可查）。 |
| `check_cameras.py` | A0.5 相机标定肉审（home/预抓/抓取三位形渲染）。 |
| `measure_sidecam_tilts.py` | 裁决 4a 数据：真实 tilt 分布 vs ik_library 支持度。 |
| `measure_true_tilts.py` | 裁决 2/3 数据：全候选 tilt 分桶 + 库陡降构型甜点区实测。 |
| `cgn_probe_clutter.py` | CGN 探针：干净单方块=双重 OOD 0 候选，加杂物后分数过线（Stack 默认 clutter=4 的由来）。 |
| `cgn_repro_gather_point.py` / `cgn_repro_pointnet_msg.py` / `cgn_repro_wrapper_synthetic.py` | ~~CUDA illegal address 最小复现三件套~~（2026-08-10 已删，git 历史可查；结论=CGN 必须容器化）。 |
| `cgn_verify_steps.py` | CGN 坐标变换链分段可视化（相机系→基座系→TCP 三步 D2/D3 判据）。 |
| `arbitrate_ik.py` | IK 裁决实验：独立多起点 DLS 直接回答"IK 有 bug 还是运动学墙属实"。 |
| `verify_ik_convergence.py` | A1/A2 验收：pyroki vs DLS 工作区网格收敛对比（pyroki 胜出任默认后端）。 |
| `hunt_phantom_ik.py` | 幻影收敛捕获（报收敛但 FK 姿态差 134° 的标本 hunt）→ 三重校验修复。 |
| `diagnose_fallback_attribution.py` | fallback 失效三模式归因（离线门禁日志解码+在线复现+库陡降分布）。 |
| `showcase_grip_vs_glyph.py` | glyph 定稿同框验证：真臂开到 top-1 候选位姿与品红 glyph 精确重合。 |
| `showcase_scene.py` | 场景锁定验收 showcase（全零 home/相机/摆放三改后的多 seed 肉审）。 |
| `view_best_effort.py` | "认证极限姿态"展品（ik_library 最陡可达构型 tilt=97.4°，证明竖直抓不可达=腕限位）。 |
| `riser_height_sweep.py` | 垫高台双高度对比实验 → **垫高负结果**（垫高台确定不加高的依据）。 |
| `verify_flip_fix.py` | CGN 反射 bug（det=-1）修复三层验证（变换链已冻结）。 |

## 5. `docs/`

| 文件 | 说明 |
|---|---|
| `api_asset_map.md`（357 行） | **现役最高权威**：API 资产对照 + 进度看板（全勾）+ 🔒冻结/封版声明 + vendor 三层规则（L1 直拿/L2 重写/L3 自写）+ 移植配方（R0 铁律）+ 两轮审计结论。改 API 前必读。 |
| `primitive_api_capx.md`（282 行） | **现役 API 权威参考**：契约 15 函数逐一签名/参数/返回/示例 + 坐标系与全局约定 + **三处有意偏差**（反投影 y 负号、joint_pos 为 (7,)、plan_grasp 返回基座系勿再左乘 pose_mat）+ 契约外成员 + 深度管线精度结论。 |
| `roadmap.md` | **Phase 1 起路线图**（2026-08-07 提炼自已删除的 ASPIRE_REPRO_PLAN.md）：论文架构对照、Skill Library 需求依据（四要素/入库流程/种子素材）、Phase 1-4（Skill 库闭环/进化搜索/长程任务/真机 sim2real）+ 长期约束。**Skill Library / Agentic coding 模块的需求源头**。 |
| `agentic_design.md` | **agentic coding + skill library 设计定稿**（2026-08-10）：论文条款（§2.1-2.3/Fig 2-3/App A/E.1-E.5/Algorithm 1）→ 实现落点逐条对照 + cap-x vendor 取舍 + 冻结层边界。改 agentic/skills/web 前必读。 |
| `cgn_container.md`（306 行） | CGN 容器化部署全录（镜像构建/default-stream 竞态修复勿回退/已知现象/变换链两次修复/标注图 F2 定稿/Piper 净开度真值 70mm）。 |
| `primitive_api.md`（287 行） | **存档**（Panda 旧线 API 文档 v0.3）：头部有存档警告，新任务一律以 cap-x 线文档为准。 |
| `sam3_vision.md`（97 行） | SAM3 视觉模块文档：MobileSAM ONNX → SAM3 迁移三坑（NCCL cu12/cu13 互覆、onnxruntime 双包遮蔽、EGL wedge 自愈）+ 接口契约 + 回退方案。 |
| `project_files.md`（本文档） | 全项目逐文件说明地图。 |

## 6. `open_details/` 复现基准（ASPIRE 官方公布，不能直接运行，需引擎注入 API）

| 文件 | 说明 |
|---|---|
| `primitive_api_cube_reset.py`（273 行） | 官方 Cube Reset 任务代码（绿块挪开、红块叠绿块）：6 步流程（定位双块→绿块挪临时位→抓红→叠放→终检）。**已被本项目复刻验证**（tasks/stack.py 以其为模式权威，2026-08-04 端到端 PASS）。 |
| `primitive_api_nut_assembly.py`（195 行） | 官方 Nut Assembly 任务代码（方螺母套柱）：Molmo 打孔心点落 3D、OBB 主轴定把手方向、dz 递降序列插入（try/except IK 失败即停当力控用）。**未复刻**（Phase 1/3 加任务时的候选）。 |
| `primitive_api_wipe.py`（103 行） | 官方 Wipe 任务代码（擦棕色污渍）：SAM3 级联 → 反投影 → bbox+margin → 双程蛇形 → 密化 → 逐点 IK+移动。**已被本项目复刻验证**（2026-08-07 Franka 擦板版端到端 PASS）。 |
| `skill_grasp.md`（233 行） | **论文 skill library 格式标杆**（BEHAVIOR-1K R1 Pro 抓取 patterns）：5 个 Pattern，每个为 **Problem → When to Apply → Strategy → Evidence** 四段式（附 Decision Matrix/Timing Data/Code Template；Evidence 必须 seed 级实证）。⚠️ 实勘：该文件**无 YAML frontmatter**（规划文档里的 frontmatter 描述的是论文附录模板，本文档以实文件为准）。Phase 1 入库的格式与验证纪律标杆。 |

## 7. `ASPIRE源论文/`（存档）

`aspire_paper.pdf`（论文原文，复现唯一权威依据）、`aspire_page.html` + `ASPIRE_files/`
（NVIDIA GEAR 项目页离线快照及网页素材，无需阅读）。看板/文档中所有"论文实锤"
（SAM3 主力/Molmo 兜底/15 函数面/skill 四段式）的出处。

## 8. 不入库资产（`.gitignore` 排除，重建见 README §4）

| 路径 | 内容 | 备注 |
|---|---|---|
| `external/cap-x/` | CaP-X 框架源码（MIT） | Primitive API 契约参照，**必需** |
| `external/contact_graspnet/` + `external/cgn_models/` | CGN 官方 TF1 代码+权重+本项目的 Dockerfile.cgn | CGN 抓取，**必需** |
| `external/piper_description/` | AgileX Piper 官方 URDF（+ 本项目生成的 piper_mjcf.urdf） | pyroki 服务用 |
| `external/agilex_arm_mujoco/` | AgileX 官方 MuJoCo 模型 | Piper MJCF 来源参照 |
| `external/cgn_venv/` | 宿主机 TF fallback 环境 | 留档调试，**正常任务禁用** |
| `external/NVIDIA_deb/` | nvidia-container-toolkit 离线包 | 无网环境装 Docker GPU |
| `SAM3/` | SAM3 模型权重（HF gated，需网页授权后下载） | 视觉，**必需** |
| `traces/` | 每次执行的证据目录（trace.json + 帧流 + 标注图） | agent 诊断的数据基础 |
| `outputs/` | 各诊断脚本的图片/npz 产物 | — |

## 9. 附录：冻结与锁定速查（2026-08-07 重组+扩锁后）

- **🔒 冻结层**（`chmod a-w`，解冻仅人类 `chmod +w`）：
  - `aspire/engine/`：`engine.py`、`engine_capx.py`
  - `aspire/api/`：`primitives_capx.py`（`primitives.py` 为 Panda 存档件，同层随锁）
  - `aspire/evidence/`：`trace.py`、`annotate.py`
  - `aspire/planning/`：`motion_planner.py`、`pyroki_client.py`（`pyroki_joint_map.json` 存档随锁）
  - `aspire/perception/`（2026-08-07 用户指令加锁）：`vision_server.py`、`vision_client.py`、`vision_sam3.py`、`cgn_server.py`
  - `aspire/robots/`（2026-08-07 用户指令加锁）：全部模型类 + assets（XML/JSON/NPZ/网格）
- **不锁**：`scripts/` 全部（任务层可自由写）、`aspire/envs/`（新场景扩展位）、
  `docs/`、`aspire/__init__.py`。
- **🔏 注释条款锁定（仅用户可改）**：robotview 相机 pos/quat（+ engine_capx.ROBOTVIEW_CAM_WORLD 同步项）、home=全零 init_qpos、场景约束（垫高台不加高/摆放区）、CGN glyph F2 规格与输出约定 0.1034（仅 CGN 模型侧；Piper 侧锚点不冻）。
- **现役服务端口**：SAM3 vision_server :8123（引擎自动拉起）、CGN docker :8117、pyroki :8116（无 /health 路由）。
- **目录重组记录（2026-08-07 用户指令）**：aspire/ 按层分 7 个子包（engine/api/evidence/planning/perception/envs/robots）；scripts/ 分 tasks/tests/tools/archive 四组 + 统一 .sh 入口（stack.sh/wipe.sh）；ASPIRE_REPRO_PLAN.md 与 handoff_piper_control_api.md 删除（前瞻内容在 docs/roadmap.md）。
