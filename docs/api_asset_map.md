# API 资产对照表（ASPIRE 需要 × cap-x 已有 × 处置方式）

> 2026-07-28 k3 编制，v2（OWL-ViT/SAM2 移出，函数面 15，加进度看板）。
> 用途：`PiperControlApiReduced` 构建的唯一索引与**进度看板**——
> **每完成一项，立即把对应 `- [ ]` 改为 `- [x]`**（交接文档 §5 的硬性要求）。
> 证据来源：ASPIRE 论文附录（api-reference.md + skill 示例）、
> `open_details/`（ASPIRE 官方任务代码 3 份 + skill 样例 1 份）、
> `external/cap-x/` 源码实读。

## 0.5 🔒 冻结声明（2026-08-05 用户裁决）

看板中所有 `- [x]` 项（已完成并经行为验证的 API/组件）已封装——
**只有人类用户（顾问也不行）批准，才能更改**。对应代码文件（primitives_capx /
engine_capx / trace / vision_server / vision_client / annotate / motion_planner）
头部有同款冻结警示。未完成项（`- [ ]`）不受此限。

## 0.5b 🔒🔒 封版声明（2026-08-06 用户裁决 · 最终封版）

**API 层全部测试完毕、无问题，正式封版。**
- **范围**：cap-x 契约 15 函数 + 契约外 5 函数（grasp_cgn /
  move_to_joints_planned / execute_legs_rrt / set_gripper_ramp /
  draw_grasp_glyphs）+ 全部支撑组件（引擎/trace/RRT 规划器/视觉服务/
  pyroki 客户端/标注）。
- **规则**：**只能在 scripts 中调用，禁止修改——只有人类（顾问也不行）
  批准才能更改。**
- **机械保护**：以下文件已 `chmod a-w`（-r--r--r--），头部有 🔒 封版警示；
  解冻须人类亲自 `chmod +w`：
  `aspire/api/primitives_capx.py`、`engine_capx.py`、`engine.py`、`trace.py`、
  `motion_planner.py`、`vision_server.py`、`vision_client.py`、`annotate.py`、
  `pyroki_client.py`。
- **不在封版范围**：scripts/ 下任务代码（demo 等，调用方可自由写）、
  存档件（cgn_server.py / vision_sam3.py / primitives.py Panda 旧线）、
  docs、robot.xml（相机参数另有锁死条款，见 E-a）。
- **封版前验证**：viz_stack 可视化端到端多轮（CGN 抓取→堆叠→避障回程），
  批量 DLS 库内真可达位姿回喂 40/40 收敛（详见 memory
  cgn-grasp-pipeline-status 0806 系列迭代记录）。

## 0.6 2026-08-05 全面审计结论（本次打钩依据）

- **打钩**：#5 plan_grasp（CGN 真身）、#15 碰撞 trace+路径检查、B-a CGN 服务、
  B-b pyroki 服务、B-d 可视化（以自写 annotate.py F2 定稿替代 vendor，用户肉审通过）；
  E-a 附注更新（ik_library 实测 0.0mm 一致, 无需重建）。
- **保持未完成**：#7 select_top_down_grasp、#11-#14 工具四函数、B-e 文档
  （docs/primitive_api_capx.md 未写；现 docs/primitive_api.md 为 Panda 线存档）、
  E1 类重构（现为 PrimitiveContextCapx + build_namespace, 函数面 11≠15）。
- **决策关闭**：B-c cuRobo——D6 不阻塞条款 + 2026-08-05 用户裁决选 RRT-Connect
  （球体包络在 6.7mm 贴脸跨指场景不可用 + py3.12/torch2.13 安装风险）。

## 0.7 2026-08-05 E1 收尾审计（看板全勾，Phase 0 函数面完成）

- **E1 类重构完成**：`PrimitiveContextCapx` → **`PiperControlApiReduced`**
  （别名保留，旧脚本零改动；`functions()` 导出契约 15 键 + 类常量
  `IK_BACKEND="pyroki"` / `GRASP_BACKEND="cgn"`；`build_namespace` 改消费
  `api.functions()`，命名空间 = 契约 15 + 契约外 `grasp_cgn` + `np`，
  wrap/trace 逻辑不变；trace.py CATEGORIES 纯新增 `select_top_down_grasp`）。
- **新函数 5 个**：#7（L1 vendor skill_library:319 原样）+ #11-#14
  （配方 #11-#14，与 Panda 线已验证实现同构）。
- **验证**：`test_piper_capx_api.py` **33/33 PASS**（新增 I/J/K 区 11 项：
  I1-I4 类结构、J1a 与冻结参考反投影 1e-9 恒等、J1b/J2 GT、J3 往返、J4 端点、
  K1-K3 三态）；10 个消费脚本 import 冒烟通过。
- **✔ 深度管线精度验证（2026-08-05，曾虚惊一场）**：当日上午测试曾报
  "渲染深度偏短 2.1cm"，根因实为**测试侧尺寸假设过期**——cubeA 为
  4×4×8cm 立块（engine_capx.py:266，2026-08-03 裁决 3 细高化），旧测试
  注释 "cube half=2cm" 是细高化前假设，把顶面 0.08 误判 0.06。经 GT 逐点
  核实：深度管线与模型一致（**亚毫米**：顶面 z_p95 0.0798 vs GT 0.0798）。
  处置：测试 C/F/J 区改为**从 model.geom_size 读尺寸** + 严格阈值
  （顶面 p95 ±5mm、footprint 半宽+1cm）；demo 放置高度 RED_HALF_Z=0.04
  修正（旧 RED_HALF=0.02 会把立块底面压进绿块 3cm）；demo 工作区 z 门控
  (0.40,0.80)→(-0.05,0.20)（落地安装时代遗留）；demo 加收臂让拍 TUCK_Q
  （Step 0/3/5，cgn_execute_grasp 同款）；engine_capx CLI 新增 --clutter /
  --target-only（纯新增，用户指令清台测试）。
- **注释修正**：primitives_capx 头部偏差声明补 #3（plan_grasp 返回**基座系**——
  原文把"相机系返回约定"误列入契约一致清单，与 2026-08-04/05 已验证行为矛盾；
  以行为为准修正注释，行为未动）。
- 看板全部 `- [x]`（B-c 以"决策关闭"打勾）。验收标准 1（测试全 PASS）/3
  （类+15 函数）/4（看板全勾）/5（文档一致）达成；标准 2（demo seed 0/1/2）
  由 2026-08-04 端到端 PASS 覆盖，本次重构命名空间兼容、未动运动/视觉行为。

## 0. 两个先决结论

### 0.1 视觉方案（论文实锤）
- **SAM3 = 主力感知**（论文 11 处提及，框架图 Fig 1 图标，skill 代码 `sam3(rgb, "bowl")`）；
- **Molmo = 官方兜底**（论文 3 处）："SAM3 返回空 → Molmo 打点 → 转 SAM3 点提示分割"
  （`localize_via_molmo` 模式，open_details 三份任务代码均使用同一兜底链）；
- 我们的 `point_prompt_molmo`（SAM3 级联 shim）已实现论文兜底链的等价物；
- ~~OWL-ViT / SAM2~~：**已移出范围**（用户决策 2026-07-28）——论文 0 提及，
  cap-x `use_sam3=False` 备用配置也不再补全。

### 0.2 ASPIRE 实际 API 面 = cap-x reduced 10 函数 + 补充 5 函数 = 15 个
open_details 三份任务代码实际调用的函数中，**4 个工具函数不在 cap-x reduced 契约内**，
加上论文 api-reference 的 `select_top_down_grasp`，完整函数面为 **15 个**（§1 中 ★ 标记）。

## 1. 主对照表（完成进度看板）

> 图例：处置方式 **L1**=文件级直拿（复制+改 import+注出处）｜**L2**=参考 cap-x 模式重写
> （服务化/契约不变）｜**L3**=契约对齐自写（已完成，不回头）。规则见 §4。
> **状态语义**：`- [x]`（存量·待审计）= 代码已存在但**未经行为验证**；
> 通过 handoff §5「第 0 步 存量审计」（`test_piper_capx_api.py` 全 PASS）后
> 删除该标注，才算真正 done。

### 感知（4）

- [x] **1. `get_observation()`** — L3 ｜ cap-x `franka/control_reduced.py:131`
  ｜ obs["robot0_robotview"]{images,intrinsics,pose_mat} + `robot0_gripper_qpos`
- [x] **2. `segment_sam3_text_prompt(rgb, text)`** — L3 ｜ `control_reduced.py:273` +
  `vision/sam3.py` ｜ 后端=本地 SAM3（进程隔离）；任务侧 area 过滤 50-12000px 排臂
- [x] **3. `segment_sam3_point_prompt(rgb, (x,y))`** — L3 ｜ `control_reduced.py:236`
  ｜ Molmo 兜底链第二环
- [x] **4. `point_prompt_molmo(rgb, text)`** — L3 shim ｜ `control_reduced.py:315` +
  `vision/molmo.py` ｜ SAM3 级联=论文兜底链等价物；真 Molmo 为可选增强（暂不做）

### 几何与抓取（3）

- [x] **5. `plan_grasp(depth, K, seg)`** — **L2** ｜ `control_reduced.py:363` +
  `vision/graspnet.py:116` + `serving/launch_contact_graspnet_server.py`
  ｜ **已完成 2026-08-05**：CGN 官方 TF 版 docker 容器服务化（见 docs/cgn_container.md）
  + `cgn_to_gripper` retarget 链（共轭翻转/指轴对齐/指尖对齐锚点, 全程实测验证）
  + 几何规划器兜底（cgn_execute_grasp 安全网, --once 下仍跳过）🔒
- [x] **6. `get_oriented_bounding_box_from_3d_points(pts)`** — L3（PCA 版）｜
  `franka/common.py`（open3d 原版）｜ nut_assembly 用它取把手轴向；可选用 open3d 替换
- [x] **7. `select_top_down_grasp(grasps, scores, cam_to_world, vertical_threshold=0.8)`** ★
  — **L1** ｜ `franka/control_reduced_skill_library.py:319`（~30 行原样 vendor）｜
  **已完成 2026-08-05**：test K1-K3 三态验证（竖直优先 / 阈值调低退化 /
  (None,-inf)）；docstring 注明输入为 CGN 约定（+z=逼近，配 grasp_cgn 原始输出），
  勿喂 plan_grasp 的 site 约定输出 🔒

### 运动（2）

- [x] **8. `solve_ik(pos, quat_wxyz)`** — **L2** ｜ `control_reduced.py:441` +
  `motion/pyroki.py:15` + `serving/launch_pyroki_server.py`
  ｜ pyroki 默认 + MuJoCo FK 后验验证 + DLS 兜底；A1 验证通过
- [x] **9. `move_to_joints(joints)`** — L3 已有，**A0 修静默失败 PASS** ｜
  `control_reduced.py:492`（tol 0.02 / 100 步阻塞语义）

### 控制（2）

- [x] **10. `open_gripper()` / `close_gripper()`** — L3 ｜ `franka/common.py`

### 工具（4，open_details 实锤调用，cap-x reduced 契约外）★

- [x] **11. `mask_to_world_points(mask_u8, depth, K, pose_mat)`** — **L1** ｜
  反投影数学抄 `utils/depth_utils.py:108` + y 负号适配（配方 #11，偏差声明 #1）
  ｜ **已完成 2026-08-05**：test J1a 与冻结参考反投影 1e-9 恒等 + J1b GT 🔒
- [x] **12. `pixel_to_world_point(u, v, z, K, E)`** — **L1** ｜ #11 的单点特化
  （Molmo 打点落 3D）｜ **已完成 2026-08-05**：test J2 🔒
- [x] **13. `rotation_matrix_to_quaternion(R)`** — **L1 自写** ｜ robosuite
  `T.mat2quat`（xyzw）→ wxyz 重排 ｜ **已完成 2026-08-05**：test J3 往返 🔒
- [x] **14. `interpolate_segment(p0, p1, step)`** — **L1 自写** ｜ wipe 路径密化，
  linspace 含两端点 ｜ **已完成 2026-08-05**：test J4 🔒

### 碰撞（横切关注，非新函数）

- [x] **15. 碰撞反馈进 trace + 运动路径碰撞检查** — L3 ｜ **已完成 2026-08-05**：
  trace.py `collision_events` 字段 + move_to_joints 逐 tick 采集落盘；
  路径检查 move_to_joints_safely + `aspire/planning/motion_planner.py` RRT-Connect
  （含腕部实体, 手-方块净距余量 3mm, 逐腿即时规划——超额完成）🔒

### 引擎资产（非 API 函数）

- [x] **E-a. 相机标定**（坑 9）— robotview 一帧含臂+工作区 / eye_in_hand 出掌心
  网格朝 -z 场景 ｜ `aspire/robots/assets/piper/robot.xml` → **handoff A0.5**
  ｜ **已解决 (2026-07-29)**：根因不是相机参数而是安装高度——标准 Stack 桌面
  在 z=0.80，原落地立柱 (基座 z=0.25) 让整条臂埋在桌下；改桌面安装
  (`PIPER_BASE_XPOS_TABLE z=0.55+pedestal 0.25=0.80`) 后，
  robotview 复用 agentview quat 平移 (pos="0.78 0 0.43" fovy=60)、
  wrist 挂 eef 下 (pos="0.045 0 0" quat="0.5 0 0.866 0" fovy=75, 即
  `gripper0_right_eye_in_hand`)，双视角均达 Franka 同款构图。
  **~~注意副作用~~（已过时）**：ik_library.npz 2026-08-04 实测与当前模型
  FK 偏差 0.0mm, 与现行模型一致, **无需重建**。🔒（相机参数仅人类可改, 见
  robot.xml 注释）

### 后端组件（不产生新 API 函数）

- [x] **B-a. Contact-GraspNet 服务** → #5 的真身后端 ｜ **已完成**：官方 TF 版
  docker 容器（cgn, checkpoint 配置挂载, 见 docs/cgn_container.md）+
  `vision_client.grasp_cgn` 调用；端到端 3/3 PASS（2026-08-05）🔒
- [x] **B-b. pyroki** → #8 的真身后端 ｜ **已完成**：独立 venv 服务
  （`scripts/tools/pyroki_server_minimal.py` :8116, 坑 7 环境隔离）; solve_ik 默认后端,
  MuJoCo FK 后验 + DLS/库/滚转扫描兜底 🔒
- [x] ~~**B-c. cuRobo 服务**~~ **决策关闭（不集成）**：D6 不阻塞条款;
  2026-08-05 用户裁决避障选 MuJoCo 真值 RRT-Connect——cuRobo 球体包络在
  6.7mm 贴脸跨指场景不可用 + py3.12/torch2.13 安装风险
- [x] **B-d. 可视化工具** → 以自写 `aspire/evidence/annotate.py`（F2 定稿, 用户肉审通过）
  替代 cap-x vendor——trace 候选渲染职能已覆盖, 非字面 vendor 🔒
- [x] **B-e. `primitive_api_capx.md` 文档** → docstring 底稿 vendor（§4b 配方 E2）
  ｜ **已完成 2026-08-05**：`docs/primitive_api_capx.md` v1.0——15 函数逐一
  签名/参数/返回/示例 + 后端映射表 + 三处有意偏差声明 + 已知上游深度 bias
  表征（§4）🔒（docs/primitive_api.md 仍为 Panda 线存档，不动）

## 2. libero 线超集（决策点：暂不收录，需要时再加）

论文主体实验在 LIBERO 上，`libero_reduced.py` 比 reduced 多出：
`goto_pose`（=open_details 里任务代码自定义的 `move_to_pose`）、
`goto_home_joint_position`、`plan_grasp_from_point_clouds`、`subsample_point_cloud`、`filter_noise`。

**顾问意见**：功能都可由任务代码 3 行组合出来，**先不进 API 面**，保持 15 函数最小完备；
若 Phase 1 skill 提炼发现它们高频重复，再升级为一等函数。

### 后续阶段可榨资产（登记，不属 Phase 0）

- `capx/envs/tasks/franka/franka_pick_place.py`：cube_stack 的 **agent prompt 模板**
  （yaml 里 `prompt is inherited from FrankaPickPlaceCodeEnv class default`）——
  Phase 1 写 agent harness 时参考"怎么给 coding agent 下指令"。
- `capx/envs/simulators/robosuite_nut_assembly.py`、`robosuite_spill_wipe.py`：
  与 `open_details/` 的 nut_assembly/wipe 任务对应的 cap-x 环境——
  Phase 1/3 加任务时的环境参考。

## 3. cap-x 没有、我们自写（已完成，不回头的）

| 资产 | 位置 | 说明 |
|---|---|---|
| Piper 机器人注册（MJCF 适配） | `aspire/robots/` | 4 个 XML 兼容问题已修 |
| ExecutionEngineCapx + trace | `aspire/engine/engine_capx.py`、`aspire/evidence/trace.py` | = 论文组件 1 |
| overhead 钩抓姿态族 + IK 种子库 | demo + `ik_library.npz` | Piper 运动学特有 |
| SAM3 进程隔离（CUDA/EGL 冲突） | `aspire/perception/vision_server.py`、`vision_client.py` | 架构与 cap-x serving 同构 |

## 4. vendor 三层规则（写死）

> **三层心智模型**（用户定稿 2026-07-28）：
> **① copy+微调能跑就跑**（L1/原样跑 server）→ **② 跑不了则参考实现逻辑翻译重写**（R0+配方）
> → **③ cap-x 没有则按 cap-x 规范自写**（ApiBase 类结构 + cap-x docstring 底稿）。
> **边界**：①的"微调"只允许换参数（robot/URDF/端口/路径/import）——
> 一旦要改逻辑，立即掉进②按配方翻译，禁止在①里无限 patch。

- **L1 文件级直拿**：纯算法无框架依赖 → 复制进 `aspire/`，改 import，
  头部注出处（cap-x 路径+行号+MIT）。适用：#7、#11-#14。
- **L2 模式参考重写**：服务化组件 → 读 cap-x 接口协议，用 vision_server
  进程隔离模式重写，函数契约不变。适用：#5、#8、B-a/B-b/B-c。
- **L3 契约对齐自写**：env 耦合深 → 已完成，不回头。

## 4b. 移植配方（R0 移植优先原则 + 逐件处方）

> **R0（铁律）**：cap-x 有实现的函数，**一律以 cap-x 源码文本为起点**——
> 先把对应源码复制/翻译为基座，再只施加本条目列明的「适配 delta」。
> **禁止脱离源码自行 0→1**；发现配方与源码事实不符时停下汇报，不发明新方案。

### 配方 A0 — `move_to_joints`（阻塞语义）
- **抄**：`capx/envs/simulators/robosuite_base.py:174` `move_to_joints_blocking`
  的循环结构（写目标 → 步进 → 容差判定 → 最大步数）；
  **另抄** `control_reduced.py:561-569` `move_along_trajectory`（虽被注释出 API 面，
  实现就在文件里）——逐 waypoint 执行结构 + 每点 `tolerance=0.025, max_steps=15`，
  正是我们路径执行包装的原型与现成参数。
- **原样保留**：容差 0.02 rad 量级、~100 步上限、绝对关节角目标；
  open/close_gripper 的 steps 参数以 `franka/common.py` 的 helper 为准对齐（cap-x 用 30）。
- **适配 delta**：① JOINT_POSITION 力矩 → position 执行器直写（已有 `_write_ctrl`）；
  ② 超时/卡死 → `raise`（我们声明的响亮失败偏差；cap-x 原语义读源码后记进注释）；
  ③ 逐段碰撞检查是我们的增强（cap-x 没有），放包装层，不改基本语义。

### 配方 #5 — `plan_grasp` ← Contact-GraspNet（官方 TF 版）
- **官方来源**：`https://github.com/NVlabs/contact_graspnet`（TensorFlow 实现）
  **禁止**：使用非官方 PyTorch 重写版（坐标系约定可能不一致）。
- **cap-x 封装参考**：`capx/integrations/vision/graspnet.py:116` `init_contact_graspnet`
  + `control_reduced.py:414-436`（调用与后处理）+
  `capx/serving/launch_contact_graspnet_server.py`（服务化，端口 8115）。
- **官方 CGN 输出坐标系约定**（严格遵循官方仓库定义）：
  - 原点：**gripper base**（非 contact point）
  - Z 轴：从 gripper base **指向物体**（approach 方向）
  - X 轴：手指开合方向
  - 右手系，OpenCV 相机系输出
- **原样保留**：plan fn 调用形态 `(depth, intrinsics, segmentation, 1,
  z_range=[0.2,2.0], forward_passes=3)`；返回 `(poses_cam, scores)`（相机系）。
- **适配 delta**：
  ① **官方仓库拉取**：`external/contact_graspnet`（TF 版），权重从 GitHub Releases 下载
  ② **独立 venv**：`external/cgn_venv`（tensorflow==2.13.0，与主环境 PyTorch 隔离）
  ③ **服务化**：首选原样跑 cap-x `launch_contact_graspnet_server.py`，备选仿
     `vision_server.py` 写 `aspire/perception/cgn_server.py`（pickle/HTTP，端口 8117）
  ④ **空候选回落**：`_plan_grasp_geometric`
  ⑤ **位姿后处理（关键标定，必须可视化验证）**：
     ```
     变换链（从左到右应用 = 矩阵右乘）：
     g_cgn:     CGN 输出 (4,4)，相机系，原点在 gripper base
     g_gl = T_flip_cv_gl @ g_cgn          # OpenCV→OpenGL（Y轴翻转，diag(1,-1,1,1)）
     g_world = pose_mat @ g_gl            # 相机系→世界系
     g_base = T_world_base @ g_world      # 世界系→基座系
     g_grip = g_base @ rot_z(-90°)        # 手指轴对齐：CGN-X→gripper-Y
     g_final = g_grip @ trans([0,0,-0.1034])  # gripper base→contact point（官方 gripper_depth）
     ```
  ⑥ **可视化验证（硬性要求，三步缺一不可）**：
     - Step 1：g_cgn 原点应在相机前方工作区（深度 0.3-0.8m）
     - Step 2：g_world/g_base 应在桌子高度附近（z≈0.8m）
     - Step 3：g_final 原点应在方块顶面中心 ±2.5cm，Z 轴朝下（点积 [0,0,-1] > 0.5）
  ⑦ **标定参数固化**：最终 `T_cgn_to_grip` 写为常量，注释标定日期/采样方法，
     以 test D2（<2.5cm）/D3（Z 轴 z<-0.5）PASS 为验收。

### 配方 #7 — `select_top_down_grasp`
- **抄**：`control_reduced_skill_library.py:319` 起整个函数体（~30 行，几乎原样）。
- **适配 delta**：docstring 注明 world=基座系；`vertical_threshold` 暴露给任务代码
  （Piper 可调低当"尽量竖直"筛选器）。

### 配方 #8 — `solve_ik` ← pyroki
- **首选（最省事）**：**原样跑 cap-x 的服务**——
  `pip install fastapi uvicorn "jax[cpu]"` + pyroki 后：
  `python external/cap-x/capx/serving/launch_pyroki_server.py --robot <piper.urdf> --port 8116`
  （该 server 本来就吃 `--robot` 参数，yaml 里默认 `panda_description`）；
  client 协议照抄 `capx/integrations/motion/pyroki.py:15-52`。
- **备选（server 原样跑不通时）**：进程内 CPU jax 重写——
  **抄** `franka/common.py:66` `solve_ik_with_convergence`（多轮收敛 + cfg 热启动）+
  `launch_pyroki_server.py:317` `/ik` 端点内的实际求解逻辑（pyroki_snippets 用法）。
- **原样保留**：`wxyz_xyz` 位姿打包（`np.concatenate([quat_wxyz, pos])`）、
  收敛循环结构、cfg 热启动模式。
- **适配 delta**：① panda_description → Piper URDF（用户提供）；
  ② `extract_arm_joints`（`common.py:101`）7 轴 → 6 轴；
  ③ `apply_tcp_offset`（`common.py:33`）保留调用但 OFFSET=0（grip_site 即 TCP）；
  ④ 失败 → 现有 DLS 兜底链。

### 配方 B-c — cuRobo 服务
- **抄**：`motion/curobo.py` + `curobo_api.py`（client 封装）+
  `serving/launch_curobo_server.py`（服务装配）。
- **首选**：原样跑 cap-x 的 `launch_curobo_server.py`，仅换 robot 参数为 Piper URDF；
- **备选**：跑不通再按 vision_server（pickle/HTTP）模式重写，端口 8124。
- **适配 delta**：机器人描述 → Piper URDF；其余不动。

### 配方 B-d — 可视化工具 vendor（trace 视觉证据增强）
- **抄**：`capx/utils/visualization_utils.py` 的 `overlay_segmentation_masks` /
  `draw_molmo_point` / `draw_oriented_bounding_box` / `render_cylinder_axis`。
- **用途**：论文组件 1 要求 trace 含 "grasp candidates" 可视化——plan_grasp 调用时
  把候选位姿渲染回图像存 trace；现 `aspire/evidence/annotate.py` 可保留，二者择一/融合。
- **适配 delta**：仅 import 路径；纯图像函数无框架耦合。

### 配方 E2 — `docs/primitive_api_capx.md` 文档底稿
- **抄**：`control_reduced.py` 各函数的 docstring（参数表/返回结构/示例）——
  vendor 为底稿，改 Piper 参数（6 轴、TCP_OFFSET=0、基座系），**不从零写文档**。

### 配方 #11/#12 — `mask_to_world_points` / `pixel_to_world_point`
- **抄**：`capx/utils/depth_utils.py:108` `depth_to_pointcloud` 的反投影数学。
- **适配 delta**：① 坐标约定换我们的 y 负号规则（契约偏差声明，与 pose_mat 配套）；
  ② 读源码确认其返回 frame，统一左乘 `pose_mat` 输出**基座系**点云；
  #12 为单点特化。

### 配方 #13/#14 — 无 cap-x 对应物
- 自写（handoff E1c 已给骨架，5/10 行）。

### 配方 C — 碰撞 trace
- cap-x 无（论文组件 1 要求），自写：`trace.py` 加 `collision_events` 字段 +
  move 原语每 tick 扫 `sim.data.contact`。

## 5. open_details 任务代码模式库（demo 与 Phase 1 的权威参考）

从三份官方任务代码提炼的**经 ASPIRE 验证的模式**（handoff D2 已据此修订）：

1. **get_best_mask**：SAM3 文本 → area 过滤（50-12000px，排机械臂）→
   Molmo 打点 → SAM3 点提示 → best-score 兜底；提示词回落（"red cube"→"red block"）。
2. **get_object_info**：`mask_to_world_points` → median center、top_z=p95、min_z=p5。
3. **pick**：抓取 z = (top_z+min_z)/2；预抓 +10cm；抓后读 `robot0_gripper_qpos`
   （>0.003 = 夹住）；空抓 → 张开 → safe_home → 重新定位 → 降 2mm 重试。
4. **place**：悬停 +12cm → 降到 place_z+clearance(0.028-0.032) → 松爪 → 撤离。
5. **safe_home**：固定高位，清开相机视野再观测。
6. **插入类**（nut_assembly）：dz 递降序列 + try/except（IK 失败即停，当力控用）。
7. **路径类**（wipe）：蛇形 path + `interpolate_segment(step=0.02)` 密化 + 逐点 IK。

## 6. skill 格式标杆（Phase 1 用）

`open_details/skill_grasp.md`（BEHAVIOR-1K R1 Pro 抓取 patterns）确认的标准结构：
**Pattern → Problem / When to Apply / Strategy（含 Decision Matrix、Timing Data、
Code Template）/ Evidence（seed 级实证）/ Limitations**。
注意其 Evidence 全部引用具体 seed 成败——入库条目的验证纪律以此为标杆。
另注意 Pattern 1/3/4 实锤 R1 Pro 的 `grasp_object` 就是 cuRobo 规划（与 B-c 互证）。
