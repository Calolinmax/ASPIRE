# PiperControlApiReduced —— cap-x 契约 API 参考（Piper 线）

> v1.0，2026-08-05（E1 重构定稿 + 看板全勾）。
> docstring 底稿 vendor 自 `external/cap-x/capx/integrations/franka/control_reduced.py`
> （FrankaControlApiReduced，MIT），按 Piper 实参改写（6 轴、TCP_OFFSET=0、基座系）——
> 非从零撰写（配方 E2）。
> 实现：`aspire/api/primitives_capx.py`（🔒 冻结，仅人类批准可改）。
> 行为验证：`MUJOCO_GL=egl .../python scripts/tests/test_piper_capx_api.py` 33/33（2026-08-05）。

## 0. 总览

`PiperControlApiReduced`（`aspire/api/primitives_capx.py`）是 cap-x `FrankaControlApiReduced`
在 Piper + 本项目基础设施上的等价实现。契约函数面 **15 个** =
cap-x reduced 10 函数 + `select_top_down_grasp`（论文 api-reference）+
工具 4 函数（open_details 三份官方任务代码实锤调用）。

```python
api = PiperControlApiReduced(engine)
fns = api.functions()          # 契约 15 函数 dict（键名与 cap-x 完全一致）
ns = build_namespace(engine)   # 任务代码命名空间 = 契约 15 + grasp_cgn（契约外）+ np
```

后端映射（与 cap-x 的对应关系）：

| cap-x (Franka) | 本实现 (Piper) |
|---|---|
| SAM3 server (HTTP) | 本地 SAM3（`aspire/perception/vision_server.py` 进程隔离，同权重） |
| Molmo server (vLLM) | SAM3 级联 shim（同名同签名，论文兜底链等价物） |
| Contact-GraspNet server | CGN docker 容器（官方 TF 版，同约定，见 `docs/cgn_container.md`） |
| pyroki IK server (panda URDF) | pyroki server（独立 venv :8116）+ DLS/IK库/滚转扫描兜底链 |
| JOINT_POSITION 力矩控制 | position 执行器直写（真机同模式） |

后端开关（类常量，A1/B1 验收结论 2026-08-05；换后端不改任何函数签名/返回结构）：

```python
PiperControlApiReduced.IK_BACKEND     # "pyroki"（cuRobo 决策关闭不集成，看板 B-c）
PiperControlApiReduced.GRASP_BACKEND  # "cgn"（空候选回落几何规划器）
```

## 1. 坐标系与全局约定

- **基座坐标系**：所有 API 输入/输出的空间量均在机械臂基座系（obs 的
  `pose_mat` 为 cam→基座系的真刚体变换）。
- **四元数 wxyz**：API 边界一律 [w, x, y, z]（模块内部用 xyzw，边界转换）。
- **阻塞语义**：`move_to_joints` / `open_gripper` / `close_gripper` 阻塞至到位；
  未到位抛 `RuntimeError`（响亮失败，不静默返回）。
- **TCP = grip_site**（`gripper0_right_grip_site`）：`solve_ik` 目标即夹爪 TCP
  位姿，Piper TCP_OFFSET=0。site 局部 +z 从指尖指向掌心（背向逼近方向）。

三处**有意的偏差**（与 cap-x 契约对照）：

1. **反投影约定**：`y = -(v-cy)*z/fy`（mujoco 相机系 y 向上，本仓库实测标定；
   cap-x 为 `y=+(v-cy)*z/fy`——其 pose_mat 内叠了修正旋转，数学上等价；
   这里保持 pose_mat 为真刚体变换，便于直接复合抓取位姿）。
2. **robot_joint_pos 为 (7,)** = 6 臂关节 + 夹爪开合度（Franka 为 7+1=(8,)）。
3. **plan_grasp 返回基座系 TCP 位姿**（cap-x 返回相机系原始候选、由调用方
   自行左乘 pose_mat）。本实现已在内部完成 cam→基座系变换 + cgn_to_gripper
   TCP 对齐（用户肉审 + 端到端验证 2026-08-04/05），任务代码直接使用返回值，
   **勿再左乘 pose_mat**。需要相机系原始候选时走契约外的 `grasp_cgn()`。

## 2. 函数参考（契约 15）

### 感知（4）

#### 2.1 `get_observation() -> dict`

获取环境观测（基座坐标系）。调用即刷新观测（保证非旧帧）。

返回 dict，关键键：

- `["robot0_robotview"]["images"]["rgb"]`：(H,W,3) uint8 第三人称彩色图（当前 640×640）
- `["robot0_robotview"]["images"]["depth"]`：(H,W) float64 深度（米）
- `["robot0_robotview"]["intrinsics"]`：(3,3) 相机内参 K
- `["robot0_robotview"]["pose_mat"]`：(4,4) 相机外参（相机系→基座系，真刚体）
- `["robot0_robotview"]["pose"]`：(7,) [xyz, quat_wxyz] 相机位姿（基座系）
- `["robot_joint_pos"]`：(7,) 6 臂关节角(rad) + 夹爪开合度（0 闭 1 开）
- `["robot_cartesian_pos"]`：(8,) 夹爪 TCP 位姿 [xyz, quat_wxyz, 开合度]（基座系）

```python
obs = get_observation()
cam = obs["robot0_robotview"]
rgb, depth, K, E = cam["images"]["rgb"], cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"]
```

#### 2.2 `segment_sam3_text_prompt(rgb, text_prompt) -> list[dict]`

SAM3 文本提示分割（GPU 本地推理，开放词汇）。后端：`vision_server` 进程隔离。

- `rgb`：(H,W,3) uint8；`text_prompt`：物体描述（英文），如 `"red cube"`
- 返回：list[dict]，按置信度降序 `{"mask": (H,W) bool, "box": [x1,y1,x2,y2], "score": float}`；
  未匹配返回 `[]`

```python
masks = segment_sam3_text_prompt(rgb, "red cube")
```

#### 2.3 `segment_sam3_point_prompt(rgb, point_coords) -> list[dict]`

SAM3 点提示分割（前景点 → 该实例的候选 mask）。Molmo 兜底链第二环。

- `point_coords`：(x, y) 像素坐标（要点在目标物体上）
- 返回结构同 2.2；未命中返回 `[]`

```python
masks = segment_sam3_point_prompt(rgb, (320, 240))
```

#### 2.4 `point_prompt_molmo(image, text_prompt) -> dict`

文本→关键点定位。接口与 cap-x 的 Molmo 一致；实现为 SAM3 级联 shim
（= 论文兜底链 "SAM3 空 → Molmo 打点 → SAM3 点提示" 的等价物）。

- 返回：`{text_prompt: (x, y)}` 像素坐标（int）；未定位到为 `(None, None)`

```python
pt = point_prompt_molmo(rgb, "red cube")["red cube"]
```

### 几何与抓取（3）

#### 2.5 `plan_grasp(depth, intrinsics, segmentation) -> (grasp_poses, grasp_scores)`

抓取规划：优先 CGN（官方 TF 版 docker 容器），失败/空候选回退几何规划器
（20~26° 倾斜 × 8 方位姿态族 + IK 库支持度排序）。

- `depth`：(H,W) 深度图（米）；`intrinsics`：(3,3) K；
  `segmentation`：(H,W) 实例分割图，整数 >0 为目标实例
- 返回：`grasp_poses` (K,4,4) 候选抓取 **TCP 位姿，基座系**（偏差 #3——
  cap-x 返回相机系，此处已变换 + TCP 对齐，勿再左乘 pose_mat）；
  `grasp_scores` (K,) 降序

```python
grasps, scores = plan_grasp(depth, K, seg)
pos, R = grasps[0][:3, 3], grasps[0][:3, :3]        # 基座系，直接可用
quat_wxyz = rotation_matrix_to_quaternion(R)
joints = solve_ik(pos, quat_wxyz, free_approach_roll=True)
```

#### 2.6 `get_oriented_bounding_box_from_3d_points(points) -> dict`

对 3D 点云做主轴拟合（PCA + 离群剔除），返回有向包围盒。

- `points`：(N,3) float64
- 返回：`{"center": (3,), "R": (3,3) 主轴旋转矩阵（右手系）, "extent": (3,) 各轴全宽}`

```python
obb = get_oriented_bounding_box_from_3d_points(pts)
```

#### 2.7 `select_top_down_grasp(grasps, scores, cam_to_world, vertical_threshold=0.8) -> tuple`

从抓取候选中选出最优 top-down（竖直）抓取
【L1 vendor：cap-x `control_reduced_skill_library.py:319`】。

- `grasps`：(N,4,4) 候选位姿（相机系，**CGN 约定：局部 +z = 逼近方向**，即
  `grasp_cgn()` 原始输出）。注意：`plan_grasp()` 返回的是 grip_site 约定
  （+z 背向物体）且已在基座系，**不要直接喂给本函数**
- `scores`：(N,) 候选得分
- `cam_to_world`：(4,4) cam→基座系（传 `pose_mat`；"world" 对 Piper 即基座系，
  z 轴竖直向上）
- `vertical_threshold`：逼近轴与竖直向下点积阈值（1.0=完全竖直）。Piper 腕限位紧、
  严格 top-down 常不可达——调低（如 0.5）退化为"尽量竖直"筛选器
- 返回：`(best_grasp_base (4,4) 基座系, best_score)` 或 `(None, -inf)`

```python
grasps_cam, scores, _ = grasp_cgn(rgb, depth, K, seg)
best_cam, s = select_top_down_grasp(grasps_cam, scores, pose_mat, vertical_threshold=0.5)
```

### 运动（2）

#### 2.8 `solve_ik(position, quaternion_wxyz, use_pyroki=True, free_approach_roll=False, seed=None, loose=False) -> (6,)`

数值 IK（pyroki 优先 + MuJoCo FK 后验 + DLS/IK库/滚转扫描兜底链），
目标为夹爪 TCP（grip_site）位姿（基座系）。

- `position`：(3,) TCP 目标位置（基座系，米）
- `quaternion_wxyz`：(4,) TCP 目标姿态 [w,x,y,z]（基座系）
- `use_pyroki`：是否优先调用 pyroki IK server（默认 True；失败自动回退 DLS 链）
- `free_approach_roll`：绕逼近轴滚转自由——只收敛位置+逼近轴方向
  （方形截面目标合法；精确 6 自由度目标常落在腕限位薄流形外）
- `seed`：可选 (6,) 种子构型，提供时先就地 DLS（分支连续，防远分支跳转）
- `loose`：中途点放宽容差（2mm/1.15°；预抓/下降中间步用，末步仍严格）
- 返回：(6,) 臂关节角 (rad)。**不收敛抛 RuntimeError**（可 try/except 做可达性检查）

```python
joints = solve_ik(np.array([0.30, 0.0, 0.85]), np.array([1.0, 0.0, 0.0, 0.0]))
```

#### 2.9 `move_to_joints(joints, tol=None, fk_tol=None) -> None`

关节空间运动（阻塞）。每 tick 采集臂-环境碰撞事件进 trace（`collision_events`）。

- `joints`：(6,) 目标关节角 (rad)。命令后等待 ‖q-target‖₂ < tol，最多约 5 秒；
  **未到位抛 RuntimeError**（响亮失败偏差；cap-x 为静默）
- `tol`：到位阈值（默认 0.02 rad；接触段可放宽到 ~0.10）
- `fk_tol`：FK 到位自检阈值（默认 0.01 m；接触段放宽）

```python
move_to_joints(joints)
```

### 控制（2）

#### 2.10 `open_gripper() -> None` / `close_gripper() -> None`

张开/闭合夹爪（阻塞至到位，~15 拍稳定）。是否夹住需通过
`robot_joint_pos[-1]` 观测判断（被物体撑住 = 开度明显大于全闭）。

```python
open_gripper()
close_gripper()
grasped = get_observation()["robot_joint_pos"][-1] > 0.15
```

### 工具（4，open_details 实锤调用，cap-x reduced 契约外）

#### 2.11 `mask_to_world_points(mask, depth, K, pose_mat) -> (N,3)`

mask 像素反投影 → 基座系 3D 点云【反投影数学 L1 自 cap-x
`utils/depth_utils.py:108`，y 负号适配（偏差 #1）】。

- `mask`：(H,W) uint8/bool，>0 的像素被反投影；`depth`：(H,W) 米；
  `K`：(3,3)；`pose_mat`：(4,4) cam→基座系
- 返回：(N,3) float64 基座系点云（无效深度剔除）；空 mask 返回 (0,3)

```python
pts = mask_to_world_points(mask.astype(np.uint8), depth, K, E)
center = np.median(pts, axis=0); top_z = np.percentile(pts[:, 2], 95)
```

#### 2.12 `pixel_to_world_point(u, v, z, K, E) -> (3,)`

单像素 + 该点深度 → 基座系 3D 点（2.11 的单点特化；Molmo 打点落 3D 用）。

```python
p = pixel_to_world_point(u, v, depth[v, u], K, E)
```

#### 2.13 `rotation_matrix_to_quaternion(R) -> (4,)`

(3,3) 旋转矩阵 → wxyz 四元数（API 边界约定；内部 robosuite mat2quat 再重排）。

```python
q_wxyz = rotation_matrix_to_quaternion(R)
```

#### 2.14 `interpolate_segment(p1, p2, step=0.02) -> list[(3,)]`

两点间按 step 密化路径点（wipe 类任务蛇形路径用）；linspace 精确含两端点，
相邻间距 = dist/ceil(dist/step) ≤ step。

```python
for i in range(len(waypoints) - 1):
    for p in interpolate_segment(waypoints[i], waypoints[i + 1], step=0.02):
        move_to_joints(solve_ik(p, quat))
```

## 3. 契约外命名空间成员

`build_namespace` 额外挂入（不在 `functions()` 15 键内）：

- `grasp_cgn(rgb, depth, K, seg, z_range=(0.2, 2.0)) -> (grasps_cam, scores, openings)`：
  CGN 相机系原始候选直连（CGN 约定：原点掌根、+z 逼近、x 开合轴）。
  与 `select_top_down_grasp` 配套；转执行位姿用 `cgn_to_gripper(g, pose_mat)`
  （模块级函数，`aspire/api/primitives_capx.py`）。
- `np`：numpy。

## 4. 深度管线精度（2026-08-05 验证结论）

反投影链（depth → `get_real_depth_map` → pose_mat × [x, -y, z]）与模型 GT
**逐点一致（亚毫米）**：cubeA 顶面 z 实测 0.0798m vs GT 0.0798m。

背景：当日一度怀疑"渲染深度系统性偏短 2.1cm"，根因实为**测试侧尺寸假设过期**——
cubeA 是 4×4×8cm 立块（`engine_capx.py:266`，2026-08-03 裁决 3 细高化），
测试注释里的"cube half=2cm"是细高化之前的旧假设。教训：**场景 GT 尺寸一律
从 `model.geom_size` 读取，不要硬编码**（`scripts/tests/test_piper_capx_api.py`
C 区已改为模型读取 + 严格阈值：顶面 p95 ±5mm、footprint 半宽+1cm）。

注：cubeA footprint 仍为 4×4cm——抓取开度相关参数（`filter_grasps_by_width`
contact_dist=0.04、demo 抓取口袋 top_z-0.02）不受影响；放置高度需用真实
半高 0.04（demo 已修正 RED_HALF_Z）。
