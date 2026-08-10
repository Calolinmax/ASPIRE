# ASPIRE Primitive API 文档（复现版 v0.3 · Panda 线存档）

> ⚠️ **状态（2026-07-28）**：本文档对应**模块 2 的 Panda 线**（`aspire/engine/engine.py` +
> `aspire/api/primitives.py`，Lift 已验证，存档保留）。**当前 API 主线是 cap-x 契约 / Piper 线**：
> `aspire/api/primitives_capx.py`（15 函数契约，2026-08-06 封版），其权威文档为
> `docs/primitive_api_capx.md`。
> 新任务代码一律以 cap-x 线文档为准，不要再以本文档为"唯一 API 依据"。

> **本文档是 Coding Agent（K3）生成任务代码的唯一 API 依据（仅限 Panda 线任务）。**
> 生成代码时**只能使用本文档列出的函数**，禁止臆造 API。
> 所有函数由执行引擎注入全局命名空间，任务代码无需 import（`numpy` 除外，`np` 可用）。

---

## 0. 全局约定

| 项目 | 约定 |
|------|------|
| 坐标系 | 世界坐标系，**Z 轴向上**，桌面高度约 z=0.8m 附近（以观测为准） |
| 位置单位 | 米（m），`np.array([x, y, z])`，shape `(3,)` |
| 四元数 | **`wxyz` 顺序**，`np.array([w, x, y, z])`，shape `(4,)` |
| 关节角 | `np.array`，shape `(7,)`（Panda 7 轴，不含夹爪），单位 rad |
| 像素坐标 | `(u, v)`：u 列（x 向右），v 行（y 向下），原点左上 |
| 图像 | `np.array`，RGB shape `(H, W, 3)` dtype `uint8`；depth shape `(H, W)` dtype `float32` 单位米 |
| 相机 | 双相机：第三人称 `"agentview"` + 腕部 `"wrist"`（robot0_eye_in_hand），分辨率 256×256 |
| 阻塞语义 | 所有运动类函数都是**阻塞的**：到达目标（或超时）后才返回 |
| 夹爪 | 二指平行夹爪；`open_gripper()` / `close_gripper()` 阻塞至动作完成 |

### 常用姿态参考

```python
# 竖直向下（top-down 抓取姿态，最常用）
QUAT_TOP_DOWN = np.array([0.0, 1.0, 0.0, 0.0])  # wxyz：绕 x 轴 180°
```

> ⚠️ `open_details/` 示例中出现过 `[0,0,1,0]`（绕 y 轴 180°），同样是 z 朝下的合法姿态，两者仅末端的 x/y 朝向不同。用 `rotation_matrix_to_quaternion` 可构造任意姿态。

### 工作空间限制（Panda + Lift 任务桌面区域）

- 可达范围大致：x ∈ [-0.20, 0.35]，y ∈ [-0.30, 0.30]，z ∈ [0.80, 1.20]
- 超出范围的运动会超时返回（不会报错），任务代码应避免激进目标点

---

## 1. 感知类（Detection）

### `get_observation(camera="agentview") -> dict`

获取指定相机的完整观测。**每次需要最新状态时调用**，不要缓存旧观测做决策。

**参数**：
- `camera`：`str`，`"agentview"`（顶部第三人称）或 `"wrist"`（腕部 eye-in-hand），默认 `"agentview"`

**返回**（常用键）：

| 键 | 类型 | 说明 |
|----|------|------|
| `"robot0_eef_pos"` | `(3,)` | 夹爪中心（eef）世界坐标 |
| `"robot0_eef_quat"` | `(4,)` wxyz | 夹爪姿态 |
| `"robot0_gripper_qpos"` | `(2,)` | 夹爪两指关节位置（大=张开） |
| `"robot0_joint_pos"` | `(7,)` | 机械臂关节角 |
| `"{camera}_image"` | `(H,W,3)` uint8 | 当前相机的 RGB 图像 |
| `"{camera}_depth"` | `(H,W)` float32 | 当前相机的深度图（米制） |
| `"camera_intrinsics"` | `(3,3)` | 当前相机内参 K |
| `"camera_extrinsics"` | `(4,4)` | 当前相机外参 E（cam→world 的 4x4 齐次矩阵） |
| `"active_camera"` | `str` | 当前请求的相机名（`"agentview"` 或 `"wrist"`） |

> **双相机用法**：先用 `agentview` 做全局定位，在预抓位附近切 `wrist`（腕部相机）近距离精化。切换时重新调用 `get_observation("wrist")`，内参/外参/图像会随之更新。

---

### `segment_sam3_text_prompt(rgb, prompt) -> list[dict]`

文本提示分割（**SAM3 GPU 真实推理**：transformers 本地权重，开放词汇概念分割，返回所有匹配实例的 mask）。

**参数**：
- `rgb`：`np.array (H,W,3)`，通常传 `obs["agentview_image"]` 或 `obs["wrist_image"]`
- `prompt`：`str`，物体描述（英文）。SAM3 原生理解开放词汇，如 `"red cube"`、`"green block"`、`"mug"` 等。由于仿真纹理简单（纯色方块），推荐直接使用颜色+类别组合（如 `"red cube"`）

**返回**：`list[dict]`，按置信度降序，每个元素：
```python
{
    "mask": np.array (H,W) uint8,    # 0/255 二值掩码
    "score": float,                   # SAM3 置信度 (0~1)
    "area": int,                      # mask 面积（像素数）
    "centroid": (float, float),       # 质心 (x, y)
    "aspect_ratio": float,            # 宽高比
}
```
未匹配到任何对象时返回 **`[]`（空列表）**。任务代码**必须处理空列表**（换 prompt 重试或抛异常）。

> **实现细节**：内部调用 `Sam3Model.from_pretrained`（848M 参数，GPU CUDA 推理），处理器配置 `threshold=0.5`。首次调用触发模型加载（~3s），后续调用复用。文本 prompt 不区分大小写，SAM3 会自动匹配语义相近的概念。

**示例**：
```python
obs = get_observation()
masks = segment_sam3_text_prompt(obs["agentview_image"], "red cube")
if not masks:
    raise RuntimeError("cannot find red cube")
mask = masks[0]["mask"]
```

---

### `segment_sam3_point_prompt(rgb, point) -> list[dict]`

点提示分割（**Sam3Tracker GPU 真实推理**：SAM2 式交互分割，给定前景点，返回该实例的多个候选 mask）。

**参数**：`point`：`(x, y)` 像素坐标（float 或 int），前景点（要点在目标物体上）

**返回**：同 `segment_sam3_text_prompt` 的 list 结构（多个候选 mask，按质量分降序）；未命中返回 `[]`。

> **实现细节**：内部调用 `Sam3TrackerModel`（SAM2 的 SAM3 替代版），输入标为前景点（label=1）。`iou_scores` 可用时按质量分排序，否则按输出顺序赋分。

---

### `point_prompt_molmo(rgb, prompt) -> dict`

文本→关键点定位。内部调用 `segment_sam3_text_prompt`，取置信度最高的 mask，返回其质心坐标。常用于文本定位→点提示的级联管道。

**返回**：`dict`，如 `{"point_0": (x, y)}`；未匹配返回 `{}`。
取值方式：`uv = list(result.values())[0]`，`uv` 可能为 `(None, None)`，需判空。

> **注意**：函数名含 "molmo" 是接口历史命名，实际不调用 Molmo 模型，直接走 SAM3 文本分割。

---

### `mask_to_world_points(mask, depth, K, E) -> np.ndarray`

将 2D mask + 深度图反投影为世界系 3D 点云。

**参数**：
- `mask`：`(H,W)` 0/1 数组
- `depth`：`(H,W)` 米制深度（即 `obs["agentview_depth"]`）
- `K`：`(3,3)` 内参（`obs["camera_intrinsics"]`）
- `E`：`(4,4)` 外参 cam→world（`obs["camera_extrinsics"]`）

**返回**：`(N, 3)` float64 世界坐标点云（已剔除无效深度）。`N` 可能为 0，调用方需检查。

**典型用法**（求物体中心与顶面高度）：
```python
pts = mask_to_world_points(mask, obs["agentview_depth"], obs["camera_intrinsics"], obs["camera_extrinsics"])
center = np.median(pts, axis=0)          # 中位数抗离群
top_z = np.percentile(pts[:, 2], 95)     # 顶面高度
```

---

### `get_oriented_bounding_box_from_3d_points(pts) -> dict`

对 3D 点云做主轴拟合（PCA），返回有向包围盒。

**返回**：
```python
{
    "center": np.array (3,),   # 包围盒中心
    "R":      np.array (3,3),  # 主轴旋转矩阵（列为轴方向）
    "extent": np.array (3,),   # 沿各轴的伸展长度（全宽，非半宽）
}
```

---

### `pixel_to_world_point(u, v, z, K, E) -> np.ndarray`

单像素 + 已知深度 → 世界坐标。`z` 为该像素深度（米）。返回 `(3,)`。

---

## 2. 运动类（Planning / Control）

### `move_to_pose(pos, quat) -> None`

**最常用的运动原语**。闭环控制 eef 到达目标位姿（阻塞）。

**参数**：
- `pos`：`(3,)` 目标位置（世界系）
- `quat`：`None` 或 `(4,)` wxyz 目标姿态；**`None` = 保持当前姿态只动位置**

**行为**：
- 内部 P 控制闭环，收敛阈值 5mm / 4°，**超时约 8 秒**后放弃并返回（不抛异常）
- 不可达目标不会报错——任务代码应在运动后重新观测确认

**示例**：
```python
move_to_pose([cx, cy, top_z + 0.10], QUAT_TOP_DOWN)   # 预抓位
move_to_pose([cx, cy, grasp_z], QUAT_TOP_DOWN)         # 下降到位
```

---

### `solve_ik(pos, quat) -> np.ndarray`

数值 IK（Jacobian 阻尼最小二乘）。返回 `(7,)` 关节角（rad）。
**无解/不收敛时抛 `RuntimeError`**——调用方可用 try/except 做可达性检查。

### `move_to_joints(joints) -> None`

关节空间运动（阻塞）。`joints` shape `(7,)`。
内部等价于移动到该关节角对应的 eef 位姿（OSC 闭环），轨迹非严格关节插值。

> 与 `solve_ik` 组合的经典写法（与 `open_details/` 示例一致）：
> ```python
> joints = solve_ik(np.asarray(pos), np.asarray(quat))
> move_to_joints(joints)
> ```

---

### `interpolate_segment(p1, p2, step=0.02) -> list[np.ndarray]`

两点间直线插值（用于擦拭等轨迹任务）。返回含端点的路点列表，间距约 `step` 米。

---

## 3. 抓取类（Grasping）

### `open_gripper() -> None`

张开夹爪（阻塞至到位，约 0.5s）。

### `close_gripper() -> None`

闭合夹爪（阻塞约 0.5s）。**不保证夹到物体**——是否夹住需通过观测判断：

```python
close_gripper()
obs = get_observation()
grip_w = float(obs["robot0_gripper_qpos"][0])
grasped = grip_w > 0.003   # 夹爪被物体撑开 > 3mm → 大概率夹住了
```

### 标准抓取流程（推荐模式）

```python
# 1. 感知定位
obs = get_observation()
masks = segment_sam3_text_prompt(obs["agentview_image"], "cube")
pts = mask_to_world_points(masks[0]["mask"], obs["agentview_depth"],
                           obs["camera_intrinsics"], obs["camera_extrinsics"])
cx, cy = np.median(pts[:, :2], axis=0)
top_z = float(np.percentile(pts[:, 2], 95))

# 2. 预抓 → 下降 → 闭合 → 抬起
open_gripper()
move_to_pose([cx, cy, top_z + 0.10], QUAT_TOP_DOWN)
move_to_pose([cx, cy, top_z - 0.01], QUAT_TOP_DOWN)
close_gripper()
move_to_pose([cx, cy, top_z + 0.20], QUAT_TOP_DOWN)

# 3. 验证
obs = get_observation()
if float(obs["robot0_gripper_qpos"][0]) < 0.003:
    print("空抓，需要重试")
```

---

## 4. 工具类

### `rotation_matrix_to_quaternion(R) -> np.ndarray`

`(3,3)` 旋转矩阵 → `(4,)` **wxyz** 四元数。

构造任意末端姿态的示例（y 轴沿 handle 方向、z 轴朝下）：
```python
z_grip = np.array([0, 0, -1.0])
y_grip = np.array([dx, dy, 0.0]); y_grip /= np.linalg.norm(y_grip)
x_grip = np.cross(y_grip, z_grip)
R = np.column_stack([x_grip, y_grip, z_grip])
quat = rotation_matrix_to_quaternion(R)
```

---

## 5. 任务代码规则（Coding Agent 必读）

1. **输出单文件 Python**：只用一个 ` ```python ` 代码块，不要拆散
2. **可用的全局名**：上述全部函数 + `np`（numpy）+ `print`。**禁止 import**（numpy 已注入，其他库不可用）
3. **必须防御**：`segment_*` 可能返回空列表；点云可能为空；运动后必须重新观测
4. **打印关键信息**：定位结果、目标点、成败判断——这些会进 trace 供调试
5. **不要写 main 守卫 / 不要 exit()**：代码从上到下顺序执行即可
6. **成功后建议自检**：结尾重新观测验证任务状态（如 Lift 验证 cube 高度）

---

*v0.3 — 2026-07-27：感知升级为 SAM3（transformers 本地权重，GPU 推理），开放词汇文本提示 + 点提示交互分割。双相机架构（agentview + wrist）。几何层为真实渲染（depth + 相机模型）。*
