# 交接 Prompt：PiperControlApiReduced —— API 构建阶段（含 4 必要组件）

> v3，2026-07-28。由 k3（高级顾问）编写，**含处方级实施细则**——每个任务给出
> 改动位置、代码骨架、已定决策、验证命令、失败退路。执行者不需要重新做架构判断，
> 按方抓药即可；遇到处方与代码现实冲突时，停下来向用户汇报，不要自行发明新方案。
> 使用时：新开对话，把下方分隔线内的全部内容粘贴为第一条消息。

---

# 任务：完成 PiperControlApiReduced（cap-x 契约 + 4 必要组件）并跑通 Stack demo

## 0. 角色与纪律

你是执行者，不是设计者。本文件已包含全部架构决策（§4「已定决策」一节写死，
**不要重新权衡**）。工作顺序 A→B→C→D→E，逐项验证通过再进下一项。
每完成一个大项，用一句话向用户汇报「做了什么 + 验证结果」。
**移植优先（D7/R0）**：写任何函数前，先打开 `docs/api_asset_map.md` §4b 找到
对应「移植配方」，以其指定的 cap-x 源码为起点翻译，只施加配方列明的 delta。
**进度纪律（硬性要求）**：`docs/api_asset_map.md` 是进度看板——每完成一个
函数/组件/工作项，**立即**把对应条目的 `- [ ]` 改为 `- [x]`（写完代码不算完，
验证通过才算完）。看板全部打勾是验收标准之一。

## 1. 项目背景

项目根目录 `/home/stouching/Desktop/ASPIRE`，ASPIRE 论文复现（长期项目）。
本任务 = `ASPIRE_REPRO_PLAN.md` 的 **Phase 0（API 构建阶段）**：

参照 `external/cap-x/capx/integrations/franka/control_reduced.py` 的
`FrankaControlApiReduced` 契约，交付 Piper 版 `PiperControlApiReduced`：
cap-x 10 函数契约 + 4 必要组件（碰撞 / Contact-GraspNet / pyroki / cuRobo）
全部集成（OWL-ViT/SAM2 已移出范围，用户决策 2026-07-28），并跑通 robosuite Stack
（红块叠绿块）demo。

**前置工作已完成 60-70%，不要推倒重写。**

## 2. 必读文件（按序读完再动手）

1. `AGENTS.md` — 环境规则：**所有 Python 必须用
   `/home/stouching/anaconda3/envs/ASPIRE/bin/python`**（conda env `ASPIRE`，py3.12）。
2. `ASPIRE_REPRO_PLAN.md` §0/§3/§4 — 现状与出口标准。
3. **`docs/api_asset_map.md` — API 资产对照表 + 进度看板（15 函数的 cap-x 出处行号 +
   L1/L2/L3 取件方式 + open_details 任务模式库）。vendor 操作一律查此表；
   每完成一项立即把对应 `- [ ]` 改为 `- [x]`（进度可视化是验收的一部分）。**
4. `external/cap-x/capx/integrations/franka/control_reduced.py` — 契约源头。
5. `aspire/primitives_capx.py` — 现有 10 函数实现（头部注释有契约对照表）。
6. `aspire/engine_capx.py` — 执行引擎。
7. `scripts/test_piper_capx_api.py`、`scripts/demo_stack_piper_capx.py`。
8. `open_details/primitive_api_cube_reset.py` — **ASPIRE 官方任务代码（Stack 同款），
   D2 的模式来源**，重点看 `get_best_mask`/`get_object_info`/`pick_object`/`place_object`。

## 3. 已验证的事实与坑（不要重复踩）

- Piper MJCF 已适配 robosuite（`aspire/robots/`）；`ExecutionEngineCapx` 直写
  position 执行器、基座坐标系、20Hz；10 函数有初版实现；SAM3 已进程隔离
  （`aspire/vision_server.py` + `vision_client.py`，引擎 main() 自动拉起）；
  IK 种子库 37,655 条（`aspire/robots/assets/piper/ik_library.npz`）。
- **视觉方案（论文实锤）**：SAM3 主力 + Molmo 兜底（SAM3 返回空 → Molmo 打点 →
  SAM3 点提示，open_details 三份任务代码同此链；我们的 molmo shim 已是等价物）。
  ~~OWL-ViT/SAM2~~ 已移出范围（用户决策 2026-07-28，论文也未使用），**不要实现**。
- **Stack 从未跑通**（traces/ 最近 6 次全 False）。
- **坑 1（运动学）**：j5 腕限位 ±70°，桌面高度顶朝下不可达；用远侧 overhead 钩抓
  （demo 里 `overhead_grasp_quat`，候选链 `[(25,0),(25,-30),(25,30),(15,0),(35,0),(45,0)]`）。
  可达带：悬停高度基座系 x≈0.28-0.40，x>0.45 IK 不收敛。
- **坑 2（执行，最严重）**：伺服撞障碍卡死后 `move_to_joints` 静默超时，TCP 偏差
  150-300mm，后续动作全在错误状态上累积。→ A0 修。
- **坑 3（渲染）**：EGL 高频 wedge，有 L1/L2 自愈，属噪声；运动拍不渲染。
- **坑 4（感知）**：SAM3 偶发漏检 "red cube" → B3 兜底。
- **坑 5（集成前提）**：`external/agilex_arm_mujoco` 只有 MJCF **没有 URDF**；
  pyroki/cuRobo 都是 URDF 驱动 → URDF **由用户提供**（A1 第一步，到时 `ls external/`
  找路径，找不到就问用户，不要猜/不要自行 clone）。
- **坑 6（进程隔离铁律）**：torch/CUDA 初始化后同进程 EGL 渲染永久损坏。
  **一切 CUDA 推理（SAM3/CGN/cuRobo）必须进程隔离**；
  jax 装 CPU 版可进程内（不碰 CUDA 即安全）。
- **坑 7（依赖死锁，2026-07-28 实测）**：pyroki → jaxls 要 jax≥0.6 → 要 numpy 2.x，
  但主环境 numpy 钉在 1.26（2.x 破坏 vision_server）→ 三角死锁。
  **解法 = 环境隔离**：pyroki server 住独立 venv（jax/numpy 随便装），主环境零改动，
  HTTP 通信（D2 首选方案天然兼容）。禁止用升级主环境全家桶的方式解。
- **坑 8（URDF/MJCF 约定不匹配，2026-07-28 诊断）**：官方 URDF 与第三方 MJCF
  （yanyuze1 仓）**几何完全一致**（link3 模长 0.28503、link4 模长 0.25171 两边
  精确相等），但**关节零位/轴系取向约定不同**——同数值关节角 FK 偏差 ~411mm。
  **解法 = 零位映射标定**（Fix 1，见 A1 第 1b 步）：逐关节拟合
  `q_mjcf_i = sign_i · q_urdf_i + offset_i`，映射层放 `_ik_pyroki` 适配层；
  验证发现轴向在 3D 中不一致（残差随角度增长）才转 Fix 2（以 MJCF 为真源
  脚本生成配套 URDF，IK 用不需要 mesh）。`mujoco.mj_saveLastXML` 只能导出
  MJCF，**不能**导出 URDF，勿走此路。
- **坑 9（相机位姿错误，2026-07-29 图像实锤）**：① `robotview`（top）只对准
  方块工作区，**臂整体出画**（trace 里"臂没动过"= 臂根本不在画面里）；
  ② `eye_in_hand`（wrist）朝 link6 **+z** 看进 piper_hand 掌心网格
  （gripper.xml：手指向 root **-z** 伸展，逼近方向 = -z）→ 渲染的是网格内表面。
  修法见 A0.5（改 `robot.xml` 两处相机 pos/target + 渲染迭代）。
- **坑 10（CGN 版本混淆，2026-07-30）**：PyTorch 重写版与官方 TF 版输出坐标系可能有差异；
  必须严格使用 NVIDIA NVlabs 官方仓库（TensorFlow），服务化参考 cap-x
  `launch_contact_graspnet_server.py`。禁止混用 PyTorch 重写版。

## 4. 已定决策（写死，不要重新权衡）

| # | 决策 | 理由 |
|---|---|---|
| D1 | IK 三层后端：`pyroki`（默认目标）→ `curobo`（装得上则对比争默认）→ `dls`（永作兜底）。后端开关集中在 `PiperControlApiReduced` 类常量 `IK_BACKEND` | pyroki 成熟轻量；cuRobo 治可达流形薄但安装有风险；DLS 已验证 |
| D2 | pyroki/cuRobo 服务**首选原样跑 cap-x 的 launch_*_server**（jax 装 CPU 版，无 CUDA）；跑不通再按 vision_server 模式重写 | 现成代码零改动效率最高；坑 6 |
| D3 | CGN 并入 `vision_server` 进程（加 endpoint），不再开新进程 | 一个 CUDA 进程够用，省显存 |
| D4 | 不做双臂、不做 `traj_plan`/`move_along_trajectory`（cap-x 源码里也注释掉了） | 简化清单：只单臂 |
| D5 | 契约冲突时以 `control_reduced.py` 源码为准；换后端不改任何函数签名/返回结构 | 契约是交付物 |
| D6 | cuRobo 安装 timebox 半天；装不上则 pyroki 为默认继续推进，cuRobo 在验收报告中标注「未集成+原因」，**不阻塞** B-E | 本机 torch 2.13 兼容史 |
| D7 | **移植优先原则（R0）**：cap-x 有实现的函数，一律以其源码文本为起点做"翻译"，只施加 `docs/api_asset_map.md` §4b 移植配方里列明的适配 delta；**禁止脱离源码自行 0→1**；配方与源码事实不符时停下汇报 | 参考现成效率最高，且保真 |

## 5. 工作清单（处方级）

### 第 0 步：存量审计（最先做，质量闸门）

现有 10 函数是上一会话（旧策略下）写的，**一律视为"未验证"**——不 review
代码怀旧，直接行为验证：

1. 运行 `MUJOCO_GL=egl /home/stouching/anaconda3/envs/ASPIRE/bin/python scripts/test_piper_capx_api.py`
   （A-H 分区恰好覆盖全部存量函数，即验收标准 1 的预演）。
2. **任何 FAIL**：打开 `docs/api_asset_map.md` 找到对应条目的 cap-x 出处与
   §4b 配方，**参考 cap-x 实现逻辑修复**（不是自由发挥）；修到 PASS。
3. 全 PASS 后：删掉看板 6 个存量项的「（存量·待审计）」标注——这才算真正 done。
4. 若按续跑指令已做过 A0 核验：那次全量结果就是存量审计，FAIL 项同样按
   配方修复。

### A0 运动可信化（`aspire/primitives_capx.py`，最先做）

**改 `move_to_joints`（现 478-498 行附近）**：
1. stall 30 拍与 MAX_TICKS_MOVE 超时的两个 `break` 改为
   `raise RuntimeError(f"move_to_joints 未到位: err={err:.4f}rad target={target}")`。
2. 新增路径执行包装（供 API 层替换原直接调用；**结构参考 cap-x 自家的
   `move_along_trajectory`，`control_reduced.py:561-569`——逐 waypoint 执行，
   每点 `tolerance=0.025, max_steps=15` 是 cap-x 现成参数**）：

```python
MAX_STEP_RAD = 0.15
def move_to_joints_safely(self, target):
    q_cur = self.engine.current_arm_qpos()
    dq = target - q_cur
    n = max(1, int(np.ceil(np.abs(dq).max() / MAX_STEP_RAD)))
    for i in range(1, n + 1):
        q_wp = q_cur + dq * i / n
        if i == n or not _collision_free_q(q_wp):   # 中间段末查碰撞
            if not _collision_free_q(q_wp):
                raise RuntimeError(f"路径第{i}/{n}段末构型碰撞")
        self.move_to_joints(q_wp)
```

3. 把 `solve_ik` 里的闭包 `collision_free` **提升为模块级函数** `_collision_free_q(q)`
   （逻辑不变，solve_ik 与路径检查共用）。
4. 到位验证：`move_to_joints` 返回前 FK 自检——`_fk_tcp(q_target)` 与
   `current TCP`（sim.data.site_xpos）误差 >1cm → raise。

**验证**：`scripts/test_piper_capx_api.py` G1-G3 PASS；手动给一个桌内/远界目标
确认 raise 而非静默。**退路**：若 G2 到位误差反复超 8mm，检查 position 执行器
gain/ctrlrange（`robot.xml`），仍不行向用户汇报，不要自行改控制器架构。

### A0.5 相机标定（坑 9，trace 可视化修复，快赢项）

**病因**（已图像实锤）：`robotview` 只框住方块工作区、臂出画；`eye_in_hand`
朝 link6 +z 看进掌心网格（手指伸向 -z）。

**修 `aspire/robots/assets/piper/robot.xml` 两处相机**：
1. `robotview`（约 116/118 行）：目标 = 一帧内同时容纳**臂 + 工作区**。
   起始值：`robotview_cam_body pos="0.85 0 1.30"`、`robotview_target pos="0.15 0 0.55"`，
   fovy 60-70。参照系：robosuite `agentview`（`table_arena.xml`：
   `pos="0.5 0 1.35"` 俯拍机器人+桌区）——我们要的是同款"看得到臂在干活"的
   第三人称，不是只看方块的监控探头。
2. `eye_in_hand`（约 225-227 行）：看向 -z（手指/场景方向）且横向偏移出掌心网格。
   起始值：camera `pos="0.05 0 0.10"`、`wrist_cam_target pos="0 0 -0.15"`。

**迭代工具**：写 `scripts/check_cameras.py`——建 env，渲染两相机于
① reset/home 位形 ② 从 IK 库取一个俯身抓取位形，存图到 `outputs/cam_check/`，
**用 Read 工具看图**，每次只调 pos/target/fovy 一个参数，≤6 轮收敛。
**验收**：home 帧 top 同时含臂与两方块的桌区且画面不滚转；wrist 帧看到桌面/
场景（手指在画面边缘可接受），不再是网格内表面；臂运动时 top 帧像素随之变化。
完成后打勾看板 E-a。

### A1 pyroki（`solve_ik` 新后端）

1. URDF **由用户提供**（已拉到 `external/piper_description/`）：选含夹爪的 urdf。
   检查 URDF 关节名与 MJCF `robot0_joint1..6` 的映射表，写进注释。

1b. **零位映射标定（坑 8，必做）**：官方 URDF 与我们的 MJCF 几何一致但零位约定
   不同（同数值 FK 差 ~411mm）。写标定脚本逐关节拟合
   `q_mjcf_i = sign_i · q_urdf_i + offset_i`（offset 预期 ±π/2 量级；注意 MJCF
   joint2 range 为 0~π）：每关节其余置 0、取 3 个角度值，URDF FK（pyroki venv）
   vs MJCF FK（mujoco）拟合 sign/offset。**验证**：100 组随机 q（双方限位交集内）
   映射后 FK 位置差 <1mm、姿态差 <0.01rad；残差随角度增长 = 轴向不一致 →
   转 Fix 2（以 MJCF 为真源生成 URDF）。映射表写为常量，放 `_ik_pyroki` 适配层
   （solve_ik 返回 URDF 解 → 映射 → 返回 MJCF 角），cuRobo（A2）复用同一映射层。
2. 安装：`.../bin/pip install "jax[cpu]"` **先钉 CPU 版**，再
   `.../bin/pip install "pyroki @ git+https://github.com/chungmin99/pyroki.git"`
   （URL 来源：cap-x `pyproject.toml` 依赖声明，已验证）。
3. **首选（D2）**：原样跑 cap-x 的 server——`.../bin/pip install fastapi uvicorn` 后：
   `.../bin/python external/cap-x/capx/serving/launch_pyroki_server.py --robot <piper.urdf> --port 8116`
   （server 自带 `--robot` 参数）；client 协议照抄
   `capx/integrations/motion/pyroki.py:15-52`。
   **备选（server 跑不通）**：进程内 CPU jax 重写——读
   `capx/serving/launch_pyroki_server.py:317` 的 `/ik` 求解逻辑 +
   `franka/common.py:66` 的收敛循环。封装 `_ik_pyroki(pos_world, quat_xyzw) -> (6,)`。
   详细移植配方见 `docs/api_asset_map.md` §4b。
4. **TCP 标定（关键）**：写一次性脚本对拍 FK——同一组关节角，pyroki FK(目标 link)
   vs MuJoCo `grip_site` 位姿，确定 pyroki 侧目标 link + 固定偏移，使误差 <1mm；
   把 link 名与偏移写为常量。
5. `solve_ik` 改为：先 pyroki（带限位），失败 → 原 DLS 链。
**验证**：工作区网格（x∈[0.25,0.45] 步 0.02 × y∈[-0.15,0.15] 步 0.03 × overhead
姿态链）收敛率对比 DLS vs pyroki，输出表格贴给用户。**退路**：先按坑 7 用
**独立 venv**（时间盒 2h）攻坚；仍不通/对拍超差 → 保持 DLS 默认，pyroki 标注
「未集成+具体原因」，继续 B。cuRobo（A2）与 jax 无关，是另一出路，不受此影响。

### A2 cuRobo（D6 timebox 半天）

0. **先读 cap-x 的现成集成**：`external/cap-x/capx/integrations/motion/curobo.py`、
   `curobo_api.py`、`capx/serving/launch_curobo_server.py`——cap-x 有完整 cuRobo
   服务化实现（libero_reduced.py 里默认注释掉的 4 个函数：
   `parse_grasp_poses_for_curobo` / `plan_grasp_trajectory` /
   `plan_with_grasped_object` / `execute_joint_trajectory`），参考其接口设计，
   不要从零造轮子。
1. `.../bin/pip install nvidia-curobo`；import 失败/段错误 → 按 D6 放弃并记录。
2. **首选（D2）**：原样跑 cap-x 的 `launch_curobo_server.py`，仅换 robot 参数为
   Piper URDF；**备选（跑不通）**：仿 `vision_server.py` 写 `aspire/curobo_server.py`
   （pickle over HTTP，端口 8124）+ `curobo_client.py`。
   详细移植配方见 `docs/api_asset_map.md` §4b。
3. 用 Piper URDF 配批量 IK（≥1024 种子并行）；可选：无碰撞轨迹替代 A0 的线性插值。
4. 与 pyroki 同网格对比（覆盖率/耗时/解质量），报告给用户选默认，更新 `IK_BACKEND`。

### B1 Contact-GraspNet（`plan_grasp` 新后端，官方 TF 版）

1. **拉取官方仓库（禁止 PyTorch 重写版）**：
   ```bash
   cd external/
   git clone https://github.com/NVlabs/contact_graspnet.git
   ```
   权重下载（~200MB）：
   ```bash
   wget -P external/contact_graspnet/checkpoints/ \
     https://github.com/NVlabs/contact_graspnet/releases/download/v1.0.0/contact_graspnet_checkpoint.zip
   unzip ...  # 得 contact_graspnet.ckpt
   ```

2. **创建独立 venv（TensorFlow 环境隔离，坑 6 + 坑 10）**：
   ```bash
   python3 -m venv external/cgn_venv
   source external/cgn_venv/bin/activate
   pip install tensorflow==2.13.0  # 官方版本
   pip install fastapi uvicorn numpy scipy opencv-python
   deactivate
   ```

3. **服务化封装（参考 cap-x）**：
   - **首选**：原样跑 cap-x `launch_contact_graspnet_server.py`（端口 8115）
   - **备选**：仿 `vision_server.py` 写 `aspire/cgn_server.py`（pickle/HTTP，端口 8117）
   - **协议**：复刻 cap-x `/grasp` endpoint 的输入输出格式

4. **官方 CGN 坐标系约定（严格遵循）**：
   - 原点：**gripper base**（非 contact point）
   - Z 轴：从 gripper base **指向物体**（approach 方向）
   - X 轴：手指开合方向
   - 右手系，OpenCV 相机系输出

5. **位姿后处理（关键标定，必须可视化验证）**：
   ```python
   # 变换链（从左到右应用）：
   g_cgn:     CGN 输出 (4,4)，相机系，原点在 gripper base
   g_gl = T_flip_cv_gl @ g_cgn          # OpenCV→OpenGL（Y轴翻转）
   g_world = pose_mat @ g_gl            # 相机系→世界系
   g_base = T_world_base @ g_world      # 世界系→基座系
   g_grip = g_base @ rot_z(-90°)        # 手指轴对齐：CGN-X→gripper-Y
   g_final = g_grip @ trans([0,0,-0.1034])  # gripper base→contact point
   ```
   **可视化验证（三步缺一不可）**：
   - Step 1：g_cgn 原点在相机前方工作区
   - Step 2：g_world 在桌子高度附近
   - Step 3：g_final 原点在方块顶面中心 ±2.5cm，Z 轴朝下

6. **验证**：test 脚本 D1-D3 PASS（候选位置 ↔ 方块中心 <2.5cm、逼近轴朝下）
   几何规划器 `_plan_grasp_geometric` 留作兜底（CGN 返回空候选时回落）。

### B2 HSV 兜底（只改 demo 任务代码）

SAM3 返回空时启用（纯 np，允许）：

```python
def hsv_locate(rgb, color):  # color: "red"/"green"
    import colorsys
    r = rgb[...,0].astype(int); g = rgb[...,1].astype(int); b = rgb[...,2].astype(int)
    if color == "red":   m = (r > 120) & (r > g*1.8) & (r > b*1.8)
    else:                m = (g > 80)  & (g > r*1.3) & (g > b*1.3)
    ys, xs = np.nonzero(m)
    return None if len(xs) < 50 else (m, (xs.mean(), ys.mean()))
```

### C 碰撞进 trace

1. `aspire/trace.py`：record schema 加可选字段 `collision_events`（缺省 `[]`，
   旧 trace 向后兼容）。
2. `primitives_capx.py` 的 move 类原语：每 tick 扫 `sim.data.contact`，收集臂几何
   （沿用 `_collision_free_q` 的 geom 名规则）与 table/cube/pedestal 的接触，
   记 `{sim_step, geom1, geom2, pos}`，随 `tracer.record` 落盘。
**验证**：跑一次 demo，`trace.json` 中 move 记录含该字段（无碰撞时为 `[]`）。

### D1 工作区收紧

写 `scripts/sweep_workspace.py`：网格撒点（范围同 A1 验证网格）× overhead 姿态链 ×
3 个 z 高度（桌面/悬停/放置），每点跑 solve_ik 候选链，输出收敛热力表；
把 `engine_capx.py` 的 `CUBE_X/Y_RANGE` 收缩到「全部 z 层 ≥3 候选收敛」的最大矩形，
结果写进注释。

### D2 抓取闭环（`scripts/demo_stack_piper_capx.py`）

**以 `open_details/primitive_api_cube_reset.py` 的模式为准重写**（ASPIRE 官方验证过，
比我们自己的处方权威；`docs/api_asset_map.md` §5 有模式库摘要）：

1. **定位 `get_object_info`**：`get_best_mask`（SAM3 文本 → **area 过滤 50-12000px
   排除机械臂 mask** → Molmo 打点 → SAM3 点提示 → best-score 兜底；提示词回落
   "red cube"→"red block"）→ `mask_to_world_points` → median center、top_z=p95、
   min_z=p5。
2. **抓取 z = (top_z+min_z)/2**；预抓 +10cm；姿态走 overhead 候选链（Piper 特有，
   替代官方的 TOP_DOWN_QUAT）。
3. **抓住判定**：`close_gripper` 后读 `robot0_gripper_qpos`（被方块撑住 = 开度明显
   大于全闭；等效判定 `robot_joint_pos[-1]` fraction 停在 0.15~0.85）。
4. **空抓重试 ≤3 次**：张开 → safe_home（固定高位清开相机视野）→ 重新定位 →
   抓取 z 每次降 2mm。
5. **放置**：悬停 +12cm 对准（dxy <1cm）→ 降到 绿块 top_z + 红块半高（0.02）→
   松爪 → 撤离。

### E 收尾

1. 重构为 `PiperControlApiReduced` 类（`aspire/primitives_capx.py`）：

```python
class PiperControlApiReduced:
    IK_BACKEND = "pyroki"      # dls | pyroki | curobo（A1/A2 验收后定）
    GRASP_BACKEND = "cgn"      # geometric | cgn
    def __init__(self, engine): ...
    def functions(self) -> dict:   # 键名与 cap-x 完全一致
        return {"get_observation": ..., "segment_sam3_text_prompt": ...,
                "segment_sam3_point_prompt": ..., "point_prompt_molmo": ...,
                "plan_grasp": ..., "get_oriented_bounding_box_from_3d_points": ...,
                "solve_ik": ..., "move_to_joints": ...,
                "open_gripper": ..., "close_gripper": ...,
                "select_top_down_grasp": ...,
                "mask_to_world_points": ..., "pixel_to_world_point": ...,
                "rotation_matrix_to_quaternion": ..., "interpolate_segment": ...}
```

   `build_namespace` 改为 `api.functions()` 消费 + 原有 wrap/trace 逻辑不变。

1b. **收编 `select_top_down_grasp`**（ASPIRE 论文附录 api-reference 里的函数，
   来源 `external/cap-x/capx/integrations/franka/control_reduced_skill_library.py:319`，
   ~30 行直接 vendor）：

```python
def select_top_down_grasp(self, grasps, scores, cam_to_world,
                          vertical_threshold=0.8):
    """(N,4,4)相机系候选 + (N,)得分 → (最佳world系4x4, score) 或 (None, -inf)。
    筛选 z 轴与竖直方向点积 > threshold 的候选中得分最高者。"""
```

   对 Piper 的特殊价值：top-down 不可达时把 threshold 调低（如 0.5），
   退化为"尽量竖直"筛选器，与 demo 的 overhead 候选链互补。
   注册进 `.functions()` 与命名空间。

1c. **收编 4 个工具函数**（open_details 三份官方任务代码实际调用，
   属 ASPIRE API 面但不在 cap-x reduced 契约内；L1 直拿/自写，
   详见 `docs/api_asset_map.md` #11-#14）：

```python
def mask_to_world_points(mask, depth, K, pose_mat):
    """(H,W)u8 mask + 深度 + 内参 + 外参 → (N,3) 基座系点云。
    反投影约定用 y = -(v-cy)*z/fy（与 pose_mat 配套，见契约偏差声明）。"""

def pixel_to_world_point(u, v, z, K, E):
    """单点版：像素 + 该点深度 → 基座系 3D 点（Molmo 打点落 3D 用）。"""

def rotation_matrix_to_quaternion(R):
    """(3,3) → wxyz 四元数（内部用 robosuite T.mat2quat 再重排，5 行）。"""

def interpolate_segment(p0, p1, step):
    """两点间按 step 密化路径点（wipe 类任务用，linspace，10 行）。"""
```

   注册进 `.functions()` 与命名空间（函数集 = 15 个）。

1d. **可视化工具 vendor**（trace 抓取候选渲染，论文组件 1 要求）：
   vendor `capx/utils/visualization_utils.py` 的 `overlay_segmentation_masks` /
   `draw_molmo_point` / `draw_oriented_bounding_box` / `render_cylinder_axis`
   进 `aspire/annotate.py`（L1 直拿，纯图像函数），plan_grasp 调用时把候选位姿
   渲染回图像存 trace。

2. 补写 `docs/primitive_api_capx.md`：15 函数逐一给 签名/参数/返回/副作用/示例 +
   后端说明 + 两处有意偏差声明（照抄 `primitives_capx.py` 头部注释的偏差段）。
   **docstring 以 `control_reduced.py` 各函数的 docstring 为底稿 vendor**
   （改 Piper 参数：6 轴、TCP_OFFSET=0、基座系），不从零写文档。
3. 扩展 `scripts/test_piper_capx_api.py`：加 B1/B2/C/A1 的用例（沿用 check() 模式）。
4. 提醒用户 git 提交（**用户确认前不要提交**）。

## 6. 验收标准（全部满足才算完成）

1. `MUJOCO_GL=egl /home/stouching/anaconda3/envs/ASPIRE/bin/python scripts/test_piper_capx_api.py` 全 PASS（含新用例）。
2. `MUJOCO_GL=egl /home/stouching/anaconda3/envs/ASPIRE/bin/python -m aspire.engine_capx --code scripts/demo_stack_piper_capx.py --task Stack --seed 0` → success: True；seed 1、2 至少 2/3。
3. `PiperControlApiReduced` 类存在，`.functions()` = 15 个函数
   （cap-x reduced 10 + select_top_down_grasp +
   工具 4：mask_to_world_points / pixel_to_world_point /
   rotation_matrix_to_quaternion / interpolate_segment）。
4. 组件集成证据：IK 后端对比报告（含 cuRobo 状态）、CGN 在 vision_server 运行、
   trace 含 `collision_events`；`docs/api_asset_map.md` 看板全部 `- [x]`。
5. `docs/primitive_api_capx.md` 与实现一致。

## 7. 约束

- API 层禁止 GT；GT 只在测试脚本。
- 已许可：pyroki（pip git 源，URL 出自 cap-x pyproject.toml）、jax[cpu]、cuRobo（pip）；
  Piper URDF 由用户提供。**其余下载（CGN checkpoint 等模型权重）先问用户；
  凡需 HuggingFace 拉取的一律先请用户人工确认仓库可访问（HF 有审批门控，
  403/401 容易误判为"项目不存在"）。**
- cap-x 契约不可动：函数名/签名/返回结构、基座系、wxyz、阻塞语义、plan_grasp 相机系约定、
  solve_ik 目标=TCP 位姿（Piper TCP_OFFSET=0）。
- 一切 CUDA 推理进程隔离（坑 6）。
- 上下文节约（上一会话死于上下文爆炸）：日志重定向到文件只读关键行；不贴大图/长日志；
  每次回复聚焦一个子任务；trace 用 python 脚本提取分析。
- 处方与现实冲突时停下汇报，不发明新方案。

---
