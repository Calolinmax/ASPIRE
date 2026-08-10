# scripts/ 任务层规范（2026-08-10 定稿）

> 本目录是**任务层**：一切"跑任务/自检/工具"的入口。规范由用户指令确立——
> 每个任务一对文件：`tasks/<task>.py`（任务代码）+ `<task>.sh`（一键启动器）。
> 新任务必须照本规范编写。

## 0. 目录结构

```
scripts/
├── stack.sh / wipe.sh      # 任务入口（统一 .sh 启动器）
├── tasks/                  # 任务代码——由引擎注入 API 后 exec 执行，非独立脚本
│   ├── stack.py            #   Stack：SAM3 定位 → CGN 抓红 → 堆绿
│   └── wipe.py             #   Wipe：SAM3 定位污渍 → 双程蛇形 → Franka 擦板擦拭
├── tests/                  # 自检
│   └── test_piper_capx_api.py   # 契约 15 函数 33 项全量自检（验收门）
├── tools/                  # 资产生产线 / 查看器 / 常驻服务
│   ├── pyroki_server_minimal.py #   pyroki IK 常驻服务（:8116，每次跑任务都需要）
│   ├── cgn_execute_grasp.py     #   CGN 抓取验收门（CLI 直驱，--view 开窗）
│   ├── build_piper_ik_library.py    # IK 种子库生产线（几何/安装变更后必重建）
│   ├── generate_piper_urdf_from_mjcf.py  # pyroki URDF 生产线（模型变更时重跑）
│   └── check_cgn_pose.py / view_cgn_poses.py / view_scene_interactive.py / watch_grasp_live.py
└── archive/                # 历史诊断留档（一次性攻坚产物，结论在文件头注释，不日常运行）
```

## 1. 任务代码规则（`tasks/<task>.py`）

由 `ExecutionEngineCapx` 注入命名空间后 `exec` 执行，**不是独立 Python 脚本**：

1. **只能用**：契约 15 函数（唯一依据 [../docs/primitive_api_capx.md](../docs/primitive_api_capx.md)，
   禁止臆造 API）+ 契约外 5 个（`grasp_cgn` / `execute_legs_rrt` / `set_gripper_ramp` /
   `move_to_joints_planned` / `draw_grasp_glyphs`）+ `np` + `print`。
   **禁止 import**（np 已注入）；可以 `def` 自己的辅助函数。
2. **坐标约定**：全部机械臂基座系；四元数 wxyz；IK 目标 = grip_site（TCP）。
3. **三条行为铁律**（stack/wipe 同款）：
   - **响亮失败**：出错抛 `RuntimeError`，不静默兜底；
   - **凡动必避障规划**：长距离移动走 `execute_legs_rrt`；只有接触段/密化小步段
     才裸 `move_to_joints`（配 graze 容差 `tol=0.06, fk_tol=0.05`）；
   - **先算后动**：整链预解 IK 收敛才执行，预解失败不碰场景。
4. **起手式**：Step 0 一律收臂让拍（`TUCK_Q = np.array([-1.5708,0,0,0,0,0])`，
   `execute_legs_rrt([TUCK_Q])`），让开相机走廊再观测。
5. **参考模板**：路径跟踪类抄 `tasks/wipe.py`；抓取类抄 `tasks/stack.py`；
   论文原始逻辑看 `open_details/` 三份官方任务代码。

骨架：

```python
print("=" * 68); print("cap-x/Piper <Task>: ..."); print("=" * 68)
TUCK_Q = np.array([-1.5708, 0.0, 0.0, 0.0, 0.0, 0.0])

# Step 0: 收臂让拍 + 观测
try:
    execute_legs_rrt([TUCK_Q])
except RuntimeError:
    move_to_joints(TUCK_Q)
obs = get_observation()
cam = obs["robot0_robotview"]
rgb, depth, K, E = cam["images"]["rgb"], cam["images"]["depth"], cam["intrinsics"], cam["pose_mat"]

# Step 1: 感知定位（SAM3 文本 → molmo 点 → 点提示分割 级联兜底）
masks = segment_sam3_text_prompt(rgb, "<object>")
# ... Step 2+: 规划 → 预解 → 执行 → 验收
print("=== TASK CODE DONE ===")
```

## 2. 场景（需要时）

- **复用现有**：`--task Stack`（官方双块）、`--task PiperWipeSpill`（污渍）、
  或 robosuite 官方环境名——跳过本节。
- **新场景**：在 `aspire/envs/<name>.py` 仿照 `wipe_spill.py` 抄官方 arena 直建
  （⚠️ 桌几何必须锁 Stack 值 🔏），然后在 `aspire/robots/__init__.py` 加一行
  import 注册。**该文件已锁🔒——须用户本人解锁**（`chmod u+w aspire/robots/__init__.py`，
  改完回锁）。`--task` 传注册的类名。
- **成功判定**两选一：场景类实现 `_check_success()`（引擎 run 末尾自动调用，
  wipe 的覆盖率判定即此模式）；或任务代码内自验（stack 的 `check_stacked` 模式）。

## 3. 启动器（`scripts/<task>.sh`）

照抄 `wipe.sh` / `stack.sh` 模板，替换任务名与场景类名：

```bash
#!/usr/bin/env bash
set -u
cd "$(dirname "$0")/.."
PY=/home/stouching/anaconda3/envs/ASPIRE/bin/python
SEED="${1:-$RANDOM}"          # 约定: 默认随机, 显式传入复现（seed 传入 env.rng）
SLOW="${2:-0.5}"              # 约定: 默认 2 倍速观看; 1=实时; 2/3=慢放
echo "本次 seed: $SEED（复现: scripts/<task>.sh $SEED $SLOW）"
echo "========== preflight =========="
code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8116/health 2>/dev/null)
[ "$code" != "000" ] && echo "  ✓ pyroki 在线" || echo "  ✗ pyroki 不在线（回落 DLS, 慢）"
# 有抓取才加 CGN 预检（curl :8117/health 须含 "model_loaded":true）
echo "==============================="
exec "$PY" -m aspire.engine.engine_capx \
    --code scripts/tasks/<task>.py \
    --task <场景类名> --seed "$SEED" \
    --render --render-slowdown "$SLOW"
```

写完 `chmod +x scripts/<task>.sh`。

## 4. 验证流程

```bash
# 1. 先 headless 排错（不开窗）
/home/stouching/anaconda3/envs/ASPIRE/bin/python -u -m aspire.engine.engine_capx \
    --code scripts/tasks/<task>.py --task <场景类名> --seed 0

# 2. 看证据: traces/ 最新目录的 trace.json + images/top/ 帧流 + 算法标注图

# 3. 回归（证明没碰坏冻结层）
python scripts/tests/test_piper_capx_api.py   # 必须 33/33

# 4. 开窗看效果
scripts/<task>.sh
```

## 5. 服务依赖

| 服务 | 端口 | 何时需要 | 不在线的行为 |
|---|---|---|---|
| vision_server（SAM3） | :8123 | 所有任务 | 引擎自动拉起，无需预检 |
| pyroki IK | :8116 | 所有任务 | 回落 DLS 兜底（慢但可用） |
| CGN 容器 | :8117 | 仅抓取任务 | 回落几何规划器（非验收链路） |

## 6. 常见坑（实测教训）

1. **任务代码禁止 import**——`np` 已注入；import 打破"只依赖契约"假设
   （agentic coding 阶段这是硬约束）。
2. **IK 别用严格顶朝下**（`[0,0,1,0]`）——Piper 腕限位（j5 ±70°）死区；
   活路 = 倾斜姿态族 + `free_approach_roll=True` + 种子链（wipe 平板贴桌要
   `free_approach_roll=False` + 倾角 38°，全姿态 IK 实测唯一整路径收敛）。
3. **拖场景约束须用户同意**——home=全零、桌几何、相机位姿、摆放区均 🔏 锁定。
4. **冻结层只能在 scripts 调用，禁止修改**（🔒 清单见
   [../docs/project_files.md](../docs/project_files.md) §9）。
