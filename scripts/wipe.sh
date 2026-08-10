#!/usr/bin/env bash
# Wipe 可视化实时仿真 · 一键启动（2026-08-06, 镜像 stack.sh）。
#
# 用法:
#   scripts/wipe.sh [seed] [slowdown]
#     seed     默认每次启动随机（$RANDOM, 换 seed 即换污渍形状）;
#              显式传入则固定复现（如 scripts/wipe.sh 0）
#     slowdown 默认 0.5（= 2 倍速观看手感）；1=20Hz 实时；2/3=慢放
#
# 场景: PiperWipeSpill（aspire/env_wipe_spill.py, 本次新增）——
#       Stack 锁定桌几何 + 官方 robosuite WipeArena 棕色污渍随机路径
#       （coverage_factor 0.25 → 世界 ±0.10 → 基座系 x∈[0.20,0.40] 可达带近旁）,
#       Piper 默认夹爪（官方 Wipe 环境的 WipingGripper 断言会拆夹爪, 不可用）。
# 链路: home → 收臂让拍(TUCK) → robotview RGB+depth → SAM3 "brown spill"
#       分割（molmo 级联兜底）→ bbox 双程蛇形 → (高度×倾角×方位)阶梯全路径
#       预解 → 闭爪悬擦执行 → RRT 收臂。成功判定 = 污渍访问覆盖率 ≥50%。
# 关窗不中断任务（自动转无窗口跑完）；执行结束窗口保持显示，关窗退出。
#
# 依赖: CGN 不需要（无抓取规划）; pyroki 需要（IK 默认后端, 不在线则回落 DLS
#       慢但可用）; vision_server 由引擎自动拉起。
# MUJOCO_GL 无需设置（同 stack.sh 说明）。
set -u
cd "$(dirname "$0")/.."

PY=/home/stouching/anaconda3/envs/ASPIRE/bin/python
SEED="${1:-$RANDOM}"
SLOW="${2:-0.5}"
echo "本次 seed: $SEED（复现: scripts/wipe.sh $SEED $SLOW）"

echo "========== preflight =========="
# pyroki（IK 默认后端；/health 无此路由, 任何 HTTP 应答即算活）
code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8116/health 2>/dev/null)
if [ "$code" != "000" ]; then
    echo "  ✓ pyroki 在线 (:8116, HTTP $code)"
else
    echo "  ✗ pyroki 不在线 (:8116)！solve_ik 将回落 DLS（慢但可用）。"
    echo "    启动: PYTHONPATH=$PWD/external/cap-x ~/venvs/pyroki/bin/python scripts/tools/pyroki_server_minimal.py &"
fi
# vision_server 由引擎自动拉起, 无需预检
echo "==============================="

exec "$PY" -m aspire.engine.engine_capx \
    --code scripts/tasks/wipe.py \
    --task PiperWipeSpill --seed "$SEED" \
    --render --render-slowdown "$SLOW"
