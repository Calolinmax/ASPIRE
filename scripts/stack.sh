#!/usr/bin/env bash
# Stack 可视化实时仿真 · 一键启动（2026-08-06 用户指令封装）。
#
# 用法:
#   scripts/stack.sh [seed] [slowdown]
#     seed     默认每次启动随机（$RANDOM, 参考 robosuite 官方 sampler+rng
#              机制——引擎把 seed 传入 env.rng, 换 seed 即换摆放）;
#              显式传入则固定复现（如 scripts/stack.sh 0 = 验收基准）
#     slowdown 默认 0.5（= 08-05 定稿的 2 倍速观看手感）；
#              1=20Hz 实时；2/3=慢放（细察抓取瞬间）
#
# 示例:
#   scripts/stack.sh          # 随机摆放, 2 倍速
#   scripts/stack.sh 3 1      # 固定 seed 3 复现, 实时
#
# 场景: 官方 robosuite Stack 原样导入（--official-stack）——红 cubeA 4cm +
#       绿 cubeB 5cm 两正方体, 无细高化、无干扰物（2026-08-06 用户指令）。
# 链路: home → 收臂让拍(TUCK) → top RGB+depth → CGN 抓取位姿 →
#       Piper 抓红块 → SAM3 定位绿块 → 堆叠。
# 关窗不中断任务（自动转无窗口跑完）；执行结束窗口保持显示，关窗退出。
#
# MUJOCO_GL 无需设置: robosuite import 时强制 egl 供离屏渲染（自愈机制覆盖）,
# 可视化窗口由 mujoco viewer 独立走 glfw 开窗, 与该变量无关。
# 排障: 窗口没出来 → 看下方 preflight 输出；执行问题看 traces/ 最新目录。
set -u
cd "$(dirname "$0")/.."

PY=/home/stouching/anaconda3/envs/ASPIRE/bin/python
SEED="${1:-$RANDOM}"
SLOW="${2:-0.5}"
echo "本次 seed: $SEED（复现: scripts/stack.sh $SEED $SLOW）"

echo "========== preflight =========="
# CGN 容器（抓取位姿真身后端）
if curl -s -m 3 http://127.0.0.1:8117/health 2>/dev/null | grep -q '"model_loaded":true'; then
    echo "  ✓ CGN 容器在线 (:8117)"
else
    echo "  ✗ CGN 容器不在线 (:8117)！demo 将回落几何规划器（非验收链路）。"
    echo "    启动: sudo docker start cgn   （首次建容器见 docs/cgn_container.md §4）"
fi
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
    --code scripts/tasks/stack.py \
    --task Stack --seed "$SEED" --official-stack \
    --render --render-slowdown "$SLOW"
