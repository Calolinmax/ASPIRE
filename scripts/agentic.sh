#!/usr/bin/env bash
# agentic coding 入口：actor 修复闭环 / evosearch 进化搜索 / coordinator 多任务入库
# 用法:
#   scripts/agentic.sh actor Stack              # 单任务修复闭环（E.3）
#   scripts/agentic.sh evosearch Stack          # 进化搜索（Algorithm 1）
#   scripts/agentic.sh coordinator Stack,PiperWipeSpill
#   其余参数原样透传给 python -m aspire.agentic.cli
set -u
cd "$(dirname "$0")/.."
PY=/home/stouching/anaconda3/envs/ASPIRE/bin/python
MODE="${1:-actor}"; TASK="${2:-Stack}"; shift 2 || shift 1

echo "========== preflight =========="
code=$(curl -s -m 3 -o /dev/null -w '%{http_code}' http://127.0.0.1:8116/health 2>/dev/null)
[ "$code" != "000" ] && echo "  ✓ pyroki 在线" || echo "  ✗ pyroki 不在线（回落 DLS, 慢）"
code=$(curl -s -m 3 http://127.0.0.1:8117/health 2>/dev/null | grep -c '"model_loaded": *true')
[ "$code" -ge 1 ] && echo "  ✓ CGN 在线" || echo "  ✗ CGN 不在线（抓取任务回落几何规划器）"
if [ ! -f .env ] && [ -z "${ASPIRE_LLM_API_KEY:-}" ]; then
    echo "  ⚠ 无 .env 且无 ASPIRE_LLM_API_KEY——LLM 未配置。"
    echo "    复制 .env.example 为 .env 填入 key；或用 --provider file 走人工桥接。"
fi
echo "==============================="

case "$MODE" in
  actor)       exec "$PY" -m aspire.agentic.cli actor --task "$TASK" "$@" ;;
  evosearch)   exec "$PY" -m aspire.agentic.cli evosearch --task "$TASK" "$@" ;;
  coordinator) exec "$PY" -m aspire.agentic.cli coordinator --tasks "$TASK" "$@" ;;
  *)           exec "$PY" -m aspire.agentic.cli "$MODE" "$TASK" "$@" ;;
esac
