#!/usr/bin/env bash
# ASPIRE Web UI 入口（FastAPI + 静态 SPA，端口 8200）
set -u
cd "$(dirname "$0")/.."
PY=/home/stouching/anaconda3/envs/ASPIRE/bin/python
echo "ASPIRE Web UI: http://127.0.0.1:8200"
exec "$PY" -m aspire.web.server
