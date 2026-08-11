# aspire/web/ —— Web UI（cap-x web 最小可用版复刻）

> 2026-08-10 新建（不锁）。参考：`external/cap-x/capx/web/`（FastAPI+React）。
> 刻意零构建：单个静态页，无 npm/React（对照 cap-x 的 nodeenv+npm build 重管线）。

## 保留的 cap-x 核心（最小可用三件套）

1. **POST /api/run + /api/stop**：后台线程跑 actor/evosearch/coordinator；
   单活跃会话（新 run 拒绝或先停旧 run）；stop = 取消事件 + 终止评估子进程。
2. **WebSocket /ws**：事件流（run 状态/LLM 响应/评估进度/skill 入库/trace 链接），
   新连接回放近 50 条事件。
3. **静态 SPA**：运行控制台（事件聊天流）+ 技能库浏览 + Traces 浏览
   （trace.json 详情 + 前后帧/标注图取图）。

## 刻意降级（相对 cap-x）

无流式 delta（一次性 LLM 响应事件）、无 viser 3D iframe（以 trace 图像代替）、
无 execution_step 埋点、无用户中途注入。

## 用法

```bash
scripts/web.sh    # http://127.0.0.1:8200
```

## 端点

| 端点 | 说明 |
|---|---|
| GET `/api/status` / `/api/services` | LLM 配置状态（不泄露 key）/ vision、pyroki、CGN 服务探活 |
| GET `/api/skills` / `/api/skills/{name}` | 技能库索引 / 条目全文 |
| GET `/api/traces` / `/api/trace_detail` / `/api/trace_image` | trace 列表 / trace.json 全文 / 目录取图（防路径穿越） |
| GET `/api/progress` | coordinator 进度 |
| POST `/api/run` `{mode,task(s),K,T,seeds,heldout,no_baseline}` | 启动 run |
| POST `/api/stop` | 停止当前 run |
| WS `/ws` | 事件流 |
