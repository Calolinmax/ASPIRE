"""LLM client —— vendor 自 cap-x `capx/llm/client.py`（MIT，commit 53e9966）。

保留设计（开源参考纪律：先读完整函数体再适配，已逐行通读）：
- 裸 requests 打 OpenAI 兼容端点，不依赖任何 provider SDK；
- 并行集成推理 = 并发 N 候选 + 再一次 LLM 综合（synthesis），**非投票硬裁决**；
  候选级失败容忍、全灭才抛错、综合固定 temperature=0.2；
- 多模态编码约定：图像 PNG→base64 data URL 进 ``image_url`` part；
  相邻 text part 必须 collapse（`collapse_text_image_inputs` 逐字保留）。

适配 delta（相对 cap-x 原文件）：
1. provider 维度：`openai`（chat completions，原逻辑）/ `anthropic`（Messages API
   格式转换）/ `file`（prompt/response 文件桥接，对接交互式 Claude Code 工作流）/
   `mock`（测试脚本化响应）。
2. 5xx/404 长退避重试**加上限**（`max_retries`，cap-x 为无限循环——复现报告建议
   加上限的落地）；重试间隔沿用 240±90s 抖动。
3. 删除 OpenRouter 强制改道与模型名单分派（本项目端点由 .env 显式配置，
   不按模型名隐式路由）；payload 按 provider 分派而非模型名。
4. 新增 `extract_code`（```python fence 抽取，cap-x 在 trial.py 的同款逻辑）与
   `image_to_data_url`（numpy RGB → data URL，cap-x launch_utils 的同款编码）。
"""

from __future__ import annotations

import concurrent.futures
import copy
import json
import os
import random
import re
import time
from collections.abc import Callable, Iterable
from typing import Any

import requests

from .config import LLMConfig

# ---------------------------------------------------------------------------
# 多模态编码（cap-x launch_utils._get_visual_feedback 同款约定）
# ---------------------------------------------------------------------------


def image_to_data_url(rgb) -> str:
    """numpy (H,W,3) uint8 RGB → ``data:image/png;base64,...``（OpenAI vision 格式）。"""
    import base64

    import cv2

    ok, buf = cv2.imencode(".png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("PNG 编码失败")
    return "data:image/png;base64," + base64.b64encode(buf.tobytes()).decode("ascii")


def image_file_to_data_url(path: str) -> str:
    """图像文件 → data URL（按扩展名推断 mime，默认 png）。"""
    import base64
    import mimetypes

    mime = mimetypes.guess_type(path)[0] or "image/png"
    with open(path, "rb") as f:
        return f"data:{mime};base64," + base64.b64encode(f.read()).decode("ascii")


def collapse_text_image_inputs(messages: list[dict]) -> list[dict]:
    """合并相邻 text part 为一个，图像保持相对位置（cap-x 逐字 vendor）。

    作用于单条 message 的 content part 列表（构造多轮 prompt 后必调）。
    """
    new_prompt = []
    current_text_input = ""
    for message in messages:
        if message["type"] == "text":
            current_text_input += message["text"] + "\n"
        else:
            if current_text_input != "":
                new_prompt.append({"type": "text", "text": current_text_input})
                current_text_input = ""
            new_prompt.append(message)
    if current_text_input != "":
        new_prompt.append({"type": "text", "text": current_text_input})
    return new_prompt


def text_message(role: str, text: str) -> dict:
    return {"role": role, "content": [{"type": "text", "text": text}]}


# ---------------------------------------------------------------------------
# 代码 fence 抽取（cap-x trial._extract_code 同款：优先 ```python，退化任意 fence）
# ---------------------------------------------------------------------------

_FENCE_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str) -> str | None:
    """从模型响应中抽取代码块。多个 fence 取最后一个（修复响应常先引述旧代码）。"""
    blocks = _FENCE_RE.findall(text or "")
    if not blocks:
        return None
    return blocks[-1].strip() + "\n"


def extract_all_code(text: str) -> list[str]:
    return [b.strip() + "\n" for b in _FENCE_RE.findall(text or "")]


# ---------------------------------------------------------------------------
# provider 后端
# ---------------------------------------------------------------------------


def _to_anthropic_messages(messages: list[dict]) -> tuple[str, list[dict]]:
    """OpenAI 形态 messages → Anthropic Messages API 形态。

    system 消息抽出为顶层 system 字符串（多段拼接）；image_url part 转
    ``{"type":"image","source":{"type":"base64",...}}``；纯字符串 content 归一为 part 列表。
    """
    system_parts: list[str] = []
    out: list[dict] = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
        if role == "system":
            system_parts.extend(p.get("text", "") for p in parts if p.get("type") == "text")
            continue
        new_parts: list[dict] = []
        for p in parts:
            if p.get("type") == "text":
                new_parts.append({"type": "text", "text": p["text"]})
            elif p.get("type") == "image_url":
                url = p["image_url"]["url"]
                m = re.match(r"data:([^;]+);base64,(.*)", url, re.DOTALL)
                if not m:
                    raise RuntimeError("anthropic provider 仅支持 base64 data URL 图像")
                new_parts.append({
                    "type": "image",
                    "source": {"type": "base64", "media_type": m.group(1), "data": m.group(2)},
                })
            else:
                raise RuntimeError(f"未知 content part 类型: {p.get('type')}")
        out.append({"role": role, "content": new_parts})
    return "\n".join(system_parts), out


def _post_with_retry(url: str, headers: dict, payload: dict, cfg: LLMConfig) -> dict:
    """POST + 重试（快速失败版——2026-08-11 教训：240s×8 的 cap-x 式长退避在
    网关持续 504 时会造成 30 分钟零实验静默；改为短退避少次数，耗尽即抛出，
    由上层（actor）收口写 findings，绝不在等待中浪费墙钟）。

    两类可重试故障：① HTTP 5xx/404（限流/服务抖动）；② 传输层异常
    （ReadTimeout/ConnectionError——大 payload 长思考实测会撞读超时）。
    """
    last_exc: Exception | None = None
    for attempt in range(cfg.max_retries + 1):
        try:
            response = requests.post(url, headers=headers, data=json.dumps(payload),
                                     timeout=cfg.timeout)
        except requests.exceptions.RequestException as e:
            last_exc = e
            if attempt >= cfg.max_retries:
                raise
            sleep_time = min(30.0 * (attempt + 1), 90) + random.uniform(-10, 10)
            print(f"[llm_client] Retry {attempt + 1}/{cfg.max_retries}. 传输异常: "
                  f"{type(e).__name__}: {e}. {sleep_time:.0f}s 后重试...", flush=True)
            time.sleep(max(5.0, sleep_time))
            continue
        if response.status_code in [403, 404, 500, 502, 503, 504, 529] and attempt < cfg.max_retries:
            # 403 在 Kimi coding 端点实测为"间歇性网关拒收"（同 payload 几分钟后
            # 即恢复 200，2026-08-11 双 episode 实证），与 5xx 同等重试；
            # 真权限类 403 最多浪费 max_retries 次短退避，代价可接受。
            sleep_time = 60 + random.uniform(-20, 20)
            print(f"[llm_client] Retry {attempt + 1}/{cfg.max_retries}. "
                  f"status={response.status_code}. Error: {response.text[:300]}. "
                  f"Retrying in {sleep_time:.0f}s...", flush=True)
            time.sleep(sleep_time)
            continue
        response.raise_for_status()
        return response.json()
    raise last_exc  # type: ignore[misc]


def _query_openai(cfg: LLMConfig, messages: list[dict]) -> dict:
    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
    }
    headers = {"Content-Type": "application/json"}
    key = cfg.api_key or os.getenv("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"
    t0 = time.time()
    body = _post_with_retry(cfg.base_url, headers, payload, cfg)
    print(f"[llm_client] openai query done in {time.time() - t0:.1f}s", flush=True)
    if cfg.debug:
        print(json.dumps(body, indent=2)[:4000])
    try:
        content = body["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as exc:
        raise RuntimeError(f"Unexpected response format: {body}") from exc
    reasoning = body.get("choices", [{}])[0].get("message", {}).get("reasoning")
    if isinstance(content, list):  # 部分端点 content 为 part 列表
        content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
    return {"content": content, "reasoning": reasoning}


def _query_anthropic(cfg: LLMConfig, messages: list[dict]) -> dict:
    system, anthro_messages = _to_anthropic_messages(messages)
    payload: dict[str, Any] = {
        "model": cfg.model,
        "max_tokens": cfg.max_tokens,
        "temperature": cfg.temperature,
        "messages": anthro_messages,
    }
    if system:
        payload["system"] = system
    if cfg.thinking_budget > 0:
        payload["thinking"] = {"type": "enabled", "budget_tokens": cfg.thinking_budget}
    headers = {
        "Content-Type": "application/json",
        "x-api-key": cfg.api_key or os.getenv("ANTHROPIC_API_KEY") or "",
        "anthropic-version": "2023-06-01",
    }
    t0 = time.time()
    body = _post_with_retry(cfg.base_url, headers, payload, cfg)
    print(f"[llm_client] anthropic query done in {time.time() - t0:.1f}s", flush=True)
    if cfg.debug:
        print(json.dumps(body, indent=2)[:4000])
    content_parts: list[str] = []
    reasoning_parts: list[str] = []
    for block in body.get("content", []):
        if block.get("type") == "text":
            content_parts.append(block.get("text", ""))
        elif block.get("type") == "thinking":
            reasoning_parts.append(block.get("thinking", ""))
    if not content_parts:
        raise RuntimeError(f"Unexpected response format: {body}")
    return {"content": "".join(content_parts),
            "reasoning": "".join(reasoning_parts) or None}


def _query_file(cfg: LLMConfig, messages: list[dict]) -> dict:
    """文件桥接：prompt 落盘 → 轮询响应文件（交互式 Claude Code / 人工当模型）。

    交换协议：写 ``<file_dir>/prompt_XXXX.md``（渲染文本；**图像 part 解码为
    ``prompt_XXXX_img_N.<ext>`` 实体文件并在文中标注路径**——读文件的 agent 可以
    用视觉工具真正看到图，多模态不断链），等待 ``<file_dir>/response_XXXX.md``
    出现（内容为模型原始输出文本）。

    适用场景（2026-08-11 实证）：直连 API 配额耗尽但 Claude Code 会话通道仍可用时，
    由本会话充当 harness 的 LLM 后端——同一个 key，走活着的通道。
    """
    import base64

    os.makedirs(cfg.file_dir, exist_ok=True)
    idx = int(time.time() * 1000) % 1_000_000
    prompt_path = os.path.join(cfg.file_dir, f"prompt_{idx:06d}.md")
    resp_path = os.path.join(cfg.file_dir, f"response_{idx:06d}.md")
    rendered = []
    img_n = 0
    for msg in messages:
        rendered.append(f"\n\n===== role: {msg['role']} =====\n")
        content = msg["content"]
        parts = content if isinstance(content, list) else [{"type": "text", "text": content}]
        for p in parts:
            if p.get("type") == "text":
                rendered.append(p["text"])
            else:
                url = p.get("image_url", {}).get("url", "")
                m = re.match(r"data:([^;]+);base64,(.*)", url, re.DOTALL)
                if m:
                    img_n += 1
                    ext = m.group(1).split("/")[-1].replace("jpeg", "jpg")
                    img_path = os.path.join(cfg.file_dir, f"prompt_{idx:06d}_img_{img_n}.{ext}")
                    with open(img_path, "wb") as f:
                        f.write(base64.b64decode(m.group(2)))
                    rendered.append(f"\n[IMAGE {img_n}: {img_path}]\n")
                else:
                    rendered.append(f"\n[IMAGE: 非 base64 来源 {url[:80]}]\n")
    with open(prompt_path, "w", encoding="utf-8") as f:
        f.write("".join(rendered))
    print(f"[llm_client] file provider: prompt 已写 {prompt_path}\n"
          f"             请将模型响应写入 {resp_path}", flush=True)
    t0 = time.time()
    while not os.path.isfile(resp_path):
        if time.time() - t0 > cfg.timeout * 30:  # 文件桥接给足人工时间（默认 5 小时）
            raise RuntimeError(f"file provider 等待响应超时: {resp_path}")
        time.sleep(2.0)
    time.sleep(0.5)  # 防读到写了一半的文件
    with open(resp_path, encoding="utf-8") as f:
        content = f.read()
    return {"content": content, "reasoning": None}


# mock provider 的脚本化响应表：全局注入，测试用
_MOCK_HANDLER: Callable[[list[dict]], str] | None = None


def set_mock_handler(handler: Callable[[list[dict]], str] | None) -> None:
    """测试用：设置 mock provider 的响应函数（输入 messages，输出 content 文本）。"""
    global _MOCK_HANDLER
    _MOCK_HANDLER = handler


def _query_mock(cfg: LLMConfig, messages: list[dict]) -> dict:
    if _MOCK_HANDLER is None:
        raise RuntimeError("mock provider 未设置 handler（set_mock_handler）")
    return {"content": _MOCK_HANDLER(messages), "reasoning": None}


# ---------------------------------------------------------------------------
# 核心查询
# ---------------------------------------------------------------------------

_PROVIDERS = {
    "openai": _query_openai,
    "anthropic": _query_anthropic,
    "file": _query_file,
    "mock": _query_mock,
}


def query_model(cfg: LLMConfig, messages: list[dict]) -> dict:
    """单次模型查询，返回 ``{"content": str, "reasoning": str|None}``。"""
    fn = _PROVIDERS.get(cfg.provider)
    if fn is None:
        raise RuntimeError(f"未知 provider: {cfg.provider}")
    return fn(cfg, messages)


def query_model_streaming(cfg: LLMConfig, messages: list[dict]) -> Iterable[dict]:
    """流式查询（仅 openai provider；其余 provider 自动降级为一次性 yield）。

    Yields: {"type": "content_delta"|"reasoning_delta"|"done", ...}
    """
    if cfg.provider != "openai":
        result = query_model(cfg, messages)
        yield {"type": "content_delta", "content": result["content"]}
        yield {"type": "done", "content": result["content"], "reasoning": result["reasoning"]}
        return

    payload = {
        "model": cfg.model,
        "messages": messages,
        "temperature": cfg.temperature,
        "max_tokens": cfg.max_tokens,
        "stream": True,
    }
    headers = {"Content-Type": "application/json"}
    key = cfg.api_key or os.getenv("OPENAI_API_KEY")
    if key:
        headers["Authorization"] = f"Bearer {key}"

    full_content = ""
    full_reasoning = ""
    with requests.post(cfg.base_url, headers=headers, data=json.dumps(payload),
                       timeout=cfg.timeout, stream=True) as response:
        response.raise_for_status()
        content_type = response.headers.get("content-type", "")
        if "application/json" in content_type and "text/event-stream" not in content_type:
            # 服务器不支持 SSE：降级为一次性返回（cap-x 同款）
            body = response.json()
            full_content = body["choices"][0]["message"]["content"]
            yield {"type": "content_delta", "content": full_content}
            yield {"type": "done", "content": full_content, "reasoning": None}
            return
        for line in response.iter_lines():
            if not line:
                continue
            line_str = line.decode("utf-8")
            if not line_str.startswith("data: "):
                continue
            data_str = line_str[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
            except json.JSONDecodeError:
                continue
            choices = data.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            if delta.get("content"):
                full_content += delta["content"]
                yield {"type": "content_delta", "content": delta["content"]}
            if delta.get("reasoning"):
                full_reasoning += delta["reasoning"]
                yield {"type": "reasoning_delta", "content": delta["reasoning"]}
    yield {"type": "done", "content": full_content,
           "reasoning": full_reasoning if full_reasoning else None}


# ---------------------------------------------------------------------------
# 并行集成推理（cap-x query_model_ensemble / query_single_model_ensemble vendor）
# 并发 N 候选 + LLM 综合；非投票硬裁决；候选级失败容忍；综合固定 temperature=0.2
# ---------------------------------------------------------------------------

SYNTHESIS_RULES = """You are synthesizing {n} candidate Python solutions into one optimal program.

SYNTHESIS RULES:
1. Analyze critically and assume no candidate is fully correct
2. Prefer explicit checks over assumptions
3. Combine the best ideas from multiple candidates when appropriate
4. If candidates disagree fundamentally, choose the more robust approach

OUTPUT FORMAT (strict):
You may include reasoning before the fenced code block.
Output ONLY ONE fenced code block (```python...```) containing the complete final solution.
Do NOT include any other code blocks or code snippets outside this single block.
"""


def _extract_original_text(messages: list[dict]) -> str:
    """取 user 消息的纯文本部分（综合 prompt 的 <original_task_description>）。"""
    original_text = ""
    for msg in messages:
        if msg["role"] == "user":
            c = msg["content"]
            if isinstance(c, list):
                original_text += "".join(x.get("text", "") for x in c if isinstance(x, dict))
            elif isinstance(c, str):
                original_text += c
    return original_text


def _synthesize(cfg: LLMConfig, messages: list[dict], successful: list[dict],
                synthesis_model: str | None) -> dict:
    """并发候选完成后的综合步（cap-x 同款 system/user 结构）。"""
    candidates = "\n\n".join(
        f"--- Candidate ({r['model']}, temp={r['temp']}) ---\n{r['content']}"
        for r in successful
    )
    synthesis_prompt = [
        {"role": "system", "content": SYNTHESIS_RULES.format(n=len(successful))},
        text_message("user",
                     "Synthesize the best solution.\n\n"
                     f"<original_task_description>\n{_extract_original_text(messages)}\n"
                     "</original_task_description>\n\n"
                     f"<candidate_solutions>\n{candidates}\n</candidate_solutions>\n"),
    ]
    synth_cfg = copy.copy(cfg)
    synth_cfg.model = synthesis_model or cfg.model
    synth_cfg.temperature = 0.2
    final = query_model(synth_cfg, synthesis_prompt)
    candidates_txt = "\n\n".join(
        f"{'=' * 60}\nModel: {r['model']}\nTemperature: {r['temp']}\nSuccess: {r['ok']}\n"
        f"{'=' * 60}\n{r['content']}"
        for r in successful
    )
    synthesis_txt = (f"Model: {synth_cfg.model}\n\n{'=' * 60}\nREASONING\n{'=' * 60}\n"
                     f"{final.get('reasoning') or '(none)'}\n\n{'=' * 60}\nOUTPUT\n"
                     f"{'=' * 60}\n{final['content']}")
    return {
        "content": final["content"],
        "reasoning": final.get("reasoning"),
        "all_responses": successful,
        "ensemble_candidates_txt": candidates_txt,
        "ensemble_synthesis_txt": synthesis_txt,
    }


def query_model_ensemble(cfg: LLMConfig, messages: list[dict],
                         ensemble_configs: list[tuple[str, list[float]]] | None = None,
                         synthesis_model: str | None = None,
                         max_workers: int = 9) -> dict[str, Any]:
    """多模型×多温度并行集成（cap-x ENSEMBLE_CONFIGS 机制，配置显式传入）。"""
    configs = ensemble_configs or cfg.ensemble_configs or [(cfg.model, [0.1, 0.5, 0.9])]

    def query_single(model: str, temp: float) -> dict:
        sub = copy.copy(cfg)
        sub.model, sub.temperature = model, temp
        try:
            result = query_model(sub, copy.deepcopy(messages))
            return {"model": model, "temp": temp, "content": result["content"], "ok": True}
        except Exception as e:
            print(f"[ensemble] {model} temp={temp} FAILED: {e}", flush=True)
            return {"model": model, "temp": temp, "content": str(e), "ok": False}

    tasks = [(m, t) for m, temps in configs for t in temps]
    responses = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {executor.submit(query_single, m, t): (m, t) for m, t in tasks}
        for future in concurrent.futures.as_completed(futures):
            resp = future.result()
            responses.append(resp)
            if resp["ok"]:
                print(f"[ensemble] {resp['model']} temp={resp['temp']} ok", flush=True)
    successful = [r for r in responses if r["ok"]]
    if not successful:
        raise RuntimeError("All ensemble queries failed: "
                           + "; ".join(f"{r['model']}:{r['content'][:100]}" for r in responses))
    out = _synthesize(cfg, messages, successful, synthesis_model)
    out["all_responses"] = responses  # 落盘用全量（含失败候选）
    return out


def query_single_model_ensemble(cfg: LLMConfig, messages: list[dict],
                                temperatures: list[float] | None = None,
                                max_workers: int = 9) -> dict[str, Any]:
    """单模型×9 温度并行集成（cap-x 同款默认 [0.1..0.9]）。"""
    temps = temperatures or [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    return query_model_ensemble(cfg, messages,
                                ensemble_configs=[(cfg.model, temps)],
                                synthesis_model=cfg.model, max_workers=max_workers)
