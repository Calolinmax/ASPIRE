"""LLM 接入配置（agentic coding 模块）。

配置来源（优先级从高到低）：
1. 显式构造参数
2. 环境变量（ASPIRE_LLM_*）
3. 仓库根目录 .env 文件（.gitignore 已排除，key 不入库）

provider 四种：
- ``openai``    OpenAI 兼容 chat completions 端点（默认；vLLM/OpenRouter/各类中转均此形态）
- ``anthropic`` Anthropic Messages API（/v1/messages），论文同构（Claude Opus）
- ``file``      文件桥接：prompt 写文件、人工/Claude Code 回写响应文件——
                与 AGENTS.md §4 现行交互式工作流同构，无 API key 也能跑全流程
- ``mock``      测试用脚本化响应（scripts/tests 用）

环境变量：
- ASPIRE_LLM_PROVIDER / ASPIRE_LLM_BASE_URL / ASPIRE_LLM_API_KEY / ASPIRE_LLM_MODEL
- ASPIRE_LLM_TEMPERATURE / ASPIRE_LLM_MAX_TOKENS / ASPIRE_LLM_TIMEOUT
- ASPIRE_LLM_FILE_DIR（file provider 的交换目录，默认 agent_runs/llm_bridge）
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1/chat/completions"
DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com/v1/messages"


def load_dotenv(path: str | None = None) -> dict[str, str]:
    """极简 .env 解析（KEY=VALUE，# 注释，忽略引号），不引入第三方依赖。"""
    path = path or os.path.join(REPO_ROOT, ".env")
    out: dict[str, str] = {}
    if not os.path.isfile(path):
        return out
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            v = v.strip().strip('"').strip("'")
            out[k.strip()] = v
    return out


def _env(key: str, dotenv: dict[str, str], default: str | None = None) -> str | None:
    return os.environ.get(key) or dotenv.get(key) or default


@dataclass
class LLMConfig:
    """单次模型查询的全部参数（对应 cap-x ModelQueryArgs，加 provider 维度）。"""

    model: str = ""
    provider: str = "openai"  # openai | anthropic | file | mock
    base_url: str = ""
    api_key: str | None = None
    temperature: float = 0.2
    max_tokens: int = 8192
    timeout: float = 600.0          # 单请求超时（秒；K3 级思考模型大多模态请求实测 180s+）
    max_retries: int = 6            # 5xx/403/传输重试上限（60s 短退避——episode 覆盖与快速失败的平衡）
    reasoning_effort: str = "medium"  # GPT 系 reasoning 模型用
    thinking_budget: int = 4096       # Claude 系 thinking budget
    file_dir: str = ""                # file provider 交换目录
    debug: bool = False
    digest_max_images: int = 8        # trace 摘要附图上限；模型不支持视觉时设 0（纯文本诊断）
    # 并行集成配置：[(model, [temps...])]，None → 用单模型 9 温度版
    ensemble_configs: list[tuple[str, list[float]]] | None = None
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_env(cls, **overrides) -> "LLMConfig":
        dotenv = load_dotenv()
        provider = (_env("ASPIRE_LLM_PROVIDER", dotenv, "openai") or "openai").lower()
        if provider == "anthropic":
            default_url = DEFAULT_ANTHROPIC_BASE_URL
            default_model = "claude-opus-4-6"
        elif provider == "openai":
            default_url = DEFAULT_OPENAI_BASE_URL
            default_model = "gpt-5.4"
        else:
            default_url = ""
            default_model = "manual"
        cfg = cls(
            provider=provider,
            base_url=_env("ASPIRE_LLM_BASE_URL", dotenv, default_url) or "",
            api_key=_env("ASPIRE_LLM_API_KEY", dotenv, None),
            model=_env("ASPIRE_LLM_MODEL", dotenv, default_model) or default_model,
            temperature=float(_env("ASPIRE_LLM_TEMPERATURE", dotenv, "0.2") or 0.2),
            max_tokens=int(_env("ASPIRE_LLM_MAX_TOKENS", dotenv, "8192") or 8192),
            timeout=float(_env("ASPIRE_LLM_TIMEOUT", dotenv, "600") or 600),
            file_dir=_env("ASPIRE_LLM_FILE_DIR", dotenv,
                          os.path.join(REPO_ROOT, "agent_runs", "llm_bridge")) or "",
            digest_max_images=int(_env("ASPIRE_DIGEST_MAX_IMAGES", dotenv, "8") or 8),
        )
        for k, v in overrides.items():
            setattr(cfg, k, v)
        return cfg

    def validate(self) -> None:
        """响亮失败：配置不可用尽早抛错，而不是第一次查询时才炸。"""
        if self.provider in ("openai", "anthropic"):
            if not self.base_url:
                raise RuntimeError(f"provider={self.provider} 需要 ASPIRE_LLM_BASE_URL")
            if not self.api_key:
                raise RuntimeError(
                    f"provider={self.provider} 需要 API key：设置 ASPIRE_LLM_API_KEY "
                    f"环境变量或写入仓库根 .env（.gitignore 已排除）")
        elif self.provider == "file":
            if not self.file_dir:
                raise RuntimeError("provider=file 需要 file_dir")
        elif self.provider == "mock":
            pass
        else:
            raise RuntimeError(f"未知 provider: {self.provider}")
