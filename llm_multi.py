"""
多模型 LLM 调用模块

支持的模型：
1. DeepSeek (deepseek-reasoner, deepseek-chat)
2. 小米 MiMo (mimo-v2-flash)

通过环境变量配置：
- LLM_PROVIDER: deepseek / mimo (默认 deepseek)
- 对应的 API_KEY 和 BASE_URL
"""
from __future__ import annotations
import os
import time
from dataclasses import dataclass
from typing import Any, Optional
from dotenv import load_dotenv
from openai import OpenAI


@dataclass
class LLMResponse:
    content: str
    reasoning_content: Optional[str]
    usage: dict[str, Any] | None
    model: str
    finish_reason: Optional[str] = None
    request_id: Optional[str] = None
    provider: str = "unknown"


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def get_provider() -> str:
    """获取当前配置的 LLM 提供商"""
    return _env_str("LLM_PROVIDER", "deepseek").lower()


def get_client_deepseek() -> OpenAI:
    """获取 DeepSeek 客户端"""
    api_key = _env_str("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY 未设置")
    base_url = _env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return OpenAI(api_key=api_key, base_url=base_url)


def get_client_mimo() -> OpenAI:
    """获取小米 MiMo 客户端"""
    api_key = _env_str("MIMO_API_KEY", "")
    if not api_key:
        raise RuntimeError("MIMO_API_KEY 未设置")
    base_url = _env_str("MIMO_BASE_URL", "https://api.xiaomimimo.com/v1")
    return OpenAI(api_key=api_key, base_url=base_url)


def chat_complete_deepseek(
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        response_format: dict[str, Any] | None = None,
        retries: int = 3,
        backoff_sec: float = 2.0,
) -> LLMResponse:
    """DeepSeek API 调用"""
    load_dotenv()
    client = get_client_deepseek()

    model = model or _env_str("DEEPSEEK_MODEL", "deepseek-reasoner")
    max_tokens = max_tokens or _env_int("DEEPSEEK_MAX_TOKENS", 3500)
    timeout = timeout or _env_int("DEEPSEEK_TIMEOUT", 120)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format

            resp = client.chat.completions.create(**kwargs)
            choice0 = resp.choices[0]
            msg = choice0.message

            return LLMResponse(
                content=msg.content or "",
                reasoning_content=getattr(msg, "reasoning_content", None),
                usage={
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                } if resp.usage else None,
                model=model,
                finish_reason=getattr(choice0, "finish_reason", None),
                request_id=getattr(resp, "id", None),
                provider="deepseek",
            )
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff_sec * attempt)

    raise last_err


def chat_complete_mimo(
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        response_format: dict[str, Any] | None = None,
        thinking_enabled: bool = True,  # MiMo 支持思考模式
        retries: int = 3,
        backoff_sec: float = 2.0,
) -> LLMResponse:
    """小米 MiMo API 调用"""
    load_dotenv()
    client = get_client_mimo()

    model = model or _env_str("MIMO_MODEL", "mimo-v2-flash")
    max_tokens = max_tokens or _env_int("MIMO_MAX_TOKENS", 4096)
    timeout = timeout or _env_int("MIMO_TIMEOUT", 120)

    last_err = None
    for attempt in range(1, retries + 1):
        try:
            kwargs = {
                "model": model,
                "messages": messages,
                "max_completion_tokens": max_tokens,
                "timeout": timeout,
                "temperature": 0.3,
                "top_p": 0.95,
                "stream": False,
            }

            # MiMo 的思考模式控制
            if thinking_enabled:
                kwargs["extra_body"] = {"thinking": {"type": "enabled"}}
            else:
                kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

            if response_format is not None:
                kwargs["response_format"] = response_format

            resp = client.chat.completions.create(**kwargs)
            choice0 = resp.choices[0]
            msg = choice0.message

            # MiMo 可能返回 reasoning_content（思考过程）
            reasoning = getattr(msg, "reasoning_content", None)

            return LLMResponse(
                content=msg.content or "",
                reasoning_content=reasoning,
                usage={
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                } if resp.usage else None,
                model=model,
                finish_reason=getattr(choice0, "finish_reason", None),
                request_id=getattr(resp, "id", None),
                provider="mimo",
            )
        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff_sec * attempt)

    raise last_err


def chat_complete(
        messages: list[dict[str, Any]],
        model: str | None = None,
        max_tokens: int | None = None,
        timeout: int | None = None,
        response_format: dict[str, Any] | None = None,
        thinking: dict[str, Any] | None = None,
        retries: int = 3,
        backoff_sec: float = 2.0,
) -> LLMResponse:
    """
    统一的 LLM 调用接口

    根据 LLM_PROVIDER 环境变量自动选择模型：
    - deepseek: 使用 DeepSeek API
    - mimo: 使用小米 MiMo API
    """
    load_dotenv()
    provider = get_provider()

    if provider == "mimo":
        # MiMo 模式
        thinking_enabled = True
        if thinking is not None and thinking.get("type") == "disabled":
            thinking_enabled = False

        return chat_complete_mimo(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            response_format=response_format,
            thinking_enabled=thinking_enabled,
            retries=retries,
            backoff_sec=backoff_sec,
        )
    else:
        # 默认 DeepSeek
        return chat_complete_deepseek(
            messages=messages,
            model=model,
            max_tokens=max_tokens,
            timeout=timeout,
            response_format=response_format,
            retries=retries,
            backoff_sec=backoff_sec,
        )


# 向后兼容的别名
def get_client() -> OpenAI:
    """向后兼容：返回当前配置的客户端"""
    provider = get_provider()
    if provider == "mimo":
        return get_client_mimo()
    return get_client_deepseek()