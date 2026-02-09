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


def _env_int(name: str, default: int) -> int:
    v = os.getenv(name, "").strip()
    if not v:
        return default
    try:
        return int(v)
    except ValueError:
        return default


def _env_str(name: str, default: str) -> str:
    v = os.getenv(name, "").strip()
    return v if v else default


def get_client() -> OpenAI:
    load_dotenv()
    api_key = _env_str("DEEPSEEK_API_KEY", "")
    if not api_key:
        raise RuntimeError("DEEPSEEK_API_KEY is not set in environment/.env")

    base_url = _env_str("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
    return OpenAI(api_key=api_key, base_url=base_url)


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
    DeepSeek API is OpenAI-compatible; deepseek-reasoner returns reasoning_content + content.
    NOTE: Do NOT feed reasoning_content back into subsequent turns (DeepSeek may return 400).
    """
    load_dotenv()
    client = get_client()

    model = model or _env_str("DEEPSEEK_MODEL", "deepseek-reasoner")
    max_tokens = max_tokens or _env_int("DEEPSEEK_MAX_TOKENS", 3500)
    timeout = timeout or _env_int("DEEPSEEK_TIMEOUT", 120)

    last_err: Exception | None = None
    for attempt in range(1, retries + 1):
        try:
            kwargs: dict[str, Any] = {
                "model": model,
                "messages": messages,
                "max_tokens": max_tokens,
                "timeout": timeout,
            }
            if response_format is not None:
                kwargs["response_format"] = response_format
            if thinking is not None:
                kwargs["thinking"] = thinking

            resp = client.chat.completions.create(**kwargs)

            choice0 = resp.choices[0]
            msg = choice0.message

            content = msg.content or ""
            reasoning = getattr(msg, "reasoning_content", None)
            finish_reason = getattr(choice0, "finish_reason", None)

            # Some SDKs expose request id differently; try best effort:
            request_id = getattr(resp, "id", None)

            usage = None
            if getattr(resp, "usage", None) is not None:
                usage = {
                    "prompt_tokens": getattr(resp.usage, "prompt_tokens", None),
                    "completion_tokens": getattr(resp.usage, "completion_tokens", None),
                    "total_tokens": getattr(resp.usage, "total_tokens", None),
                }

            return LLMResponse(
                content=content,
                reasoning_content=reasoning,
                usage=usage,
                model=model,
                finish_reason=finish_reason,
                request_id=request_id,
            )

        except Exception as e:
            last_err = e
            if attempt < retries:
                time.sleep(backoff_sec * attempt)
                continue
            raise

    assert last_err is not None
    raise last_err
