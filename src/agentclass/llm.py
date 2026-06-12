"""Multi-provider LLM client (OpenAI + OpenRouter) with disk caching.

Routing is by model id (see ``Settings.provider_for``): ids without a "/" go to
OpenAI directly (e.g. ``gpt-4.1-nano``); ids with a "/" go to OpenRouter (e.g.
``google/gemini-2.5-flash-lite``). This lets the project spend the OpenAI ($30)
and OpenRouter ($70) budgets independently.

Every ``chat`` result is a raw ``response.model_dump()`` and therefore carries a
``usage`` block; use ``tokens_of`` to read (prompt, completion) tokens for the
compute-matched budgeting in EXPERIMENTS.md §5.2.
"""

from __future__ import annotations

import hashlib
import json
import re
from typing import Any

import diskcache
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from agentclass.config import settings

_clients: dict[str, OpenAI] = {}
_cache: diskcache.Cache | None = None


def get_client(provider: str) -> OpenAI:
    """Return a cached OpenAI-SDK client for the given provider."""
    if provider not in _clients:
        if provider == "openai":
            if not settings.openai_api_key:
                raise RuntimeError("OPENAI_API_KEY is not set but an OpenAI model was requested.")
            _clients[provider] = OpenAI(
                api_key=settings.openai_api_key,
                base_url=settings.openai_base_url,
                timeout=settings.llm_timeout_seconds,
            )
        elif provider == "openrouter":
            if not settings.openrouter_api_key:
                raise RuntimeError(
                    "OPENROUTER_API_KEY is not set but an OpenRouter model was requested."
                )
            _clients[provider] = OpenAI(
                api_key=settings.openrouter_api_key,
                base_url=settings.openrouter_base_url,
                default_headers=settings.openrouter_headers() or None,
                timeout=settings.llm_timeout_seconds,
            )
        else:  # pragma: no cover - defensive
            raise ValueError(f"Unknown provider: {provider!r}")
    return _clients[provider]


def get_cache() -> diskcache.Cache:
    global _cache
    if _cache is None:
        settings.cache_dir.mkdir(parents=True, exist_ok=True)
        _cache = diskcache.Cache(str(settings.cache_dir))
    return _cache


def _cache_key(model: str, messages: list[dict], **kwargs: Any) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages, "kwargs": kwargs},
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def parse_json(content: str | None) -> dict:
    """Robustly parse a JSON object from model output.

    Handles markdown code fences (```json ... ```), which Anthropic/Claude models
    emit even in JSON mode, and falls back to the first balanced {...} block.
    Returns {} on failure rather than raising.
    """
    if not content:
        return {}
    s = content.strip()
    if s.startswith("```"):
        s = re.sub(r"^```[a-zA-Z]*\n?", "", s)
        s = re.sub(r"\n?```$", "", s).strip()
    try:
        return json.loads(s)
    except (json.JSONDecodeError, ValueError):
        i, j = s.find("{"), s.rfind("}")
        if 0 <= i < j:
            try:
                return json.loads(s[i:j + 1])
            except (json.JSONDecodeError, ValueError):
                return {}
        return {}


def tokens_of(response: dict) -> tuple[int, int]:
    """Return (prompt_tokens, completion_tokens) from a chat result dict."""
    usage = response.get("usage") or {}
    return int(usage.get("prompt_tokens", 0)), int(usage.get("completion_tokens", 0))


@retry(stop=stop_after_attempt(8), wait=wait_exponential(multiplier=1, min=2, max=60))
def chat(
    messages: list[dict],
    model: str | None = None,
    response_format: dict | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    use_cache: bool | None = None,
    cache_nonce: str | None = None,
) -> dict:
    model = model or settings.llm_cheap
    provider = settings.provider_for(model)
    temperature = settings.llm_temperature if temperature is None else temperature
    max_tokens = max_tokens or settings.llm_max_tokens
    use_cache = settings.cache_enabled if use_cache is None else use_cache

    kwargs: dict[str, Any] = {
        "temperature": temperature,
        "max_tokens": max_tokens,
    }
    if response_format is not None:
        kwargs["response_format"] = response_format

    cache = get_cache() if use_cache else None
    # Cache key includes the model id (encodes provider) and, for sampled calls,
    # a nonce so N samples at the same temperature don't collapse to one cached
    # response. The nonce is NOT sent to the API.
    key = _cache_key(model, messages, _nonce=cache_nonce, **kwargs) if cache is not None else None
    if cache is not None and key in cache:
        return cache[key]

    client = get_client(provider)
    try:
        response = client.chat.completions.create(model=model, messages=messages, **kwargs)
    except Exception as e:  # noqa: BLE001 - fall back if model rejects json mode
        if "response_format" in kwargs and "response_format" in str(e).lower():
            kwargs.pop("response_format")
            response = client.chat.completions.create(model=model, messages=messages, **kwargs)
        else:
            raise
    result = response.model_dump()
    if cache is not None and key is not None:
        cache[key] = result
    return result
