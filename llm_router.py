from __future__ import annotations

import json
import os
import threading
import time
from dataclasses import dataclass
from typing import Any, Mapping
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from .redaction import redact_secrets

_TIMEOUT_SECONDS = 30
_COOLDOWN_SECONDS = 60
_MAX_RESPONSE_BYTES = 2_000_000


@dataclass(frozen=True)
class Provider:
    name: str
    key_env: str
    base_url: str
    default_model: str
    model_env: str


PROVIDERS = (
    Provider(
        "groq",
        "GROQ_API_KEY",
        "https://api.groq.com/openai/v1",
        "openai/gpt-oss-120b",
        "GROQ_MODEL",
    ),
    Provider(
        "gemini",
        "GEMINI_API_KEY",
        "https://generativelanguage.googleapis.com/v1beta/openai",
        "gemini-2.5-flash",
        "GEMINI_MODEL",
    ),
    Provider(
        "mistral",
        "MISTRAL_API_KEY",
        "https://api.mistral.ai/v1",
        "mistral-small-latest",
        "MISTRAL_MODEL",
    ),
    Provider(
        "openrouter",
        "OPENROUTER_API_KEY",
        "https://openrouter.ai/api/v1",
        "recursive/flint-v1.0:free",
        "OPENROUTER_MODEL",
    ),
)

_cooldown_until: dict[str, float] = {}
_cooldown_lock = threading.Lock()


class ProviderError(RuntimeError):
    pass


def configured_generator():
    """Return a generator when any supported provider token is set."""
    return generate_text if configured_provider_names() else None


def configured_provider_names(environ: Mapping[str, str] | None = None) -> list[str]:
    values = os.environ if environ is None else environ
    return [
        provider.name
        for provider in PROVIDERS
        if values.get(provider.key_env, "").strip()
    ]


def generate_text(
    prompt: str,
    system_prompt: str = "You are ULTRON, a helpful personal AI assistant.",
) -> str:
    return chat(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ]
    )


def chat(
    messages: list[Mapping[str, str]],
    *,
    environ: Mapping[str, str] | None = None,
    now: float | None = None,
    request_func=None,
    cooldown_seconds: int = _COOLDOWN_SECONDS,
) -> str:
    """Try configured OpenAI-compatible providers in priority order."""
    values = os.environ if environ is None else environ
    clock = time.time if now is None else lambda: now
    requester = _request_provider if request_func is None else request_func
    if not _valid_messages(messages):
        raise ProviderError("AI request messages must be valid role/content objects.")

    failures: list[str] = []
    configured = [
        provider for provider in PROVIDERS if values.get(provider.key_env, "").strip()
    ]
    for provider in configured:
        current = clock()
        with _cooldown_lock:
            blocked_until = _cooldown_until.get(provider.name, 0)
        if blocked_until > current:
            failures.append(f"{provider.name}: cooling down")
            continue
        model = values.get(provider.model_env, provider.default_model).strip()
        if not model:
            failures.append(f"{provider.name}: model configuration is empty")
            continue
        try:
            content = requester(provider, values[provider.key_env].strip(), model, messages)
        except ProviderError as exc:
            failures.append(f"{provider.name}: {exc}")
            with _cooldown_lock:
                _cooldown_until[provider.name] = current + cooldown_seconds
            continue
        with _cooldown_lock:
            _cooldown_until.pop(provider.name, None)
        return content

    if not configured:
        raise ProviderError("No AI provider API key is configured.")
    summary = "; ".join(failures) or "all providers are unavailable"
    raise ProviderError(
        "All configured AI providers failed or are cooling down: " + summary
    )


def _valid_messages(messages: list[Mapping[str, str]]) -> bool:
    if not isinstance(messages, list) or not messages or len(messages) > 30:
        return False
    total_size = 0
    for message in messages:
        if not isinstance(message, Mapping):
            return False
        if set(message) - {"role", "content"}:
            return False
        role = message.get("role")
        content = message.get("content")
        if role not in {"system", "user", "assistant"} or not isinstance(content, str):
            return False
        total_size += len(content.encode("utf-8"))
    return total_size <= _MAX_RESPONSE_BYTES


def _request_provider(
    provider: Provider,
    token: str,
    model: str,
    messages: list[Mapping[str, str]],
) -> str:
    payload = json.dumps(
        {"model": model, "messages": messages},
        ensure_ascii=False,
        separators=(",", ":"),
    ).encode("utf-8")
    request = Request(
        f"{provider.base_url}/chat/completions",
        data=payload,
        headers={
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urlopen(request, timeout=_TIMEOUT_SECONDS) as response:
            body = response.read(_MAX_RESPONSE_BYTES + 1)
    except HTTPError as exc:
        if exc.code in {401, 403}:
            raise ProviderError(f"authentication rejected (HTTP {exc.code})") from None
        if exc.code == 429:
            raise ProviderError("rate limited (HTTP 429)") from None
        raise ProviderError(f"request rejected (HTTP {exc.code})") from None
    except (URLError, TimeoutError, OSError):
        raise ProviderError("connection failed or timed out") from None
    if len(body) > _MAX_RESPONSE_BYTES:
        raise ProviderError("response exceeded the 2 MB limit")
    try:
        data: Any = json.loads(body)
    except (UnicodeError, json.JSONDecodeError):
        raise ProviderError("provider returned invalid JSON") from None
    try:
        content = data["choices"][0]["message"]["content"]
    except (KeyError, IndexError, TypeError):
        raise ProviderError("provider returned an unexpected response") from None
    if not isinstance(content, str) or not content.strip():
        raise ProviderError("provider returned an empty response")
    return redact_secrets(content.strip())


def reset_cooldowns() -> None:
    """Clear in-memory cooldowns, primarily for focused tests."""
    with _cooldown_lock:
        _cooldown_until.clear()
