"""LLM factory with graceful degradation.

The digest step is optional: if no provider/key is configured (or the call
fails), the agent still produces a feed from heuristics.
"""

from __future__ import annotations

import os
from functools import lru_cache

from .config import _load_dotenv

DEFAULT_MODELS = {
    "openrouter": "deepseek/deepseek-v4.1-flash",
    "deepseek": "deepseek-chat",
    "openai": "gpt-4o-mini",
}

PROVIDERS = {
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "deepseek": ("https://api.deepseek.com", "DEEPSEEK_API_KEY"),
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "custom": ("NEWS_AGENT_BASE_URL", "NEWS_AGENT_API_KEY"),
}


@lru_cache(maxsize=8)
def _make(provider: str, model: str, base_url: str, api_key: str):
    from langchain_openai import ChatOpenAI

    return ChatOpenAI(
        model=model,
        base_url=base_url or None,
        api_key=api_key,
        temperature=0.2,
        timeout=90,
        max_retries=2,
    )


def build_llm(settings) -> tuple[object | None, str]:
    """Return ``(chat_model, reason_if_unavailable)``."""
    _load_dotenv()
    if not getattr(settings, "use_llm", True):
        return None, "LLM disabled (--no-llm)"

    provider = (settings.provider or "").lower()
    if not provider:
        provider = "openrouter" if os.getenv("OPENROUTER_API_KEY") else "deepseek"
    if provider not in PROVIDERS:
        return None, f"unknown provider '{provider}'"

    if provider == "custom":
        base_url = os.getenv("NEWS_AGENT_BASE_URL", "")
        key = os.getenv("NEWS_AGENT_API_KEY", "")
        if not base_url or not key:
            return None, "NEWS_AGENT_BASE_URL / NEWS_AGENT_API_KEY not set"
    else:
        base_url, key_env = PROVIDERS[provider]
        key = os.getenv(key_env, "")
        if not key:
            return None, f"{key_env} not set"

    model = settings.model or DEFAULT_MODELS.get(provider, "")
    if not model:
        return None, f"no model configured for provider '{provider}'"
    try:
        chat = _make(provider, model, base_url, key)
    except Exception as exc:  # pragma: no cover - import/config errors
        return None, f"failed to init chat model: {exc}"
    # resolve the auto-detected provider/model back into settings so that
    # logging and the digest metadata show what actually ran
    try:
        settings.provider, settings.model = provider, model
    except Exception:  # pragma: no cover - settings passed as a plain dict
        pass
    return chat, ""