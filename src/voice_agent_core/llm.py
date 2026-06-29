"""LLM provider builders.

Three concrete builders, registered into the provider registry (see ``providers.py``):

- ``build_livekit_llm`` — LiveKit Inference (free for LiveKit Cloud users; no extra key)
- ``build_openrouter_llm`` — OpenRouter (default; 50+ models; requires ``OPENROUTER_API_KEY``)
- ``build_custom_llm`` — any OpenAI-compatible endpoint (self-hosted SGLang/vLLM, a
  gateway, …); reads its own ``CUSTOM_LLM_*`` config so it never shares ``LLM_MODEL``
  with the OpenRouter/LiveKit builders.

The public dispatcher ``build_llm(settings)`` lives in ``providers.py`` and selects the
builder by ``settings.llm_provider``. The livekit/openrouter builders read the model id
from ``settings.llm_model``; the custom builder reads ``CUSTOM_LLM_MODEL`` (falling back
to ``settings.llm_model``).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from livekit.agents import llm as agents_llm
from pydantic_settings import BaseSettings, SettingsConfigDict

from voice_agent_core.observability import get_logger

if TYPE_CHECKING:
    from voice_agent_core.config import BaseAgentSettings

log = get_logger(__name__)

_OPENROUTER_APP_NAME = "voice-agent-core"


class OpenRouterSettings(BaseSettings):
    """OpenRouter-provider config, env-driven with the ``OPENROUTER_`` prefix.

    Provider-owned (read by ``build_openrouter_llm``) so the key isn't on the generic
    ``BaseAgentSettings``. Env name preserved: ``OPENROUTER_API_KEY``.
    """

    model_config = SettingsConfigDict(
        env_prefix="OPENROUTER_",
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    api_key: str = ""


class CustomLLMSettings(BaseSettings):
    """Config for an OpenAI-compatible custom endpoint, env-driven (``CUSTOM_LLM_``).

    Provider-owned (read by ``build_custom_llm``) and deliberately namespaced with its
    own ``CUSTOM_LLM_`` prefix so the endpoint's model id (``CUSTOM_LLM_MODEL``) never
    collides with the generic ``LLM_MODEL`` that the openrouter/livekit builders read.
    Point this at a self-hosted SGLang/vLLM server, a gateway, or any API that speaks
    the OpenAI chat-completions protocol.
    """

    model_config = SettingsConfigDict(
        env_prefix="CUSTOM_LLM_",
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    base_url: str = ""
    model: str = ""
    api_key: str = "EMPTY"
    # Hard cap on reply length, sent as the legacy `max_tokens` field via extra_body
    # (SGLang/vLLM compatibility). 0 = no cap.
    max_tokens: int = 0
    temperature: float = 0.6


def build_custom_llm(settings: BaseAgentSettings) -> agents_llm.LLM:
    """Build an LLM against any OpenAI-compatible endpoint.

    Reads ``CUSTOM_LLM_*`` from :class:`CustomLLMSettings` — its own ``base_url``,
    ``model``, ``api_key`` and optional ``max_tokens`` — so it's a direct client with
    no proxy hop and no ``reasoning_effort`` (non-OpenAI models reject it). The model
    id comes from ``CUSTOM_LLM_MODEL`` and falls back to the generic ``LLM_MODEL`` only
    if unset.
    """
    cfg = CustomLLMSettings()
    if not cfg.base_url:
        raise ValueError("CUSTOM_LLM_BASE_URL is required when llm_provider=custom")
    model = cfg.model or settings.llm_model

    from livekit.plugins import openai

    log.info(
        "llm.build",
        provider="custom",
        model=model,
        base_url=cfg.base_url,
        max_tokens=cfg.max_tokens,
    )
    kwargs: dict = {
        "model": model,
        "base_url": cfg.base_url,
        "api_key": cfg.api_key or "EMPTY",
        "temperature": cfg.temperature,
    }
    if cfg.max_tokens > 0:
        kwargs["extra_body"] = {"max_tokens": cfg.max_tokens}
    return openai.LLM(**kwargs)


def build_livekit_llm(settings: BaseAgentSettings) -> agents_llm.LLM:
    """Build a LiveKit Inference LLM. Requires LiveKit Cloud credentials."""
    if not settings.livekit_api_key or not settings.livekit_api_secret:
        raise ValueError(
            "LIVEKIT_API_KEY and LIVEKIT_API_SECRET are both required when "
            "llm_provider=livekit (LiveKit Inference authenticates against your "
            "LiveKit Cloud project)"
        )

    from livekit.agents import inference

    log.info("llm.build", provider="livekit", model=settings.llm_model)
    return inference.LLM(model=settings.llm_model)


def build_openrouter_llm(settings: BaseAgentSettings) -> agents_llm.LLM:
    """Build an OpenRouter-backed LLM. ``OPENROUTER_API_KEY`` from ``OpenRouterSettings``."""
    openrouter = OpenRouterSettings()
    if not openrouter.api_key:
        raise ValueError("OPENROUTER_API_KEY is required when llm_provider=openrouter")

    from livekit.plugins import openai

    log.info("llm.build", provider="openrouter", model=settings.llm_model)
    return openai.LLM.with_openrouter(
        model=settings.llm_model,
        api_key=openrouter.api_key,
        app_name=_OPENROUTER_APP_NAME,
    )


__all__ = [
    "CustomLLMSettings",
    "OpenRouterSettings",
    "build_custom_llm",
    "build_livekit_llm",
    "build_openrouter_llm",
]
