"""LLM provider builders.

Two concrete builders, registered into the provider registry (see ``providers.py``):

- ``build_livekit_llm`` — LiveKit Inference (free for LiveKit Cloud users; no extra key)
- ``build_openrouter_llm`` — OpenRouter (default; 50+ models; requires ``OPENROUTER_API_KEY``)

The public dispatcher ``build_llm(settings)`` lives in ``providers.py`` and selects the
builder by ``settings.llm_provider``. Both builders read the model id from
``settings.llm_model``.
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


__all__ = ["OpenRouterSettings", "build_livekit_llm", "build_openrouter_llm"]
