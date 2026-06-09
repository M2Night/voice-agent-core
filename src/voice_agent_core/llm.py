"""LLM factory: build the right LLM client from settings.

Two backends, switched via ``settings.llm_backend`` (env: ``LLM_BACKEND``):

- ``livekit`` — LiveKit Inference (free for LiveKit Cloud users; no extra API key needed)
- ``openrouter`` — OpenRouter (50+ models; requires ``OPENROUTER_API_KEY``)

Usage::

    from voice_agent_core import BaseAgentSettings, build_llm

    settings = BaseAgentSettings()
    llm = build_llm(settings)
    session = AgentSession(llm=llm, ...)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from livekit.agents import llm as agents_llm

from voice_agent_core.observability import get_logger

if TYPE_CHECKING:
    from voice_agent_core.config import BaseAgentSettings

log = get_logger(__name__)

_OPENROUTER_APP_NAME = "voice-agent-core"


def build_llm(settings: BaseAgentSettings) -> agents_llm.LLM:
    """Construct an LLM client from settings.

    Dispatches on ``settings.llm_backend``. Raises ``ValueError`` if
    ``llm_backend=openrouter`` but ``OPENROUTER_API_KEY`` is missing.
    """
    backend = settings.llm_backend

    if backend == "livekit":
        if not settings.livekit_api_key or not settings.livekit_api_secret:
            raise ValueError(
                "LIVEKIT_API_KEY and LIVEKIT_API_SECRET are both required when "
                "llm_backend=livekit (LiveKit Inference authenticates against "
                "your LiveKit Cloud project)"
            )

        from livekit.agents import inference

        log.info("llm.build", backend="livekit", model=settings.llm_model)
        return inference.LLM(model=settings.llm_model)

    if backend == "openrouter":
        if not settings.openrouter_api_key:
            raise ValueError(
                "OPENROUTER_API_KEY is required when llm_backend=openrouter"
            )

        from livekit.plugins import openai

        log.info("llm.build", backend="openrouter", model=settings.openrouter_model)
        return openai.LLM.with_openrouter(
            model=settings.openrouter_model,
            api_key=settings.openrouter_api_key,
            app_name=_OPENROUTER_APP_NAME,
        )

    raise ValueError(f"Unknown llm_backend: {backend!r}")


__all__ = ["build_llm"]
