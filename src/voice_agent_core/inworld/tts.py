"""Inworld TTS builder.

The LiveKit Inworld plugin is the provider implementation; this module only maps
voice-agent-core's generic settings/env conventions to that plugin.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from voice_agent_core.inworld.settings import InworldSettings

if TYPE_CHECKING:
    from voice_agent_core.config import BaseAgentSettings

def build_inworld_tts(settings: BaseAgentSettings):
    """Construct LiveKit's Inworld TTS plugin from a settings object.

    The provider registry resolves Inworld's default model before calling this
    builder; direct callers must pass ``settings.tts_model`` explicitly.
    """
    inworld_settings = InworldSettings()
    if not inworld_settings.api_key:
        raise ValueError("INWORLD_API_KEY is required to build Inworld TTS")
    if not settings.tts_model:
        raise ValueError("TTS_MODEL is required to build Inworld TTS")

    try:
        from livekit.plugins import inworld
    except ImportError as exc:  # pragma: no cover - depends on optional install state
        raise ImportError(
            "livekit-plugins-inworld is required for TTS_PROVIDER=inworld"
        ) from exc

    kwargs: dict[str, Any] = {
        "api_key": inworld_settings.api_key,
        "model": settings.tts_model,
        "encoding": "PCM",
    }
    if settings.tts_voice:
        kwargs["voice"] = settings.tts_voice
    if inworld_settings.tts_delivery_mode:
        kwargs["delivery_mode"] = inworld_settings.tts_delivery_mode
    if inworld_settings.tts_language:
        kwargs["language"] = inworld_settings.tts_language

    return inworld.TTS(**kwargs)


__all__ = ["build_inworld_tts"]
