"""Inworld provider settings.

Provider-specific knobs live here so ``BaseAgentSettings`` stays provider-neutral.
Generic TTS selection still uses ``TTS_PROVIDER`` / ``TTS_MODEL`` / ``TTS_VOICE``.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

InworldTTSDeliveryMode = Literal[
    "",
    "DELIVERY_MODE_UNSPECIFIED",
    "STABLE",
    "BALANCED",
    "CREATIVE",
]


class InworldSettings(BaseSettings):
    """Env-driven config for Inworld TTS."""

    model_config = SettingsConfigDict(
        env_prefix="INWORLD_",
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    api_key: str = Field(default="", description="Inworld API key")
    tts_delivery_mode: InworldTTSDeliveryMode = Field(
        default="",
        description=(
            "Optional Inworld TTS-2 delivery mode. Empty means do not send the field "
            "and let Inworld use its server default."
        ),
    )
    tts_language: str = Field(
        default="",
        description="Optional BCP-47 language tag passed to Inworld TTS, e.g. en-US.",
    )


__all__ = ["InworldSettings", "InworldTTSDeliveryMode"]
