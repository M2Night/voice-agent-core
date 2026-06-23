"""Fish Audio provider settings.

Fish-specific config lives here, NOT on ``BaseAgentSettings`` — so the generic config
stays brand-agnostic and a frontend can render "Fish settings" only when Fish is the
selected provider (via ``FishSettings.model_json_schema()``).

Only knobs with a genuine per-deployment tradeoff are exposed: the API key and the
Fish-native latency/quality mode. Everything else about Fish TTS (pcm output, native
streaming, min-chunk-length, onset fade, sample rate) is a validated optimization with
no per-scenario tradeoff, so it's hardcoded in ``fish/tts.py`` rather than configurable.

Env names are preserved via ``env_prefix="FISH_"`` (``FISH_API_KEY``,
``FISH_TTS_LATENCY_MODE``).
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

FishTTSLatencyMode = Literal["normal", "balanced", "low"]
"""Fish Audio TTS latency mode (matches fishaudio plugin enum).

- ``low``: lowest latency, may trade quality
- ``balanced``: default tradeoff
- ``normal``: standard latency, highest quality
"""


class FishSettings(BaseSettings):
    """Fish-provider config, env-driven with the ``FISH_`` prefix.

    Read by ``build_fish_stt`` / ``build_fish_tts``. Holds only the Fish API key (shared
    by Fish STT + TTS) and the latency/quality mode. Generic selection (model / voice /
    language) stays on ``BaseAgentSettings``.
    """

    model_config = SettingsConfigDict(
        env_prefix="FISH_",
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    api_key: str = Field(default="", description="Fish Audio API key (STT + TTS)")
    tts_latency_mode: FishTTSLatencyMode = Field(
        default="balanced",
        description="Fish TTS latency/quality tradeoff (low | balanced | normal)",
    )


__all__ = ["FishSettings", "FishTTSLatencyMode"]
