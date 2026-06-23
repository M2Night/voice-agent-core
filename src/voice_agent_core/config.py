"""Configuration loading: env-driven settings + YAML config files.

Two pieces:

- ``BaseAgentSettings`` — pydantic-settings base class for env vars. Apps subclass it
  to add app-specific fields (e.g. SLACK_WEBHOOK_URL). All secrets live here, never
  in YAML.
- ``load_yaml`` / ``load_env_walking_up`` — utilities for loading YAML config files
  and locating a ``.env`` file when the agent is run from a workspace subdirectory.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

LogFormat = Literal["json", "console"]
LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]
OTelExporter = Literal["console", "none"]
TurnDetectionMode = Literal["multilingual", "vad", "stt"]
"""How to detect end-of-user-turn.

- ``multilingual``: LiveKit's semantic transformer detector — best quality;
  constructed inside the session (needs a job context), so it's resolved in
  ``build_session``, not ``build_pipeline``.
- ``vad``: silence-based via VAD — lightest, loads no extra model.
- ``stt``: end-of-turn from the STT endpoint.
"""


class BaseAgentSettings(BaseSettings):
    """Env-driven settings shared by every voice agent built on voice-agent-core.

    Apps extend this with their own fields::

        class LeadQualSettings(BaseAgentSettings):
            slack_webhook_url: str | None = None

        settings = LeadQualSettings()  # reads from environment

    Call :func:`load_env_walking_up` BEFORE instantiating if the .env file lives
    above the agent's working directory (common in monorepos).
    """

    model_config = SettingsConfigDict(
        env_file=None,  # we use load_env_walking_up instead for monorepo-friendly discovery
        case_sensitive=False,
        extra="ignore",
    )

    # --- LiveKit ---
    livekit_url: str = Field(default="", description="LiveKit server URL (wss://...)")
    livekit_api_key: str = Field(default="", description="LiveKit API key")
    livekit_api_secret: str = Field(default="", description="LiveKit API secret")

    # NOTE: provider credentials (FISH_API_KEY, DEEPGRAM_API_KEY, OPENROUTER_API_KEY)
    # and provider-specific knobs live with their provider, not here — see
    # voice_agent_core.fish.settings.FishSettings etc. This keeps BaseAgentSettings
    # brand-agnostic: only generic selection (provider/model/voice/language) +
    # provider-independent behavior live below.

    # --- STT (provider → model → language) ---
    stt_provider: str = Field(
        default="deepgram",
        description=(
            "STT provider name; must be registered in providers.py. Defaults to "
            "'deepgram' (native streaming, low transcription latency; requires "
            "DEEPGRAM_API_KEY). Set 'fish' for Fish batch ASR — non-streaming providers "
            "are automatically wrapped with a VAD StreamAdapter."
        ),
    )
    stt_model: str = Field(
        default="",
        description="STT model id for the chosen provider ('' = provider default; Fish ASR has one model)",
    )
    stt_language: str = Field(
        default="en",
        description=(
            "Provider-specific STT language hint. 'auto' uses provider-specific "
            "behavior and is supported by Fish; Deepgram streaming requires an "
            "explicit value such as 'en' or 'multi'."
        ),
    )

    # --- TTS (provider → model → voice) ---
    tts_provider: str = Field(
        default="fish",
        description="TTS provider name; must be registered in providers.py (default 'fish')",
    )
    tts_model: str = Field(
        default="s2-pro",
        description=(
            "TTS model id for the chosen provider. For Fish: 's2-pro' (default) is "
            "clean in LiveKit 1.5.x; 's2.1-pro' produces audible static — listen-test "
            "before switching."
        ),
    )
    tts_voice: str = Field(
        default="",
        description=(
            "Voice selection for the chosen TTS provider ('' = provider default). "
            "Provider-defined value space (Fish: reference/voice id; OpenAI TTS: a named "
            "voice; etc.)."
        ),
    )

    # --- LLM (provider → model) ---
    llm_provider: str = Field(
        default="openrouter",
        description="LLM provider name; must be registered in providers.py (default 'openrouter')",
    )
    llm_model: str = Field(
        default="openai/gpt-5.4-mini",
        description=(
            "LLM model id for the chosen provider. For 'openrouter' (default): "
            "OpenRouter notation (e.g. 'openai/gpt-5.4-mini'). For 'livekit': "
            "LiveKit Inference model id configured for your LiveKit account."
        ),
    )

    # --- Turn detection ---
    turn_detection_mode: TurnDetectionMode = Field(
        default="multilingual",
        description=(
            "Turn-detection strategy: 'multilingual' (semantic transformer, default), "
            "'vad' (silence-based, lightest), or 'stt' (STT-endpoint based)."
        ),
    )

    # --- Session behavior ---
    preemptive_generation: bool = Field(
        default=True,
        description=(
            "Start LLM generation before end-of-turn is fully confirmed. True (default) "
            "lowers perceived response latency for simple conversational flows; set false "
            "for tool-heavy / high-precision flows (e.g. customer support) where premature "
            "generations may be discarded, waste tokens, or produce stale tool-call plans. "
            "Provider-independent LiveKit session knob."
        ),
    )
    min_endpointing_delay: float | None = Field(
        default=None,
        ge=0,
        description=(
            "Minimum silence (seconds) to wait before treating the user's turn as "
            "ended. Lower = snappier responses but more risk of cutting users off / "
            "barging in; higher = safer but laggier. None (default) keeps the mode-aware "
            "default: 0.5 for TURN_DETECTION_MODE=vad (the silence buffer is load-bearing), "
            "0 for multilingual/stt (they already carry a strong end-of-turn signal). "
            "Set a non-negative float to override; negative values fail validation. "
            "Provider-independent LiveKit session knob."
        ),
    )

    # --- Observability ---
    log_level: LogLevel = Field(default="INFO", description="Root log level")
    log_format: LogFormat = Field(
        default="json",
        description="Log output format ('json' for production, 'console' for local dev)",
    )
    otel_metrics_exporter: OTelExporter = Field(
        default="console",
        description="Where to send OTEL metrics ('console' prints to stdout for local dev, 'none' disables)",
    )


def load_yaml(path: str | Path) -> dict[str, Any]:
    """Load a YAML file into a dict.

    Returns an empty dict for empty files. Raises ``FileNotFoundError`` if the path
    doesn't exist, ``ValueError`` if the root isn't a mapping, ``yaml.YAMLError`` on
    parse failure.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {p}")
    with p.open(encoding="utf-8") as f:
        data = yaml.safe_load(f)
    if data is None:
        return {}
    if not isinstance(data, dict):
        raise ValueError(
            f"Expected YAML mapping at root of {p}, got {type(data).__name__}"
        )
    return data


def load_env_walking_up(
    start: Path | str | None = None,
    name: str = ".env",
    override: bool = False,
) -> Path | None:
    """Walk up from ``start`` looking for a ``.env`` file and load it.

    Useful in monorepos where the agent runs from ``apps/<name>/`` but ``.env``
    lives at the workspace root. Walks from ``start`` (default: current working
    directory) through every parent until either the file is found or filesystem
    root is reached.

    Returns the path of the file that was loaded, or ``None`` if no file was found.
    By default does not override values already present in the environment.
    """
    from dotenv import load_dotenv

    p = Path(start).resolve() if start else Path.cwd().resolve()
    for parent in [p, *p.parents]:
        candidate = parent / name
        if candidate.is_file():
            load_dotenv(candidate, override=override)
            return candidate
    return None


__all__ = [
    "BaseAgentSettings",
    "LogFormat",
    "LogLevel",
    "OTelExporter",
    "TurnDetectionMode",
    "load_env_walking_up",
    "load_yaml",
]
