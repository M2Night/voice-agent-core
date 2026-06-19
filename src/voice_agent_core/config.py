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
TTSLatencyMode = Literal["normal", "balanced", "low"]
"""Fish Audio TTS latency mode (matches fishaudio plugin enum).

- ``low``: lowest latency, may trade quality
- ``balanced``: default tradeoff
- ``normal``: standard latency, highest quality
"""
OTelExporter = Literal["console", "none"]


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

    # --- Provider credentials (secrets) ---
    # Kept separate from provider/model selection: secrets live here, the
    # "which provider / which model" knobs live in the per-layer sections below.
    fish_api_key: str = Field(default="", description="Fish Audio API key (STT + TTS)")
    openrouter_api_key: str = Field(
        default="",
        description="OpenRouter API key (used when llm_provider=openrouter, the default)",
    )
    deepgram_api_key: str = Field(
        default="",
        description="Deepgram API key (only used when stt_provider=deepgram)",
    )

    # --- STT (provider → model) ---
    stt_provider: str = Field(
        default="fish",
        description="STT provider name; must be registered in providers.py (default 'fish')",
    )
    stt_model: str = Field(
        default="",
        description="STT model id for the chosen provider ('' = provider default; Fish ASR has one model)",
    )
    stt_language: str = Field(
        default="auto",
        description="STT language ('auto' for auto-detect)",
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
    tts_voice_id: str = Field(
        default="",
        description="TTS voice id ('' = provider default voice)",
    )
    tts_latency_mode: TTSLatencyMode = Field(
        default="balanced",
        description="TTS latency/quality tradeoff (Fish: low | balanced | normal)",
    )

    # --- LLM (provider → model) ---
    llm_provider: str = Field(
        default="openrouter",
        description="LLM provider name; must be registered in providers.py (default 'openrouter')",
    )
    llm_model: str = Field(
        default="anthropic/claude-sonnet-4-6",
        description=(
            "LLM model id for the chosen provider. For 'openrouter' (default): "
            "OpenRouter notation (e.g. 'anthropic/claude-sonnet-4-6'). For 'livekit': "
            "LiveKit Inference notation (e.g. 'openai/gpt-5.2-chat-latest')."
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
    "TTSLatencyMode",
    "load_env_walking_up",
    "load_yaml",
]
