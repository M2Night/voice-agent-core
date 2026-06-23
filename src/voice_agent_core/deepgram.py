"""Deepgram STT builder (streaming) — registered as the ``deepgram`` STT provider.

Deepgram is a streaming ASR: unlike Fish's batch endpoint it emits interim/final
transcripts in real time, so the LLM can start the moment the user stops talking —
the simplest way to close Fish STT's batch latency gap for latency-sensitive
scenarios (e.g. customer support). Select it with ``STT_PROVIDER=deepgram``.

Requires the Deepgram plugin (``uv add 'livekit-agents[deepgram]'``) and a
``DEEPGRAM_API_KEY``. The plugin import is lazy so the rest of the library — and the
provider registry — work whether or not the extra is installed; the helpful error
only fires if you actually select Deepgram without the package present.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from livekit.agents import stt as agents_stt
from pydantic_settings import BaseSettings, SettingsConfigDict

from voice_agent_core.observability import get_logger

if TYPE_CHECKING:
    from voice_agent_core.config import BaseAgentSettings

log = get_logger(__name__)

_DEFAULT_MODEL = "nova-3"
_DEFAULT_LANGUAGE = "en"


class DeepgramSettings(BaseSettings):
    """Deepgram-provider config, env-driven with the ``DEEPGRAM_`` prefix.

    Provider-owned (read by ``build_deepgram_stt``) so the Deepgram key isn't on the
    generic ``BaseAgentSettings``. Env name preserved: ``DEEPGRAM_API_KEY``.
    """

    model_config = SettingsConfigDict(
        env_prefix="DEEPGRAM_",
        env_file=None,
        case_sensitive=False,
        extra="ignore",
    )

    api_key: str = ""


def build_deepgram_stt(settings: BaseAgentSettings) -> agents_stt.STT:
    """Construct a streaming Deepgram STT from a settings object.

    Reads ``stt_model`` (defaults to ``nova-3``) and ``stt_language``. Deepgram
    streaming needs an explicit language hint; the default is ``en`` for the common
    English smoke/dev path. Set ``stt_language=multi`` explicitly for Nova-3
    multilingual/code-switching. ``DEEPGRAM_API_KEY`` comes from ``DeepgramSettings``.
    """
    deepgram_settings = DeepgramSettings()
    if not deepgram_settings.api_key:
        raise ValueError("DEEPGRAM_API_KEY is required to build Deepgram STT")

    model = settings.stt_model or _DEFAULT_MODEL
    language = settings.stt_language or _DEFAULT_LANGUAGE

    # RISK: this onboards Deepgram via LiveKit's first-party plugin. It's the lowest-
    # effort, lowest-latency path today (in-process, direct to Deepgram with our key —
    # no LiveKit proxy), but it couples Deepgram to LiveKit. If we ever off-board
    # LiveKit we'll have to re-onboard Deepgram directly (deepgram-sdk + a custom
    # stt.STT adapter, like fish/stt.py).
    try:
        from livekit.plugins import deepgram
    except ImportError as exc:
        raise ImportError(
            "Deepgram STT requires the 'deepgram' extra — "
            "install with: uv add 'livekit-agents[deepgram]'"
        ) from exc

    log.info("stt.build", provider="deepgram", model=model, language=language)
    return deepgram.STT(
        model=model,
        language=language,
        api_key=deepgram_settings.api_key,
    )


__all__ = ["DeepgramSettings", "build_deepgram_stt"]
