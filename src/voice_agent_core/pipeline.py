"""Pipeline factory: assemble a complete LiveKit voice agent pipeline from settings.

``build_pipeline(settings)`` returns a :class:`PipelineComponents` bundle with all
the moving parts an ``AgentSession`` needs: STT, TTS, LLM, VAD, and a turn-detection
selection.

Turn detection is carried as a *mode marker*, not a constructed object: the value is
either a string (``"multilingual"`` / ``"vad"`` / ``"stt"``, from
``settings.turn_detection_mode``) or a turn-detector instance you inject. The
``"multilingual"`` transformer model needs a LiveKit job context, so it is resolved
lazily in :func:`voice_agent_core.runtime.build_session` (which runs inside the
session entrypoint) — **not** here. This keeps ``build_pipeline`` callable anywhere
(tests, tooling), with no job context required.

Usage (inside the session entrypoint)::

    pipeline = build_pipeline(settings, vad=ctx.proc.userdata["vad"])
    session = build_session(pipeline)  # resolves the turn detector in-context

VAD note: silero loads a PyTorch model — slow on first call. For production, prewarm
in ``JobProcess.prewarm`` and pass it in via ``vad=``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from livekit.agents import llm as agents_llm
from livekit.agents import stt as agents_stt
from livekit.agents import tts as agents_tts
from livekit.plugins import silero

from voice_agent_core.observability import get_logger
from voice_agent_core.providers import build_llm, build_stt, build_tts
from voice_agent_core.stt import StreamAdapter

if TYPE_CHECKING:
    from livekit.plugins.turn_detector.multilingual import MultilingualModel

    from voice_agent_core.config import BaseAgentSettings

log = get_logger(__name__)


@dataclass
class PipelineComponents:
    """All the pieces an ``AgentSession`` needs, in one bundle.

    Apps construct this via :func:`build_pipeline` and hand it to
    :func:`voice_agent_core.runtime.build_session`.
    """

    stt: agents_stt.STT
    tts: agents_tts.TTS
    llm: agents_llm.LLM
    vad: silero.VAD
    turn_detection: str | MultilingualModel
    # Provider-independent session behavior carried from settings, mirroring
    # turn_detection: build_pipeline sets it from settings.preemptive_generation,
    # build_session reads it off the bundle (unless an explicit arg overrides).
    # Defaulted so direct PipelineComponents construction stays non-breaking.
    preemptive_generation: bool = True
    # Mode-aware end-of-turn delay override, same carry pattern as above.
    # None = use build_session's mode-aware default (0.5 for vad, 0 otherwise);
    # a non-negative float overrides it.
    min_endpointing_delay: float | None = None


def build_pipeline(
    settings: BaseAgentSettings,
    *,
    vad: silero.VAD | None = None,
    stream_adapter_vad: silero.VAD | None = None,
    turn_detection: str | MultilingualModel | None = None,
    stream_adapt: bool | None = None,
) -> PipelineComponents:
    """Assemble a LiveKit voice pipeline from settings.

    No job context required: ``turn_detection`` defaults to the mode marker from
    ``settings.turn_detection_mode`` and the (context-bound) transformer model is
    constructed later in ``build_session``. Pass ``vad=`` a prewarmed instance in
    production; pass ``stream_adapter_vad=`` to give the batch-STT adapter a
    dedicated VAD instance; pass ``turn_detection=`` to inject your own detector.

    Stream adaptation is **automatic**: a non-streaming STT (e.g. Fish batch ASR) is
    wrapped in a VAD-based ``StreamAdapter`` so it presents a streaming interface;
    natively-streaming STT (Deepgram) is used as-is. Pass ``stream_adapt=True/False`` to
    force it on/off (default ``None`` = auto by capability).
    """
    log.info(
        "pipeline.build_start",
        stt_provider=settings.stt_provider,
        tts_provider=settings.tts_provider,
        llm_provider=settings.llm_provider,
        turn_detection_mode=settings.turn_detection_mode,
        preemptive_generation=settings.preemptive_generation,
        min_endpointing_delay=settings.min_endpointing_delay,
    )

    stt = build_stt(settings)
    pipeline_vad = vad if vad is not None else silero.VAD.load()
    # Auto: wrap non-streaming STT; explicit stream_adapt overrides the capability check.
    should_adapt = (not stt.capabilities.streaming) if stream_adapt is None else stream_adapt
    if should_adapt:
        if stt.capabilities.streaming:
            log.info(
                "stt.stream_adapter_skipped",
                provider=settings.stt_provider,
                reason="provider_already_streaming",
            )
        else:
            stt = StreamAdapter(
                stt=stt,
                vad=stream_adapter_vad if stream_adapter_vad is not None else pipeline_vad,
            )
            log.info("stt.stream_adapter_enabled", provider=settings.stt_provider)

    pipeline = PipelineComponents(
        stt=stt,
        tts=build_tts(settings),
        llm=build_llm(settings),
        vad=pipeline_vad,
        turn_detection=(
            turn_detection if turn_detection is not None else settings.turn_detection_mode
        ),
        preemptive_generation=settings.preemptive_generation,
        min_endpointing_delay=settings.min_endpointing_delay,
    )

    log.info("pipeline.build_done")
    return pipeline


__all__ = ["PipelineComponents", "build_pipeline"]
