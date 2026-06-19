"""Pipeline factory: assemble a complete LiveKit voice agent pipeline from settings.

``build_pipeline(settings)`` returns a :class:`PipelineComponents` bundle with all
the moving parts an ``AgentSession`` needs: STT, TTS, LLM, VAD, turn detection.

Usage::

    pipeline = build_pipeline(settings)
    session = AgentSession(
        stt=pipeline.stt,
        tts=pipeline.tts,
        llm=pipeline.llm,
        vad=pipeline.vad,
        turn_handling=TurnHandlingOptions(turn_detection=pipeline.turn_detection),
    )

``vad`` and ``turn_detection`` are accepted as optional kwargs:

- VAD: silero loads a PyTorch model — slow on first call. For production, prewarm
  in ``JobProcess.prewarm`` and pass it in.
- MultilingualModel: its constructor requires a LiveKit job context, so it can
  only be created inside a ``@server.rtc_session`` entrypoint. If you call
  ``build_pipeline`` from outside such a context (e.g. unit tests), pass a
  sentinel via ``turn_detection=...`` to skip the default construction.

Production usage (everything default, called inside the session entrypoint)::

    pipeline = build_pipeline(settings)

Production with prewarmed VAD::

    def prewarm(proc: JobProcess):
        proc.userdata["vad"] = silero.VAD.load()

    pipeline = build_pipeline(settings, vad=ctx.proc.userdata["vad"])
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from livekit.agents import llm as agents_llm
from livekit.agents import stt as agents_stt
from livekit.agents import tts as agents_tts
from livekit.plugins import silero
from livekit.plugins.turn_detector.multilingual import MultilingualModel

from voice_agent_core.observability import get_logger
from voice_agent_core.providers import build_llm, build_stt, build_tts

if TYPE_CHECKING:
    from voice_agent_core.config import BaseAgentSettings

log = get_logger(__name__)


@dataclass
class PipelineComponents:
    """All the pieces an ``AgentSession`` needs, in one bundle.

    Apps construct this via :func:`build_pipeline` and unpack the fields into
    their ``AgentSession(...)`` call site.
    """

    stt: agents_stt.STT
    tts: agents_tts.TTS
    llm: agents_llm.LLM
    vad: silero.VAD
    turn_detection: MultilingualModel


def build_pipeline(
    settings: BaseAgentSettings,
    *,
    vad: silero.VAD | None = None,
    turn_detection: MultilingualModel | None = None,
) -> PipelineComponents:
    """Assemble a complete LiveKit voice pipeline from settings.

    Both ``vad`` and ``turn_detection`` are constructed eagerly if not provided.
    See module docstring for production prewarm patterns and the LiveKit job
    context requirement on ``MultilingualModel``.
    """
    log.info(
        "pipeline.build_start",
        stt_provider=settings.stt_provider,
        tts_provider=settings.tts_provider,
        llm_provider=settings.llm_provider,
    )

    pipeline = PipelineComponents(
        stt=build_stt(settings),
        tts=build_tts(settings),
        llm=build_llm(settings),
        vad=vad if vad is not None else silero.VAD.load(),
        turn_detection=turn_detection if turn_detection is not None else MultilingualModel(),
    )

    log.info("pipeline.build_done")
    return pipeline


__all__ = ["PipelineComponents", "build_pipeline"]
