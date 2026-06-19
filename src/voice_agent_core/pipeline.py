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


def build_pipeline(
    settings: BaseAgentSettings,
    *,
    vad: silero.VAD | None = None,
    turn_detection: str | MultilingualModel | None = None,
) -> PipelineComponents:
    """Assemble a LiveKit voice pipeline from settings.

    No job context required: ``turn_detection`` defaults to the mode marker from
    ``settings.turn_detection_mode`` and the (context-bound) transformer model is
    constructed later in ``build_session``. Pass ``vad=`` a prewarmed instance in
    production; pass ``turn_detection=`` to inject your own detector.
    """
    log.info(
        "pipeline.build_start",
        stt_provider=settings.stt_provider,
        tts_provider=settings.tts_provider,
        llm_provider=settings.llm_provider,
        turn_detection_mode=settings.turn_detection_mode,
    )

    pipeline = PipelineComponents(
        stt=build_stt(settings),
        tts=build_tts(settings),
        llm=build_llm(settings),
        vad=vad if vad is not None else silero.VAD.load(),
        turn_detection=(
            turn_detection if turn_detection is not None else settings.turn_detection_mode
        ),
    )

    log.info("pipeline.build_done")
    return pipeline


__all__ = ["PipelineComponents", "build_pipeline"]
