"""Runtime helpers that wire up the Fish voice agent infrastructure.

These exist so every demo (lead-qualification, outbound-sales, …) inherits the
same input-side audio enhancement, warmup short-circuit, prewarmed VAD, and
TTS connection prewarm without copy-pasting the patterns into each ``main.py``.

Typical demo entrypoint::

    from voice_agent_core import (
        build_pipeline,
        build_session,
        default_prewarm,
        default_room_options,
        is_warmup_session,
        warm_tts,
    )

    server.setup_fnc = default_prewarm

    @server.rtc_session(agent_name="my-demo")
    async def entry(ctx):
        if is_warmup_session(ctx):
            return
        pipeline = build_pipeline(settings, vad=ctx.proc.userdata["vad"])
        asyncio.create_task(warm_tts(pipeline.tts))
        session = build_session(pipeline)
        await session.start(
            agent=MyAgent(),
            room=ctx.room,
            room_options=default_room_options(),
        )
        await ctx.connect()
        await session.say("Greeting…")
"""

from __future__ import annotations

from typing import Any

from livekit.agents import (
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    room_io,
)
from livekit.agents import tts as agents_tts
from livekit.plugins import ai_coustics, silero

from voice_agent_core.observability import get_logger
from voice_agent_core.pipeline import PipelineComponents

log = get_logger(__name__)


def default_prewarm(proc: JobProcess) -> None:
    """Load silero VAD once per worker process and stash it on ``proc.userdata``.

    Demos that need additional prewarm work should compose with this rather
    than replace it::

        def prewarm(proc):
            default_prewarm(proc)
            proc.userdata["custom_model"] = ...
        server.setup_fnc = prewarm
    """
    proc.userdata["vad"] = silero.VAD.load()


def is_warmup_session(ctx: JobContext) -> bool:
    """True when the room name starts with ``warmup-``.

    Frontends fire a phantom agent dispatch on page load (room name
    ``warmup-<id>``) to wake the cold worker before the user clicks Talk.
    The entry function should return immediately on warmup sessions — no
    pipeline build, no audio plumbing, no notification side-effects.
    """
    return ctx.room.name.startswith("warmup-")


def default_room_options() -> room_io.RoomOptions:
    """LiveKit ``RoomOptions`` with AI Coustics mic-side audio enhancement.

    QUAIL_VF_S runs noise cancellation + echo cancellation + dereverberation
    on the visitor's mic feed. Cleaner input → better STT → more accurate
    turn detection → fewer overlapping audio frames in playback.
    """
    return room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=ai_coustics.audio_enhancement(
                model=ai_coustics.EnhancerModel.QUAIL_VF_S,
            ),
        ),
    )


def build_session(
    pipeline: PipelineComponents,
    *,
    preemptive_generation: bool = True,
    **overrides: Any,
) -> AgentSession:
    """Assemble an ``AgentSession`` from a pipeline with the standard defaults.

    Defaults applied:

    - STT / TTS / LLM / VAD wired from the pipeline
    - ``turn_handling`` wraps ``pipeline.turn_detection`` in
      ``TurnHandlingOptions``
    - ``preemptive_generation=True`` so the LLM starts generating before
      end-of-turn is fully confirmed (lower first-token latency at the cost
      of occasional discarded responses)

    Any extra kwargs in ``overrides`` flow straight to ``AgentSession`` and
    take precedence — pass e.g. ``preemptive_generation=False`` to disable.
    """
    return AgentSession(
        stt=pipeline.stt,
        tts=pipeline.tts,
        llm=pipeline.llm,
        vad=pipeline.vad,
        turn_handling=TurnHandlingOptions(turn_detection=pipeline.turn_detection),
        preemptive_generation=preemptive_generation,
        **overrides,
    )


async def warm_tts(tts: agents_tts.TTS, *, text: str = ".") -> None:
    """Drive a tiny streaming synth round-trip to open the TTS WebSocket pool.

    Fire as ``asyncio.create_task(warm_tts(pipeline.tts))`` in parallel with
    ``session.start`` + ``ctx.connect``. By the time the real opener fires
    (typically a few hundred ms later) the connection is hot — the first
    audio chunk arrives faster, giving WebRTC's adaptive jitter buffer more
    data before playback starts and reducing the first-word stutter.

    Important: this must use the streaming path (``tts.stream()``), not
    ``synthesize()``. ``session.say`` and ``session.generate_reply`` go
    through the streaming WebSocket, which is a different connection pool
    from ``synthesize()``'s HTTP POST — warming the wrong path is wasted.

    Best-effort: failures are logged and swallowed. The real synth will
    surface any actual provider error to the user via its normal failure
    path.
    """
    try:
        stream = tts.stream()
        try:
            stream.push_text(text)
            stream.end_input()
            async for _ in stream:
                pass
        finally:
            await stream.aclose()
    except Exception as exc:
        log.warning("tts.warmup_failed", error=repr(exc))


__all__ = [
    "build_session",
    "default_prewarm",
    "default_room_options",
    "is_warmup_session",
    "warm_tts",
]
