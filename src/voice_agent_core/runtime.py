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
    )

    server.setup_fnc = default_prewarm

    @server.rtc_session(agent_name="my-demo")
    async def entry(ctx):
        if is_warmup_session(ctx):
            return
        pipeline = build_pipeline(settings, vad=ctx.proc.userdata["vad"])
        session = build_session(pipeline)
        await session.start(
            agent=MyAgent(),
            room=ctx.room,
            room_options=default_room_options(),
        )
        await ctx.connect()
        await session.say("Greeting…")

Note: ``warm_tts`` is exported but currently not recommended with Fish
Audio — see its docstring. The connection-pool prewarm pattern is only
beneficial with TTS providers whose plugins pool WebSocket connections.
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
from livekit.agents.voice.turn import PreemptiveGenerationOptions
from livekit.plugins import noise_cancellation, silero

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
    """LiveKit ``RoomOptions`` with LiveKit's BVC server-side noise cancellation.

    Switched 2026-06 from AI Coustics ``QUAIL_VF_S`` (client-side ONNX
    inference running in the agent container) to LiveKit's ``BVC()``
    (Background Voice Cancellation) which runs on the LiveKit SFU
    server-side. Two wins:

    1. **CPU freed up in the agent container.** AI Coustics QUAIL_VF_S
       was burning 10-30% CPU per session, which on a 2-core cloud
       worker was crowding silero VAD inference — we observed
       "inference is slower than realtime" warnings with delays up to
       4.5 seconds. BVC runs on the SFU, so the agent gets that CPU
       back for VAD + STT + LLM serialization.
    2. **Tighter integration with LiveKit's audio pipeline.** BVC is
       LiveKit's own model tuned for their codec/jitter behavior; less
       chance of artifact interaction with the rest of the path.

    The AI Coustics plugin is still a transitive dep (kept in
    pyproject for now in case a demo needs the client-side option).
    """
    return room_io.RoomOptions(
        audio_input=room_io.AudioInputOptions(
            noise_cancellation=noise_cancellation.BVC(),
        ),
    )


def build_session(
    pipeline: PipelineComponents,
    *,
    preemptive_generation: bool | None = None,
    min_endpointing_delay: float | None = None,
    **overrides: Any,
) -> AgentSession:
    """Assemble an ``AgentSession`` from a pipeline with the standard defaults.

    Defaults applied:

    - STT / TTS / LLM / VAD wired from the pipeline
    - ``turn_handling`` wraps ``pipeline.turn_detection`` and
      ``preemptive_generation`` inside a single ``TurnHandlingOptions`` —
      this is the v1.5+ API; passing ``preemptive_generation`` directly to
      ``AgentSession`` was deprecated and removed in v2.0.

    ``preemptive_generation`` resolution (highest precedence first):

    1. an explicit ``preemptive_generation=`` argument here (``None`` = unset)
    2. ``pipeline.preemptive_generation`` (set by ``build_pipeline`` from
       ``settings.preemptive_generation`` / the ``PREEMPTIVE_GENERATION`` env var)
    3. the ``PipelineComponents`` field default, ``True``

    The sentinel ``None`` default is what lets the env-driven setting win when the
    caller doesn't pass the arg, while still letting a caller force a value. When on,
    the LLM starts generating before end-of-turn is confirmed (lower first-token
    latency at the cost of occasional discarded responses); turn it off for tool-heavy
    flows where premature generations waste tokens.

    Any extra kwargs in ``overrides`` flow straight to ``AgentSession`` and
    take precedence — e.g. pass your own ``turn_handling=...`` to override the
    wrapper (and the resolved ``preemptive_generation``) entirely.

    Turn detection: ``pipeline.turn_detection`` is a mode marker. The
    ``"multilingual"`` transformer model is constructed *here* (inside the session
    entrypoint = a valid job context), which is why ``build_pipeline`` itself stays
    context-free. ``"vad"`` / ``"stt"`` pass straight through as LiveKit strings; an
    injected detector instance is used as-is.

    ``min_endpointing_delay`` resolution mirrors ``preemptive_generation`` (explicit
    arg > ``pipeline.min_endpointing_delay`` from settings/env > mode-aware default).
    The mode-aware default is ``0`` for the transformer / STT detector (already a strong
    end-of-turn signal — saves 200-500ms/turn) and ``0.5`` for VAD-only mode, where the
    silence buffer is load-bearing (delay 0 would clip users mid-pause). ``None`` at any
    layer defers to the next; pass a non-negative float to override.
    """
    td = pipeline.turn_detection
    if td == "multilingual":
        # Lazy + in-context: MultilingualModel() needs a running LiveKit job context,
        # which exists here (session entrypoint) but not in build_pipeline.
        from livekit.plugins.turn_detector.multilingual import MultilingualModel

        td = MultilingualModel()

    default_min_endpointing = 0.5 if pipeline.turn_detection == "vad" else 0

    # Explicit arg wins; otherwise fall back to the value carried on the pipeline
    # (settings/env). `is not None` (not truthiness) so an explicit False still wins.
    enabled = (
        preemptive_generation
        if preemptive_generation is not None
        else pipeline.preemptive_generation
    )

    # Same precedence as preemptive_generation: explicit arg > pipeline/settings >
    # mode-aware default. None at each layer means "defer to the next"; a 0.0 the user
    # set survives (it's not None), so they can force "no delay" on VAD mode.
    configured_delay = (
        min_endpointing_delay
        if min_endpointing_delay is not None
        else pipeline.min_endpointing_delay
    )
    effective_min_endpointing = (
        configured_delay if configured_delay is not None else default_min_endpointing
    )

    kwargs: dict[str, Any] = {
        "stt": pipeline.stt,
        "tts": pipeline.tts,
        "llm": pipeline.llm,
        "vad": pipeline.vad,
        "turn_handling": TurnHandlingOptions(
            turn_detection=td,
            preemptive_generation=PreemptiveGenerationOptions(
                enabled=enabled,
            ),
        ),
        "min_endpointing_delay": effective_min_endpointing,
    }
    kwargs.update(overrides)
    return AgentSession(**kwargs)


async def warm_tts(tts: agents_tts.TTS, *, text: str = "hi") -> None:
    """Drive a tiny streaming synth round-trip to open a TTS WebSocket.

    **Currently NOT recommended for use with Fish Audio via
    livekit-plugins-fishaudio.** Empirical measurement (see voice-agent-core
    issue tracker / commit history) showed that:

    1. The Fish plugin does not pool WebSocket connections — every
       ``tts.stream()`` opens a fresh socket, so the warmup connection is
       closed before the real ``session.say`` synth can reuse it.
       Connection-reused metrics in production logs were always ``false``.
    2. Single-character inputs like ``"."`` confused Fish's TTS into
       generating multi-second blobs of "silence audio" (one production
       sample produced 35 seconds of audio for a single ``.``), which
       wastes Fish billing and occasionally triggered an
       ``Inference backend returned empty audio`` error.

    The helper is kept in voice-agent-core because the *pattern* is sound —
    it will pay off when:

    - A TTS provider whose plugin actually pools WebSocket connections is
      added (e.g. ElevenLabs, OpenAI TTS)
    - The Fish plugin gains connection pooling upstream
    - We swap to a provider-agnostic abstraction

    Until one of those happens, demos should NOT call this. Default ``text``
    was changed from ``"."`` to ``"hi"`` so any future invocation hits a
    well-defined synth path rather than the empty-audio degenerate case.

    Implementation notes (when re-enabled): uses the streaming path
    (``tts.stream()``), not ``synthesize()`` — the latter goes through a
    separate HTTP POST endpoint that wouldn't share state with
    ``session.say``'s streaming WebSocket even with a pooling provider.
    Best-effort: failures are logged and swallowed.
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
