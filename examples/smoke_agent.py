"""Smoke agent — minimum runnable agent built on voice-agent-core.

No business logic; just demonstrates the canonical wiring with the v0.2.x
runtime helpers. Use it to verify voice-agent-core works end-to-end in your
environment, and as a reference template for building your own agents.

Run::

    # From repo root (so uv finds the project's venv):
    cp examples/.env.example examples/.env  # then fill in real keys
    uv run python examples/smoke_agent.py dev

Then open https://agents-playground.livekit.io/ in a browser, paste your
LiveKit URL, generate a token, and talk through your microphone.
"""

import asyncio
import time
from pathlib import Path
from typing import Literal

from livekit.agents import Agent, AgentServer, JobContext, cli

from voice_agent_core import (
    BaseAgentSettings,
    NotificationField,
    NotificationPayload,
    SlackNotifier,
    build_pipeline,
    build_session,
    default_prewarm,
    default_room_options,
    format_transcript,
    get_logger,
    is_warmup_session,
    load_env_walking_up,
    setup_observability,
    summarize_transcript,
)

log = get_logger(__name__)
_TRANSCRIPT_CHAR_CAP = 1500


class SmokeSettings(BaseAgentSettings):
    """Extends BaseAgentSettings with the SLACK_WEBHOOK_URL knob.

    Shows the canonical pattern for downstream apps adding their own
    env-driven fields on top of voice-agent-core's base settings.
    """

    slack_webhook_url: str = ""
    # Opener: 'say' speaks a canned line (TTS only, ~0.6-1s — no LLM round-trip on
    # turn 0) for the snappiest first utterance; 'generate' has the LLM ideate the
    # greeting (exercises the full pipeline, but adds LLM TTFT to the first line).
    greeting_mode: Literal["say", "generate"] = "say"
    greeting_text: str = (
        "Hi! I'm a test agent for the voice pipeline. What's on your mind?"
    )


# Start the .env search from this script's directory (examples/) so it works
# whether you run from repo root or from examples/. cwd-based defaults wouldn't
# find examples/.env when the script is invoked as `python examples/smoke_agent.py`.
load_env_walking_up(start=Path(__file__).parent)
settings = SmokeSettings()
setup_observability(settings, service_name="voice-agent-core-smoke")

# LiveKit decides whether to start the local inference executor before any job
# entrypoint runs. Importing the multilingual turn-detector plugin here registers
# its inference runner early enough for local dev mode.
if settings.turn_detection_mode == "multilingual":
    from livekit.plugins.turn_detector.multilingual import (
        MultilingualModel as _MultilingualModel,  # noqa: F401
    )

server = AgentServer()
# Module-level notifier — each session subprocess gets its own copy (forked from
# the worker). Stateless: sharing initial config across forks is safe.
# Empty webhook_url triggers SlackNotifier's dev-log mode (no HTTP call).
notifier = SlackNotifier(webhook_url=settings.slack_webhook_url)
# default_prewarm loads the main silero VAD into proc.userdata["vad"]. This smoke
# demo uses Deepgram (streaming STT), so it does NOT opt into the second
# stream-adapter VAD; a Fish-batch-STT deployment would use
# `lambda proc: default_prewarm(proc, stream_adapter_vad=True)` instead. Demos that
# need extra prewarm work should wrap this: call default_prewarm(proc) first, then
# stash whatever else on proc.userdata.
server.setup_fnc = default_prewarm


class SmokeAgent(Agent):
    """A trivial agent — just chats. No tools, no domain logic."""

    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are a friendly voice assistant for testing the "
                "voice-agent-core library. Keep replies brief — 1 to 2 "
                "sentences. If asked what you can do, just say you're a "
                "simple test agent for verifying the voice pipeline."
            )
        )


@server.rtc_session(agent_name="smoke")
async def entry(ctx: JobContext) -> None:
    # Short-circuit page-load warmup dispatches (room names "warmup-*").
    # Frontends fire these to wake a cold worker before the user clicks Talk;
    # see voice-agent-core README's worker-warmup pattern. Returning here
    # leaves the worker hot but skips the rest of the pipeline + side effects.
    if is_warmup_session(ctx):
        log.info("session.warmup", room=ctx.room.name)
        return

    pipeline = build_pipeline(
        settings,
        vad=ctx.proc.userdata["vad"],
        # Only prewarmed when default_prewarm(stream_adapter_vad=True) was used
        # (batch STT); None here on the Deepgram path, which build_pipeline ignores.
        stream_adapter_vad=ctx.proc.userdata.get("stream_adapter_vad"),
    )
    session_start = time.monotonic()

    # NB: voice_agent_core also exports `warm_tts(pipeline.tts)` for prewarming
    # the TTS WebSocket pool, but it currently provides no benefit with Fish
    # (the livekit-plugins-fishaudio client doesn't pool connections — every
    # synth opens a new socket). It also burns Fish billing on a junk synth.
    # See warm_tts docstring; the helper is kept for future providers.

    # build_session wraps turn_detection AND preemptive_generation inside
    # TurnHandlingOptions — the v1.5+ API. preemptive_generation is carried on
    # the pipeline from settings.preemptive_generation (the PREEMPTIVE_GENERATION
    # env var, default true), so flip it via env without touching this call. Pass
    # build_session(pipeline, preemptive_generation=...) to force a value, or a
    # full turn_handling=... kwarg to override the wrapper entirely.
    session = build_session(pipeline)

    async def notify_session_ended(close_event) -> None:
        """Fires when AgentSession closes (participant disconnect, error, etc.).
        One per session — concurrent users get independent notifications since
        each session runs in its own subprocess.

        We hook ``session.on("close")`` rather than ``ctx.add_shutdown_callback``
        because the latter only fires on full job shutdown, which in ``dev`` mode
        means Ctrl+C — by then the subprocess may exit forcefully before the
        notification can complete.
        """
        duration_s = round(time.monotonic() - session_start, 1)
        transcript = format_transcript(session.history)

        summary_text = "(no conversation content)"
        if transcript:
            try:
                summary_text = await summarize_transcript(pipeline.llm, transcript)
            except Exception as exc:
                log.warning("notify.summary_failed", error=repr(exc))
                summary_text = "(summary unavailable — LLM call failed)"

        # Slack field cap ~3000 chars; keep transcript visible but bounded.
        transcript_display = transcript[:_TRANSCRIPT_CHAR_CAP] + (
            "…" if len(transcript) > _TRANSCRIPT_CHAR_CAP else ""
        )

        await notifier.send(
            NotificationPayload(
                title="Smoke session ended",
                summary=summary_text,
                fields=[
                    NotificationField(label="Room", value=ctx.room.name),
                    NotificationField(label="Duration", value=f"{duration_s} s"),
                    NotificationField(label="Reason", value=close_event.reason.value),
                    NotificationField(
                        label="Transcript", value=transcript_display or "(empty)"
                    ),
                ],
                urgency="normal",
            )
        )

    # session.on is a sync emitter; schedule the async work on the running loop.
    session.on(
        "close",
        lambda close_event: asyncio.create_task(notify_session_ended(close_event)),
    )

    # default_room_options() returns RoomOptions with AI Coustics QUAIL_VF_S
    # mic-side noise cancellation. For demos that need different audio config,
    # construct RoomOptions inline instead.
    await session.start(
        agent=SmokeAgent(),
        room=ctx.room,
        room_options=default_room_options(),
    )
    await ctx.connect()

    # Opener. GREETING_MODE=say (default) speaks a canned line — TTS only, no LLM
    # round-trip on turn 0, so the first utterance lands in ~0.6-1s. GREETING_MODE=
    # generate has the LLM ideate the greeting (exercises the full pipeline, slower).
    if settings.greeting_mode == "say":
        await session.say(settings.greeting_text)
    else:
        await session.generate_reply(
            instructions="Greet the user warmly and ask what's on their mind."
        )


if __name__ == "__main__":
    cli.run_app(server)
