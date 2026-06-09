"""Smoke agent — minimum runnable agent built on voice-agent-core.

No business logic; just demonstrates wiring: settings → pipeline → AgentSession.
Use it to verify voice-agent-core works end-to-end in your environment, and as
a reference template for building your own agents.

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

from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    JobProcess,
    TurnHandlingOptions,
    cli,
)
from livekit.plugins import silero

from voice_agent_core import (
    BaseAgentSettings,
    NotificationField,
    NotificationPayload,
    SlackNotifier,
    build_pipeline,
    format_transcript,
    get_logger,
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


# Start the .env search from this script's directory (examples/) so it works
# whether you run from repo root or from examples/. cwd-based defaults wouldn't
# find examples/.env when the script is invoked as `python examples/smoke_agent.py`.
load_env_walking_up(start=Path(__file__).parent)
settings = SmokeSettings()
setup_observability(settings, service_name="voice-agent-core-smoke")

server = AgentServer()
# Module-level notifier — each session subprocess gets its own copy (forked from
# the worker). Stateless: sharing initial config across forks is safe.
# Empty webhook_url triggers SlackNotifier's dev-log mode (no HTTP call).
notifier = SlackNotifier(webhook_url=settings.slack_webhook_url)


def prewarm(proc: JobProcess) -> None:
    """Load silero VAD once per process; reused across all sessions."""
    proc.userdata["vad"] = silero.VAD.load()


server.setup_fnc = prewarm


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
    pipeline = build_pipeline(settings, vad=ctx.proc.userdata["vad"])
    session_start = time.monotonic()

    session = AgentSession(
        stt=pipeline.stt,
        tts=pipeline.tts,
        llm=pipeline.llm,
        vad=pipeline.vad,
        turn_handling=TurnHandlingOptions(turn_detection=pipeline.turn_detection),
    )

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

    await session.start(agent=SmokeAgent(), room=ctx.room)
    await ctx.connect()
    await session.generate_reply(
        instructions="Greet the user warmly and ask what's on their mind."
    )


if __name__ == "__main__":
    cli.run_app(server)
