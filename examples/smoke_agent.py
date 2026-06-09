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
    build_pipeline,
    load_env_walking_up,
    setup_observability,
)

# Start the .env search from this script's directory (examples/) so it works
# whether you run from repo root or from examples/. cwd-based defaults wouldn't
# find examples/.env when the script is invoked as `python examples/smoke_agent.py`.
load_env_walking_up(start=Path(__file__).parent)
settings = BaseAgentSettings()
setup_observability(settings, service_name="voice-agent-core-smoke")

server = AgentServer()


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

    session = AgentSession(
        stt=pipeline.stt,
        tts=pipeline.tts,
        llm=pipeline.llm,
        vad=pipeline.vad,
        turn_handling=TurnHandlingOptions(turn_detection=pipeline.turn_detection),
    )

    await session.start(agent=SmokeAgent(), room=ctx.room)
    await ctx.connect()
    await session.generate_reply(
        instructions="Greet the user warmly and ask what's on their mind."
    )


if __name__ == "__main__":
    cli.run_app(server)
