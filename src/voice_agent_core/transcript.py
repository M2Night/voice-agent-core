"""Transcript utilities: extract a session's chat history as text + LLM summary.

Voice agents commonly need to (a) flatten an :class:`AgentSession`'s chat history
into plain text and (b) ask an LLM to produce a short summary at session end
(for notifications, post-call analysis, log enrichment, etc.). These two
helpers consolidate the pattern so each consumer app doesn't reimplement them.

Usage::

    from voice_agent_core import format_transcript, summarize_transcript

    # inside a session-end callback:
    transcript = format_transcript(session.history)
    summary = await summarize_transcript(pipeline.llm, transcript)
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from livekit.agents.llm import LLM, ChatContext


DEFAULT_SUMMARY_INSTRUCTION = (
    "Summarize the following voice conversation in ONE short sentence. "
    "Focus on what was discussed, not on greetings. If the exchange is "
    "trivial, just say 'A short test exchange.'"
)


def format_transcript(history: ChatContext) -> str:
    """Render a session ``history`` as plain ``User: ... / Agent: ...`` text.

    Skips items that aren't message-type, messages with roles other than
    ``user`` / ``assistant`` (e.g. system, developer), and messages whose text
    content is empty or whitespace-only. Returns ``""`` if the history has no
    user/assistant exchanges.
    """
    lines: list[str] = []
    for item in history.items:
        if getattr(item, "type", None) != "message":
            continue
        if item.role not in ("user", "assistant"):
            continue
        # ``content`` is typically ``list[ChatContent]`` — usually str fragments,
        # but can contain non-str (image refs, tool refs, etc.) which we ignore.
        if isinstance(item.content, str):
            text = item.content
        else:
            text = " ".join(c for c in item.content if isinstance(c, str))
        text = text.strip()
        if not text:
            continue
        prefix = "User" if item.role == "user" else "Agent"
        lines.append(f"{prefix}: {text}")
    return "\n".join(lines)


async def summarize_transcript(
    llm: LLM,
    transcript: str,
    *,
    instruction: str = DEFAULT_SUMMARY_INSTRUCTION,
) -> str:
    """Ask ``llm`` to summarize ``transcript`` per ``instruction``.

    Single non-streaming round-trip — collects the whole response into one
    string. Override ``instruction`` to bias the summary for your domain
    (e.g. ``"Summarize this lead-qualification conversation as: company, use
    case, monthly volume, decision-maker status, recommended plan."``).

    Returns the model's reply with surrounding whitespace stripped, or the
    literal ``"(empty summary)"`` if the model produced no output.
    """
    from livekit.agents.llm import ChatContext

    ctx = ChatContext.empty()
    ctx.add_message(role="user", content=f"{instruction}\n\n{transcript}")
    parts: list[str] = []
    async for chunk in llm.chat(chat_ctx=ctx).to_str_iterable():
        parts.append(chunk)
    return "".join(parts).strip() or "(empty summary)"


__all__ = [
    "DEFAULT_SUMMARY_INSTRUCTION",
    "format_transcript",
    "summarize_transcript",
]
