"""Tests for voice_agent_core.transcript."""

from __future__ import annotations

from collections.abc import AsyncIterator
from unittest.mock import MagicMock

from voice_agent_core.transcript import (
    DEFAULT_SUMMARY_INSTRUCTION,
    format_transcript,
    summarize_transcript,
)


def _msg(role: str, content) -> MagicMock:
    """Build a fake ChatMessage with type='message' for transcript tests."""
    m = MagicMock()
    m.type = "message"
    m.role = role
    m.content = content
    return m


def _history(items: list) -> MagicMock:
    h = MagicMock()
    h.items = items
    return h


class TestFormatTranscript:
    def test_empty_history(self) -> None:
        assert format_transcript(_history([])) == ""

    def test_user_and_agent_rendered(self) -> None:
        h = _history([_msg("user", "hi"), _msg("assistant", "hello there")])
        assert format_transcript(h) == "User: hi\nAgent: hello there"

    def test_skips_non_message_items(self) -> None:
        non_msg = MagicMock()
        non_msg.type = "tool_call"
        h = _history([non_msg, _msg("user", "hi")])
        assert format_transcript(h) == "User: hi"

    def test_skips_system_and_developer_roles(self) -> None:
        h = _history(
            [
                _msg("system", "be nice"),
                _msg("developer", "use json"),
                _msg("user", "hi"),
            ]
        )
        assert format_transcript(h) == "User: hi"

    def test_skips_whitespace_only_content(self) -> None:
        h = _history([_msg("user", "   "), _msg("assistant", "hi")])
        assert format_transcript(h) == "Agent: hi"

    def test_list_content_concatenates_str_parts_ignoring_others(self) -> None:
        # MultiModal content can mix str fragments with non-str refs
        h = _history([_msg("user", ["hel", "lo", 42, None])])
        assert format_transcript(h) == "User: hel lo"


def _fake_llm(response_chunks: list[str]) -> MagicMock:
    """Build a fake LLM whose .chat(...).to_str_iterable() yields the given strings."""

    async def _async_iter() -> AsyncIterator[str]:
        for c in response_chunks:
            yield c

    stream = MagicMock()
    stream.to_str_iterable = MagicMock(return_value=_async_iter())

    llm = MagicMock()
    llm.chat = MagicMock(return_value=stream)
    return llm


class TestSummarizeTranscript:
    async def test_collects_stream_into_string(self) -> None:
        llm = _fake_llm(["This ", "is ", "a ", "summary."])
        result = await summarize_transcript(llm, "User: hi\nAgent: hello")
        assert result == "This is a summary."

    async def test_strips_surrounding_whitespace(self) -> None:
        llm = _fake_llm(["  ok  "])
        assert await summarize_transcript(llm, "x") == "ok"

    async def test_empty_stream_returns_fallback(self) -> None:
        llm = _fake_llm([])
        assert await summarize_transcript(llm, "x") == "(empty summary)"

    async def test_default_instruction_sent_to_llm(self) -> None:
        captured: list = []
        llm = _fake_llm(["ok"])
        llm.chat = MagicMock(
            side_effect=lambda *, chat_ctx, **k: (
                captured.append(chat_ctx),
                llm.chat.return_value,
            )[1]
        )
        # Re-stub return_value after MagicMock reset above
        stream = MagicMock()

        async def _gen():
            yield "ok"

        stream.to_str_iterable = MagicMock(return_value=_gen())
        llm.chat.return_value = stream

        await summarize_transcript(llm, "User: hi")
        # The single message we added should contain the default instruction
        sent_content = str(captured[0].items[0].content)
        assert DEFAULT_SUMMARY_INSTRUCTION in sent_content
        assert "User: hi" in sent_content

    async def test_custom_instruction_overrides_default(self) -> None:
        captured: list = []
        llm = MagicMock()
        stream = MagicMock()

        async def _gen():
            yield "ok"

        stream.to_str_iterable = MagicMock(return_value=_gen())

        def _chat(*, chat_ctx, **k):
            captured.append(chat_ctx)
            return stream

        llm.chat = MagicMock(side_effect=_chat)

        await summarize_transcript(
            llm, "User: hi", instruction="Summarize as JSON only:"
        )
        sent_content = str(captured[0].items[0].content)
        assert "Summarize as JSON only:" in sent_content
        assert DEFAULT_SUMMARY_INSTRUCTION not in sent_content


class TestDefaults:
    def test_default_instruction_is_nonempty(self) -> None:
        assert isinstance(DEFAULT_SUMMARY_INSTRUCTION, str)
        assert len(DEFAULT_SUMMARY_INSTRUCTION) > 20
