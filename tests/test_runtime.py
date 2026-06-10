"""Tests for the runtime helpers (build_session, default_prewarm, etc.)."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from voice_agent_core.runtime import (
    build_session,
    default_prewarm,
    default_room_options,
    is_warmup_session,
    warm_tts,
)


class TestIsWarmupSession:
    """Boundary cases for the room-name prefix check."""

    def _ctx(self, room_name: str) -> SimpleNamespace:
        return SimpleNamespace(room=SimpleNamespace(name=room_name))

    def test_warmup_prefix_returns_true(self) -> None:
        assert is_warmup_session(self._ctx("warmup-abc12345")) is True

    def test_real_room_returns_false(self) -> None:
        assert is_warmup_session(self._ctx("lead-qual-abc12345")) is False

    def test_empty_room_name_returns_false(self) -> None:
        assert is_warmup_session(self._ctx("")) is False

    def test_substring_match_not_prefix_returns_false(self) -> None:
        # "warmup-" appearing in the middle must NOT trigger
        assert is_warmup_session(self._ctx("real-warmup-1234")) is False

    def test_just_the_prefix_with_no_id_still_matches(self) -> None:
        # Defensive: even a bare "warmup-" counts as warmup (defensive default)
        assert is_warmup_session(self._ctx("warmup-")) is True


class TestDefaultPrewarm:
    """default_prewarm should populate proc.userdata['vad'] using silero."""

    def test_loads_vad_into_userdata(self) -> None:
        proc = SimpleNamespace(userdata={})
        sentinel = object()
        with patch("voice_agent_core.runtime.silero.VAD.load", return_value=sentinel) as load:
            default_prewarm(proc)
            load.assert_called_once_with()
        assert proc.userdata["vad"] is sentinel


class TestDefaultRoomOptions:
    """default_room_options should hand back RoomOptions with AI Coustics on input."""

    def test_returns_room_options_with_noise_cancellation(self) -> None:
        opts = default_room_options()
        # Don't assert on internal attribute names of the AI Coustics object —
        # just that audio_input has SOMETHING set as noise_cancellation, since
        # the upstream type is the contract we depend on.
        assert opts.audio_input is not None
        assert opts.audio_input.noise_cancellation is not None


class TestBuildSession:
    """build_session should pass the pipeline's components into AgentSession
    with sane defaults (preemptive_generation=True, turn_handling wrapper)."""

    def _fake_pipeline(self) -> SimpleNamespace:
        return SimpleNamespace(
            stt=MagicMock(name="stt"),
            tts=MagicMock(name="tts"),
            llm=MagicMock(name="llm"),
            vad=MagicMock(name="vad"),
            turn_detection=MagicMock(name="turn_detection"),
        )

    def test_passes_pipeline_components_through(self) -> None:
        pipeline = self._fake_pipeline()
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline)
        kwargs = agent_session_cls.call_args.kwargs
        assert kwargs["stt"] is pipeline.stt
        assert kwargs["tts"] is pipeline.tts
        assert kwargs["llm"] is pipeline.llm
        assert kwargs["vad"] is pipeline.vad

    def test_wraps_turn_detection_in_turn_handling_options(self) -> None:
        pipeline = self._fake_pipeline()
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline)
        kwargs = agent_session_cls.call_args.kwargs
        # TurnHandlingOptions is a TypedDict (dict subclass) in livekit-agents 1.5.x,
        # so use dict access. The contract is just "turn_detection lands here."
        assert kwargs["turn_handling"]["turn_detection"] is pipeline.turn_detection

    def test_preemptive_generation_defaults_true(self) -> None:
        pipeline = self._fake_pipeline()
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline)
        assert agent_session_cls.call_args.kwargs["preemptive_generation"] is True

    def test_preemptive_generation_can_be_disabled(self) -> None:
        pipeline = self._fake_pipeline()
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline, preemptive_generation=False)
        assert agent_session_cls.call_args.kwargs["preemptive_generation"] is False

    def test_extra_kwargs_passed_through(self) -> None:
        pipeline = self._fake_pipeline()
        with patch("voice_agent_core.runtime.AgentSession") as agent_session_cls:
            build_session(pipeline, allow_interruptions=False)
        assert agent_session_cls.call_args.kwargs["allow_interruptions"] is False


class TestWarmTts:
    """warm_tts should drive a tiny streaming synth through the streaming path
    (not the synthesize HTTP path) so the connection pool that session.say
    uses actually gets warmed."""

    def _stream(self) -> MagicMock:
        """Fake the async-iterable stream returned by tts.stream()."""
        stream = MagicMock(name="stream")
        stream.push_text = MagicMock()
        stream.end_input = MagicMock()
        # async iteration support — yields nothing, terminates immediately
        stream.__aiter__ = lambda self: self
        stream.__anext__ = AsyncMock(side_effect=StopAsyncIteration())
        stream.aclose = AsyncMock()
        return stream

    @pytest.mark.asyncio
    async def test_calls_streaming_path(self) -> None:
        stream = self._stream()
        tts = MagicMock()
        tts.stream = MagicMock(return_value=stream)
        await warm_tts(tts)
        tts.stream.assert_called_once_with()
        stream.push_text.assert_called_once_with(".")
        stream.end_input.assert_called_once_with()
        stream.aclose.assert_awaited_once()

    @pytest.mark.asyncio
    async def test_custom_text(self) -> None:
        stream = self._stream()
        tts = MagicMock()
        tts.stream = MagicMock(return_value=stream)
        await warm_tts(tts, text="hi")
        stream.push_text.assert_called_once_with("hi")

    @pytest.mark.asyncio
    async def test_swallows_provider_failure(self) -> None:
        """If TTS provider is unreachable, warmup must not propagate the error
        — the real synth path is what surfaces failures to the user."""
        tts = MagicMock()
        tts.stream = MagicMock(side_effect=RuntimeError("fish unreachable"))
        # Should NOT raise
        await warm_tts(tts)

    @pytest.mark.asyncio
    async def test_closes_stream_even_on_iteration_error(self) -> None:
        """aclose runs in finally so a flaky iteration still releases the
        connection back to the pool."""
        stream = self._stream()
        stream.__anext__ = AsyncMock(side_effect=RuntimeError("mid-stream boom"))
        tts = MagicMock()
        tts.stream = MagicMock(return_value=stream)
        await warm_tts(tts)  # caught at the outer try
        stream.aclose.assert_awaited_once()
