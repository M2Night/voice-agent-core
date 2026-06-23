"""Tests for the idle re-engagement watcher.

We drive the watcher with a fake AgentSession that records ``on``/``off``
registrations and lets us fire ``user_state_changed`` events synchronously (as
LiveKit does from its event loop). ``retry_interval`` is tiny so the nudge loop
runs fast; assertions poll for the expected attempt count rather than sleeping a
fixed time, to stay non-flaky.
"""

from __future__ import annotations

import asyncio
from types import SimpleNamespace

import pytest

from voice_agent_core.idle import IdleWatcher, attach_idle_watcher


class _FakeSession:
    """Minimal AgentSession double: event registry + mutable user_state."""

    def __init__(self, user_state: str = "listening") -> None:
        self.user_state = user_state
        self._handlers: dict[str, list] = {}

    def on(self, event: str, cb) -> None:
        self._handlers.setdefault(event, []).append(cb)

    def off(self, event: str, cb) -> None:
        self._handlers.get(event, []).remove(cb)

    def handler_count(self, event: str) -> int:
        return len(self._handlers.get(event, []))

    def emit_state(self, new_state: str) -> None:
        """Flip user_state and fire the event, like the real session does."""
        ev = SimpleNamespace(new_state=new_state, old_state="listening")
        self.user_state = new_state
        for cb in list(self._handlers.get("user_state_changed", [])):
            cb(ev)

    def emit_close(self) -> None:
        """Fire a CloseEvent. LiveKit does NOT emit a final user_state_changed on
        close, so the watcher must react to this directly."""
        ev = SimpleNamespace(type="close", error=None, reason="user_initiated")
        for cb in list(self._handlers.get("close", [])):
            cb(ev)


async def _wait_for(predicate, *, timeout: float = 1.0) -> None:
    """Poll ``predicate`` until true or timeout (keeps timing assertions robust)."""
    elapsed = 0.0
    while not predicate() and elapsed < timeout:
        await asyncio.sleep(0.005)
        elapsed += 0.005
    assert predicate(), "condition not met within timeout"


class TestAttach:
    def test_registers_handler(self) -> None:
        session = _FakeSession()
        attach_idle_watcher(session, lambda attempt: None)
        assert session.handler_count("user_state_changed") == 1

    def test_rejects_bad_args(self) -> None:
        session = _FakeSession()
        with pytest.raises(ValueError, match="max_consecutive"):
            IdleWatcher(session, lambda a: None, max_consecutive=0)
        with pytest.raises(ValueError, match="retry_interval"):
            IdleWatcher(session, lambda a: None, retry_interval=0)


class TestNudging:
    async def test_nudges_on_away(self) -> None:
        session = _FakeSession()
        attempts: list[int] = []

        async def on_idle(attempt: int) -> None:
            attempts.append(attempt)

        attach_idle_watcher(session, on_idle, max_consecutive=1, retry_interval=0.01)
        session.emit_state("away")
        await _wait_for(lambda: attempts == [1])

    async def test_repeats_up_to_max_then_stops(self) -> None:
        session = _FakeSession()
        attempts: list[int] = []

        async def on_idle(attempt: int) -> None:
            attempts.append(attempt)  # user stays "away" the whole time

        attach_idle_watcher(session, on_idle, max_consecutive=3, retry_interval=0.01)
        session.emit_state("away")
        await _wait_for(lambda: attempts == [1, 2, 3])
        # Capped: no further nudges even after more time passes.
        await asyncio.sleep(0.05)
        assert attempts == [1, 2, 3]

    async def test_user_return_cancels_nudging(self) -> None:
        session = _FakeSession()
        attempts: list[int] = []

        async def on_idle(attempt: int) -> None:
            attempts.append(attempt)

        attach_idle_watcher(session, on_idle, max_consecutive=5, retry_interval=0.02)
        session.emit_state("away")
        await _wait_for(lambda: len(attempts) >= 1)
        # User speaks again → in-flight loop is cancelled.
        session.emit_state("speaking")
        count_at_return = len(attempts)
        await asyncio.sleep(0.08)
        assert len(attempts) == count_at_return
        assert count_at_return < 5

    async def test_rearms_after_user_returns(self) -> None:
        session = _FakeSession()
        episodes: list[int] = []

        async def on_idle(attempt: int) -> None:
            episodes.append(attempt)

        attach_idle_watcher(session, on_idle, max_consecutive=1, retry_interval=0.01)
        session.emit_state("away")
        await _wait_for(lambda: episodes == [1])
        session.emit_state("speaking")  # re-engaged → counter resets
        await asyncio.sleep(0.02)
        session.emit_state("away")  # fresh idle episode nudges again from 1
        await _wait_for(lambda: episodes == [1, 1])

    async def test_non_away_states_do_not_nudge(self) -> None:
        session = _FakeSession()
        attempts: list[int] = []

        attach_idle_watcher(
            session, lambda a: attempts.append(a), retry_interval=0.01
        )
        session.emit_state("speaking")
        session.emit_state("listening")
        await asyncio.sleep(0.05)
        assert attempts == []

    async def test_nudge_failure_is_swallowed_and_loop_continues(self) -> None:
        session = _FakeSession()
        attempts: list[int] = []

        async def on_idle(attempt: int) -> None:
            attempts.append(attempt)
            raise RuntimeError("generate_reply blew up")

        attach_idle_watcher(session, on_idle, max_consecutive=2, retry_interval=0.01)
        session.emit_state("away")
        # Both attempts still run despite the callback raising each time.
        await _wait_for(lambda: attempts == [1, 2])


class TestDetach:
    async def test_detach_unregisters_and_cancels(self) -> None:
        session = _FakeSession()
        attempts: list[int] = []

        async def on_idle(attempt: int) -> None:
            attempts.append(attempt)

        watcher = attach_idle_watcher(
            session, on_idle, max_consecutive=5, retry_interval=0.02
        )
        session.emit_state("away")
        await _wait_for(lambda: len(attempts) >= 1)
        watcher.detach()
        assert session.handler_count("user_state_changed") == 0
        assert session.handler_count("close") == 0
        count_at_detach = len(attempts)
        await asyncio.sleep(0.08)
        assert len(attempts) == count_at_detach

    def test_detach_is_idempotent(self) -> None:
        session = _FakeSession()
        watcher = attach_idle_watcher(session, lambda a: None)
        watcher.detach()
        watcher.detach()  # must not raise (handlers already removed)
        assert session.handler_count("user_state_changed") == 0

    async def test_session_close_cancels_and_unregisters(self) -> None:
        # LiveKit emits "close" (not a final user_state_changed) when the session
        # ends mid-episode — the watcher must stop nudging and release handlers.
        session = _FakeSession()
        attempts: list[int] = []

        async def on_idle(attempt: int) -> None:
            attempts.append(attempt)

        attach_idle_watcher(session, on_idle, max_consecutive=5, retry_interval=0.02)
        session.emit_state("away")
        await _wait_for(lambda: len(attempts) >= 1)

        session.emit_close()
        assert session.handler_count("user_state_changed") == 0
        assert session.handler_count("close") == 0
        count_at_close = len(attempts)
        await asyncio.sleep(0.08)
        assert len(attempts) == count_at_close  # loop cancelled, no more nudges
