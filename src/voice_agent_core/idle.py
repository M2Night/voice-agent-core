"""Idle re-engagement: nudge a user who has gone quiet, without nagging.

LiveKit's ``AgentSession`` already *detects* idleness: with ``user_away_timeout``
(default 15s) it flips ``user_state`` to ``"away"`` once the user and agent have
both been silent that long, emitting a ``user_state_changed`` event. That 15s window
is the debounce — a brief pause never trips it.

This module turns that signal into an opt-in *action*: call a re-engagement callback
when the user goes away, optionally repeat it every ``retry_interval`` seconds while
they stay away, and stop after ``max_consecutive`` unanswered nudges so we never spam
someone who has actually left. Any return to speech cancels in-flight nudging and
re-arms the watcher, so a later silence is treated fresh.

Usage::

    from voice_agent_core import attach_idle_watcher

    async def _nudge(attempt: int) -> None:
        # attempt is 1-based; escalate or close the call on the last one.
        await session.generate_reply(
            instructions="The user has gone quiet — gently check if they're still there."
        )

    attach_idle_watcher(session, on_idle=_nudge, max_consecutive=2)

Tune the entry debounce via ``AgentSession(user_away_timeout=...)`` (or disable
``away`` detection entirely with ``user_away_timeout=None``, which makes this watcher
inert). The cap-and-reset logic here is the portable robustness pattern; the 15s
"are they really idle?" judgement is delegated to the framework that already owns it.
"""

from __future__ import annotations

import asyncio
from collections.abc import Awaitable
from typing import TYPE_CHECKING, Protocol

from voice_agent_core.observability import get_logger

if TYPE_CHECKING:
    from livekit.agents.voice import AgentSession
    from livekit.agents.voice.events import CloseEvent, UserStateChangedEvent

log = get_logger(__name__)


class OnIdle(Protocol):
    """Async re-engagement callback. ``attempt`` is the 1-based nudge count.

    Must return an awaitable — ``_nudge_loop`` always ``await``\\ s the result. A
    plain (non-async) callback would raise ``TypeError`` at await time, which the
    loop logs and treats as a failed nudge.
    """

    def __call__(self, attempt: int) -> Awaitable[object]: ...


class IdleWatcher:
    """Re-engage a silent user via ``AgentSession``'s ``away`` state, capped.

    Prefer the :func:`attach_idle_watcher` helper, which constructs this and wires
    the event handler in one call. Hold a reference if you need to :meth:`detach`
    later (e.g. mid-session policy change); the watcher otherwise lives as long as
    the session emits events to it.
    """

    def __init__(
        self,
        session: AgentSession,
        on_idle: OnIdle,
        *,
        max_consecutive: int = 2,
        retry_interval: float = 10.0,
    ) -> None:
        if max_consecutive < 1:
            raise ValueError("max_consecutive must be >= 1")
        if retry_interval <= 0:
            raise ValueError("retry_interval must be > 0")
        self._session = session
        self._on_idle = on_idle
        self._max_consecutive = max_consecutive
        self._retry_interval = retry_interval
        self._task: asyncio.Task[None] | None = None
        self._detached = False

    def attach(self) -> IdleWatcher:
        """Start listening for state changes. Returns self for chaining.

        Also listens for ``close``: when the session ends mid-episode LiveKit does
        *not* emit a final ``user_state_changed``, so without this a nudge loop
        sleeping on ``retry_interval`` would linger past session close.
        """
        self._session.on("user_state_changed", self._on_state_changed)
        self._session.on("close", self._on_close)
        return self

    def detach(self) -> None:
        """Stop listening and cancel any in-flight nudging. Idempotent."""
        if self._detached:
            return
        self._detached = True
        self._session.off("user_state_changed", self._on_state_changed)
        self._session.off("close", self._on_close)
        self._cancel_task()

    def _on_close(self, _ev: CloseEvent) -> None:
        # Session is going away — release handlers + cancel the loop so neither the
        # in-flight coroutine nor this watcher keeps the session object alive.
        self.detach()

    def _cancel_task(self) -> None:
        if self._task is not None and not self._task.done():
            self._task.cancel()
        self._task = None

    def _on_state_changed(self, ev: UserStateChangedEvent) -> None:
        if ev.new_state == "away":
            # Arm one nudge loop per idle episode; ignore re-entrant "away" events.
            if self._task is None or self._task.done():
                self._task = asyncio.create_task(
                    self._nudge_loop(), name="idle_watcher_nudge"
                )
        else:
            # Back to "speaking"/"listening" — the user re-engaged; stop nudging and
            # re-arm so a future idle episode starts its count from scratch.
            self._cancel_task()

    async def _nudge_loop(self) -> None:
        for attempt in range(1, self._max_consecutive + 1):
            # State can flip between the sleep and here; only nudge if still away.
            if self._session.user_state != "away":
                return
            log.info("idle.nudge", attempt=attempt, max=self._max_consecutive)
            try:
                await self._on_idle(attempt)
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                log.warning("idle.nudge_failed", attempt=attempt, error=repr(exc))
            await asyncio.sleep(self._retry_interval)
        log.info("idle.exhausted", attempts=self._max_consecutive)


def attach_idle_watcher(
    session: AgentSession,
    on_idle: OnIdle,
    *,
    max_consecutive: int = 2,
    retry_interval: float = 10.0,
) -> IdleWatcher:
    """Attach an :class:`IdleWatcher` to ``session`` and start it.

    ``on_idle(attempt)`` is awaited when the user goes ``away`` and again every
    ``retry_interval`` seconds while they stay away, up to ``max_consecutive`` times.
    Returns the watcher so callers can :meth:`IdleWatcher.detach` it later.
    """
    return IdleWatcher(
        session,
        on_idle,
        max_consecutive=max_consecutive,
        retry_interval=retry_interval,
    ).attach()


__all__ = ["IdleWatcher", "OnIdle", "attach_idle_watcher"]
