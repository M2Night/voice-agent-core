"""Pytest fixtures for voice_agent_core tests."""

from __future__ import annotations

import pytest


@pytest.fixture(scope="session", autouse=True)
def _shutdown_otel_at_session_end():
    """Cleanly shut down OTEL meter provider when the test session ends.

    Without this, the periodic console exporter background thread tries to flush
    after pytest has already closed stdout, producing harmless but ugly tracebacks.
    """
    yield
    from voice_agent_core.observability import shutdown_observability

    shutdown_observability(timeout_ms=1000)
