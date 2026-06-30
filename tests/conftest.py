"""Pytest fixtures for voice_agent_core tests."""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _isolate_http_proxy_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep host proxy settings from changing HTTP client construction in tests."""
    for name in (
        "ALL_PROXY",
        "all_proxy",
        "HTTP_PROXY",
        "http_proxy",
        "HTTPS_PROXY",
        "https_proxy",
    ):
        monkeypatch.delenv(name, raising=False)


@pytest.fixture(scope="session", autouse=True)
def _shutdown_otel_at_session_end():
    """Cleanly shut down OTEL meter provider when the test session ends.

    Without this, the periodic console exporter background thread tries to flush
    after pytest has already closed stdout, producing harmless but ugly tracebacks.
    """
    yield
    from voice_agent_core.observability import shutdown_observability

    shutdown_observability(timeout_ms=1000)
