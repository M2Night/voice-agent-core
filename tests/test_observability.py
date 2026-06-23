"""Tests for voice_agent_core.observability — smoke-level coverage.

Real OTEL backend testing isn't done here (you'd need a fixture exporter); we just
verify the setup functions wire correctly and return usable loggers/meters.
"""

from __future__ import annotations

import pytest

from voice_agent_core.config import BaseAgentSettings
from voice_agent_core.observability import (
    MetricNames,
    configure_logging,
    configure_metrics,
    get_logger,
    get_meter,
    setup_observability,
)


class TestLogging:
    def test_configure_and_log_json(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="INFO", log_format="json")
        log = get_logger("test")
        log.info("smoke_event", key="value", count=1)
        out = capsys.readouterr().out
        assert "smoke_event" in out
        assert '"key"' in out and '"value"' in out

    def test_log_below_level_suppressed(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="WARNING", log_format="json")
        log = get_logger("test")
        log.debug("should_not_appear")
        log.info("should_not_appear_either")
        out = capsys.readouterr().out
        assert "should_not_appear" not in out

    def test_console_format(self, capsys: pytest.CaptureFixture[str]) -> None:
        configure_logging(level="INFO", log_format="console")
        log = get_logger("test")
        log.info("console_event")
        out = capsys.readouterr().out
        assert "console_event" in out


class TestMetrics:
    def test_configure_console(self) -> None:
        configure_metrics(service_name="test-service", exporter="console")
        meter = get_meter("test")
        hist = meter.create_histogram("test.metric", unit="ms")
        # Doesn't raise — recording is no-op-friendly
        hist.record(42)

    def test_configure_none(self) -> None:
        configure_metrics(service_name="test-service", exporter="none")
        meter = get_meter("test")
        counter = meter.create_counter("test.counter")
        counter.add(1)

    def test_configure_unknown_raises(self) -> None:
        with pytest.raises(ValueError, match="Unknown OTEL exporter"):
            configure_metrics(service_name="t", exporter="bogus")  # type: ignore[arg-type]

    def test_metric_names_are_strings(self) -> None:
        # Sanity check that constants exist and look right
        assert MetricNames.FISH_TTS_TTFB_MS == "fish_tts.ttfb_ms"
        assert MetricNames.LLM_LATENCY_MS == "llm.latency_ms"
        assert MetricNames.SESSION_COUNT == "session.count"


class TestSetupObservability:
    def test_one_shot_setup(
        self, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
    ) -> None:
        monkeypatch.setenv("LOG_LEVEL", "INFO")
        monkeypatch.setenv("LOG_FORMAT", "json")
        monkeypatch.setenv("OTEL_METRICS_EXPORTER", "none")
        settings = BaseAgentSettings()

        setup_observability(settings, service_name="test-suite")

        log = get_logger("test")
        log.info("post_setup_event")
        out = capsys.readouterr().out
        assert "post_setup_event" in out
