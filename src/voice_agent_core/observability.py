"""Observability: structured logging + OpenTelemetry metrics.

Design:

- **Logging** via ``structlog`` — JSON output for production (machine-readable, ships
  cleanly to any log aggregator); console renderer for local TTY dev. Same call sites,
  different rendering.
- **Metrics** via ``opentelemetry`` — emits to stdout by default (``ConsoleMetricExporter``)
  so they're visible during local dev. Swap to OTLP (Honeycomb, Datadog, Grafana, etc.)
  via one env var without touching call sites.

Call :func:`setup_observability` once at process startup. Use :func:`get_logger` and
:func:`get_meter` from anywhere afterward.
"""

from __future__ import annotations

import contextlib
import logging
import sys
from typing import Any

import structlog
from opentelemetry import metrics
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import (
    ConsoleMetricExporter,
    PeriodicExportingMetricReader,
)
from opentelemetry.sdk.resources import Resource

from voice_agent_core.config import (
    BaseAgentSettings,
    LogFormat,
    LogLevel,
    OTelExporter,
)


def configure_logging(level: LogLevel = "INFO", log_format: LogFormat = "json") -> None:
    """Configure structlog. Idempotent — safe to call multiple times.

    With ``log_format='json'`` each log line is a single JSON object suitable for any
    log aggregator; timestamps are ISO 8601 UTC. With ``log_format='console'`` lines
    are colorized and human-readable for local development; timestamps are
    ``HH:MM:SS.us`` local time so they align visually with LiveKit's own
    stdlib-logger formatter when both appear in the same terminal.

    Note: this configures *our* structlog only. LiveKit's stdlib loggers
    (``livekit.agents``, ``livekit_api::*`` Rust SDK) are independent and follow
    their own conventions — in particular, ``cli dev`` defaults to DEBUG.
    """
    log_level = getattr(logging, level.upper(), logging.INFO)

    # Short local time for console (visually aligns with LiveKit's own logger);
    # ISO 8601 UTC for JSON (machine-friendly for log aggregators).
    if log_format == "json":
        timestamper = structlog.processors.TimeStamper(fmt="iso", utc=True)
    else:
        timestamper = structlog.processors.TimeStamper(fmt="%H:%M:%S.%f", utc=False)

    shared_processors: list[Any] = [
        structlog.contextvars.merge_contextvars,
        structlog.processors.add_log_level,
        timestamper,
        structlog.processors.StackInfoRenderer(),
        structlog.dev.set_exc_info,
    ]

    renderer: Any
    if log_format == "json":
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=sys.stdout.isatty())

    structlog.configure(
        processors=[*shared_processors, renderer],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Get a structlog logger.

    Usage::

        log = get_logger(__name__)
        log.info("session_started", session_id="abc", lead_company="acme")
    """
    return structlog.get_logger(name)


def configure_metrics(
    service_name: str,
    exporter: OTelExporter = "console",
    export_interval_ms: int = 60_000,
) -> None:
    """Configure OpenTelemetry metrics.

    With ``exporter='console'``, metrics are printed to stdout every ``export_interval_ms``
    — useful for local dev to visually verify instrumentation. With ``exporter='none'``,
    metric recording becomes a no-op (no exporter is wired).

    Production OTLP-backed exporters (Honeycomb, Datadog, Grafana Cloud) aren't wired
    here yet; add an ``"otlp"`` branch when needed.
    """
    resource = Resource.create({"service.name": service_name})

    if exporter == "none":
        provider = MeterProvider(resource=resource)
    elif exporter == "console":
        reader = PeriodicExportingMetricReader(
            ConsoleMetricExporter(),
            export_interval_millis=export_interval_ms,
        )
        provider = MeterProvider(resource=resource, metric_readers=[reader])
    else:
        raise ValueError(f"Unknown OTEL exporter: {exporter!r}")

    metrics.set_meter_provider(provider)


def get_meter(name: str) -> metrics.Meter:
    """Get an OTEL meter for emitting metrics.

    Usage::

        meter = get_meter(__name__)
        ttfb = meter.create_histogram(
            MetricNames.FISH_TTS_TTFB_MS,
            unit="ms",
            description="Fish TTS time-to-first-byte",
        )
        ttfb.record(150)
    """
    return metrics.get_meter(name)


def shutdown_observability(timeout_ms: int = 5_000) -> None:
    """Flush and shut down OTEL exporters. Call once before process exit.

    Production-grade apps should call this from their signal handlers (SIGTERM/SIGINT)
    to avoid losing metric batches sitting in the export queue.
    """
    provider = metrics.get_meter_provider()
    if hasattr(provider, "shutdown"):
        # Best-effort: never crash on exit if the exporter is already torn down.
        with contextlib.suppress(Exception):
            provider.shutdown(timeout_millis=timeout_ms)


def setup_observability(settings: BaseAgentSettings, service_name: str) -> None:
    """One-shot observability setup driven by a settings object.

    Apps typically call this once at startup before anything else::

        from voice_agent_core import BaseAgentSettings, setup_observability

        settings = MyAppSettings()
        setup_observability(settings, service_name="lead-qualification")
    """
    configure_logging(level=settings.log_level, log_format=settings.log_format)
    configure_metrics(
        service_name=service_name,
        exporter=settings.otel_metrics_exporter,
    )


class MetricNames:
    """Canonical metric names used across voice-agent-core.

    Keep these consistent so dashboards/alerts in a real OTEL backend don't break
    when different consumer applications add instrumentation independently.
    """

    # --- Fish Audio ---
    FISH_TTS_TTFB_MS = "fish_tts.ttfb_ms"
    """Fish TTS time-to-first-byte (network ack)."""
    FISH_TTS_TTFT_MS = "fish_tts.ttft_ms"
    """Fish TTS time-to-first-audio-frame."""
    FISH_TTS_RTF = "fish_tts.rtf"
    """Fish TTS real-time-factor (synth_duration / audio_duration). <1.0 is realtime."""
    FISH_TTS_ERRORS = "fish_tts.errors"
    FISH_STT_LATENCY_MS = "fish_stt.latency_ms"
    FISH_STT_ERRORS = "fish_stt.errors"

    # --- LLM ---
    LLM_LATENCY_MS = "llm.latency_ms"
    LLM_TOKENS_INPUT = "llm.tokens.input"
    LLM_TOKENS_OUTPUT = "llm.tokens.output"
    LLM_ERRORS = "llm.errors"

    # --- Tool calls ---
    TOOL_CALL_LATENCY_MS = "tool.call.latency_ms"
    TOOL_CALL_ERRORS = "tool.call.errors"

    # --- Notifier ---
    NOTIFY_DISPATCH_LATENCY_MS = "notify.dispatch.latency_ms"
    NOTIFY_ERRORS = "notify.errors"
    NOTIFY_DLQ_SIZE = "notify.dlq.size"

    # --- Session ---
    SESSION_COUNT = "session.count"
    SESSION_DURATION_MS = "session.duration_ms"


__all__ = [
    "MetricNames",
    "configure_logging",
    "configure_metrics",
    "get_logger",
    "get_meter",
    "setup_observability",
    "shutdown_observability",
]
