# voice-agent-core

Shared Python library powering Fish Audio's voice agent by leveraging with LiveKit
## What it provides

- **Fish Audio STT/TTS adapters** with built-in metrics (TTFB, TTFT, RTF) and retry
- **LLM factory** — LiveKit Inference (default) + OpenRouter backends, switchable via env var
- **Pipeline factory** — assemble a complete LiveKit agent pipeline from one YAML config
- **AgentBase** — LiveKit `Agent` base class with structured event publishing and tool error boundaries
- **Notifier** abstraction — Slack (Block Kit) + Console fallback + generic Webhook
- **Observability** — `structlog` JSON logging + OpenTelemetry metrics (console exporter by default; swap to Honeycomb/Datadog/Prometheus via one env var)

## Install

For local development from a sibling clone:

```bash
uv add "voice-agent-core @ file://$(pwd)/../voice-agent-core"
```

For production / published consumers, pin to a tagged version:

```bash
uv add "voice-agent-core @ git+https://github.com/M2Night/voice-agent-core.git@v0.1.0"
```

## Status

Pre-1.0. Breaking changes possible until v1.0.0. Pin to a specific git tag in production.

## Used by

- [fish-voice-demos](https://github.com/M2Night/fish-voice-demos) — 5 production voice agents built on this library
