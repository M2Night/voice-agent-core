"""Slack webhook notifier with retry and dev log fallback.

When ``webhook_url`` is configured, notifications POST to a Slack incoming
webhook with Block Kit formatting (emoji + urgency mention + structured fields +
optional "View details" button). When ``webhook_url`` is empty (dev mode), the
notifier logs the payload structurally instead of making an HTTP call — local
development requires no Slack workspace.

Transient errors (network + 5xx) retry with exponential backoff via tenacity.
Permanent errors (4xx) abort immediately and propagate to the caller.

Usage::

    from voice_agent_core import (
        NotificationField, NotificationPayload, SlackNotifier,
    )

    notifier = SlackNotifier(webhook_url=settings.slack_webhook_url)
    await notifier.send(NotificationPayload(
        title="New lead from acme.com",
        summary="Use case: realtime voice agents. Volume: 80k min/month.",
        fields=[
            NotificationField(label="Score", value="8 / 10 (enterprise)"),
            NotificationField(label="Use case", value="realtime_agents"),
        ],
        urgency="urgent",
        link_url="https://your-app.com/leads/abc",
    ))
"""

from __future__ import annotations

import time
from typing import Any, Literal

import httpx
from pydantic import BaseModel, Field
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from voice_agent_core.observability import MetricNames, get_logger, get_meter

log = get_logger(__name__)
_meter = get_meter("voice_agent_core.notify")
_h_latency = _meter.create_histogram(
    MetricNames.NOTIFY_DISPATCH_LATENCY_MS,
    unit="ms",
    description="Notification dispatch latency",
)
_c_errors = _meter.create_counter(
    MetricNames.NOTIFY_ERRORS,
    description="Notification dispatch errors",
)


class NotificationField(BaseModel):
    """One labeled value to render in the Slack message body."""

    label: str
    value: str


class NotificationPayload(BaseModel):
    """Provider-agnostic notification payload.

    Notifiers only read the public fields declared here. Subclass to attach
    extra structured data if your code needs it; SlackNotifier ignores anything
    not in the base schema.
    """

    title: str
    summary: str
    fields: list[NotificationField] = Field(default_factory=list)
    urgency: Literal["normal", "urgent"] = "normal"
    link_url: str | None = None


class _SlackServerError(Exception):
    """Raised on Slack 5xx responses so the retry layer can catch + back off."""


class SlackNotifier:
    """Posts notifications to a Slack incoming webhook.

    Construction is cheap — no network call until ``send()`` is invoked. Pass
    an empty ``webhook_url`` (default) to enable dev mode: notifications are
    logged via structlog instead of POSTed, so local dev needs no real Slack.
    """

    def __init__(
        self,
        *,
        webhook_url: str = "",
        urgent_mention: str = "<!here>",
        timeout_s: float = 5.0,
        max_retries: int = 3,
    ) -> None:
        self._webhook_url = webhook_url
        self._urgent_mention = urgent_mention
        self._timeout_s = timeout_s
        self._max_retries = max_retries

    async def send(self, payload: NotificationPayload) -> None:
        """Dispatch a notification.

        Dev mode (no ``webhook_url``): logs the payload structurally and returns.
        Configured: POSTs Block Kit JSON to Slack with retry on transient errors.
        Raises ``httpx.HTTPStatusError`` on permanent (4xx) failure after retries.
        """
        if not self._webhook_url:
            log.info(
                "notify.dev_mode",
                title=payload.title,
                summary=payload.summary,
                urgency=payload.urgency,
                fields=[f.model_dump() for f in payload.fields],
                link_url=payload.link_url,
            )
            return

        body = self._format_blocks(payload)
        started = time.perf_counter()
        attrs = {"urgency": payload.urgency}

        try:
            await self._post_with_retry(body)
        except Exception:
            _c_errors.add(1, attributes=attrs)
            log.error(
                "notify.send_failed",
                title=payload.title,
                urgency=payload.urgency,
                exc_info=True,
            )
            raise

        latency_ms = round((time.perf_counter() - started) * 1000, 2)
        _h_latency.record(latency_ms, attributes=attrs)
        log.info(
            "notify.sent",
            title=payload.title,
            urgency=payload.urgency,
            latency_ms=latency_ms,
        )

    def _format_blocks(self, payload: NotificationPayload) -> dict[str, Any]:
        """Build Slack Block Kit JSON for the payload."""
        is_urgent = payload.urgency == "urgent"
        mention = f"{self._urgent_mention} " if is_urgent else ""
        emoji = "🚨" if is_urgent else "✅"

        blocks: list[dict[str, Any]] = [
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"{emoji} {payload.title}"},
            },
            {
                "type": "section",
                "text": {"type": "mrkdwn", "text": f"{mention}{payload.summary}"},
            },
        ]

        if payload.fields:
            blocks.append({
                "type": "section",
                "fields": [
                    {"type": "mrkdwn", "text": f"*{f.label}*\n{f.value}"}
                    for f in payload.fields
                ],
            })

        if payload.link_url:
            blocks.append({
                "type": "actions",
                "elements": [{
                    "type": "button",
                    "text": {"type": "plain_text", "text": "View details"},
                    "url": payload.link_url,
                }],
            })

        return {"blocks": blocks}

    async def _post_with_retry(self, body: dict[str, Any]) -> None:
        """POST body to webhook with exponential-backoff retry.

        Retries on network errors and 5xx responses. 4xx raises immediately
        (the request itself is wrong; retrying won't help).
        """
        # httpx.RequestError covers network/timeout failures (transient — retry).
        # httpx.HTTPStatusError is intentionally NOT in this list: it's raised by
        # raise_for_status() on 4xx responses (permanent — do not retry).
        # Our explicit _SlackServerError signals 5xx so we retry on those.
        async for attempt in AsyncRetrying(
            retry=retry_if_exception_type((httpx.RequestError, _SlackServerError)),
            stop=stop_after_attempt(self._max_retries),
            wait=wait_exponential(multiplier=0.5, min=0.5, max=2.0),
            reraise=True,
        ):
            with attempt:
                async with httpx.AsyncClient(timeout=self._timeout_s) as client:
                    resp = await client.post(self._webhook_url, json=body)
                    if 500 <= resp.status_code < 600:
                        raise _SlackServerError(
                            f"Slack returned {resp.status_code}: {resp.text[:200]}"
                        )
                    resp.raise_for_status()


__all__ = ["NotificationField", "NotificationPayload", "SlackNotifier"]
