"""Tests for voice_agent_core.notify.SlackNotifier.

We don't hit real Slack — for HTTP-path tests, we monkeypatch
``httpx.AsyncClient.post`` to return canned responses.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import httpx
import pytest

from voice_agent_core.notify import (
    NotificationField,
    NotificationPayload,
    SlackNotifier,
)


def _sample_payload(urgency: str = "normal") -> NotificationPayload:
    return NotificationPayload(
        title="Test lead from acme.com",
        summary="Use case: realtime voice agents.",
        fields=[
            NotificationField(label="Score", value="8/10 (enterprise)"),
            NotificationField(label="Use case", value="realtime_agents"),
        ],
        urgency=urgency,  # type: ignore[arg-type]
        link_url="https://example.com/leads/1",
    )


class TestDevMode:
    async def test_empty_webhook_skips_http(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # If any HTTP call leaks through in dev mode, this would blow up
        async def boom(*a: Any, **k: Any) -> None:
            raise AssertionError("dev mode should NOT make HTTP calls")

        monkeypatch.setattr(httpx.AsyncClient, "post", boom)

        notifier = SlackNotifier(webhook_url="")
        await notifier.send(_sample_payload())  # must not raise


class TestFormatBlocks:
    def test_urgent_has_alarm_emoji_and_mention(self) -> None:
        notifier = SlackNotifier(webhook_url="x", urgent_mention="<!here>")
        body = notifier._format_blocks(_sample_payload(urgency="urgent"))

        header_text = body["blocks"][0]["text"]["text"]
        summary_text = body["blocks"][1]["text"]["text"]

        assert "🚨" in header_text
        assert "<!here>" in summary_text

    def test_normal_has_check_emoji_no_mention(self) -> None:
        notifier = SlackNotifier(webhook_url="x")
        body = notifier._format_blocks(_sample_payload(urgency="normal"))

        header_text = body["blocks"][0]["text"]["text"]
        summary_text = body["blocks"][1]["text"]["text"]

        assert "✅" in header_text
        assert "<!here>" not in summary_text

    def test_fields_render_as_section(self) -> None:
        notifier = SlackNotifier(webhook_url="x")
        body = notifier._format_blocks(_sample_payload())

        fields_block = body["blocks"][2]
        assert fields_block["type"] == "section"
        assert len(fields_block["fields"]) == 2
        # Labels are bolded markdown
        assert "*Score*" in fields_block["fields"][0]["text"]

    def test_link_renders_as_button(self) -> None:
        notifier = SlackNotifier(webhook_url="x")
        body = notifier._format_blocks(_sample_payload())

        last_block = body["blocks"][-1]
        assert last_block["type"] == "actions"
        assert last_block["elements"][0]["url"] == "https://example.com/leads/1"

    def test_payload_with_no_fields_no_link(self) -> None:
        notifier = SlackNotifier(webhook_url="x")
        body = notifier._format_blocks(
            NotificationPayload(title="t", summary="s")
        )
        # Only header + summary blocks; no fields section or button
        assert len(body["blocks"]) == 2


class TestPostPath:
    async def test_successful_post(self, monkeypatch: pytest.MonkeyPatch) -> None:
        posts_made: list[dict[str, Any]] = []

        async def fake_post(self: Any, url: str, *, json: Any) -> MagicMock:
            posts_made.append({"url": url, "body": json})
            resp = MagicMock()
            resp.status_code = 200
            resp.raise_for_status = MagicMock()
            return resp

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        await notifier.send(_sample_payload())

        assert len(posts_made) == 1
        assert posts_made[0]["url"] == "https://hooks.slack.com/test"
        assert "blocks" in posts_made[0]["body"]

    async def test_4xx_raises_without_retry(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        call_count = 0

        async def fake_post_400(self: Any, url: str, *, json: Any) -> MagicMock:
            nonlocal call_count
            call_count += 1
            resp = MagicMock()
            resp.status_code = 400
            resp.raise_for_status = MagicMock(
                side_effect=httpx.HTTPStatusError(
                    "400 Bad Request", request=MagicMock(), response=resp
                )
            )
            return resp

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post_400)

        notifier = SlackNotifier(webhook_url="https://hooks.slack.com/test")
        with pytest.raises(httpx.HTTPStatusError):
            await notifier.send(_sample_payload())

        assert call_count == 1  # No retry on 4xx

    async def test_5xx_retries_then_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        responses = [503, 503, 200]  # fail twice, succeed on third
        call_count = 0

        async def fake_post(self: Any, url: str, *, json: Any) -> MagicMock:
            nonlocal call_count
            status = responses[call_count]
            call_count += 1
            resp = MagicMock()
            resp.status_code = status
            resp.text = "service unavailable"
            resp.raise_for_status = MagicMock()
            return resp

        monkeypatch.setattr(httpx.AsyncClient, "post", fake_post)

        notifier = SlackNotifier(
            webhook_url="https://hooks.slack.com/test", max_retries=3
        )
        await notifier.send(_sample_payload())

        assert call_count == 3
