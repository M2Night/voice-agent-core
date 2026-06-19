"""Tests for voice_agent_core.llm.build_llm.

We don't make real LLM calls here — just verify that build_llm dispatches to the
right provider and that config errors raise clearly.
"""

from __future__ import annotations

import pytest

from voice_agent_core.config import BaseAgentSettings
from voice_agent_core.providers import build_llm


class TestBuildLLM:
    def test_livekit_backend_returns_inference_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "livekit")
        monkeypatch.setenv("LLM_MODEL", "openai/gpt-5.2-chat-latest")
        monkeypatch.setenv("LIVEKIT_API_KEY", "test-key")
        monkeypatch.setenv("LIVEKIT_API_SECRET", "test-secret")
        s = BaseAgentSettings()

        llm = build_llm(s)

        from livekit.agents import inference

        assert isinstance(llm, inference.LLM)

    def test_livekit_backend_requires_api_key_and_secret(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "livekit")
        monkeypatch.delenv("LIVEKIT_API_KEY", raising=False)
        monkeypatch.delenv("LIVEKIT_API_SECRET", raising=False)
        s = BaseAgentSettings()

        with pytest.raises(ValueError, match="LIVEKIT_API_KEY and LIVEKIT_API_SECRET"):
            build_llm(s)

    def test_openrouter_backend_requires_api_key(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
        s = BaseAgentSettings()

        with pytest.raises(ValueError, match="OPENROUTER_API_KEY"):
            build_llm(s)

    def test_openrouter_backend_with_key_returns_openai_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("LLM_PROVIDER", "openrouter")
        monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test-fake")
        monkeypatch.setenv("LLM_MODEL", "anthropic/claude-sonnet-4-6")
        s = BaseAgentSettings()

        llm = build_llm(s)

        from livekit.plugins import openai

        assert isinstance(llm, openai.LLM)
