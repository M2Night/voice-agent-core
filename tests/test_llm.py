"""Tests for voice_agent_core.llm.build_llm.

We don't make real LLM calls here — just verify that build_llm dispatches to the
right provider and that config errors raise clearly.
"""

from __future__ import annotations

import pytest

from voice_agent_core.config import BaseAgentSettings
from voice_agent_core.providers import build_llm


def _fake_openai_plugin(monkeypatch: pytest.MonkeyPatch) -> list[dict[str, object]]:
    """Replace ``livekit.plugins.openai.LLM`` with a stub that records its kwargs.

    Lets us assert what ``build_custom_llm`` passes to ``openai.LLM(**kwargs)``
    without constructing a real client (which would open an httpx connection). We
    patch the attribute on the real module rather than swapping sys.modules, since
    other tests may already have bound ``openai`` as an attribute of the
    ``livekit.plugins`` package (which a sys.modules swap would not override).
    """
    calls: list[dict[str, object]] = []

    class FakeLLM:
        def __init__(self, **kwargs: object) -> None:
            calls.append(kwargs)

    from livekit.plugins import openai

    monkeypatch.setattr(openai, "LLM", FakeLLM)
    return calls


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


class TestBuildCustomLLM:
    def test_custom_backend_requires_base_url(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Base-url check happens before the (lazy) openai import, so this passes
        # whether or not the plugin is importable in the test env.
        monkeypatch.setenv("LLM_PROVIDER", "custom")
        monkeypatch.delenv("CUSTOM_LLM_BASE_URL", raising=False)

        with pytest.raises(ValueError, match="CUSTOM_LLM_BASE_URL"):
            build_llm(BaseAgentSettings())

    def test_custom_backend_passes_config_to_openai_llm(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _fake_openai_plugin(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "custom")
        monkeypatch.setenv("CUSTOM_LLM_BASE_URL", "https://endpoint.example/v1")
        monkeypatch.setenv("CUSTOM_LLM_MODEL", "google/gemma-4")
        monkeypatch.setenv("CUSTOM_LLM_API_KEY", "secret")
        monkeypatch.setenv("CUSTOM_LLM_TEMPERATURE", "0.3")
        # LLM_MODEL must be ignored when CUSTOM_LLM_MODEL is set.
        monkeypatch.setenv("LLM_MODEL", "openai/should-not-be-used")

        build_llm(BaseAgentSettings())

        kwargs = calls[-1]
        assert kwargs["base_url"] == "https://endpoint.example/v1"
        assert kwargs["model"] == "google/gemma-4"
        assert kwargs["api_key"] == "secret"
        assert kwargs["temperature"] == 0.3
        # No cap by default -> no extra_body.
        assert "extra_body" not in kwargs

    def test_custom_model_falls_back_to_llm_model(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _fake_openai_plugin(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "custom")
        monkeypatch.setenv("CUSTOM_LLM_BASE_URL", "https://endpoint.example/v1")
        monkeypatch.delenv("CUSTOM_LLM_MODEL", raising=False)
        monkeypatch.setenv("LLM_MODEL", "fallback/model")

        build_llm(BaseAgentSettings())

        assert calls[-1]["model"] == "fallback/model"

    def test_custom_max_tokens_sets_extra_body(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _fake_openai_plugin(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "custom")
        monkeypatch.setenv("CUSTOM_LLM_BASE_URL", "https://endpoint.example/v1")
        monkeypatch.setenv("CUSTOM_LLM_MODEL", "m")
        monkeypatch.setenv("CUSTOM_LLM_MAX_TOKENS", "60")

        build_llm(BaseAgentSettings())

        assert calls[-1]["extra_body"] == {"max_tokens": 60}

    def test_custom_api_key_defaults_to_empty_sentinel(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls = _fake_openai_plugin(monkeypatch)
        monkeypatch.setenv("LLM_PROVIDER", "custom")
        monkeypatch.setenv("CUSTOM_LLM_BASE_URL", "https://endpoint.example/v1")
        monkeypatch.setenv("CUSTOM_LLM_MODEL", "m")
        monkeypatch.delenv("CUSTOM_LLM_API_KEY", raising=False)

        build_llm(BaseAgentSettings())

        assert calls[-1]["api_key"] == "EMPTY"
