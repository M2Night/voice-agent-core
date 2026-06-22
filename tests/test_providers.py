"""Tests for voice_agent_core.providers (registry + catalog + dispatch).

No network calls — we register fakes and assert dispatch/catalog behavior, plus
verify the built-in Fish/LLM providers are registered.
"""

from __future__ import annotations

import sys
from types import ModuleType

import pytest

from voice_agent_core import providers
from voice_agent_core.config import BaseAgentSettings
from voice_agent_core.providers import (
    LLMProvider,
    STTProvider,
    TTSProvider,
    build_stt,
    build_tts,
    list_llm_providers,
    list_stt_providers,
    list_tts_providers,
    register_stt,
    register_tts,
    tts_models,
)


class TestBuiltinRegistration:
    def test_fish_registered_for_stt_and_tts(self) -> None:
        assert "fish" in list_stt_providers()
        assert "fish" in list_tts_providers()

    def test_llm_providers_registered(self) -> None:
        assert set(list_llm_providers()) >= {"livekit", "openrouter"}

    def test_fish_tts_catalog(self) -> None:
        # Catalog feeds a future provider -> model dropdown.
        assert tts_models("fish") == ("s1", "s2-pro", "s2.1-pro")

    def test_deepgram_registered_with_catalog(self) -> None:
        assert "deepgram" in list_stt_providers()
        assert providers.stt_models("deepgram") == ("nova-3", "nova-2")


class TestDeepgram:
    def test_build_without_key_raises(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Key check happens before the (lazy) plugin import, so this passes whether
        # or not the deepgram extra is installed.
        monkeypatch.setenv("STT_PROVIDER", "deepgram")
        monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)
        with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
            build_stt(BaseAgentSettings())

    def test_default_language_passed_to_plugin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, object]] = []

        class FakeDeepgramSTT:
            def __init__(self, **kwargs: object) -> None:
                calls.append(kwargs)

        fake = ModuleType("livekit.plugins.deepgram")
        fake.STT = FakeDeepgramSTT  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "livekit.plugins.deepgram", fake)
        monkeypatch.setenv("STT_PROVIDER", "deepgram")
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        monkeypatch.delenv("STT_LANGUAGE", raising=False)

        build_stt(BaseAgentSettings())

        assert calls[-1]["language"] == "en"

    def test_explicit_multi_language_passed_to_plugin(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        calls: list[dict[str, object]] = []

        class FakeDeepgramSTT:
            def __init__(self, **kwargs: object) -> None:
                calls.append(kwargs)

        fake = ModuleType("livekit.plugins.deepgram")
        fake.STT = FakeDeepgramSTT  # type: ignore[attr-defined]
        monkeypatch.setitem(sys.modules, "livekit.plugins.deepgram", fake)
        monkeypatch.setenv("STT_PROVIDER", "deepgram")
        monkeypatch.setenv("DEEPGRAM_API_KEY", "test-key")
        monkeypatch.setenv("STT_LANGUAGE", "multi")

        build_stt(BaseAgentSettings())

        assert calls[-1]["language"] == "multi"

class TestDispatch:
    def test_build_tts_uses_selected_provider(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        sentinel = object()
        register_tts(TTSProvider(name="_fake_tts", build=lambda s: sentinel))
        monkeypatch.setenv("TTS_PROVIDER", "_fake_tts")
        try:
            assert build_tts(BaseAgentSettings()) is sentinel
        finally:
            providers._TTS_PROVIDERS.pop("_fake_tts", None)

    def test_unknown_provider_raises_with_available_list(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("STT_PROVIDER", "does-not-exist")
        with pytest.raises(ValueError, match="Unknown STT provider 'does-not-exist'"):
            build_stt(BaseAgentSettings())


class TestRegistryExtensibility:
    def test_register_then_appears_in_catalog(self) -> None:
        register_stt(STTProvider(name="_fake_stt", build=lambda s: object(), models=("m1",)))
        try:
            assert "_fake_stt" in list_stt_providers()
            assert providers.stt_models("_fake_stt") == ("m1",)
        finally:
            providers._STT_PROVIDERS.pop("_fake_stt", None)

    def test_register_overrides_same_name(self) -> None:
        first = object()
        second = object()
        register_stt(STTProvider(name="_dup", build=lambda s: first))
        register_stt(STTProvider(name="_dup", build=lambda s: second))
        try:
            assert providers._STT_PROVIDERS["_dup"].build(BaseAgentSettings()) is second
        finally:
            providers._STT_PROVIDERS.pop("_dup", None)


def test_llmprovider_models_default_free_form() -> None:
    p = LLMProvider(name="x", build=lambda s: object())
    assert p.models == ()
