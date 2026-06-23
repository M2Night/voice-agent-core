"""Provider abstraction: a registry + catalog for STT / TTS / LLM.

Each layer (STT, TTS, LLM) has a registry mapping a provider name to a provider spec
— a build function plus the list of model ids that provider offers. The registry is the
single source of truth for two things:

1. **Dispatch** — :func:`build_stt` / :func:`build_tts` / :func:`build_llm` select a
   builder by the matching ``*_provider`` setting and call it. The builder reads its own
   model / voice / credentials off the settings object.
2. **Catalog** — :func:`list_stt_providers` / :func:`stt_models` (and the tts/llm
   equivalents) expose what's available, so a UI (e.g. a future dashboard) can render
   provider → model dropdowns from the same data.

Deepgram is the default STT, Fish is the default TTS, and OpenRouter is the default
LLM. Adding a new
provider is a single ``register_*`` call — no change to this module or to
``pipeline.py`` — and can be done from outside the library::

    from voice_agent_core import register_tts, TTSProvider
    register_tts(TTSProvider("elevenlabs", build=my_builder, models=("eleven_flash_v2_5",)))
    # then set TTS_PROVIDER=elevenlabs

Fallback chains (wrapping the chosen component in LiveKit's ``FallbackAdapter``) plug in
at the ``build_*`` boundary and are intentionally not implemented yet.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, TypeVar

from livekit.agents import llm as agents_llm
from livekit.agents import stt as agents_stt
from livekit.agents import tts as agents_tts

from voice_agent_core.deepgram import DeepgramSettings, build_deepgram_stt
from voice_agent_core.fish import FishSettings, build_fish_stt, build_fish_tts
from voice_agent_core.llm import (
    OpenRouterSettings,
    build_livekit_llm,
    build_openrouter_llm,
)

if TYPE_CHECKING:
    from pydantic_settings import BaseSettings

    from voice_agent_core.config import BaseAgentSettings


@dataclass(frozen=True)
class STTProvider:
    """An STT provider: how to build it + which model ids it offers."""

    name: str
    build: Callable[[BaseAgentSettings], agents_stt.STT]
    models: tuple[str, ...] = ()
    """Selectable model ids. Empty = single/fixed model or free-form."""
    settings_cls: type[BaseSettings] | None = None
    """Provider-specific settings class (env-driven). Powers a frontend's
    provider-specific config form via :func:`provider_config_schema`."""


@dataclass(frozen=True)
class TTSProvider:
    """A TTS provider: how to build it + which model ids it offers."""

    name: str
    build: Callable[[BaseAgentSettings], agents_tts.TTS]
    models: tuple[str, ...] = ()
    """Selectable model ids. Empty = single/fixed model or free-form."""
    settings_cls: type[BaseSettings] | None = None
    """Provider-specific settings class (env-driven)."""


@dataclass(frozen=True)
class LLMProvider:
    """An LLM provider: how to build it + which model ids it offers."""

    name: str
    build: Callable[[BaseAgentSettings], agents_llm.LLM]
    models: tuple[str, ...] = ()
    """Selectable model ids. Empty = free-form (provider accepts arbitrary ids)."""
    settings_cls: type[BaseSettings] | None = None
    """Provider-specific settings class (env-driven)."""


_STT_PROVIDERS: dict[str, STTProvider] = {}
_TTS_PROVIDERS: dict[str, TTSProvider] = {}
_LLM_PROVIDERS: dict[str, LLMProvider] = {}

_P = TypeVar("_P", STTProvider, TTSProvider, LLMProvider)


def _require(registry: dict[str, _P], name: str, layer: str) -> _P:
    try:
        return registry[name]
    except KeyError:
        raise ValueError(
            f"Unknown {layer} provider {name!r}; available: {sorted(registry)}"
        ) from None


# --- Registration (idempotent; re-registering a name overrides it) ---


def register_stt(provider: STTProvider) -> None:
    """Register (or override) an STT provider by name."""
    _STT_PROVIDERS[provider.name] = provider


def register_tts(provider: TTSProvider) -> None:
    """Register (or override) a TTS provider by name."""
    _TTS_PROVIDERS[provider.name] = provider


def register_llm(provider: LLMProvider) -> None:
    """Register (or override) an LLM provider by name."""
    _LLM_PROVIDERS[provider.name] = provider


# --- Dispatch (selects builder by the *_provider setting) ---


def build_stt(settings: BaseAgentSettings) -> agents_stt.STT:
    """Build the STT component for ``settings.stt_provider``."""
    return _require(_STT_PROVIDERS, settings.stt_provider, "STT").build(settings)


def build_tts(settings: BaseAgentSettings) -> agents_tts.TTS:
    """Build the TTS component for ``settings.tts_provider``."""
    return _require(_TTS_PROVIDERS, settings.tts_provider, "TTS").build(settings)


def build_llm(settings: BaseAgentSettings) -> agents_llm.LLM:
    """Build the LLM component for ``settings.llm_provider``."""
    return _require(_LLM_PROVIDERS, settings.llm_provider, "LLM").build(settings)


# --- Catalog (for UIs: provider list + per-provider model list) ---


def list_stt_providers() -> list[str]:
    """Registered STT provider names, sorted."""
    return sorted(_STT_PROVIDERS)


def list_tts_providers() -> list[str]:
    """Registered TTS provider names, sorted."""
    return sorted(_TTS_PROVIDERS)


def list_llm_providers() -> list[str]:
    """Registered LLM provider names, sorted."""
    return sorted(_LLM_PROVIDERS)


def stt_models(provider: str) -> tuple[str, ...]:
    """Selectable model ids for an STT provider (empty = fixed/free-form)."""
    return _require(_STT_PROVIDERS, provider, "STT").models


def tts_models(provider: str) -> tuple[str, ...]:
    """Selectable model ids for a TTS provider (empty = fixed/free-form)."""
    return _require(_TTS_PROVIDERS, provider, "TTS").models


def llm_models(provider: str) -> tuple[str, ...]:
    """Selectable model ids for an LLM provider (empty = free-form)."""
    return _require(_LLM_PROVIDERS, provider, "LLM").models


_REGISTRIES = {"stt": _STT_PROVIDERS, "tts": _TTS_PROVIDERS, "llm": _LLM_PROVIDERS}


def provider_config_schema(layer: str, name: str) -> dict | None:
    """JSON schema of a provider's provider-specific settings, or None if it has none.

    Lets a frontend render "show Fish's config only when Fish is selected": e.g.
    ``provider_config_schema("tts", "fish")`` returns ``FishSettings``'s JSON schema
    (its ``FISH_*`` fields). Generic selection (model/voice/language) is separate, on
    ``BaseAgentSettings``.
    """
    try:
        registry = _REGISTRIES[layer]
    except KeyError:
        raise ValueError(f"Unknown layer {layer!r}; expected stt/tts/llm") from None
    provider = _require(registry, name, layer.upper())
    if provider.settings_cls is None:
        return None
    return provider.settings_cls.model_json_schema()


# --- Built-in providers (Fish-first) ---

register_stt(STTProvider(name="fish", build=build_fish_stt, settings_cls=FishSettings))
register_stt(
    STTProvider(
        name="deepgram",
        build=build_deepgram_stt,
        models=("nova-3", "nova-2"),
        settings_cls=DeepgramSettings,
    )
)
register_tts(
    TTSProvider(
        name="fish",
        build=build_fish_tts,
        models=("s1", "s2-pro", "s2.1-pro"),
        settings_cls=FishSettings,
    )
)
register_llm(LLMProvider(name="livekit", build=build_livekit_llm))
register_llm(
    LLMProvider(
        name="openrouter",
        build=build_openrouter_llm,
        settings_cls=OpenRouterSettings,
    )
)


__all__ = [
    "LLMProvider",
    "STTProvider",
    "TTSProvider",
    "build_llm",
    "build_stt",
    "build_tts",
    "list_llm_providers",
    "list_stt_providers",
    "list_tts_providers",
    "llm_models",
    "provider_config_schema",
    "register_llm",
    "register_stt",
    "register_tts",
    "stt_models",
    "tts_models",
]
