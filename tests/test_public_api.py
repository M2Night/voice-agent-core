"""Tests for the package-root convenience API."""

from __future__ import annotations

import subprocess
import sys


def test_package_root_import_is_lazy() -> None:
    code = """
import sys
import voice_agent_core

heavy = [
    "voice_agent_core.fish.tts",
    "voice_agent_core.pipeline",
    "voice_agent_core.runtime",
]
print("before", any(name in sys.modules for name in heavy))
print("submodule-before", "voice_agent_core.providers" in sys.modules)
print("submodule-after", voice_agent_core.providers.__name__)

from voice_agent_core import BaseAgentSettings, Notifier, build_pipeline

print("after", BaseAgentSettings.__name__, Notifier.__name__, callable(build_pipeline))
"""
    result = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )

    assert "before False" in result.stdout
    assert "after BaseAgentSettings Notifier True" in result.stdout
    assert "submodule-before False" in result.stdout
    assert "submodule-after voice_agent_core.providers" in result.stdout
