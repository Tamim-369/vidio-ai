"""Shared pytest fixtures for the unit suite.

Run with an explicit path so the manual harnesses that sit beside this
directory are never collected:

    pytest src/tests/unit/

``src/tests/test_youtube_upload.py`` is named test_* but performs a REAL
YouTube upload, so it must stay outside any auto-collected path.

Import note: `src` is a real top-level package imported as `src.*`, so the
project root has to be importable. Keeping the insert here avoids repeating it
in every test module.
"""
from __future__ import annotations

import sys
from pathlib import Path

import pytest

# src/tests/unit/conftest.py -> repo root
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture
def project_root() -> Path:
    """Absolute path to the repository root."""
    return ROOT


@pytest.fixture(autouse=True)
def isolated_voice_rotation(tmp_path, monkeypatch):
    """Keep the test suite out of the real voice-rotation cursor.

    ``pick_voice()`` persists the last character to src/state/voice_rotation.json
    so separate CLI runs advance the cycle. A test calling it without an explicit
    path would move that cursor, and the user's next real video would start on an
    arbitrary character instead of the intended next one. Redirecting the module
    default to tmp_path means forgetting the path argument is harmless.
    """
    from src.agents.voice_cast import agent as cast

    monkeypatch.setattr(cast, "ROTATION_FILE", str(tmp_path / "voice_rotation.json"))
