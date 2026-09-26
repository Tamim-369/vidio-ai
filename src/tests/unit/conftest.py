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
