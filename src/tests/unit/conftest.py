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

import json
import sys
from pathlib import Path

import pytest

# src/tests/unit/conftest.py -> repo root
ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# used_quotes.json is gitignored: it is per-machine run history, not source, so
# it is absent on a fresh clone and in CI. Tests that need realistic quotes
# therefore cannot depend on it existing, or the suite passes locally and fails
# everywhere else. This is a frozen sample of ten real accepted quotes chosen to
# span the pool's measured length range (58-98, median 81), which is what the
# YouTube title-limit tests are calibrated against. Quotes at 86 and above are
# the ones that overflow once a pseudonym prefix is added, so the range matters.
SAMPLE_QUOTES = [
    "An army marches on its stomach; I march on my snack stash.",  # 58
    "Hard work beats talent when talent doesn't show up for coffee.",  # 62
    "Deadlines are like last sets. I hit them before the barbell drops.",  # 66
    "An army wins by surprise; I win when the surprise is a pizza delivery.",  # 70
    "A leader plans ahead; I plan to leave the battlefield when the Wi-Fi dies.",  # 74
    "A bad day feels heavy. I add more plates to my schedule and call it a leg day.",  # 78
    "A true leader plans ahead, which is why I schedule my naps between board meetings.",  # 82
    "A good commander reads the battlefield. I read the menu to know where the enemy dines.",  # 86
    "An army conquers the terrain before the enemy. I conquer the terrain after the enemy eats.",  # 90
    "A general knows when to strike. I strike when the microwave beeps, assuming the enemy is hungry.",  # 96
]


@pytest.fixture
def project_root() -> Path:
    """Absolute path to the repository root."""
    return ROOT


def _read_quotes(pool_file: Path) -> list:
    """Accepted quote texts from a pool file, or [] if there is nothing to read."""
    try:
        with open(pool_file) as fh:
            return [e["text"] for e in json.load(fh)["quotes"] if e.get("text")]
    except (OSError, ValueError):
        return []


def load_quote_pool(root: Path) -> list:
    """The real pool if this root has one, else the frozen sample.

    A plain function rather than fixture logic so the fallback is directly
    testable; the fixture is a one-line wrapper over it.
    """
    return _read_quotes(root / "used_quotes.json") or list(SAMPLE_QUOTES)


@pytest.fixture
def quote_pool() -> list:
    """Real accepted quotes when this machine has them, else the frozen sample.

    Tests should read the real pool where it exists, because the length
    distribution it produced is the thing under test. But a gitignored file
    cannot be a hard dependency, so a clone falls back to the sample rather
    than erroring.
    """
    return load_quote_pool(ROOT)


@pytest.fixture(autouse=True)
def isolated_voice_rotation(tmp_path, monkeypatch):
    """Keep the test suite out of the real voice-rotation cursor.

    ``pick_voice()`` persists the last character to voice_rotation.json at the
    repo root
    so separate CLI runs advance the cycle. A test calling it without an explicit
    path would move that cursor, and the user's next real video would start on an
    arbitrary character instead of the intended next one. Redirecting the module
    default to tmp_path means forgetting the path argument is harmless.
    """
    from src.agents.voice_cast import agent as cast

    monkeypatch.setattr(cast, "ROTATION_FILE", str(tmp_path / "voice_rotation.json"))
