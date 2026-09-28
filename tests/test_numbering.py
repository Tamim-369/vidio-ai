"""The video counter, whose only job is to never issue a number twice.

A duplicate number is the one failure a viewer can see: two videos in a channel
that is explicitly a numbered library, both titled "#42". A gap is invisible.
Every rule here follows from that asymmetry, including the surprising one that a
dry run must not consume anything.
"""
import json

import pytest

from src.agents.video import numbering


@pytest.fixture
def counter(tmp_path):
    """An isolated counter file, so no test can touch the real channel's."""
    path = tmp_path / "video_number.json"
    numbering.reset(str(path))
    return str(path)


def test_a_fresh_channel_starts_at_one(counter):
    assert numbering.peek(counter) == numbering.FIRST_NUMBER == 1


def test_an_absent_counter_file_reads_as_one_and_does_not_create_it(tmp_path):
    """A missing file is "no videos yet", not an error.

    Creating it here would be wrong: reading the schedule must not be a write.
    """
    path = str(tmp_path / "never_written.json")
    assert numbering.peek(path) == 1
    assert not (tmp_path / "never_written.json").exists()


@pytest.mark.parametrize("corrupt", [
    "{",                          # truncated write
    "[]",                         # wrong shape
    '{"next": "3"}',              # number as a string
    '{"next": 0}',                # below the first number
    '{"next": -7}',               # negative
    "null",
    "",
])
def test_a_corrupt_counter_falls_back_to_one_rather_than_raising(counter, corrupt):
    """A half-written file must not stop the channel.

    Falling back to 1 risks reissuing a number, which is the worse error, so this
    is a deliberate trade rather than a crash. The test pins the behaviour so a
    future change to it is a conscious one.
    """
    with open(counter, "w", encoding="utf-8") as f:
        f.write(corrupt)
    assert numbering.peek(counter) == 1


def test_peek_does_not_consume(counter):
    assert numbering.peek(counter) == 1
    assert numbering.peek(counter) == 1
    assert numbering.peek(counter) == 1


def test_advance_consumes_and_returns_the_next_number(counter):
    assert numbering.advance(counter) == 2
    assert numbering.peek(counter) == 2


def test_advance_survives_a_process_restart(counter):
    """The reason this is a file and not a module global.

    A batch runs in one process but successive invocations do not, so an
    in-memory counter would restart at zero and every run would open on #1.
    """
    numbering.advance(counter)
    numbering.advance(counter)
    assert json.load(open(counter))["next"] == 3
    assert numbering.peek(counter) == 3


def test_reset_goes_back_to_the_first_number(counter):
    numbering.advance(counter)
    numbering.advance(counter)
    numbering.reset(counter)
    assert numbering.peek(counter) == 1


def test_the_path_override_keeps_validation_off_the_real_counter(tmp_path, monkeypatch):
    """VIDEO_NUMBER_FILE is what makes a dry run safe.

    Asserted by behaviour rather than by reading the module's constant: what
    matters is that setting the variable moves the file the counter reads *and*
    writes, which a constant captured at import time would silently not do.
    """
    path = str(tmp_path / "override.json")
    monkeypatch.setenv("VIDEO_NUMBER_FILE", path)
    assert numbering.peek() == 1
    numbering.advance()
    assert json.load(open(path))["next"] == 2
    assert numbering.peek() == 2
