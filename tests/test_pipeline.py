"""When a video number is spent, and when it is not.

This is the pipeline's contract with the counter, tested with the expensive
stages stubbed out. Three rules, and the third is the one that was wrong until
it was pinned here:

- a real build that finishes spends its number;
- a real build that fails does not, so the retry and the next batch can reuse it;
- a dry run does not, so inspecting the schedule is free and repeatable.
"""
import pytest

import src.agents.video.pipeline as pipeline
from src.agents.video import numbering
from src.agents.video.plan import plan_batch

ENABLED = ["donald-trump", "andrew-tate", "arnold-schwarzenegger"]


@pytest.fixture
def counter(tmp_path, monkeypatch):
    """Point the counter at a temporary file for the duration of the test.

    Set through the environment rather than passed as an argument, because the
    pipeline calls peek/advance with no path. Without this the tests below would
    advance the real channel counter -- which is exactly what happened the first
    time this file was written.
    """
    path = str(tmp_path / "video_number.json")
    monkeypatch.setenv("VIDEO_NUMBER_FILE", path)
    numbering.reset(path)
    return path


@pytest.fixture
def stub_render(monkeypatch):
    """Replace the whole build with a filename, so no TTS or card work happens."""
    calls = []

    def fake_create_video(plan, number, publish, script_only):
        calls.append({"number": number, "type_id": plan.type_id,
                      "cast": plan.characters})
        return f"/tmp/fake_{number}.mp4"

    monkeypatch.setattr(pipeline, "create_video", fake_create_video)
    return calls


def test_a_finished_real_build_spends_its_number(counter, stub_render):
    plan = plan_batch(1, ENABLED, start=0)[0]
    pipeline._attempt(plan, numbering.peek(counter), False, False, 1, "x")
    assert numbering.peek(counter) == 2


def test_a_failed_real_build_does_not_spend_its_number(counter, monkeypatch):
    """A build that put no video on the channel must leave the number free.

    Otherwise the next batch re-enters a different number, which shifts every
    later video and quietly renumbers the whole back catalogue.
    """
    def boom(**kwargs):
        raise RuntimeError("render failed")

    monkeypatch.setattr(pipeline, "create_video", boom)
    plan = plan_batch(1, ENABLED, start=0)[0]
    pipeline._attempt(plan, numbering.peek(counter), False, False, 1, "x")
    assert numbering.peek(counter) == 1


def test_a_retry_reuses_the_number_and_keeps_the_cast(counter, monkeypatch):
    """The retry must be the same video, not a different one wearing its number."""
    plans = plan_batch(1, ENABLED, start=0)
    plan = plans[0]
    attempts = []

    def flaky(**kwargs):
        attempts.append(kwargs["number"])
        if len(attempts) == 1:
            raise RuntimeError("transient TTS failure")
        return f"/tmp/fake_{kwargs['number']}.mp4"

    monkeypatch.setattr(pipeline, "create_video", flaky)
    number = numbering.peek(counter)
    video = pipeline._attempt(plan, number, False, False, 2, "x")

    assert attempts == [1, 1], "the retry used a different number"
    assert plan.characters == ("donald-trump",), "the cast moved between attempts"
    assert video == "/tmp/fake_1.mp4"
    assert numbering.peek(counter) == 2, "spent exactly once, not once per attempt"


def test_a_dry_run_spends_nothing(counter, stub_render):
    """--script-only produces no video, so it must not consume a number.

    This is the footgun: a dry run used to advance the real counter, so checking
    the schedule left a permanent gap in the channel's numbering.
    """
    plan = plan_batch(1, ENABLED, start=0)[0]
    pipeline._attempt(plan, numbering.peek(counter), False, True, 1, "x")
    assert numbering.peek(counter) == 1
    assert stub_render[0]["number"] == 1, "it still reports the number it would take"


def test_a_dry_run_batch_reports_the_numbers_it_would_use(counter, stub_render):
    """A dry run should answer "what would these be?", not "what is next?".

    It reports the contiguous run it is previewing while the counter stays put,
    so the answer is the same every time it is asked.
    """
    made = pipeline.run_batch(3, publish=False, script_only=True, attempts=1)
    assert len(made) == 3
    assert [c["number"] for c in stub_render] == [1, 2, 3]
    assert numbering.peek(counter) == 1


def test_running_the_same_dry_run_twice_gives_the_same_answer(counter, stub_render):
    for _ in range(2):
        pipeline.run_batch(3, publish=False, script_only=True, attempts=1)
    assert [c["number"] for c in stub_render] == [1, 2, 3, 1, 2, 3]
    assert numbering.peek(counter) == 1


def test_a_real_batch_advances_once_per_video(counter, stub_render):
    made = pipeline.run_batch(3, publish=False, script_only=False, attempts=1)
    assert len(made) == 3
    assert [c["number"] for c in stub_render] == [1, 2, 3]
    assert numbering.peek(counter) == 4


def test_a_real_batch_never_reissues_a_number_when_a_slot_fails(counter, monkeypatch):
    """The invariant the whole counter exists for, under the awkward case.

    If video 2 of 3 fails, its number stays free, so the next slot takes that
    number rather than the one after it. Two videos sharing a number is the one
    error a viewer can see, so the assertion is on numbers that were actually
    *issued* -- a failed attempt legitimately touches the same number twice.
    """
    issued, tried_two = [], []

    def flaky(**kwargs):
        number = kwargs["number"]
        if number == 2:
            tried_two.append(number)
            if len(tried_two) == 1:          # fail the first attempt only
                raise RuntimeError("render failed")
        issued.append(number)
        return f"/tmp/fake_{number}.mp4"

    monkeypatch.setattr(pipeline, "create_video", flaky)
    pipeline.run_batch(3, publish=False, script_only=False, attempts=1)

    assert len(issued) == len(set(issued)), f"a number was issued twice: {issued}"
    assert issued == [1, 2], "the freed number was skipped instead of reused"
    assert numbering.peek(counter) == 3, "two of three videos were made"


def test_a_failed_slot_is_abandoned_rather_than_reserved(counter, monkeypatch):
    """Pins a consequence of the above, because it is a real trade-off.

    The number is reused, but the *slot* is not: the next planned video is built
    with the freed number, so the published channel skips the layout the plan
    called for. Numbering stays dense and correct; the layout sequence loses an
    entry. Retrying harder reduces how often this happens but cannot remove it,
    since a render can fail for reasons outside the pipeline's control.
    """
    built, tried_two = [], []

    def flaky(**kwargs):
        number = kwargs["number"]
        if number == 2:
            tried_two.append(number)
            if len(tried_two) == 1:
                built.append(number)
                raise RuntimeError("render failed")
        built.append(number)
        return f"/tmp/fake_{number}.mp4"

    monkeypatch.setattr(pipeline, "create_video", flaky)
    made = pipeline.run_batch(3, publish=False, script_only=False, attempts=1)
    assert len(made) == 2
    # Number 2 was rebuilt, but as the *next* slot, not the one that failed.
    plans = pipeline.plan_batch(3, ENABLED, start=0)
    assert plans[1].type_id == 2, "the failed slot was the two-character video"
    assert built == [1, 2, 2], "the freed number was handed to the following slot"
