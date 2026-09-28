"""The number -> cast mapping, which is the whole channel's shape.

Every other stage is a detail of one video. This is the one thing that has to be
right for video 400 as well as video 4, and it is the cheapest thing to get
wrong: change the cycle length or the phase and a hundred videos are in the
wrong order with nothing to indicate it.
"""
import pytest

from src.agents.video.plan import (
    CYCLE,
    TYPE_CYCLE,
    plan_batch,
    with_lead,
)

ENABLED = ["donald-trump", "andrew-tate", "arnold-schwarzenegger"]

# The schedule the channel is specified to produce, written out rather than
# computed, so that a change to either constant fails here instead of quietly
# redefining what "correct" means.
EXPECTED_FIRST_TEN = [
    (1, "donald-trump", ["donald-trump"]),
    (2, "andrew-tate", ["andrew-tate", "arnold-schwarzenegger", "donald-trump"]),
    (3, "arnold-schwarzenegger", ["arnold-schwarzenegger", "donald-trump"]),
    (4, "donald-trump", ["donald-trump"]),
    (5, "andrew-tate", ["andrew-tate", "arnold-schwarzenegger"]),
    (6, "arnold-schwarzenegger", ["arnold-schwarzenegger", "donald-trump", "andrew-tate"]),
    (7, "donald-trump", ["donald-trump"]),
    (8, "andrew-tate", ["andrew-tate", "arnold-schwarzenegger", "donald-trump"]),
    (9, "arnold-schwarzenegger", ["arnold-schwarzenegger", "donald-trump"]),
    (10, "donald-trump", ["donald-trump"]),
]


@pytest.mark.parametrize("number,lead,cast", EXPECTED_FIRST_TEN)
def test_first_ten_slots_match_the_specified_schedule(number, lead, cast):
    """Video N is the Nth slot of a batch planned from the first video."""
    plan = plan_batch(number, ENABLED, start=0)[-1]
    assert plan.characters[0] == lead
    assert list(plan.characters) == cast


def test_the_layout_cycle_visits_every_shape_without_one_dominating():
    """The reason the layouts vary at all: no single shape takes the channel.

    Over the specified ten, the monologue is the plurality (4 of 10) but not a
    majority, and the other six slots are two- and three-character videos. The
    requirement is the spread, not any particular count, so that is what is
    asserted -- a batch of ten identical monologues passes no test here.
    """
    types = [p.type_id for p in plan_batch(10, ENABLED, start=0)]
    assert types == [1, 2, 3, 1, 3, 2, 1, 2, 3, 1]
    assert set(types) == {1, 2, 3}
    assert max(types.count(t) for t in {1, 2, 3}) <= 4
    # The layouts run out of phase with the cast cycle rather than in step: the
    # three-character videos are not all the same slot number.
    assert len({i % 6 for i, t in enumerate(types) if t == 2}) > 1


def test_quotes_each_makes_the_batch_add_up_to_the_slot_count():
    """One call per character, and the totals are the point of the plan."""
    plans = plan_batch(10, ENABLED, start=0)
    assert sum(len(p.characters) for p in plans) == 19
    assert sum(len(p.characters) * p.quotes_each for p in plans) == 27
    for p in plans:
        assert p.quotes_each == (3 if p.type_id == 1 else 1)


def test_a_later_batch_continues_the_channel_instead_of_restarting():
    """Two consecutive batches must read as one channel, not the same ten twice."""
    first = plan_batch(4, ENABLED, start=0)
    second = plan_batch(4, ENABLED, start=4)
    assert [p.type_id for p in first + second] == [1, 2, 3, 1, 3, 2, 1, 2]
    assert [p.characters[0] for p in first + second] == [
        "donald-trump", "andrew-tate", "arnold-schwarzenegger", "donald-trump",
        "andrew-tate", "arnold-schwarzenegger", "donald-trump", "andrew-tate",
    ]


def test_the_cycle_never_repeats_a_lead_adjacent_to_itself():
    """Consecutive videos must not be narrated by the same character."""
    leads = [p.characters[0] for p in plan_batch(30, ENABLED, start=0)]
    assert all(a != b for a, b in zip(leads, leads[1:]))


def test_every_cast_member_is_enabled_and_the_lead_is_first():
    plans = plan_batch(10, ENABLED, start=0)
    for p in plans:
        assert set(p.characters) <= set(ENABLED)
        assert len(set(p.characters)) == len(p.characters), "a character appears twice"
        assert p.characters[0] in p.characters


def test_with_lead_moves_the_lead_without_changing_the_shape():
    """--voice is an override of who leads, not of what the video is.

    Promoting somebody who is not already in the cast should leave the co-stars
    exactly as they were: only the lead moves.
    """
    plan = plan_batch(5, ENABLED, start=0)[-1]          # type 3, tate + arnold
    assert plan.characters == ("andrew-tate", "arnold-schwarzenegger")

    moved = with_lead(plan, "donald-trump", ENABLED)
    assert moved.type_id == plan.type_id
    assert moved.quotes_each == plan.quotes_each
    assert moved.characters == ("donald-trump", "arnold-schwarzenegger")
    assert plan.characters[0] == "andrew-tate", "the original is untouched"


def test_with_lead_never_puts_one_character_in_the_video_twice():
    """Promoting somebody who is already a co-star must swap, not duplicate.

   donald-trump is the co-star of this slot, so making him the lead and leaving
    him in place yields the same voice and the same face on two cards. The
    replacement has to come from the enabled voices, not from the cycle, or a
    disabled character gets promoted into a cast nobody asked for.
    """
    plan = plan_batch(3, ENABLED, start=0)[-1]          # arnold + donald
    assert "donald-trump" in plan.characters

    moved = with_lead(plan, "donald-trump", ENABLED)
    assert moved.characters[0] == "donald-trump"
    assert len(set(moved.characters)) == len(moved.characters)
    assert len(moved.characters) == 2
    assert set(moved.characters) <= set(ENABLED)


def test_with_lead_cannot_promote_a_voice_outside_the_enabled_list():
    plan = plan_batch(3, ENABLED, start=0)[-1]
    enabled = ["donald-trump", "arnold-schwarzenegger"]   # tate switched off
    moved = with_lead(plan, "donald-trump", enabled)
    assert set(moved.characters) <= set(enabled)
    assert "andrew-tate" not in moved.characters


def test_cycles_are_long_enough_to_stay_out_of_phase():
    """Guards the assumption the out-of-phase test above rests on."""
    assert len(TYPE_CYCLE) == 6
    assert len(CYCLE) == 3
