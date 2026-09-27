"""What each video in a batch is: how many characters speak, and how many quotes each.

A video is one of three shapes:

    type 1   one character, three quotes   (a monologue)
    type 2   three characters, one quote each
    type 3   two characters, one quote each

Types 2 and 3 are the same rule at different widths -- N characters speaking
once each. Only type 1 is one character speaking more than once, which is why
it is the only shape that needs its own entry rather than falling out of the
rule.

The type cycle is deliberately *not* in phase with the character cycle. Both
advance one step per video, so a type sequence of 1,2,3,1,2,3 alongside a
character cycle of trump,tate,arnold would repeat every three videos and a
viewer four videos in could predict the whole schedule. Offsetting the two makes
the run look varied while staying deterministic.

Nothing here touches the network or the filesystem: planning a batch is pure, so
it can be checked without spending a single token or second of TTS.
"""
from __future__ import annotations

from dataclasses import dataclass

# Repeating cycle of video shapes. Balanced 2/2/2, and no shape ever repeats
# back to back -- including across the wrap from 2 back to 1.
TYPE_CYCLE = (1, 2, 3, 1, 3, 2)

TYPE_SHAPES = {
    1: {"characters": 1, "quotes_each": 3},
    2: {"characters": 3, "quotes_each": 1},
    3: {"characters": 2, "quotes_each": 1},
}

# The order characters cycle in. This is *not* the registry order: the registry
# happens to list donald-trump, arnold-schwarzenegger, andrew-tate, and cycling
# that would put Brolexander second. Any enabled voice missing from this list is
# appended in registry order, so adding a voice cannot silently drop it.
CYCLE = ("donald-trump", "andrew-tate", "arnold-schwarzenegger")


@dataclass(frozen=True)
class VideoPlan:
    """One video's slot in a batch.

    characters is ordered lead-first. The lead is whoever the character cycle
    lands on for this video; the rest follow cycle order from there, so a
    three-character video always starts on the character the viewer came for.
    """

    index: int
    type_id: int
    characters: tuple
    quotes_each: int

    @property
    def total_quotes(self) -> int:
        return len(self.characters) * self.quotes_each

    @property
    def lead(self) -> str:
        return self.characters[0]

    def __str__(self) -> str:
        shape = (f"{len(self.characters)} char x {self.quotes_each} quote"
                 if len(self.characters) > 1 else f"1 char x {self.quotes_each} quotes")
        return f"type {self.type_id} ({shape}): {', '.join(self.characters)}"


def character_cycle(enabled) -> tuple:
    """The cycle order, restricted to the voices that are actually enabled.

    A disabled voice still holds its slot in CYCLE, so enabling one later
    resumes the sequence where it was rather than shifting everything.
    """
    enabled = list(enabled)
    ordered = [v for v in CYCLE if v in enabled]
    ordered += [v for v in enabled if v not in ordered]
    return tuple(ordered)


def plan_batch(count: int, enabled, start: int = 0) -> list:
    """Plan ``count`` videos, continuing from batch offset ``start``.

    ``start`` lets a later run pick up mid-cycle instead of replaying the same
    opening types, which is what makes two consecutive batches look like one
    continuous channel rather than the same ten videos twice.
    """
    cycle = character_cycle(enabled)
    if not cycle:
        raise ValueError("no enabled voices to plan a batch with")
    if count < 0:
        raise ValueError(f"count must not be negative, got {count}")

    plans = []
    for i in range(count):
        step = start + i
        type_id = TYPE_CYCLE[step % len(TYPE_CYCLE)]
        shape = TYPE_SHAPES[type_id]
        # A shape asking for more characters than exist is clamped rather than
        # refused: with two voices enabled, a three-character video is still a
        # perfectly good two-character video, and failing the whole batch over
        # it would be worse than running the narrower version.
        width = min(shape["characters"], len(cycle))
        lead = step % len(cycle)
        characters = tuple(cycle[(lead + k) % len(cycle)] for k in range(width))
        plans.append(VideoPlan(
            index=i + 1,
            type_id=type_id,
            characters=characters,
            quotes_each=shape["quotes_each"],
        ))
    return plans


def with_lead(plan: VideoPlan, voice: str) -> VideoPlan:
    """The same slot, but spoken by ``voice`` instead of the cycle's choice.

    ``--voice`` is an override of who leads, not of what shape the video is: a
    three-character video stays three characters. Only the lead moves, and the
    co-stars are re-derived from the override so the cast is not somebody
    followed by a cast that was picked around them.
    """
    if voice == plan.lead:
        return plan
    rest = [c for c in plan.characters[1:]]
    return VideoPlan(
        index=plan.index,
        type_id=plan.type_id,
        characters=(voice,) + tuple(rest),
        quotes_each=plan.quotes_each,
    )
