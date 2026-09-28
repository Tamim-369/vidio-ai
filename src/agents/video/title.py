"""The one line of metadata a video is named by.

Kept in its own module because two very different callers need it and neither
should import the other: the pipeline builds a title when it makes a video, and
the publisher builds one from an arbitrary script. Both must agree on the limit,
so there is exactly one definition of it here.
"""
from __future__ import annotations

from src.agents.voice_cast.voices import VOICES

# YouTube rejects a title over 100 characters with invalidArgument, so this is a
# hard API limit rather than a style choice. Not the feed's ~40-character display
# window, though: a title built from a real quote runs 60-111 characters, and
# cutting at 40 would slice off the punchline, which is the second sentence of
# every quote this channel writes. YouTube truncates the display itself and
# indexes the full string, so only the API limit needs respecting.
TITLE_MAX_CHARS = 100


def build_title(quote: str, voice_id: str = "", name: str = "") -> str:
    """Title for a video: the channel's pseudonym, then the quote.

    Built from the quote rather than a template, which makes it naturally varied
    -- each video's opening line is its own, so titles do not read as one
    repeated string.

    The pseudonym is the channel's, not the real name. These are fabricated
    quotes in the voice of living public figures, so the parody name is the one
    that belongs in metadata; "Don Tzu: Paying bills is stressful..." both stays
    honest and is the string people actually search for on this channel.

    `name` lets a caller that already holds the pseudonym pass it in rather than
    re-deriving it from an id. The publisher needs that: it reads lines that
    carry the voice config directly, and a config has no "id" key, so resolving
    by id alone would drop the name from the title while the description built
    from the same line kept it.
    """
    who = name or VOICES.get(voice_id, {}).get("pseudonym") or _short_name(voice_id)
    title = f"{who}: {quote}".strip() if who else quote.strip()
    return shorten_title(title)


def shorten_title(title: str) -> str:
    """Trim to the API limit, cutting on a word boundary.

    Cutting mid-word is the one option guaranteed to look wrong, and the
    pseudonym and first clause of the quote are what the reader actually sees,
    so those are what the boundary is protecting.
    """
    if len(title) <= TITLE_MAX_CHARS:
        return title
    room = TITLE_MAX_CHARS - 1
    head = title[:room]
    cut = head.rfind(" ")
    if cut > len(head) * 0.5:          # only bother if a boundary is near
        head = head[:cut]
    return head.rstrip(" ,.;:-") + "…"


def _short_name(vid: str) -> str:
    return (VOICES.get(vid, {}) or {}).get("short_name") or vid


# The channel's name as it appears in metadata. The registered channel is
# "Wordz Of Wizdom"; the real name of the person behind it is nobody's business
# in a title.
BRAND = "Wordz Of Wizdom"


def numbered_suffix(number: int) -> str:
    return f" | {BRAND} #{number}"


def build_numbered_title(quote: str, number: int) -> str:
    """Title for a published video: the quote, then the channel and its number.

    The pseudonym is deliberately absent. The suffix costs 25 characters of a
    100-character budget, and a real quote runs 58-98 characters, so a
    ``Don Tzu: `` prefix would leave the quote clipped to roughly 63. The quote
    is the hook and gets the room; the speaker is named in the description,
    where there is space for it.

    The quote is fitted to what is left *before* the suffix is added, rather than
    building the whole string and letting shorten_title() cut it. Cutting the
    finished title would land the word boundary inside the number, which is the
    one part of the string a viewer uses to find the video.
    """
    suffix = numbered_suffix(number)
    budget = TITLE_MAX_CHARS - len(suffix)
    return f"{_fit(quote, budget)}{suffix}"


def _fit(text: str, budget: int) -> str:
    """Trim to ``budget`` characters on a word boundary.

    An ellipsis is only worth the character when something was actually cut: a
    quote that happens to fit is returned untouched, so short quotes do not all
    end in a dangling dot.
    """
    text = (text or "").strip()
    if len(text) <= budget:
        return text
    room = budget - 1                      # leave room for the ellipsis
    head = text[:room]
    cut = head.rfind(" ")
    if cut > len(head) * 0.5:              # only bother if a boundary is near
        head = head[:cut]
    return head.rstrip(" ,.;:-") + "…"
