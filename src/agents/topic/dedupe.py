"""Dedupe leads against each other and against everything already made.

Reuses the pipeline's ground truth: used_topics.json plus the titles of the
.mp4 files actually rendered (output/*.mp4), via the same fuzzy matcher the
rest of the repo uses so behavior is consistent end to end.
"""
from __future__ import annotations

from src.agents.topic.helpers import _is_duplicate, _load_used, _scan_made_videos
from src.agents.topic.leads import Lead


def load_used_titles() -> list:
    """All titles that must not be re-proposed: recorded + rendered files."""
    used = list(_load_used())
    used += _scan_made_videos()
    return used


def dedupe_leads_leads(leads: list[Lead]) -> list[Lead]:
    """Remove near-duplicate leads within the agent's own collected list."""
    kept: list[Lead] = []
    seen_titles = [l.title for l in leads[:1]] if leads else []
    for lead in leads:
        if _is_duplicate(lead.title, seen_titles, fuzzy=False):
            continue
        # Skip a lead whose title is a fuzzy hit on any already-kept one
        if any(_token_overlap(lead.title, t) >= 0.7 for t in seen_titles):
            continue
        seen_titles.append(lead.title)
        kept.append(lead)
    return kept


def _token_overlap(a: str, b: str) -> float:
    ta, tb = set(a.lower().split()), set(b.lower().split())
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / min(len(ta), len(tb))


def filter_unseen(leads: list[Lead], used: list) -> list[Lead]:
    """Keep only leads that are not already used/made."""
    return [lead for lead in leads if not _is_duplicate(lead.title, used)]