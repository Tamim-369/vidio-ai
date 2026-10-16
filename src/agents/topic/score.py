"""Virality scoring for leads: specificity, freshness, authority.

Formula (transparent, tunable via weights below):

  score = 100 * (0.20*authority + 0.15*freshness + 0.30*specificity
                 + 0.35*engagement)

  authority   — how much the source is trusted for the niche (0..1).
  freshness   — recency decay; history niches age over ~60d. Missing dates
                imply a freshly-crawled page → treated as fresh.
  specificity — density of factual tokens (years, counts, %, proper nouns)
                in the available text; topic-finding wants numbers & names.
  engagement  — raw Reddit upvotes on a log scale (0..1). This is the only
                popularity signal Reddit exposes — there is no view count —
                so "top most viewed stories" is enforced with it.
"""
from __future__ import annotations

import math
import re
from datetime import datetime, timezone

from src.agents.topic.leads import Lead
from src.agents.topic.sources import AUTHORITY

_HISTORY_HALF_LIFE_H = 60 * 24  # history stories age slowly
_YEAR_HALF_LIFE_H = 365 * 24  # year-window reddit posts stay rankable

_FACT_YEAR = re.compile(r"\b(1[6-9]\d{2}|20\d{2})\b")
_FACT_NUM = re.compile(r"\d[\d,]*\.?\d*\s*(?:%|km|miles?|kg|tons?|deaths?|soldiers?|troops?|patients?|victims?|men|planes?|ships?|tanks?)", re.I)
_FACT_MONEY = re.compile(r"\$\s?\d[\d,]*")
_CAP_NAMES = re.compile(r"\b[A-Z][a-z]{2,}(?:\s[A-Z][a-z]+){1,3}\b")


def _specificity(lead: Lead) -> float:
    text = f"{lead.title} {lead.summary} {(lead.body or '')[:2000]}"
    facts = 0
    facts += len(_FACT_YEAR.findall(text))
    facts += len(_FACT_NUM.findall(text))
    facts += len(_FACT_MONEY.findall(text))
    facts += len(_CAP_NAMES.findall(text))
    # Reasonable stories land 2–6 facts; anything past 6 is all yield.
    return min(1.0, facts / 6)


def _freshness(lead: Lead, half_life_h: float) -> float:
    published = lead.published or lead.collected_at
    if published is None:
        return 1.0
    if published.tzinfo is None:
        published = published.replace(tzinfo=timezone.utc)
    age_h = (datetime.now(timezone.utc) - published).total_seconds() / 3600
    if age_h <= 0:
        return 1.0
    return 2 ** (-age_h / half_life_h)


def _engagement(lead: Lead) -> float:
    """Raw upvotes -> 0..1 on a log scale; 100k votes saturates."""
    if lead.engagement <= 0:
        return 0.0
    return min(1.0, math.log10(1 + lead.engagement) / 5.0)


def score_lead(lead: Lead) -> tuple[int, dict]:
    """Return (score, breakdown) for a single lead."""
    freshest = {
        "experiments": _HISTORY_HALF_LIFE_H,
        "dark_legends": _HISTORY_HALF_LIFE_H,
        "ww1_ww2_stories": _HISTORY_HALF_LIFE_H,
        # The story subs are crawled on a t=year window, so a top post is
        # routinely months old. Decaying that at the 60-day history rate
        # would rank the whole window near zero and make the crawl useless.
        "frontier_history": _YEAR_HALF_LIFE_H,
        "true_crime": _YEAR_HALF_LIFE_H,
        "ancient_world": _YEAR_HALF_LIFE_H,
        "history_debate": _YEAR_HALF_LIFE_H,
    }.get(lead.niche, _HISTORY_HALF_LIFE_H)

    authority = AUTHORITY.get(lead.source_key, 0.5)
    specificity = _specificity(lead)
    freshness = _freshness(lead, freshest)
    engagement = _engagement(lead)

    # Engagement carries the largest weight: within a niche every lead shares
    # one authority value and a t=year window barely moves freshness, so this
    # is the term that makes "top most viewed" mean anything.
    raw = 100 * (0.20 * authority + 0.15 * freshness + 0.30 * specificity
                 + 0.35 * engagement)
    breakdown = {
        "authority": round(authority, 3),
        "freshness": round(freshness, 3),
        "specificity": round(specificity, 3),
        "engagement": round(engagement, 3),
    }
    return round(raw), breakdown