"""Virality scoring for leads: specificity, freshness, authority.

Formula (transparent, tunable via weights below):

  score = 100 * (0.35*authority + 0.35*freshness + 0.30*specificity)

  authority   — how much the source is trusted for the niche (0..1).
  freshness   — recency decay; history niches age over ~60d. Missing dates
                imply a freshly-crawled page → treated as fresh.
  specificity — density of factual tokens (years, counts, %, proper nouns)
                in the available text; topic-finding wants numbers & names.
"""
from __future__ import annotations

import re
from datetime import datetime, timezone

from src.services.topic_agent.leads import Lead
from src.services.topic_agent.sources import AUTHORITY

_HISTORY_HALF_LIFE_H = 60 * 24  # history stories age slowly

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


def score_lead(lead: Lead) -> tuple[int, dict]:
    """Return (score, breakdown) for a single lead."""
    freshest = {
        "experiments": _HISTORY_HALF_LIFE_H,
        "dark_legends": _HISTORY_HALF_LIFE_H,
        "ww1_ww2_stories": _HISTORY_HALF_LIFE_H,
    }.get(lead.niche, _HISTORY_HALF_LIFE_H)

    authority = AUTHORITY.get(lead.source_key, 0.5)
    specificity = _specificity(lead)
    freshness = _freshness(lead, freshest)

    raw = 100 * (0.35 * authority + 0.35 * freshness + 0.30 * specificity)
    breakdown = {
        "authority": round(authority, 3),
        "freshness": round(freshness, 3),
        "specificity": round(specificity, 3),
    }
    return round(raw), breakdown