"""Unified lead model plus the helpers that turn raw HTML/RSS/JSON into it.

A Lead is a discovered story: a title, a canonical URL back to the source,
a snippet, and — when the collector fetched the article body — enough
long-form text to score specificity and to seed the script-later.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

MAX_BODY_CHARS = 6000

_WS = re.compile(r"\s+")


def clean_text(text: str) -> str:
    """Collapse whitespace and strip markdown-ish cruft from scraped text."""
    if not text:
        return ""
    text = text.replace("\u00a0", " ")
    text = _WS.sub(" ", text)
    return text.strip()


def first_paragraph(text: str, words: int = 170) -> str:
    """First ~`words` words of the first real paragraph, for summaries."""
    text = clean_text(text)
    if not text:
        return ""
    toks = text.split()
    return " ".join(toks[:words])


@dataclass
class Lead:
    title: str
    url: str
    source_key: str
    source_name: str
    niche: str
    published: Optional[datetime] = None
    summary: str = ""
    body: str = ""
    collected_at: Optional[datetime] = None

    def __post_init__(self) -> None:
        self.title = clean_text(self.title)[:300]
        if len(self.body) > MAX_BODY_CHARS:
            self.body = self.body[:MAX_BODY_CHARS] + " …"


def lead_to_topic(lead: Lead, angle: str, score: int, niche: str) -> dict:
    """Map a lead into the batch_*.json topic dict the pipeline consumes."""
    return {
        "title": lead.title,
        "hook": lead.title,
        "category": niche,
        "angle": angle,
        "source_urls": [lead.url],
        "summary": lead.summary or lead.title,
        "content": lead.body or f"{lead.title}\n(no body captured)",
        "novelty_score": 50,
        "est_video_length_min": 3,
        "source": f"topicagent:{lead.source_key}",
        "score": score,
        "upvote_ratio": 1.0,
        "confidence": 5,
        "outline": [],
        "lead_url": lead.url,
    }