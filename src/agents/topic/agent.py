"""Orchestrates the topic agent: crawl sources -> dedupe -> score -> batch.

Writes a batch file in the exact shape the pipeline expects
(topics/batch_YYYYMMDD_HHMMSS.json, list of topic dicts read by
load_latest_topics()). Selection is niche-weighted so the crazy experiment
pillar leads without starving dark legends and WW1/WW2 war stories.

Standalone use:
    uv run python -m src.agents.topic.agent --target 12
    uv run python -m src.agents.topic.agent --niche experiments --target 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from src.agents.topic import collectors, sources
from src.agents.topic.dedupe import dedupe_leads_leads, filter_unseen, load_used_titles
from src.agents.topic.leads import Lead, lead_to_topic
from src.agents.topic.helpers import _passes_filters
from src.agents.topic.score import score_lead
from src.agents.topic.wikipedia import has_source_imagery

# Niches the research pipeline still defines, so an explicit --niche keeps
# working for pillars that NICHE_WEIGHTS has parked.
_RESEARCH_NICHES = {"experiments", "dark_legends", "ww1_ww2_stories"}

BATCH_DIR = os.path.join("topics")

# The channel's three current pillars, weighted by proven topic shape:
# a named subject, physical stakes, truth-vs-lies framing. The legacy
# WW2/experiments/legends pillars are carried at zero weight so the old
# sources still work via an explicit `--niche` override, but they never
# take a slot in a default run.
NICHE_WEIGHTS = {
    "frontier_history": 0.34,   # r/history: named subject + physical stakes
    "true_crime": 0.34,         # r/TrueCrime, r/UnresolvedMysteries
    "ancient_world": 0.32,      # r/AncientHistory, r/medievalhistory
    "experiments": 0.0,
    "ww1_ww2_stories": 0.0,
    "dark_legends": 0.0,
}


def _active_niches(niches: list | None) -> set:
    """Niches a run can actually select from (an explicit pick re-enables a
    pillar the default weights have parked)."""
    targets = _niche_targets(niches)
    return {n for n, w in targets.items() if w > 0}


def _crawl_all_sources(niches: list | None = None) -> tuple[Counter, dict]:
    """Collect from every live SOURCES spec in parallel.

    Specs for parked pillars are skipped rather than crawled and discarded —
    that is a dozen RSS/index/MediaWiki fetches per run spent collecting
    leads no quota will ever look at.
    Returns (counts, per_niche).
    """
    counts: Counter = Counter()
    per_niche: dict = {}
    active = _active_niches(niches)
    specs = [spec for spec in sources.SOURCES if spec["niche"] in active]
    skipped = len(sources.SOURCES) - len(specs)
    if skipped:
        print(f"    ({skipped} spec(s) skipped — niche parked)")

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(collectors.collect, spec): spec for spec in specs}
        for fut in as_completed(futures):
            spec = futures[fut]
            try:
                ls = fut.result()
            except Exception:
                ls = []
            counts[spec["key"]] = len(ls)
            for lead in ls:
                per_niche.setdefault(spec["niche"], []).append(lead)

    for key, n in counts.most_common():
        print(f"    {key:<16} {n:>3} leads")
    print(f"    {sum(counts.values())} raw leads")
    return counts, per_niche


def _niche_targets(niches: list | None) -> dict:
    """Weight map restricted to `niches` when given; falls back to all."""
    if niches:
        # Asking for a niche by name re-enables it even when its default
        # weight is parked at zero (the legacy WW2 pillars).
        targets = {n: (NICHE_WEIGHTS.get(n, 0) or 1.0)
                   for n in niches if n in NICHE_WEIGHTS or n in _RESEARCH_NICHES}
    else:
        targets = {n: w for n, w in NICHE_WEIGHTS.items()}
    return targets or dict(NICHE_WEIGHTS)


def _select_from_niche(niche: str, leads: list, used: set, quota: int) -> list:
    """Score/dedupe/filter one niche's leads, keep the best `quota` as topics.

    Every candidate also has to clear the source-imagery gate: a subject
    Commons does not picture cannot be rendered correctly, and finding that
    out after research costs a full video's worth of work. The gate is
    positive-only (it blocks what it can prove, never what it cannot judge),
    and results are cached so a retry is free.
    """
    niche_leads = dedupe_leads_leads(leads)
    niche_leads = filter_unseen(niche_leads, used)
    before = len(niche_leads)
    niche_leads = [l for l in niche_leads if _lead_passes(l)]
    dropped = before - len(niche_leads)
    if dropped:
        print(f"    [{niche}] {dropped} lead(s) filtered out "
              f"(meta post, low score, off-niche or too thin)")
    scored = [(score_lead(lead), lead) for lead in niche_leads]
    scored.sort(key=lambda pair: pair[0][0], reverse=True)

    picked = []
    gated = 0
    imagery_cache: dict = {}
    # A generous but bounded search: vet + imagery rejections can cost
    # several candidates each, but the gate is two API calls apiece so the
    # scan must not run the whole list.
    for (score, breakdown), lead in scored[: max(quota * 3, quota + 6)]:
        if len(picked) >= quota:
            break
        angle = _vet(lead)
        if angle is None:
            continue
        if not has_source_imagery(lead.title, imagery_cache):
            gated += 1
            continue
        topic = lead_to_topic(lead, angle=angle, score=score, niche=niche)
        topic["confidence"] = 5 if lead.body else 3
        picked.append(topic)
    if gated:
        print(f"    [imagery] {gated} topic(s) had no free imagery — skipped")
    return picked


def _write_batch(picked: list) -> str | None:
    """Persist the selection; returns the path (or None when nothing picked)."""
    if not picked:
        print("\n⚠️  Topic agent found no usable topics.")
        return None
    os.makedirs(BATCH_DIR, exist_ok=True)
    path = os.path.join(BATCH_DIR, f"batch_{datetime.now():%Y%m%d_%H%M%S}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(picked, f, ensure_ascii=False, indent=2)
    print(f"\n📦 Wrote {len(picked)} topics -> {path}")
    return path


def run_topic_agent(target: int = 24, niches: list | None = None) -> list:
    """Full agent run. Returns the selected topic dicts (also saves a batch)."""
    print("\n🕸️  Topic agent: crawling source inventory...")
    _counts, per_niche = _crawl_all_sources(niches)

    used = load_used_titles()
    picked: list = []
    for niche, weight in _niche_targets(niches).items():
        if weight <= 0:
            continue
        quota = max(1, round(weight * target))
        kept = _select_from_niche(niche, per_niche.get(niche, []), used, quota)
        picked += kept
        print(f"    {niche:<14} kept {len(kept)}/{len(per_niche.get(niche, []))} unseen scored")

    picked.sort(key=lambda t: t["score"], reverse=True)
    picked = picked[:target]
    _write_batch(picked)
    return picked


def _lead_passes(lead: Lead) -> bool:
    """Apply the shared topic filters to a lead.

    Only reddit leads are filtered: every other spec is a curated feed whose
    niche is already the filter, whereas the story subs mix real stories with
    polls, link-drops and "what's your favourite" threads that score well on
    upvotes and are worthless as videos.
    """
    if not lead.source_name.startswith("reddit:"):
        return True
    return _passes_filters({
        "source": lead.source_name,
        "title": lead.title,
        "content": lead.body or lead.title,
        "score": lead.engagement,
        "upvote_ratio": lead.upvote_ratio,
    })


def _vet(lead: Lead) -> str | None:
    """Angle for the topic, or None if off-niche/locked. Reuses pipeline rules."""
    from src.agents.research.helpers import _vet_idea

    out = _vet_idea({"title": lead.title, "niche": lead.niche})
    return out.get("angle") if isinstance(out, dict) else None


def main() -> None:
    if os.path.basename(sys.argv[0]).startswith("src"):
        sys.path.insert(0, os.path.dirname(sys.argv[0]) + "/../")
    parser = argparse.ArgumentParser(description="Topic agent (sourced topic generation)")
    parser.add_argument("--target", type=int, default=24, help="Topics to select (default 24)")
    parser.add_argument("--niche", nargs="+", default=None,
                        help="Restrict to niche(s): frontier_history true_crime "
                             "ancient_world (legacy: experiments dark_legends "
                             "ww1_ww2_stories)")
    args = parser.parse_args()
    run_topic_agent(target=args.target, niches=args.niche)


if __name__ == "__main__":
    main()