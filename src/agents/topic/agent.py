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
from src.agents.topic.score import score_lead

BATCH_DIR = os.path.join("topics")

NICHE_WEIGHTS = {
    "experiments": 0.45,
    "dark_legends": 0.35,
    "ww1_ww2_stories": 0.20,
}


def _crawl_all_sources() -> tuple[Counter, dict]:
    """Collect from every SOURCES spec in parallel. Returns (counts, per_niche)."""
    counts: Counter = Counter()
    per_niche: dict = {}

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(collectors.collect, spec): spec for spec in sources.SOURCES}
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
        targets = {n: w for n, w in NICHE_WEIGHTS.items() if n in niches}
    else:
        targets = dict(NICHE_WEIGHTS)
    return targets or dict(NICHE_WEIGHTS)


def _select_from_niche(niche: str, leads: list, used: set, quota: int) -> list:
    """Score/dedupe/filter one niche's leads, keep the best `quota` as topics."""
    niche_leads = dedupe_leads_leads(leads)
    niche_leads = filter_unseen(niche_leads, used)
    scored = [(score_lead(lead), lead) for lead in niche_leads]
    scored.sort(key=lambda pair: pair[0][0], reverse=True)

    picked = []
    for (score, breakdown), lead in scored[:quota]:
        angle = _vet(lead)
        if angle is None:
            continue
        topic = lead_to_topic(lead, angle=angle, score=score, niche=niche)
        topic["confidence"] = 5 if lead.body else 3
        picked.append(topic)
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
    _counts, per_niche = _crawl_all_sources()

    used = load_used_titles()
    picked: list = []
    for niche, weight in _niche_targets(niches).items():
        quota = max(1, round(weight * target))
        kept = _select_from_niche(niche, per_niche.get(niche, []), used, quota)
        picked += kept
        print(f"    {niche:<14} kept {len(kept)}/{len(per_niche.get(niche, []))} unseen scored")

    picked.sort(key=lambda t: t["score"], reverse=True)
    picked = picked[:target]
    _write_batch(picked)
    return picked


def _vet(lead: Lead) -> str | None:
    """Angle for the topic, or None if off-niche/locked. Reuses pipeline rules."""
    from src.agents.research.helpers import _vet_idea

    out = _vet_idea({"title": lead.title})
    return out.get("angle") if isinstance(out, dict) else None


def main() -> None:
    if os.path.basename(sys.argv[0]).startswith("src"):
        sys.path.insert(0, os.path.dirname(sys.argv[0]) + "/../")
    parser = argparse.ArgumentParser(description="Topic agent (sourced topic generation)")
    parser.add_argument("--target", type=int, default=24, help="Topics to select (default 24)")
    parser.add_argument("--niche", nargs="+", default=None,
                        help="Restrict to niche(s): experiments dark_legends ww1_ww2_stories")
    args = parser.parse_args()
    run_topic_agent(target=args.target, niches=args.niche)


if __name__ == "__main__":
    main()