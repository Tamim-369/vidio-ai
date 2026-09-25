"""Orchestrates the topic agent: crawl sources -> dedupe -> score -> batch.

Writes a batch file in the exact shape the pipeline expects
(topics/batch_YYYYMMDD_HHMMSS.json, list of topic dicts read by
load_latest_topics()). Selection is niche-weighted so the crazy experiment
pillar leads without starving dark legends and WW1/WW2 war stories.

Standalone use:
    uv run python -m src.services.topic_agent.agent --target 12
    uv run python -m src.services.topic_agent.agent --niche experiments --target 6
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime

from src.services.topic_agent import collectors, sources
from src.services.topic_agent.dedupe import dedupe_leads_leads, filter_unseen, load_used_titles
from src.services.topic_agent.leads import Lead, lead_to_topic
from src.services.topic_agent.score import score_lead

BATCH_DIR = os.path.join("topics")

NICHE_WEIGHTS = {
    "experiments": 0.45,
    "dark_legends": 0.35,
    "ww1_ww2_stories": 0.20,
}


def run_topic_agent(target: int = 24, niches: list | None = None) -> list:
    """Full agent run. Returns the selected topic dicts (also saves a batch)."""
    print("\n🕸️  Topic agent: crawling source inventory...")
    counts: Counter = Counter()
    per_niche: dict = {}

    def work(spec: dict) -> list:
        return collectors.collect(spec)

    with ThreadPoolExecutor(max_workers=5) as pool:
        futures = {pool.submit(work, spec): spec for spec in sources.SOURCES}
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

    used = load_used_titles()

    if niches:
        targets = {n: w for n, w in NICHE_WEIGHTS.items() if n in niches}
    else:
        targets = dict(NICHE_WEIGHTS)
    if not targets:
        targets = dict(NICHE_WEIGHTS)

    picked: list = []
    for niche, weight in targets.items():
        niche_leads = per_niche.get(niche, [])
        niche_leads = dedupe_leads_leads(niche_leads)
        niche_leads = filter_unseen(niche_leads, used)
        scored = [(score_lead(lead), lead) for lead in niche_leads]
        scored.sort(key=lambda pair: pair[0][0], reverse=True)
        quota = max(1, round(weight * target))
        taken = scored[:quota]
        for (score, breakdown), lead in taken:
            angle = _vet(lead)
            if angle is None:
                continue
            topic = lead_to_topic(lead, angle=angle, score=score, niche=niche)
            topic["confidence"] = 5 if lead.body else 3
            picked.append(topic)
        print(f"    {niche:<14} kept {len(taken)}/{len(niche_leads)} unseen scored")

    picked.sort(key=lambda t: t["score"], reverse=True)
    picked = picked[:target]

    if picked:
        os.makedirs(BATCH_DIR, exist_ok=True)
        path = os.path.join(BATCH_DIR, f"batch_{datetime.now():%Y%m%d_%H%M%S}.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump(picked, f, ensure_ascii=False, indent=2)
        print(f"\n📦 Wrote {len(picked)} topics -> {path}")
    else:
        print("\n⚠️  Topic agent found no usable topics.")
    return picked


def _vet(lead: Lead) -> str | None:
    """Angle for the topic, or None if off-niche/locked. Reuses pipeline rules."""
    from src.services.research_pipeline import _vet_idea

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