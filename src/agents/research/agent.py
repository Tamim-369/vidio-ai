"""Research agent: sources fresh material per niche, extracts structured video
ideas with the local model, dedups, scores, and writes the approved queue to
topics/batch_*.json — the same queue the video pipeline reads.

Two paths live here:
  * run_research_pipeline()  — deep: Wikipedia categories + search per niche,
    chunked local-LLM idea extraction, channel mining, quota-balanced selection.
  * generate_first_topics()  — light: one cheap LLM call per niche proposes
    titles; we take the FIRST `target` titles we have not already made.

Deliberately standalone: run on its own schedule, with a human review gate
between idea generation and video production.
"""
import json
import os
import time

from src.agents.common.llm import _local as _llm
from src.agents.research.helpers import (
    IDEAS_CHUNK_SOURCES,
    IDEAS_SOURCE_CHARS,
    LIGHT_SUBJECTS,
    NICHES,
    TOPICS_OUTPUT_DIR,
    _fetch_niche_sources,
    _parse_idea_json,
    _parse_titles,
    _score_idea,
    _vet_idea,
)
from src.agents.research.prompts import IDEAS_PROMPT, LIGHT_IDEAS_PROMPT
from src.agents.topic.helpers import _is_duplicate, _load_used, _scan_made_videos
from src.agents.topic.miner import fetch_channel_candidates


def _generate_ideas(niche: str, sources: list, max_ideas: int = 10) -> list:
    """Chunked LLM calls per niche window -> deduped structured idea list.

    One small call per source-chunk (not one giant call): each request stays
    under the runtime's token ceiling, and a broken/rate-limited chunk can be
    skipped without losing the rest of the window.
    """
    if not sources:
        return []

    ideas, seen = [], set()
    for start in range(0, len(sources), IDEAS_CHUNK_SOURCES):
        chunk = sources[start:start + IDEAS_CHUNK_SOURCES]
        raw = "\n\n".join(
            f"[{i}] {s['title']}\n{s['content'][:IDEAS_SOURCE_CHARS]}\nSource: {s['url']}"
            for i, s in enumerate(chunk, 1)
        )
        prompt = IDEAS_PROMPT.format(
            niche=NICHES[niche]["prompt"], raw=raw
        )
        try:
            raw_out = _llm(prompt, temperature=0.7, tag="research")
        except Exception as e:
            print(f"    [llm] chunk {start//IDEAS_CHUNK_SOURCES + 1} failed: {str(e)[:80]}")
            continue

        # Retry once with repair prompt if JSON parse fails
        parsed = _parse_idea_json(raw_out)
        if not parsed:
            repair_prompt = (
                "Your previous response was not valid JSON. Output ONLY a valid JSON array "
                "matching the schema exactly. No prose, no markdown fences."
            )
            try:
                raw_out = _llm(
                    f"{prompt}\n\n(previous response: {raw_out[:800]})\n{repair_prompt}",
                    temperature=0.3,  # Lower temp for structured repair
                    tag="research",
                )
                parsed = _parse_idea_json(raw_out)
            except Exception as e:
                print(f"    [llm] repair attempt failed: {str(e)[:80]}")

        for idea in parsed:
            key = idea["title"].strip().lower()
            if key in seen:
                continue
            seen.add(key)
            ideas.append(idea)
        if len(ideas) >= max_ideas:
            break
        time.sleep(1.5)
    return ideas[:max_ideas]


def _propose_topics(niche: str, k: int = 8) -> list:
    """ONE LLM call for one niche -> list of {"title": ...} candidates."""
    prompt = LIGHT_IDEAS_PROMPT.format(subject=LIGHT_SUBJECTS.get(niche, niche), k=k)
    try:
        raw = _llm(prompt, temperature=0.8, tag="research")
    except Exception as e:
        print(f"    [light] idea call failed for {niche}: {str(e)[:90]}")
        return []
    titles = _parse_titles(raw)
    if not titles:
        if raw:
            print(f"    [light] unparseable reply for {niche}: {raw[:120]!r}")
        else:
            print(f"    [light] empty reply for {niche}")
        try:
            raw = _llm(
                f"{prompt}\n\n(previous response: {raw[:800]})\nReturn ONLY a JSON array of strings, nothing else.",
                temperature=0.3,
                tag="research",
            )
            titles = _parse_titles(raw)
            if not titles and raw:
                print(f"    [light] still unparseable for {niche}: {raw[:120]!r}")
        except Exception as e:
            print(f"    [light] repair failed for {niche}: {str(e)[:90]}")
            return []
    return [{"title": t} for t in titles]


def _save_batch(topics: list, caption: str = "Selected") -> str:
    """Print the picked list and persist it to topics/batch_<timestamp>.json.

    Shared by both paths (first-topics and deep research) so the queue on disk
    and the summary printed for the human gate stay identical.
    """
    print(f"\n{caption} {len(topics)} topics:")
    for i, t in enumerate(topics, 1):
        angle = f"{t.get('angle', '?'):>8}" if t.get("angle") else ""
        score = f"  (score {t['score']})" if "score" in t else ""
        print(f"{i:2d}. [{t['category'][:16]:16} |{angle}] {t['title']}{score}")

    batch_file = os.path.join(
        TOPICS_OUTPUT_DIR, f"batch_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(batch_file, "w") as f:
        json.dump(topics, f, indent=2)
    print(f"\nQueue saved to {batch_file}")
    return batch_file


def _accept_topic(prop: dict, niche: str, used: list) -> dict | None:
    """Vet one proposed topic; returns a pipeline-standard topic or None."""
    title = prop["title"]
    if _is_duplicate(title, used, fuzzy=True):
        print(f"  skip (already made): {title}")
        return None
    vetted = _vet_idea({**prop, "title": title})
    if vetted is None:
        print(f"  skip (vet-block): {title}")
        return None
    return {
        "title": title,
        "hook": title,
        "category": niche,
        "angle": vetted["angle"],
        "source_urls": [],
        "summary": title,
        "content": f"{title}\n(story researched on demand at script time)",
        "novelty_score": 50,
        "est_video_length_min": 3,
        "source": f"research:first:{niche}",
        "score": 50,
        "upvote_ratio": 1.0,
        "confidence": 5,
        "outline": [],
    }


def generate_first_topics(target: int = 2, k_per_niche: int = 8) -> list:
    """Pick the FIRST `target` topics we haven't already made. No ranking.

    Walks niches in order, one cheap LLM call each for candidate titles, stops
    as soon as `target` fresh (not-already-made) titles are collected. Writes
    the same batch_*.json the pipeline reads.
    """
    os.makedirs(TOPICS_OUTPUT_DIR, exist_ok=True)
    used = list(dict.fromkeys(_load_used() + _scan_made_videos()))

    accepted, seen_titles = [], []
    for niche in NICHES:
        if len(accepted) >= target:
            break
        print(f"\n=== {niche} ===")
        proposals = _propose_topics(niche, k=max(k_per_niche, target))
        fresh = 0
        for prop in proposals:
            if len(accepted) >= target:
                break
            topic = _accept_topic(prop, niche, used + seen_titles)
            if topic is None:
                continue
            accepted.append(topic)
            seen_titles.append(topic["title"])
            fresh += 1
        print(f"  +{fresh} fresh (total {len(accepted)}/{target})")

    if not accepted:
        print("\n  No fresh topics — run the deep research pipeline instead.\n")
        return []

    _save_batch(accepted)
    return accepted


def _topic_from_idea(vetted: dict, niche: str, score: int) -> dict:
    """Build a pipeline-standard topic dict from a vetted research idea."""
    hook = vetted["hook"] or vetted["why_viral"]
    content = "\n\n".join([
        hook,
        vetted["why_viral"],
        "Outline: " + " | ".join(vetted["outline"]),
    ])
    return {
        "title": vetted["title"],
        "hook": hook,
        "category": niche,
        "angle": vetted["angle"],
        "source_urls": [vetted["source_url"]] if vetted["source_url"] else [],
        "summary": vetted["why_viral"],
        "content": content,
        "novelty_score": score,
        "est_video_length_min": 3,
        "source": f"research:{niche}",
        "score": score,
        "upvote_ratio": 1.0,
        "confidence": vetted["confidence"],
        "outline": vetted["outline"],
    }


def _collect_niche_topics(niche: str, used: list) -> list:
    """Fetch sources for one niche, run idea passes, vet + score into topics."""
    print(f"\n=== {niche} ===")
    print("  fetching sources...")
    sources = _fetch_niche_sources(niche, limit=25)
    print(f"  {len(sources)} source articles")

    # Two passes over different source windows to multiply fresh ideas;
    # dedup within the niche so we don't repeat the same story.
    niche_ideas = []
    windows = [sources[:15], sources[15:]] if len(sources) > 15 else [sources]
    for wi, window in enumerate(windows, 1):
        if not window:
            continue
        print(f"  pass {wi}/{len(windows)}: {len(window)} sources -> LLM...")
        ideas = _generate_ideas(niche, window, max_ideas=10)
        fresh = 0
        for idea in ideas:
            if any(_is_duplicate(idea["title"], [n["title"]]) for n in niche_ideas):
                continue
            niche_ideas.append(idea)
            fresh += 1
        print(f"    +{fresh} fresh ideas")
        if fresh == 0:
            break  # model is repeating itself; stop burning calls
    print(f"  {len(niche_ideas)} fresh niche ideas")

    topics = []
    for idea in niche_ideas:
        if _is_duplicate(idea["title"], used, fuzzy=True):
            print(f"    skip (dup): {idea['title']}")
            continue
        vetted = _vet_idea(idea)
        if vetted is None:
            print(f"    skip (vet-block): {idea['title']}")
            continue
        topics.append(_topic_from_idea(
            vetted, niche, _score_idea(vetted, {})
        ))
    return topics


def _merge_channel_candidates(topics: list, used: list) -> None:
    """Append cached channel-mined seeds (proven viral, cheap). In place."""
    print("\n=== channel mining (cached) ===")
    try:
        for cand in fetch_channel_candidates(used_titles=used):
            topics.append({
                "title": cand["title"],
                "hook": cand["content"][:100].strip(),
                "category": cand["category"],
                "angle": cand.get("category", "mystery"),
                "source_urls": cand["source_urls"],
                "summary": cand["content"][:200],
                "content": cand["content"],
                "novelty_score": 100,
                "est_video_length_min": 4,
                "source": cand["source"],
                "score": cand["score"],
                "upvote_ratio": 1.0,
                "confidence": 10,
            })
    except Exception as e:
        print(f"  [miner] skipped: {e}")


def _balance_quota(topics: list, target: int, min_per_niche: int) -> list:
    """Pick `target` topics with per-niche floors and a bucket ceiling.

    Pass 1 guarantees each research niche its floor (min_per_niche + bonus);
    pass 2 fills the rest by score while no bucket exceeds ~40% of target.
    The experiments pillar is the money niche but the cap still forces the
    legends and WW1/WW2-story pillars to get airtime.
    """
    per_niche_floor = {
        n: min_per_niche + NICHES[n].get("floor_bonus", 0) for n in NICHES
    }
    max_bucket = max(1, int(target * 0.4))

    selected, counts = [], {}
    # Pass 1: guarantee each research niche its floor.
    for t in sorted(topics, key=lambda t: t["score"], reverse=True):
        if len(selected) >= target:
            break
        bucket = t["category"]
        if bucket in NICHES and counts.get(bucket, 0) < per_niche_floor[bucket]:
            selected.append(t)
            counts[bucket] = counts.get(bucket, 0) + 1

    # Pass 2: fill the rest by score, enforcing the bucket ceiling.
    for t in topics:
        if len(selected) >= target:
            break
        bucket = t["category"]
        if counts.get(bucket, 0) >= max_bucket:
            continue
        selected.append(t)
        counts[bucket] = counts.get(bucket, 0) + 1

    if len(selected) < target:
        print(f"\n  Note: only {len(selected)} unique ideas this run (need {target}). "
              "Rerun to mine deeper, or raise --target on the next pass.")
    return selected


def run_research_pipeline(target: int = 24, min_per_niche: int = 6) -> list:
    """Run the standalone research pipeline and write topics/batch_*.json.

    Returns the approved topic list (same shape as run_topic_generation output)
    so main.py --use-saved picks it up unchanged.
    """
    os.makedirs(TOPICS_OUTPUT_DIR, exist_ok=True)
    used = list(dict.fromkeys(_load_used() + _scan_made_videos()))

    topics = []
    for niche in NICHES:
        topics += _collect_niche_topics(niche, used)

    _merge_channel_candidates(topics, used)

    if not topics:
        print("\n  No ideas generated. Try again or widen sources.\n")
        return []

    # Rank: proven-viral channel seeds first, then LLM ideas by score.
    topics.sort(key=lambda t: t["score"], reverse=True)
    selected = _balance_quota(topics, target, min_per_niche)

    _save_batch(selected)
    return selected