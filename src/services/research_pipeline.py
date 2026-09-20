"""Standalone content-research pipeline.

Sources fresh material per niche (ancient mysteries / WW2 secret experiments /
obscure secret history) from Wikipedia categories + search, feeds it to the
Ollama LLM (minimax-m3:cloud) to extract structured video ideas, dedups
against everything already produced, scores for novelty/angle-balance, and
writes an approved queue to topics/batch_*.json — the same queue the video
pipeline already reads.

Deliberately NOT wired into run_topic_generation: this is the standalone
"research" half, run on its own schedule, with a human review gate between
idea generation and video production.
"""
import json
import os
import re
import time

from src.services.llm import call_text
from src.services.topic_generator import (
    _load_used,
    _scan_made_videos,
    _is_duplicate,
    _wiki_category_members,
    _wiki_search,
    _wiki_extract_many,
)
from src.services.channel_miner import fetch_channel_candidates

TOPICS_OUTPUT_DIR = "topics"

# ---------------------------------------------------------------- vetting

# Hard reject: true crime / serial killers / missing persons / paranormal /
# cryptid / folklore. Mirror of the channel-miner gate.
# "was killed" is intentionally ABSENT — casualty language is the substance of
# war-story topics.
VET_BLOCK = [
    "serial killer", "murder", "rapist", "stalker", "cctv",
    "kidnap", "abduction", "missing person", "body found",
    "school shooting", "mass shooter", "arrested", "police", "criminal",
    "manslaughter", "ghost", "haunted", "ghost ship", "poltergeist",
    "cryptid", "skinwalker", "werewolf", "vampire", "zombie", "bigfoot",
    "sasquatch", "chupacabra", "loch ness", "alien", "aliens", "alien abduction",
    "ufo", "extraterrestrial", "paranormal", "supernatural", "occult ritual",
    "satanic", "demonic", "possession", "exorcism", "curse of the mummy",
    "sea monster", "giant squid attack", "man-eating", "man eater",
    "human trafficking", "drug cartel", "cartel", "gang", "gangster",
]

# Angle classifier: mystery | scary | injustice | heroic
ANGLE_MYSTERY = ["secret", "mystery", "mysterious", "unexplained", "hidden",
                 "cover-up", "cover up", "classified", "vanished", "disappeared",
                 "lost", "unknown", "forgot", "forgotten", "conspiracy",
                 "declassified", "code", "cipher", "coded"]
ANGLE_SCARY = ["horror", "terrify", "horrify", "fear", "dread", "nightmare",
               "death", "dead", "die", "killed", "disaster", "collapse",
               "crash", "explosion", "massacre", "slaughter", "doomed",
               "trapped", "suffocating", "burned", "drowned"]
ANGLE_INJUSTICE = ["injustice", "betrayal", "cover-up", "cover up", "neglect",
                   "cost-cutting", "cost cutting", "greed", "corruption",
                   "abuse", "exploited", "victims", "innocent", "suppressed",
                   "silenced", "forgotten victims", "lies", "lie", "lied",
                   "scandal", "conspiracy against"]
ANGLE_HEROIC = ["hero", "heroic", "bravery", "brave", "stood", "holding",
                "underdog", "last stand", "against", "defended", "defense",
                "saved", "rescued", "legend", "impossible", "outnumbered"]

ANGLE_RULES = [
    ("injustice", ANGLE_INJUSTICE),
    ("scary", ANGLE_SCARY),
    ("heroic", ANGLE_HEROIC),
    ("mystery", ANGLE_MYSTERY),
]


def _vet_idea(idea: dict) -> dict:
    """Reject off-niche ideas and tag an angle. Returns None if rejected."""
    text = f"{idea['title']} {idea.get('hook', '')} {idea.get('why_viral', '')}".lower()
    if any(b in text for b in VET_BLOCK):
        return None
    for angle, keywords in ANGLE_RULES:
        if any(k in text for k in keywords):
            idea["angle"] = angle
            return idea
    # No signal — default to mystery (the channel's core pillar).
    idea["angle"] = "mystery"
    return idea


# ---------------------------------------------------------------- niches

# Each niche has its own Wikipedia categories + search queries + prompt focus.
# This keeps sourcing and prompt tuning separate per category (no dilution).
# The "war_*" niches additionally merge in the LIVE source pool
# (src/services/war_stories.py: Google News RSS for the current wars, and
# long-form war-history magazine feeds) so modern conflicts surface as they
# break. War/story/war-experiment topics replace the old mystery niches.
NICHES = {
    # 2026 US–Iran war — the flagship modern-war niche, news-driven.
    "iran_360": {
        "categories": [
            "Category:Iran–United States relations",
            "Category:2020s in Iran",
            "Category:2026 in Iran",
            "Category:Wars involving Iran",
            "Category:Red Sea crisis",
        ],
        "queries": [
            "US Iran war 2026", "Iran war February 2026", "Iran missile attack 2026",
            "US strikes Iran 2026", "Houthi Yemen 2026", "Houthi Red Sea attacks 2026",
            "Strait of Hormuz 2026", "Iran naval blockade 2026", "US garrison attack 2026",
        ],
        "prompt": (
            "the 2026 US–Iran war and the wider Middle East crisis it triggered: "
            "aerial strikes, missile/drone barrages on US bases and Gulf states, "
            "the Houthi offensive in Yemen, the Strait of Hormuz naval blockade, "
            "and the troops caught in it"
        ),
        "live": True,
        "floor_bonus": 2,  # war news is the money niche — feed more of it
    },
    # Iraq war — insurgent operations, battles, and the gates-of-hades fighting.
    "iraq_war": {
        "categories": [
            "Category:Iraq War",
            "Category:Military operations of the Iraq War",
            "Category:Military history of Iraq",
            "Category:People of the Iraq War",
        ],
        "queries": [
            "Iraq war battle fallen city", "Iraq war ambush convoy",
            "insurgent attack Iraq", "Fallujah battle", "Baghdad war 2003",
            "Iraq war helicopter downed", "Iraq war soldier story",
        ],
        "prompt": (
            "stories of the Iraq War: urban battles like Fallujah, ambushed convoys "
            "and downed helicopters, brutal insurgency operations, and the soldiers "
            "who fought through them"
        ),
    },
    # Vietnam War — first-hand combat stories, tunnels, and big-unit battles.
    "vietnam_war": {
        "categories": [
            "Category:Vietnam War",
            "Category:Battles and operations of the Vietnam War",
            "Category:Vietnam War casualties",
        ],
        "queries": [
            "Vietnam war battle Ia Drang", "Khe Sanh siege 1968",
            "Tet Offensive battle", "Vietnam war tunnel rat",
            "Vietnam war ambush platoon", "Cu Chi tunnels",
        ],
        "prompt": (
            "stories from the Vietnam War: desperate firefights and sieges like Ia Drang "
            "and Khe Sanh, tunnel operations, ambushed patrols, and the grunts who "
            "fought a war they weren't told they could win"
        ),
    },
    # Cold War / proxy conflicts — Berlin, nuclear close-calls, secret missions.
    "cold_war": {
        "categories": [
            "Category:Cold War conflicts",
            "Category:Cold War",
            "Category:Cuban Missile Crisis",
        ],
        "queries": [
            "cold war brush with nuclear war", "Cuban missile crisis day by day",
            "spy plane shot down cold war", "Cold war submarine incident",
            "proxy war cold war Africa Asia", "Berlin blockade airlift",
            "Kursk submarine disaster", "nuclear submarine collision",
        ],
        "prompt": (
            "Cold War stories: nuclear close calls and hair-trigger incidents, "
            "spy-plane shootdowns and submarine collisions, Berlin and the airlift, "
            "and the proxy wars waged around the world from Africa to Asia"
        ),
    },
    # Weird military experiments (kept) — secret projects, test failures,
    # radiation/nuclear mishaps, and bizarre prototypes.
    "war_experiments": {
        "categories": [
            "Category:V-weapons",
            "Category:World War II weapons of Germany",
            "Category:Secret military programs",
            "Category:Human subject research",
        ],
        "queries": [
            "secret military experiment", "nuclear test gone wrong",
            "weird war weapon prototype", "soldier guinea pig experiment",
            "secret ww2 project", "chemical weapon test soldier",
        ],
        "prompt": (
            "weird and classified military experiments: secret weapon programs, "
            "nuclear and chemical tests that went horribly wrong, and the soldiers "
            "used as guinea pigs"
        ),
    },
}

# ---------------------------------------------------------------- source layer

def _fetch_niche_sources(niche: str, limit: int = 25) -> list:
    """Gather raw candidate articles for one niche (title + url + intro).

    Live-marked niches (e.g. the 2026 US–Iran war) also merge in the external
    source pool — Google News RSS + war-history magazines — so the current war
    surfaces alongside the archival Wikipedia material.
    """
    spec = NICHES[niche]
    titles = []
    seen = set()

    for cat in spec["categories"]:
        try:
            for t in _wiki_category_members(cat, limit=30):
                if t.lower() not in seen:
                    seen.add(t.lower())
                    titles.append(t)
        except Exception:
            continue

    for q in spec["queries"]:
        try:
            for t in _wiki_search(q, limit=10):
                if t.lower() not in seen:
                    seen.add(t.lower())
                    titles.append(t)
        except Exception:
            continue

    extracts = _wiki_extract_many(titles[:limit])
    sources = []
    for t in titles[:limit]:
        ext = (extracts.get(t) or "").strip()
        if len(ext) < 300:
            continue
        sources.append({
            "title": t,
            "url": f"https://en.wikipedia.org/wiki/{t.replace(' ', '_')}",
            "content": ext,
        })

    # Live layer for current-war niches: news + long-form history feeds.
    if spec.get("live"):
        try:
            from src.services.war_stories import fetch_all_war_sources
            extra = fetch_all_war_sources(limit=limit)
            known = {s["title"].lower() for s in sources}
            for s in extra:
                if s["title"].lower() not in known:
                    sources.append(s)
                    known.add(s["title"].lower())
        except Exception as e:
            print(f"    [war] live source pool failed: {e}")
        print(f"  {len(sources)} sources (incl. live war pool)")

    if not sources:
        print(f"  [source] {niche}: nothing usable")
    return sources


# ---------------------------------------------------------------- LLM idea-gen

IDEAS_PROMPT = """You are a YouTube content strategist for a faceless channel about {niche}.

Given the raw source material below, extract up to 3 video ideas.

For each idea output ONLY valid JSON in this schema:
{{
  "title": "clickable but not clickbait-lie title, under 70 chars",
  "hook": "first 15 seconds script, must create an open question",
  "why_viral": "one sentence on the curiosity gap or stakes",
  "outline": ["beat 1", "beat 2", "beat 3", "beat 4"],
  "source_url": "...",
  "confidence": 1-10 (how obscure/novel is this, avoid oversaturated topics)
}}

Reject anything that is: already extremely well-covered on YouTube
(D-Day, Roswell, Atlantis basics, Titanic), unverifiable pure speculation
with zero primary source, or requires reproducible technical/harmful detail.
Prefer stories a general audience has NOT heard — obscure but documented.

Return ONLY a JSON array of idea objects, no prose, no markdown fences.

Raw material:
{raw}
"""


# Max sources per single LLM call, and max chars per source, so an individual
# request stays well under the 8k-TPM free-tier ceiling of Groq/Ollama.
# (Chunked, not RAG-indexed: the goal is small prompt windows, not retrieval.)
IDEAS_CHUNK_SOURCES = 4
IDEAS_SOURCE_CHARS = 500


def _parse_idea_json(raw_out: str) -> list:
    """Strip fences/prose, grab the JSON array, normalize to idea dicts."""
    m = re.search(r"\[.*\]", raw_out, re.S)
    if not m:
        print(f"    [llm] no JSON in response")
        return []
    try:
        data = json.loads(m.group(0))
    except Exception:
        print(f"    [llm] unparseable JSON")
        return []

    ideas = []
    for item in data if isinstance(data, list) else []:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title", "")).strip()
        if not title:
            continue
        confidence = 0
        try:
            confidence = min(10, max(1, int(item.get("confidence", 5))))
        except (TypeError, ValueError):
            confidence = 5
        ideas.append({
            "title": title,
            "hook": str(item.get("hook", "")).strip(),
            "why_viral": str(item.get("why_viral", "")).strip(),
            "outline": item.get("outline", []),
            "source_url": str(item.get("source_url", "")).strip(),
            "confidence": confidence,
        })
    return ideas


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
            raw_out = call_text(
                [{"role": "user", "content": prompt}],
                temperature=0.7,
                max_tokens=2000,  # Ensure enough budget for JSON array output
            )
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
                raw_out = call_text(
                    [
                        {"role": "user", "content": prompt},
                        {"role": "assistant", "content": raw_out},
                        {"role": "user", "content": repair_prompt},
                    ],
                    temperature=0.3,  # Lower temp for structured repair
                    max_tokens=2000,
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


# ---------------------------------------------------------------- scoring

def _score_idea(idea: dict, angle_map: dict) -> dict:
    """Novelty (confidence) is the controllable proxy for virality."""
    score = idea["confidence"] * 10
    # Slight lift for ideas backed by a real source URL (verifiable).
    if idea.get("source_url"):
        score += 5
    return score


# ---------------------------------------------------------------- main

def run_research_pipeline(target: int = 24, min_per_niche: int = 6) -> list:
    """Run the standalone research pipeline and write topics/batch_*.json.

    Returns the approved topic list (same shape as run_topic_generation output)
    so main.py --use-saved picks it up unchanged.
    """
    os.makedirs(TOPICS_OUTPUT_DIR, exist_ok=True)
    used = list(dict.fromkeys(_load_used() + _scan_made_videos()))

    topics = []
    for niche in NICHES:
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

        for idea in niche_ideas:
            if _is_duplicate(idea["title"], used, fuzzy=True):
                print(f"    skip (dup): {idea['title']}")
                continue
            vetted = _vet_idea(idea)
            if vetted is None:
                print(f"    skip (vet-block): {idea['title']}")
                continue
            score = _score_idea(vetted, {})
            hook = vetted["hook"] or vetted["why_viral"]
            content = "\n\n".join([
                hook,
                vetted["why_viral"],
                "Outline: " + " | ".join(vetted["outline"]),
            ])
            topics.append({
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
            })

    # Optionally merge cached channel-mined seeds (proven viral, cheap).
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

    if not topics:
        print("\n  No ideas generated. Try again or widen sources.\n")
        return []

    # Rank: proven-viral channel seeds first, then LLM ideas by score.
    topics.sort(key=lambda t: t["score"], reverse=True)

    # Category quota balance: don't let one bucket flood the queue.
    # Research niches get at least min_per_niche(+floor_bonus); no bucket exceeds
    # ~40% of target. The 2026 Iran war niche gets a bigger floor because the
    # live news layer feeds it fresh material continuously.
    selected, counts = [], {}
    per_niche_floor = {
        n: min_per_niche + NICHES[n].get("floor_bonus", 0) for n in NICHES
    }
    max_bucket = max(1, int(target * 0.4))

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

    print(f"\nSelected {len(selected)} topics:")
    for i, t in enumerate(selected, 1):
        angle = f"{t.get('angle','?'):>8}" if t.get("angle") else ""
        print(f"{i:2d}. [{t['category'][:16]:16} |{angle}] {t['title']}  (score {t['score']})")

    batch_file = os.path.join(
        TOPICS_OUTPUT_DIR, f"batch_{time.strftime('%Y%m%d_%H%M%S')}.json"
    )
    with open(batch_file, "w") as f:
        json.dump(selected, f, indent=2)
    print(f"\nQueue saved to {batch_file}")

    return selected