"""Competitor-channel mining — turns viral competitor videos into topic seeds.

For each competitor channel: rank videos by views (yt-dlp flat playlist),
fetch the transcript for the top N, and LLM-resolve each into a concrete
viral topic seed. Returns candidates in the SAME dict shape as the Wikipedia
and Reddit paths, so filtering/ranking/output stay shared.

Uses `uvx yt-dlp` so we always get the latest build (the apt-installed binary
is too old to parse YouTube's current page model).
"""
import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

from src.agents.topic.prompts import resolve_topic_prompt
from src.agents.common.llm import _local

# Competitor channels (NuttyHistory is frequently blocked — it simply yields nothing).
CHANNELS = [
    ("DarkDocs", "https://www.youtube.com/@DarkDocs/videos"),
    ("FascinatingHorror", "https://www.youtube.com/@FascinatingHorror/videos"),
    ("HistoryDose", "https://www.youtube.com/@HistoryDose/videos"),
    ("NuttyHistory", "https://www.youtube.com/@NuttyHistory/videos"),
]

TOP_N = 5
PLAYLIST_END = 200
CACHE_DIR = "channel_cache"  # persists transcripts + resolved seeds across runs


def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def _top_videos(channel_url, top_n=TOP_N):
    """Rank a channel's videos by view count (flat playlist, no downloads)."""
    out = _run(["uvx", "yt-dlp", "--flat-playlist", "--playlist-end", str(PLAYLIST_END),
                "--print", "%(id)s\t%(view_count)s\t%(title)s", channel_url])
    rows = []
    for line in out.stdout.splitlines():
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        vid, views, title = parts[0], parts[1], "\t".join(parts[2:])
        try:
            views = int(views or 0)
        except ValueError:
            views = 0
        rows.append((views, vid, title))
    rows.sort(key=lambda r: r[0], reverse=True)
    return rows[:top_n]


def _fetch_transcript(vid, tmp):
    """Download the video's auto-captions (en, falling back to en-GB) as text."""
    for lang in ("en", "en-GB"):
        path = os.path.join(tmp, f"{vid}.{lang}.vtt")
        _run(["uvx", "yt-dlp", "--skip-download", "--write-auto-subs", "--sub-langs", lang,
              "--sub-format", "vtt", "--write-subs",
              "-o", os.path.join(tmp, f"{vid}.%(ext)s"),
              f"https://www.youtube.com/watch?v={vid}"])
        if os.path.exists(path):
            break
    else:
        return None
    raw = open(path).read()
    lines = [l for l in raw.split("\n")
             if l.strip() and not l.startswith("WEBVTT") and not l.startswith("Kind:")
             and not l.startswith("Language:") and not re.match(r"^\d+$", l.strip())
             and "-->" not in l]
    text = re.sub(r"<[^>]+>", "", " ".join(l.strip() for l in lines))
    return re.sub(r"\s+", " ", text).strip()


def _resolve_topic(vid, title, transcript):
    prompt = resolve_topic_prompt(title, transcript)
    raw = _local(prompt, temperature=0.3, tag="research")
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        seed = json.loads(m.group(0))
    except Exception:
        return None
    if not isinstance(seed, dict):
        return None
    topic = str(seed.get("topic", "")).strip()
    if not topic or len(topic.split()) > 14:
        return None
    return seed


def _cache_path(vid):
    return os.path.join(CACHE_DIR, f"{vid}.json")


def _load_cache(vid):
    try:
        with open(_cache_path(vid)) as f:
            return json.load(f)
    except (OSError, json.JSONDecodeError):
        return None


def _save_cache(vid, data):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(_cache_path(vid), "w") as f:
        json.dump(data, f)


def _process_video(cname, views, vid, title, tmp):
    """Fetch transcript + resolve seed for one video. Returns candidate dict or None."""
    cached = _load_cache(vid)
    if cached:
        seed = cached
        transcript = cached.get("_transcript", "")
    else:
        transcript = _fetch_transcript(vid, tmp)
        if not transcript:
            return None
        seed = _resolve_topic(vid, title, transcript)
        if not seed:
            return None
        seed = dict(seed)
        seed["_transcript"] = transcript
        _save_cache(vid, seed)

    topic = str(seed.get("topic", "")).strip()
    if not topic or len(topic.split()) > 14:
        return None

    # Enforce the resolver's own niche verdict: a competitor's top video is
    # only a candidate when it is genuinely on-niche (WW1/WW2 stories, human/
    # animal experiments, or history-rooted dark legends). Rejecting off-niche
    # viral videos here keeps the batch queue free of random stories.
    niche_fit = str(seed.get("niche_fit", "")).strip().lower()
    if not niche_fit or niche_fit.startswith("no"):
        print(f"    [miner] off-niche (rejected): {topic}")
        return None

    return {
        "source": f"channel:{cname}",
        "category": seed.get("angle", "mystery"),
        "title": topic,
        "content": transcript or seed.get("_transcript", ""),
        "score": views,
        "upvote_ratio": 1.0,
        "source_urls": [f"https://www.youtube.com/watch?v={vid}"],
    }


def fetch_channel_candidates(top_n: int = TOP_N, used_titles: list = None) -> list:
    """Return topic-seed candidates from competitor channels.

    Same dict shape as the Wikipedia/Reddit paths:
    source, category, title, content, score, upvote_ratio, source_urls.

    Transcripts and LLM seeds are cached on disk (channel_cache/) so re-runs
    skip yt-dlp + LLM entirely; the LLM calls for a channel's top videos run
    in parallel.
    """
    used = {t.lower() for t in (used_titles or [])}
    candidates = []
    tmp = tempfile.mkdtemp(prefix="channel_miner_")

    for cname, url in CHANNELS:
        rows = _top_videos(url, top_n)
        if not rows:
            print("    no videos fetched (blocked?)")
            continue
        t0 = time.time()
        with ThreadPoolExecutor(max_workers=top_n) as ex:
            futures = [ex.submit(_process_video, cname, views, vid, title, tmp)
                       for views, vid, title in rows]
            for fut in as_completed(futures):
                cand = fut.result()
                if cand and cand["title"].lower() not in used:
                    candidates.append(cand)
        print(f"    {len(candidates)} candidates so far ({time.time()-t0:.1f}s)")

    return candidates