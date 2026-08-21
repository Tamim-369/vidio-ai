"""Standalone channel mining test — NO repo wiring.

For each competitor channel: rank videos by views, fetch transcripts for the
top N, and LLM-resolve each into a concrete viral topic seed.
Run:  .venv/bin/python test_channel_miner.py
"""
import os
import re
import json
import subprocess
import tempfile
import time

from src.services.script_builder import _call_ollama

CHANNELS = [
    ("DarkDocs", "https://www.youtube.com/@DarkDocs/videos"),
    ("FascinatingHorror", "https://www.youtube.com/@FascinatingHorror/videos"),
    ("HistoryDose", "https://www.youtube.com/@HistoryDose/videos"),
    ("NuttyHistory", "https://www.youtube.com/@NuttyHistory/videos"),
]

TOP_N = 5
PLAYLIST_END = 200
TMP = tempfile.mkdtemp(prefix="channel_miner_")


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True)


def fetch_top_videos(channel_url):
    out = run(["uvx", "yt-dlp", "--flat-playlist", "--playlist-end", str(PLAYLIST_END),
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
    return rows[:TOP_N]


def fetch_transcript(vid):
    for lang in ["en", "en-GB"]:
        path = os.path.join(TMP, f"{vid}.{lang}.vtt")
        out = run(["uvx", "yt-dlp", "--skip-download", "--write-auto-subs", "--sub-langs", lang,
                   "--sub-format", "vtt", "--write-subs", "-o", os.path.join(TMP, f"{vid}.%(ext)s"),
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


def resolve_topic(vid, title, transcript):
    prompt = (
        "You extract viral topic seeds from faceless documentary transcripts.\n"
        "Return ONLY a JSON object:\n"
        '{"topic": "concrete historical subject (named entity: person/operation/place/experiment), '
        'max 12 words", "angle": "mystery|scary|injustice|heroic", '
        '"niche_fit": "yes or no + 1-line reason"}\n'
        "Angle rules: mystery = secrets/cover-ups/unsolved unknowns; scary = horror/fear/terror "
        "in the story itself; injustice = institutional betrayal, innocent victims, cruel neglect; "
        "heroic = underdog soldiers, impossible stands, bravery. Never default to 'other'.\n"
        "Reject (niche_fit=no) if it's true crime, serial killers, missing persons, boring weapon "
        "specs, OR paranormal/cryptid/folklore content (ghosts, skinwalkers, aliens, giants, "
        "man-eating creatures, monsters). Stick to documented history.\n"
        f"Video title: {title}\nTranscript (first 6000 chars):\n{transcript[:6000]}"
    )
    raw = _call_ollama([{"role": "user", "content": prompt}], temperature=0.3)
    try:
        m = re.search(r"\{.*\}", raw, re.S)
        return json.loads(m.group(0))
    except Exception:
        return {"topic": raw[:200], "angle": "?", "niche_fit": "?"}


def main():
    results = {}
    for name, url in CHANNELS:
        print(f"\n=== {name} ===")
        t0 = time.time()
        tops = fetch_top_videos(url)
        if not tops:
            print("  no videos fetched (blocked?)")
            continue
        print(f"  fetched {len(tops)} top videos ({time.time()-t0:.1f}s)")
        channel_results = []
        for views, vid, title in tops:
            print(f"  [{views:>10,} views] {title[:60]}")
            transcript = fetch_transcript(vid)
            if not transcript:
                print("    no transcript")
                continue
            print(f"    transcript {len(transcript)} chars")
            seed = resolve_topic(vid, title, transcript)
            print(f"    -> {seed}")
            channel_results.append({"title": title, "views": views, "seed": seed})
        results[name] = channel_results

    print("\n\n===== SUMMARY =====")
    for name, items in results.items():
        print(f"\n--- {name} ---")
        for it in items:
            seed = it["seed"]
            print(f"  {it['views']:>10,} | {seed.get('topic','?')} "
                  f"| {seed.get('angle','?')} | {seed.get('niche_fit','?')}")


if __name__ == "__main__":
    main()