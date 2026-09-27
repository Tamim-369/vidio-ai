"""Title/description/tag generation for a rendered video.

Two paths. The deterministic one builds metadata from the video's own quotes, so
it is unique by construction and costs nothing. The LLM path asks a model, then
verifies, then repairs once, and is kept as the fallback for scripts that do not
carry the fields the deterministic builder needs. Pure text work -- no YouTube
client, no network.
"""

import json

from src.agents.completion.llm import call_text
from src.agents.publish.prompts import (
    get_metadata_fix_prompt,
    get_metadata_prompt,
    get_metadata_verifier_prompt,
)
from src.agents.video.title import (
    BRAND,
    TITLE_MAX_CHARS,
    build_title,
    shorten_title,
)
from src.agents.voice_cast.voices import VOICES

# Stated plainly in every description. These are fabricated quotes in the voices
# of living public figures, and a viewer who finds one on a channel that looks
# like the real person's should be able to tell within a second that it is not.
_DISCLAIMER = "Parody quotes written in the style of public figures. Not real quotes."


def _voice_cfg(line: dict) -> dict:
    """The registry entry for whoever speaks ``line``.

    A line carries the voice config inline, and a config has no ``id`` key, so
    resolution falls back to the line's ``character`` field.
    """
    cfg = line.get("voice") or {}
    if cfg.get("pseudonym") or cfg.get("short_subject"):
        return cfg
    return VOICES.get(cfg.get("id") or line.get("character", ""), {})


def _pseudonym(line: dict) -> str:
    return _voice_cfg(line).get("pseudonym", "")


def _short_subject(line: dict) -> str:
    return _voice_cfg(line).get("short_subject", "")


def build_description(script: dict, number: int = 0) -> str:
    """Description assembled from the video's own quotes.

    The first quote leads unnumbered because it is also the title, so numbering
    it "1." would imply a list it is not the first item of. The rest are numbered
    and attributed to the pseudonym that spoke them, which is what lets a viewer
    tell a three-character video apart at a glance.

    The quote lines come from a pool already deduplicated against
    used_quotes.json, so the description cannot duplicate another video's: an LLM
    asked for N descriptions writes N near-identical paragraphs, and the only
    cure is watching for collisions and regenerating. It also means no model
    call, which drops a whole step off a batch.

    The disclaimer is not decoration. These are fabricated quotes in the voices
    of living public figures, and it has to be visible without expanding
    anything.
    """
    lines = [l for l in (script.get("lines") or []) if l.get("text")]
    if not lines:
        return _DISCLAIMER
    # Fall back to the script's own number, so a planned video gets its footer
    # from any caller rather than only from the one that happens to pass it.
    number = number or script.get("number") or 0

    parts = [lines[0]["text"].strip()]
    for i, l in enumerate(lines[1:], start=1):
        parts.append(f"{i}. {l['text'].strip()}\n   — {_pseudonym(l)}")
    body = "\n\n".join(parts)

    hashtags = "#quotes #shorts"
    for l in lines:
        tag = "#" + _pseudonym(l).replace(" ", "")
        if tag not in hashtags:
            hashtags += " " + tag

    footer = []
    if number:
        footer.append(f"{BRAND} #{number}")
    return "\n\n".join([body, _DISCLAIMER] + footer + [hashtags]).strip()


def build_tags(script: dict) -> list:
    """Tags drawn from who is speaking and what about.

    The pseudonym leads: it is what a viewer of this channel searches for, and
    leading with the real name would advertise a channel that is not this one.
    The speaker's own topic follows, which is what the video actually contains.
    """
    tags: list = []
    for line in script.get("lines") or []:
        for name in (_pseudonym(line), line.get("character"),
                     _short_subject(line)):
            if name and name.lower() not in [t.lower() for t in tags]:
                tags.append(name.lower())
    tags += ["parody quotes", "motivation", "gym motivation", "shorts"]
    seen, out = set(), []
    for t in tags:
        if t.lower() not in seen:
            seen.add(t.lower())
            out.append(t)
    return out[:15]


def deterministic_metadata(script: dict, number: int = 0) -> dict:
    """Title, description and tags for a script, with no model call.

    The number is read off the script when not passed, so it reaches the
    description without every caller in the publish path having to grow a
    parameter: a script that was planned carries it, and a hand-written one
    simply has no number and gets no footer.

    The title is passed through shorten_title() even when the script already
    carries one. A title built by build_title() is already safe, but a title
    inherited from an older script is not, and YouTube rejects the whole
    upload over a 101st character -- so the cap is enforced at the single point
    where a title becomes an API argument rather than trusted upstream.
    """
    lines = [l for l in (script.get("lines") or []) if l.get("text")]
    number = number or script.get("number") or 0
    title = (script.get("title") or "").strip()
    if not title and lines:
        first = lines[0]
        voice_id = (first.get("voice") or {}).get("id") or first.get("character", "")
        title = build_title(first["text"].strip(), voice_id, _pseudonym(first))
    return {
        "title": shorten_title(title),
        "description": build_description(script, number),
        "tags": build_tags(script),
    }


def _format_script(script: dict) -> str:
    """Flatten the script lines into readable narration text."""
    topic = script.get("topic", "")
    lines = script.get("lines", [])
    body = "\n".join(f"{i + 1}. {l.get('text', '')}" for i, l in enumerate(lines))
    return f"Topic: {topic}\n\n{body}"


def _fallback_metadata(topic: str, script_text: str) -> dict:
    """Metadata used when the LLM call fails.

    This is the path a video takes precisely when something has already gone
    wrong, so it must not be allowed to invent a second one: the copy below
    describes a parody-quote channel, because a channel that posts fabricated
    quotes under living public figures' names must never describe itself as a
    true-crime documentary, however convenient that framing would be.
    """
    keyword = " ".join(topic.split()[:5]).strip(" .") or "parody quotes"
    title = f"{keyword} | parody quotes"
    if len(title) > TITLE_MAX_CHARS:
        title = title[:TITLE_MAX_CHARS - 1].rstrip() + "…"

    covered = "\n".join(
        f"- {l.strip()}" for l in script_text.splitlines() if l.strip()
    )
    description = (
        f"{_DISCLAIMER}\n\n"
        f"{covered}\n\n"
        f"Parody quotes on {keyword.lower()}.\n\n"
        "#parodyquotes #shorts"
    )
    tags = ["parody quotes", keyword.lower(), "motivation", "gym motivation",
            "shorts"]
    return {"title": title, "description": description, "tags": tags}


def _extract_json(raw: str) -> dict:
    """Pull the first {...} JSON object out of an LLM response."""
    start, end = raw.find("{"), raw.rfind("}")
    if start == -1 or end <= start:
        return {}
    try:
        return json.loads(raw[start:end + 1])
    except json.JSONDecodeError:
        return {}


def _clean_metadata(meta: dict) -> dict:
    """Validate and normalize a metadata dict; empty values fall back to fallback."""
    title = (meta.get("title") or "").strip()
    description = (meta.get("description") or "").strip()
    tags = meta.get("tags") or []

    if not title or not description or len(title) > 100 or len(description) < 50:
        return {}

    return {
        "title": title,
        "title_alternates": [t for t in (meta.get("title_alternates") or []) if isinstance(t, str)][:3],
        "description": description,
        "tags": [str(t).lower().replace(" ", "-") for t in tags if str(t).strip()][:15],
    }


def generate_metadata(topic: str, script: dict) -> dict:
    """Generate title, description, and tags using minimax-m3:cloud (Ollama).

    Two-pass with a verifier:
      1. The generator is told to think through title/description tactics first.
      2. A verifier (same model) audits the result against every rule and returns
         a PASS/FAIL verdict. On FAIL it is fed the verifier's feedback and
         rewrites, up to MAX_VERIFY_ROUNDS.
    Falls back to deterministic metadata if the LLM calls fail.
    """
    MAX_VERIFY_ROUNDS = 2
    script_text = _format_script(script)

    def _call(prompt: str, temperature: float = 0.6) -> dict:
        try:
            raw = call_text(
                [{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            return _extract_json(raw)
        except Exception as e:
            print(f"    [youtube] LLM call failed: {e}")
            return {}

    # Pass 1: generate (think-first prompt).
    meta = _call(get_metadata_prompt(topic, script_text), temperature=0.7)
    if not meta:
        print("    [youtube] Metadata generation failed, using fallback")
        return _fallback_metadata(topic, script_text)

    # Pass 2+: verify, and rewrite on FAIL until it passes.
    for round_no in range(1, MAX_VERIFY_ROUNDS + 1):
        verifier = _call(
            get_metadata_verifier_prompt(topic, script_text, json.dumps(meta, indent=2)),
            temperature=0.2,
        )
        verdict = (verifier.get("verdict") or "").strip().upper()
        print(
            f"    [youtube] Verifier round {round_no}: {verdict}"
            f" (title {verifier.get('title_score')}/100,"
            f" desc {verifier.get('description_score')}/100,"
            f" emotion: {verifier.get('emotion_triggered')})"
        )

        if verdict == "PASS":
            # Prefer the verifier's fixed title if it supplied a stronger one.
            fixed_title = (verifier.get("fixed_title") or "").strip()
            if fixed_title and _clean_metadata({"title": fixed_title, "description": meta.get("description", ""), "tags": meta.get("tags", [])}):
                meta["title"] = fixed_title
            break

        # FAIL: rewrite with the verifier's feedback.
        meta = _call(
            get_metadata_fix_prompt(topic, script_text, json.dumps(meta, indent=2), json.dumps(verifier, indent=2)),
            temperature=0.7,
        )
        if not meta:
            break

    cleaned = _clean_metadata(meta)
    if not cleaned:
        print("    [youtube] LLM metadata invalid after verification, using fallback")
        return _fallback_metadata(topic, script_text)

    return cleaned
