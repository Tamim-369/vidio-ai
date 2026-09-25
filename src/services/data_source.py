import re

import wikipedia
from ddgs import DDGS

from src.services.topic_agent.collectors import extract_article_body

# Snippet lines that are pure noise and must never reach the story stage.
_JUNK_RE = re.compile(r"""(?x)
    \b(?:please\s+be\s+advised|typically\s+delivery|delivery\s+times|may\s+take\s+longer)
    |\b\d[\d,]*\s*views?\s*•
    |\ball\s+episodes\b
    |\bsubscrib(e|ing)\b
    |newsletter
    |cookie[sp]?\s+policy
    |privacy\s+policy
    |terms\s+of\s+service
    |@\S{2,}\s*/\s*\w+
    |^\s*Advertisement\s*$
""", re.MULTILINE | re.IGNORECASE)


def _is_junk(line: str) -> bool:
    return bool(_JUNK_RE.search(line)) or len(line.strip()) < 20


def research(topic: str, urls: list[str] | None = None) -> str:
    """Gather raw research for a topic: Wikipedia + DDG + real article bodies.

    urls (optional) are the topic's known source pages — their full article
    text is fetched directly so thin RSS summaries never starve the story
    stage. When a real article body is fetched it is the strongest signal and
    the fuzzy web signals (Wikipedia, DDG) are suppressed: they have dragged
    in off-topic tangents (a different Wikipedia article, a snippet about an
    unrelated unit) that the story stage then follows as fact.
    """
    bodies: list[str] = []

    # Known source URLs: fetch the real article body (best signal).
    if urls:
        for url in urls:
            try:
                body = extract_article_body(url)
            except Exception:
                continue
            if body and len(body) >= 400:
                bodies.append(f"Source ({url}):\n{body}")

    if bodies:
        return "\n\n".join(bodies)

    # No usable article body — fall back to the fuzzy signals.
    results = []
    try:
        summary = wikipedia.summary(topic, sentences=10, auto_suggest=True)
        results.append(f"Wikipedia:\n{summary}")
    except Exception:
        pass

    # DuckDuckGo text search — top 5 results, filtered for junk.
    try:
        with DDGS() as ddgs:
            hits = ddgs.text(topic, max_results=5)
            snippets = [
                h["body"] for h in hits
                if h.get("body") and not _is_junk(h["body"])
            ]
            if snippets:
                results.append("Web search:\n" + "\n".join(snippets))
    except Exception:
        pass

    if not results:
        return topic  # fallback: just return the topic itself

    return "\n\n".join(results)