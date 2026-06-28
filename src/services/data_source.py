import wikipedia
from ddgs import DDGS


def research(topic: str) -> str:
    """Gather raw research data for a topic from DuckDuckGo + Wikipedia."""
    results = []

    # Wikipedia summary
    try:
        summary = wikipedia.summary(topic, sentences=10, auto_suggest=True)
        results.append(f"Wikipedia:\n{summary}")
    except Exception:
        pass

    # DuckDuckGo text search — top 5 results
    try:
        with DDGS() as ddgs:
            hits = ddgs.text(topic, max_results=5)
            snippets = [h["body"] for h in hits if "body" in h]
            if snippets:
                results.append("Web search:\n" + "\n".join(snippets))
    except Exception:
        pass

    if not results:
        return topic  # fallback: just return the topic itself

    return "\n\n".join(results)
