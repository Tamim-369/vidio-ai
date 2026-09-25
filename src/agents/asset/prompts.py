"""Asset agent prompts: topic keywords, image->line assignment, repair.

All three personalities run on the local model. Prompts are split out here so
the agent modules stay logic-only and the tuning text is discoverable in one
place.
"""

N_KEYWORDS = 6

KEYWORD_PROMPT = (
    'Video topic: "{topic}"\n\n'
    "I need search keywords to find REAL photos for this video. Generate "
    f"{N_KEYWORDS} DISTINCT keyword phrases (2-5 words each) that an image "
    "search would actually return, all specifically about THIS topic. Use real, "
    "photographable subject matter: named people, places, monuments, artifacts, "
    "equipment, historical photos, period scenes, buildings, weapons, vehicles, "
    "uniforms, maps.\n\n"
    "- Every keyword must be ON-TOPIC. Nothing generic, decorative, or unrelated.\n"
    "- Cover different facets (the person/thing itself, the place, key artifacts, "
    "the event, the era).\n"
    "- Reply ONLY with a JSON list of strings, no markdown, no numbering, no "
    'explanation. Example: ["Port Arthur naval base", "destroyer Yamato", ...]'
)


def assign_prompt(paths: list, catalog: str, expectations: str, line_ids: list, topic: str) -> str:
    ids = ", ".join(str(i) for i in line_ids)
    return (
        f'Video topic: "{topic}"\n\n'
        "I have images downloaded for the topic via targeted per-line searches "
        "(each image tagged with the search query that found it) and a narration "
        "script. Assign images to narration lines so each line's on-screen photo "
        "matches its required feature checklist.\n\n"
        "Image catalog (search query that found the photo):\n"
        f"{catalog}\n\n"
        "Script lines (the feature checklist the on-screen image must satisfy):\n"
        f"{expectations}\n\n"
        "Rules:\n"
        "- EVERY line MUST get at least one image. Never leave a line empty.\n"
        "- Prefer the image that was searched for with that line in mind, but a "
        "clearly better visual for another line wins.\n"
        "- A line may get 1-2 images if the visual demands it.\n"
        "- An image may be reused, but spread them out where possible.\n"
        "- Reply ONLY with JSON, no markdown, using the 1-based image numbers above: "
        '{"line_id": ["image number", ...], ...} Example: {"3": ["1","3"], "5": ["2"]}\n'
        f'Line ids to cover: {ids}'
    )


def repair_prompt(topic: str, catalog: str, expectations: str, missing: list) -> str:
    return (
        f'Video topic: "{topic}"\n\n'
        f"Lines still missing an image: {', '.join(str(m) for m in missing)}\n\n"
        "Image catalog (search query that found the photo):\n"
        f"{catalog}\n\n"
        "Script lines (feature checklist the on-screen image must satisfy):\n"
        f"{expectations}\n\n"
        'Assign at least one image to EVERY line listed above, from the catalog, '
        'matching its checklist. Reply ONLY with the JSON mapping using 1-based '
        'image numbers, no markdown.\n'
        '{"line_id": ["image number", ...], ...}'
    )