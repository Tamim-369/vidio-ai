import json
from groq import Groq
from src.config.settings import GROQ_API_KEY, GROQ_MODEL, VIDEO_STYLE

client = Groq(api_key=GROQ_API_KEY)

STYLE_PROMPTS = {
    "educational": "You are a scriptwriter for short educational videos. Write in a clear, engaging, fact-driven tone.",
    "motivational": "You are a scriptwriter for motivational short videos. Write in an inspiring, energetic, emotional tone.",
    "ad": "You are a scriptwriter for short product/service ads. Write in a persuasive, punchy, benefit-focused tone.",
    "storytelling": "You are a scriptwriter for short story videos. Write in a narrative, suspenseful, engaging tone.",
    "attraction": (
        "You are a scriptwriter for short viral male self-improvement and attractiveness videos. "
        "Write in a direct, confident, slightly provocative tone that hooks men aged 18-35. "
        "Use specific, visual, data-driven language. Make every line feel like a revelation. "
        "The goal is to educate men on what makes them more physically attractive and how to improve. "
        "Subtly hint that an AI tool can analyze and score their attractiveness. "
        "Never be preachy. Be the cool older brother who gives real advice."
    ),
}

SYSTEM_PROMPT = """
You are given a topic and research data. Your job is to write a short video script (8-12 lines).

Return ONLY valid JSON. No explanation, no markdown, no code blocks. Just raw JSON.

Format:
{{
  "topic": "<topic>",
  "style": "<style>",
  "lines": [
    {{
      "id": 1,
      "text": "<one sentence to be narrated>",
      "search_term": "<descriptive image search query for this sentence>",
      "image_type": "<stock|search>",
      "duration": <estimated seconds as integer>
    }}
  ]
}}

Rules:
- Each line is ONE sentence, narrated by TTS
- search_term must describe what a CAMERA would photograph — think like a photographer, not a summarizer
  - BAD: "subtle effect in the eyes" → matches anything with eyes
  - GOOD: "attractive young man confident direct eye contact close up portrait"
- search_term must ALWAYS include the main topic context so results stay on-topic
- duration should match how long it takes to say the text naturally (3-8 seconds)
- 8 to 12 lines total
- image_type must be "stock" for common/generic visuals (people, nature, gym, city, doctor, etc.) or "search" for specific/niche visuals (specific person, scientific diagram, historical image, specific event, etc.)
"""


def build_script(topic: str, raw_data: str) -> dict:
    """Call Groq and return structured script JSON."""
    style_instruction = STYLE_PROMPTS.get(VIDEO_STYLE, STYLE_PROMPTS["educational"])

    user_prompt = f"""
Topic: {topic}
Style: {VIDEO_STYLE}

Research data:
{raw_data}

Write the video script now.
"""

    response = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": style_instruction + "\n\n" + SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.7,
    )

    raw = response.choices[0].message.content.strip()

    # Strip markdown code blocks if model wraps response anyway
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    return json.loads(raw)
