# Automated Video Generator — Pipeline Plan

## Goal
Build a fully automated, niche-agnostic video generation system.
Changing the niche = changing a config value. Nothing else.

---

## High-Level Flow

```
Topic Input
     ↓
data_source.py       — research raw data for the topic (DuckDuckGo + Wikipedia)
     ↓
script_builder.py    — Groq LLM → structured JSON script (text + search terms per line)
     ↓
asset_fetcher.py     — fetch best matching image per line (DDG Images → Pollinations fallback)
     ↓
tts.py               — generate voiceover audio per line
     ↓
video_assembler.py   — combine visuals + audio + subtitles → .mp4
     ↓
output.mp4
```

---

## Folder Structure

```
src/
  config/
    voices.py           — voice name constants (already exists)
    settings.py         — global config (groq model, format, style, voice, dirs)
  services/
    data_source.py      — research layer (DuckDuckGo search + Wikipedia)
    script_builder.py   — calls Groq, returns structured script JSON
    asset_fetcher.py    — fetches best image per line (DDG → Pollinations fallback)
    tts.py              — generates .wav per script line
    video_assembler.py  — assembles final video using moviepy
  utils/
    file_helpers.py     — save/load JSON, manage temp files, output paths
main.py                 — top-level orchestrator, runs the full pipeline
```

---

## Module Breakdown

### 1. `src/config/settings.py`
Central config. Changing this file changes the entire behavior of the pipeline.

```python
GROQ_MODEL = "llama3-70b-8192"
VIDEO_FORMAT = "9:16"        # or "16:9"
VIDEO_STYLE = "educational"  # drives the LLM prompt template (educational, ad, motivational...)
VOICE = "M5"                 # from voices.py
OUTPUT_DIR = "output/"
TEMP_DIR = "temp/"
```

---

### 2. `src/services/data_source.py`
Gathers raw context/data about the topic before passing to the LLM.
Gives the script real facts to work with and prevents hallucination.

**Methods:**
- `research(topic: str) -> str` — main entry point, returns combined raw text

**How it works:**
1. DuckDuckGo text search via `duckduckgo_search` library — pulls top results for the topic
2. Wikipedia summary via `wikipedia` library — grabs the intro/summary of the topic page
3. Combines both into one raw text string and returns it

**Why separate:**
Swapping the data source (news API, product description, user input) never touches anything else.

---

### 3. `src/services/script_builder.py`
Takes raw research data + topic, calls Groq, returns a validated script JSON.

**Interface:**
```python
build_script(topic: str, raw_data: str) -> dict
```

**Output — Script JSON:**
```json
{
  "topic": "Black Holes",
  "style": "educational",
  "lines": [
    {
      "id": 1,
      "text": "A black hole is a region of space where gravity is so strong that nothing can escape.",
      "search_term": "black hole space gravity visualization",
      "duration": 6
    },
    {
      "id": 2,
      "text": "They form when massive stars collapse at the end of their life cycle.",
      "search_term": "star collapse supernova space",
      "duration": 5
    }
  ]
}
```

**How it works:**
- Prompt template is selected based on `VIDEO_STYLE` from settings
- Groq is told to return strict JSON only
- Response is parsed and validated before returning
- `search_term` per line is crafted by Groq to be descriptive enough for accurate image search
- Changing `VIDEO_STYLE` → different prompt template → completely different video type

**Groq SDK:** official `groq` Python package.

---

### 4. `src/services/asset_fetcher.py`
For each line in the script, finds and downloads the most accurate image.

**Interface:**
```python
fetch_assets(lines: list) -> list  # returns lines with asset_path added
```

**Each line gets:**
```json
{
  "id": 1,
  "text": "...",
  "search_term": "black hole space gravity",
  "duration": 6,
  "asset_path": "temp/assets/1.jpg"
}
```

**Image search strategy:**

1. **Primary — DuckDuckGo Image Search** (`duckduckgo_search` library)
   - No API key needed
   - Searches the entire web for images
   - Uses `search_term` from the script line as the query
   - Downloads first valid/accessible image URL
   - Fast and accurate for any topic

2. **Fallback — Pollinations.ai AI Generation**
   - Used when DDG returns no results or all URLs are broken/inaccessible
   - Free, no API key, no rate limits
   - Just an HTTP GET: `https://image.pollinations.ai/prompt/{search_term}`
   - Generates an AI image directly from the search term
   - Guarantees an image is always returned, for any topic

**Notes:**
- All assets downloaded to `temp/assets/`
- Cleaned up after video is assembled

---

### 5. `src/services/tts.py`
Generates one audio file per script line.

**Interface:**
```python
generate_audio(lines: list) -> list  # returns lines with audio_path + actual_duration added
```

**Each line gets:**
```json
{
  "id": 1,
  "text": "...",
  "audio_path": "temp/audio/1.wav",
  "actual_duration": 5.3
}
```

**TTS engine:** `supertonic` (local, high quality, already in the project)
- `edge-tts` kept as alternative, selectable via settings
- `actual_duration` is read from the generated wav and used by assembler for precise timing

---

### 6. `src/services/video_assembler.py`
Takes the fully enriched script JSON and produces the final `.mp4`.

**Interface:**
```python
assemble(script: dict) -> str  # returns path to output .mp4
```

**Per line:**
1. Load image as `ImageClip`
2. Resize/crop to target format (9:16 or 16:9) from settings
3. Set duration to `actual_duration` of the audio
4. Burn subtitle text onto the frame
5. Attach audio

**Then:**
- Concatenate all clips in order
- Export final `.mp4` to `output/`

**Subtitles:** full line displayed during its clip duration (can upgrade to word-by-word later)

---

### 7. `src/utils/file_helpers.py`
Small shared utilities.

- `save_json(data, path)` / `load_json(path)`
- `ensure_dirs()` — creates `output/` and `temp/` if they don't exist
- `cleanup_temp()` — wipes temp folder after assembly
- `output_path(topic)` — generates a clean filename from topic string

---

### 8. `main.py`
Thin orchestrator. Just calls each service in order.

```python
topic = "Black Holes"

raw_data = research(topic)
script   = build_script(topic, raw_data)
lines    = fetch_assets(script["lines"])
lines    = generate_audio(lines)
output   = assemble({**script, "lines": lines})

print(f"Video saved to {output}")
```

---

## What Makes This Niche-Agnostic

| What you want to change              | What you actually change              |
|--------------------------------------|---------------------------------------|
| Topic/niche                          | `topic` variable in main.py           |
| Video style (edu, ad, motivational)  | `VIDEO_STYLE` in settings.py          |
| Short vs long form                   | `VIDEO_FORMAT` in settings.py         |
| Voice                                | `VOICE` in settings.py                |
| Data source                          | Swap implementation in data_source.py |

---

## Dependencies to Add

- `groq` — Groq API SDK
- `duckduckgo_search` — DDG image + text search, no API key
- `wikipedia` — Wikipedia summaries, no API key
- `requests` — downloading images (already in project)
- `Pillow` — image resizing/processing before feeding to moviepy

Everything else (`moviepy`, `supertonic`, `edge-tts`, `soundfile`) already in the project.

---

## Environment Variables needed (`.env`)

```
GROQ_API_KEY=your_key_here
```

That's the only key needed. Everything else is free with no authentication.
