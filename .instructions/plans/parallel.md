# Parallelization Plan

## Current Pipeline (Sequential)

```
1. Research (data_source.py)
2. Build Script (script_builder.py) - includes generating image expectations
3. Review Script (script_builder.py)
4. Revise Script (if needed) (script_builder.py)
5. Fetch Assets (asset_fetcher.py) - loops through lines sequentially
6. Generate Audio (tts.py) - loops through lines sequentially
7. Assemble Video (video_assembler.py)
```

## Bottlenecks Identified

1. **Image Expectation Generation** - Each line's `image_expectation` is generated sequentially using `_generate_image_expectation()`. Can be parallelized.

2. **Asset Fetching** - Each line's image fetch is independent. Can be parallelized with a thread pool.

3. **TTS Generation** - Each line's audio generation is independent. Can be parallelized with a thread pool.

## Parallelization Strategy

### Use ThreadPoolExecutor for I/O-bound tasks

```python
from concurrent.futures import ThreadPoolExecutor, as_completed
```

### 1. Parallel Image Expectations

```python
def build_script(...) -> dict:
    script = _call_groq(...)
    
    # Parallel image expectation generation
    with ThreadPoolExecutor(max_workers=5) as executor:
        futures = {
            executor.submit(_generate_image_expectation, topic, line): line 
            for line in script["lines"]
        }
        for future in as_completed(futures):
            line = futures[future]
            line["image_expectation"] = future.result()
```

### 2. Parallel Asset Fetching

```python
def fetch_assets(lines: list, topic: str = "") -> list:
    # Process all lines in parallel
    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(_fetch_single_asset, line, topic, assets_dir): line["id"] 
            for line in lines
        }
        for future in as_completed(futures):
            line_id = futures[future]
            lines[line_id - 1]["asset_path"] = future.result()
```

### 3. Parallel TTS Generation

```python
def generate_audio(lines: list) -> list:
    pipeline = _get_pipeline()
    audio_dir = os.path.join(TEMP_DIR, "audio")
    
    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = {
            executor.submit(_generate_single_audio, line, pipeline, audio_dir): line["id"]
            for line in lines
        }
        for future in as_completed(futures):
            line_id = futures[future]
            # Update line with audio info
```

## Parallelization Order

```
Research → Script Build (with parallel image expectations)
    ↓
Review/Revision (sequential, must wait for previous result)
    ↓
Fetch Assets (parallel)
    ↓
Generate Audio (parallel)
    ↓
Assemble Video (sequential)
```

## Thread Pool Configuration

| Task | max_workers | Reason |
|------|-------------|--------|
| Image Expectations | 5 | Moderate parallelism, LLM calls |
| Asset Fetching | 10 | High parallelism, I/O bound (network requests) |
| TTS Generation | 8 | Moderate parallelism, local computation + I/O |

## Benefits

- **Image expectations**: 9 lines → ~9x faster (from ~30s to ~3s)
- **Asset fetching**: 9 lines → ~10x faster (from ~45s to ~4s)
- **TTS generation**: 9 lines → ~8x faster (from ~60s to ~7s)
- **Total time reduction**: ~30-40% faster end-to-end

## Considerations

1. **Thread safety**: Groq client is thread-safe, Kokoro pipeline may need locks
2. **Error handling**: Each parallel task should handle errors independently
3. **Progress tracking**: May need to adjust logging to show parallel progress
4. **Rate limits**: Asset fetching may hit DDG/Pexels rate limits; consider rate limiter
