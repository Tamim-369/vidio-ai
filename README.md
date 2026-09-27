# videoai — parody quote shorts

A batch pipeline that turns one AI-written joke into a finished, uploaded
YouTube Short. One character, one quote, one video.

```bash
uv run src/main.py --batch=10 --no-upload
```

Roughly 45 seconds per video. Most of that is neural TTS on CPU, not the
language model.

---

## What it does

```
character → quote (Groq) → voiceover (Chatterbox) → quote card → music → upload
```

1. **Pick a character.** Rotates through the enabled voices so a batch spreads
   across all of them.
2. **Write one quote.** The character decides the subject — Don Tzu writes
   military strategy, Brolexander writes gym discipline — so the joke and the
   narrator always agree.
3. **Speak it.** Chatterbox clones the voice from a short reference clip, so
   each personality is audibly itself.
4. **Render one card.** The quote over the character's photo, cut to the length
   of the audio it was spoken in.
5. **Mix, title, publish.** Background music, a title, a description, and an
   upload — or keep it local with `--no-upload`.

## Why one quote per video

The quote card is a still frame. A second line adds narration and a second card
and nothing else, and stacking several characters into one video made the
output look like a debate rather than a punchline. One strong line per
character, rotating, is what the cards were designed for. The batch loop asks
one question — which character next — and nothing else.

## Setup

```bash
uv sync
cp .env.example .env      # then fill in the keys
```

| Variable | Required | Why |
| --- | --- | --- |
| `GROQ_API_KEY` | yes | Quote generation. |
| `GROQ_API_KEY_SECOND`, `GROQ_API_KEY_THIRD` | no | The client rotates to these on a rate limit, which happens on batches over about ten videos. |
| `YOUTUBE_CLIENT_SECRETS` | only for `--upload` | Path to the OAuth client secrets JSON. Defaults to `client_secrets.json` in the project root. |
| `AUTO_PUBLISH` | no | `1` publishes by default; `--upload`/`--no-upload` override it. Off by default. |
| `HF_HOME` | no | Where model weights are cached. |
| `GROQ_QUOTE_MODEL` | no | Defaults to `openai/gpt-oss-20b`. |

Speech synthesis runs on CPU and needs no key. It does need the Chatterbox
weights, about 3.3GB, which land in `HF_HOME` on first run.

YouTube uses the OAuth installed-app flow: the first `--upload` prints a URL,
you paste it back, and the resulting `token.json` is reused. Both that and
`client_secrets.json` are gitignored.

## Commands

```bash
uv run src/main.py                              # one video
uv run src/main.py --batch=10 --no-upload       # ten, kept local
uv run src/main.py --batch=5 --script-only      # just the quotes, no render
uv run src/main.py --voice=donald-trump         # force one character
uv run src/main.py --batch=3 --upload           # publish to YouTube
```

Five flags, all of them. There is no format, seed, or line-count knob, because
there is nothing left to configure.

## Layout

```
src/
  main.py                    CLI: four flags
  agents/
    quotes/                  joke generation + validation + dedupe
    voice_cast/              the character registry (who, subject, voice, face)
    voiceover/               Chatterbox TTS: chunking, workers, DSP, timing
    visuals/                 the quote card renderer
    soundtrack/              background music
    publish/                 YouTube upload + deterministic metadata
    video/                   the pipeline loop and title rules
    completion/              the LLM client, with key rotation
  assets/voice_refs/         reference clips the voices are cloned from
  faces/                     the portrait on each card
  tests/unit/                the tests worth keeping (see below)
```

Each agent owns one job and is callable on its own. `agents/video/pipeline.py`
is the only place that knows the order.

## Design notes

**Quotes are validated, not trusted.** A candidate is rejected for being
meta-commentary, for being multi-line, for falling outside the length window its
character's prompt asks for, for reading like a fact rather than a joke, or for
duplicating anything already used. The prompt treats sincere wisdom as a
failure, so the filter is deliberately strict.

**No quote is ever used twice.** Every accepted quote is appended to
`used_quotes.json` and checked against on the way in. The file is local mutable
state, deliberately untracked.

**The text client has somewhere else to go.** A batch of ten is ten calls
against a rate limit, and a hard 429 mid-batch means a half-finished run. The
client falls back Groq → Gemini → Cloudflare, rotates through three Groq keys
first, and only gives up once all three tiers are exhausted. Everything after
the first tier is optional insurance, so the required setup stays at one key.

**Metadata is computed, not generated.** An LLM asked for ten descriptions
writes ten near-identical paragraphs, and the only cure is watching for
collisions and regenerating. Building the description from the video's own
quote makes a duplicate structurally impossible and takes a step off the batch.

**Titles are built from the quote.** `Don Tzu: <the quote>`, capped at 100
characters on a word boundary because YouTube rejects longer titles outright.
Since quotes are deduplicated, titles are unique for free.

**Parody, and labelled as such.** These are fabricated jokes written in the
style of public figures, credited to invented names. Every description carries
a disclaimer, and Chatterbox's audio watermark is explicitly stripped — the
model ships one that marks output as machine-generated, and that is the correct
behaviour for this content, so it stays off.

## Tests

```bash
uv run pytest src/tests/
```

323 tests, 18 files, about 25 seconds. The bar for keeping one was a single
question: *does this guard a failure that is silent, expensive, or both?*

The ones that earned their place:

- `test_youtube_limits.py` — title, description and tag caps. These fail at
  upload time, after a full render, which is the most expensive place a bug can
  surface.
- `test_quote_agent.py` — validation rules and the dedupe pool, including a
  missing or corrupt state file. A bug here is invisible until quotes start
  repeating on the channel.
- `test_quote_card.py` — that text shrinks to fit its box instead of running off
  the frame, and that card duration tracks the audio it was cut to.
- `test_pipeline.py` — each step's output reaches the next, and a video that
  fails does not abort the batch around it.
- `test_single_video.py` — the face is validated before the first card renders,
  and a silent line is an error rather than a blank video.
- `test_extracted_helpers.py` — the JSON salvage that digs the candidate array
  out of a prose-wrapped model response.
- `test_voice_models.py` — the watermark stays off. The watermark is inaudible,
  so nothing else would ever catch it.
- `test_architecture.py` — the quote pool lives outside `temp/`, which
  `cleanup_temp()` rmtree's on every run.

This started at 545 tests in 20 files. The 222 that went were not all
decorative:

- **Prompt wording.** 60-odd tests asserted that a prompt still contained the
  words I had written — `test_the_style_examples_survived`,
  `test_the_prompt_text_is_locked`, `test_the_bad_and_good_pairs_survived`.
  They broke on every prompt edit and caught no bugs. Prompt *wiring* is still
  tested: that a character resolves to its own prompt, and that an unknown one
  falls back instead of raising.
- **Pixel pinning.** `test_square_centres_on_360_1560_at_1080x1920` asserted an
  exact coordinate. The geometry contract it was standing in for — text fits
  the box, the byline clears the face — is tested; the coordinate is not.
- **Duplication.** The voice registry checked eight fields across three voices
  as 24 separate tests, two of which asserted the same thing because the
  "required" and "enabled" lists had quietly become the same tuple. It is now
  one test per voice, checked as a whole, because a half-configured voice is the
  only failure that matters.
- **Deleted design.** `test_multichar_video.py` covered per-line faces and
  multi-character voice clashes. One quote per video makes all of that
  unreachable.

What was deliberately left untested is anything whose failure is loud: an
exception on a missing argument, a library raising on unreadable input, or the
existence of a file that any code path touching it would already fail to open.

