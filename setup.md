# Setup

Builds faceless YouTube Shorts: research/viral-topic generation → story/script → Pocket-TTS voiceover → image assembly → MP4 → optional YouTube upload.

## Prerequisites

- Python 3.13 (see `.python-version`; a 3.12 venv also works)
- System binaries (must be on `PATH`):
  - `ffmpeg` — video assembly (`video_assembler.py`)
  - `sox` — TTS DSP (pitch, tempo, EQ; `tts_dsp.py`)
  - `tesseract` — OCR text-slop filter for images (`pytesseract`; `agents/asset/agent.py`)
- [uv](https://docs.astral.sh/uv/), or plain `pip`/`venv`

## Install

One command does everything: checks prerequisites, creates `.venv`, installs
requirements, installs Playwright Chromium, creates `.env` from the template,
and checks the narrator voice.

```bash
git clone https://github.com/Tamim-369/vidio-ai.git && cd vidio-ai
bash setup.sh
```

Flags: `--no-playwright` (skip the optional ~150 MB browser download),
`--no-voice-check`. Then edit `.env` and add your keys — see below.

---

Manual install (if you prefer the steps yourself):

If using pip directly:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
playwright install chromium
```

## Environment (.env)

Start from the template and fill in only what you have:

```bash
cp .env.example .env
```

It is loaded automatically by the CLI and every module that reads a knob —
`dotenv` is handled internally, so no shell `export` is needed. Never commit
`.env` (already gitignored).

| Variable | Required | Purpose |
|---|---|---|
| `GROQ_API_KEY` | yes* | Primary LLM (script/story/research/JSON). `_SECOND`/`_THIRD` keys rotate on failure. |
| `GEMINI_API_KEY` (`_ONE`..`_FIVE`) | fallback | Gemini text fallback + topic brainstormer. |
| `CLOUDFLARE_API_TOKEN` + `CLOUDFLARE_ACCOUNT_ID` | fallback | Cloudflare Workers AI, last-resort LLM. |
| `PEXELS_API_KEY` | assets | Image search/asset fetching. |
| `POCKET_VOICE_REF` | if no voice | Path to a reference `.wav` used to build the narrator voice state (see Voice). |
| `HF_HOME` | recommended | Where HuggingFace caches TTS models (e.g. `<repo>/.models`). |
| `REDDIT_CLIENT_ID` / `REDDIT_CLIENT_SECRET` / `REDDIT_USERNAME` / `REDDIT_PASSWORD` | topic source | Optional Reddit OAuth ("script" app at reddit.com/prefs/apps). |
| `YOUTUBE_CLIENT_SECRETS` / `YOUTUBE_TOKEN_FILE` | upload | YouTube OAuth files (defaults `client_secrets.json`/`token.json`). |
| `YOUTUBE_PRIVACY_STATUS` | | `public`/`private`/`unlisted` (default `private`). |
| `YOUTUBE_REDIRECT_PORT` | | OAuth loopback port (default `8080`). |
| `YOUTUBE_CATEGORY_ID` | | Video category (default `27` = Education). |
| `YOUTUBE_TAGS` | | Comma-separated tags appended to every upload. |
| `TTS_MIN_WPS` / `TTS_MAX_WPS` / `LOCAL_MODEL` / `CAPTIONS_ENABLED` / `ASSET_*` / `STAGE_TIMEOUT_S` / `MAX_*` | tuning | Optional behavioral knobs; each has a sane default. |

\* At least one text LLM must be configured; the chain is Groq → Gemini → Cloudflare.

## Voice

The pipeline uses the Pocket-TTS "narrator" voice exclusively (see
`agents/voice/registry.py`). **The voice does not ship in the repo** —
`voices/` is gitignored, so a fresh clone has no voice. Pick one:

1. **Prebuilt state** — place `voices/narrator.safetensors` from an existing
   run/drive into the repo (works with either model variant).
2. **Build it from a reference wav** — put a 5–15s clean narration `.wav`
   somewhere and set `POCKET_VOICE_REF` in `.env`. The first run builds
   `voices/narrator.safetensors` from it (then it can be removed). **This
   path requires the gated model — see "TTS model".**

Without one of these, every TTS run fails with a `FileNotFoundError`.

## TTS model

The narrator runs on Kyutai's Pocket-TTS. The model is NOT bundled in the repo —
it is downloaded from HuggingFace on the first TTS run and cached under
`$HF_HOME/hub` (default `HF_HOME=~/.cache/huggingface`; point it inside the
project, e.g. `HF_HOME=<repo>/.models`, so it is reused and gitignored).

There are two model variants, and which one you get depends on whether you
already have a voice state file:

- **Open model — `kyutai/pocket-tts-without-voice-cloning`** (~a few GB, no
  login). This is what downloads if you have no HuggingFace auth. It works with
  a **prebuilt** `voices/narrator.safetensors` (the repo's normal path).
- **Gated model — `kyutai/pocket-tts`** (needed to **create** a new voice from
  a reference `.wav`). To enable it:

  ```bash
  # 1. Accept the model terms:
  #    https://huggingface.co/kyutai/pocket-tts  -> "Agree and access repository"
  uvx hf auth login          # or set HF_TOKEN=<token> in .env
  ```

  On first run the model downloads into `$HF_HOME/hub/models--kyutai--pocket-tts*`.

> Cold-start note: the download is a few GB and takes a while on slow
> connections. If `TTSModel.load_model()` can't fetch the gated model, the
> library silently uses the open variant — so a missing login only surfaces
> when you later try to build a new voice (`POCKET_VOICE_REF`), not at load.

## LLM assets

Groq/Gemini/Cloudflare credentials from `.env` are enough. See "TTS model" for
the TTS model download.

## YouTube upload (optional)

Handled on the first `--upload` run:

1. Put your Google Cloud OAuth client in `client_secrets.json`
   (or set `YOUTUBE_CLIENT_SECRETS`).
2. If it is a "Web application" client, register the redirect URI
   `http://localhost:8080/` (and `http://127.0.0.1:8080/`) in the Cloud
   Console; desktop-app clients accept any loopback port automatically.
3. Run the pipeline once; the tool opens a browser, you authorize, and
   `token.json` is saved (gitignored). Subsequent runs reuse the token.

`--no-upload` skips this entirely.

## Run

The CLI bootstraps `sys.path` and loads `.env`, so it works from any CWD:

```bash
# Single video for one topic
uv run python src/cli/main.py "The mystery of the Dyatlov Pass"

# No YouTube upload
uv run python src/cli/main.py "Topic" --no-upload

# Research + batch-render topics, or reuse a saved topic batch
uv run python src/cli/main.py --batch
uv run python src/cli/main.py --batch --use-saved

# Stages / helpers
uv run python src/cli/generate_topics.py --target 20 --save-only
uv run python src/cli/research_topics.py --no-miner
uv run python src/cli/main.py --list-voices
uv run python src/cli/main.py --script-only          # topics + scripts only
```

Equivalent module form: `uv run python -m src ...`.

## Layout

```
src/cli/            entry points (main, generate_topics, research_topics)
src/agents/         per-domain agents (topic, research, story, script, scene, asset, voice)
src/services/       TTS, DSP, providers/LLM, captions, video assembly, YouTube
src/pipeline/       orchestration (create_video, run_batch, lab)
src/utils/          shared file helpers
src/tests/          unit tests (pytest)
```

Run the standalone tests directly (each is self-running):

```bash
.venv/bin/python -u src/tests/script_lab_test.py            # staged script pipeline
.venv/bin/python src/tests/test_channel_miner.py            # channel mining
.venv/bin/python src/tests/test_youtube_upload.py           # YouTube upload
```

## First-run checklist

1. `ffmpeg`, `sox`, `tesseract` on PATH.
2. `cp .env.example .env` with at least one LLM key + `PEXELS_API_KEY`.
3. Voice present (`voices/narrator.safetensors`) or `POCKET_VOICE_REF` set.
4. `output/`, `temp/`, `.models/` are created on demand and gitignored.