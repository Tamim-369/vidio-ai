# Setup

## What this project does

Builds faceless YouTube Shorts videos automatically. The pipeline:

1. picks viral/mystery topics (Reddit, competitor channel mining, Wikipedia)
2. researches each topic
3. writes a story and a script
4. generates voiceover audio with Pocket-TTS
5. fetches images and captions
6. assembles everything into an MP4
7. optionally uploads to YouTube

## What you need before you start

### Hardware

- Any modern CPU is enough. A GPU makes TTS generation faster but is not required.

### System packages

These three programs must be on your `PATH`. The code calls them directly.

| Program | Used for | Ubuntu/Debian | macOS | Windows |
|---|---|---|---|---|
| `ffmpeg` | video assembly | `sudo apt install ffmpeg` | `brew install ffmpeg` | `winget install ffmpeg` |
| `sox` | TTS audio DSP | `sudo apt install sox` | `brew install sox` | `winget install sox` |
| `tesseract` | OCR text filter on images | `sudo apt install tesseract-ocr` | `brew install tesseract` | `winget install tesseract` |

### Software

- Python 3.13 (with a `.python-version` file, `uv` picks it automatically; 3.12 also works)
- [uv](https://docs.astral.sh/uv/) (recommended) or plain `pip`/`venv`
- `git`

> If you use `uv`, install it with: `curl -LsSf https://astral.sh/uv/install.sh | sh`

### Accounts and API keys (optional except the first one)

The pipeline needs at least one AI text model to write scripts. Everything else
is optional and only enables that feature.

| Account | Where to get it | Needed for | Required? |
|---|---|---|---|
| Groq | https://console.groq.com/keys | main text LLM (best output) | at least one of the three LLM options |
| Gemini | https://aistudio.google.com/apikey | text LLM fallback + topic brainstorming | fallback |
| Cloudflare Workers AI | https://dash.cloudflare.com → Workers AI | text LLM last resort | last resort |
| Pexels | https://www.pexels.com/api/ | image search for video | for image fetching |
| HuggingFace | https://huggingface.co/settings/tokens | download TTS model | if you build a new voice (see TTS model) |
| Reddit | https://www.reddit.com/prefs/apps | topic source (script app) | optional |
| Google Cloud / YouTube | https://console.cloud.google.com | uploading to YouTube | optional |

You do not need all of them. The absolute minimum to render a video is:
- one LLM key (Groq preferred)
- a Pexels API key
- a voice (see Voice section)

## Step 1. Clone the repository

```bash
git clone https://github.com/Tamim-369/vidio-ai.git
cd vidio-ai
```

## Step 2. Install dependencies

### Option A: one command (recommended)

```bash
bash setup.sh
```

The script does everything for you:

1. checks that `ffmpeg`, `sox`, and `tesseract` are installed
2. detects `uv` or `pip`
3. creates a virtual environment in `.venv`
4. installs everything in `requirements.txt`
5. installs the Playwright Chromium browser (optional feature)
6. creates `.env` from `.env.example` if you do not have one
7. checks whether the narrator voice exists and tells you the next step

Useful flags:

```bash
bash setup.sh --no-playwright     # skip the optional ~150 MB browser download
bash setup.sh --no-voice-check    # do not check for the narrator voice
```

### Option B: manual steps

With uv:

```bash
uv venv
uv pip install -r requirements.txt
uv run playwright install chromium   # optional
```

With plain pip:

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt
playwright install chromium          # optional
```

## Step 3. Configure `.env`

The project reads its configuration from a file named `.env` in the repo root.
Start from the template:

```bash
cp .env.example .env
```

Then open `.env` and fill in your keys. The file is loaded automatically by the
CLI and by every module that reads a setting. You never need to `export`
anything in your shell, and the file is gitignored so it will never be
committed.

### All supported variables

| Variable | Purpose | Required |
|---|---|---|
| `GROQ_API_KEY` | primary LLM | at least one LLM key |
| `GROQ_API_KEY_SECOND` | second Groq key, rotated on failure | no |
| `GROQ_API_KEY_THIRD` | third Groq key, rotated on failure | no |
| `LLM_PROVIDER` | `groq` (default) or `ollama`; when `groq`, the 3 Groq keys above are used, otherwise local Ollama | no |
| `CLOUDFLARE_API_TOKEN` | Cloudflare Workers AI access token | no |
| `CLOUDFLARE_ACCOUNT_ID` | Cloudflare account id | no |
| `PEXELS_API_KEY` | image search | for images |
| `HF_HOME` | where TTS models are cached, e.g. `<repo>/.models` | recommended |
| `HF_TOKEN` | HuggingFace token, only if building a new voice | only for new voice |
| `POCKET_VOICE_REF` | path to a reference `.wav` to build the narrator voice | only if no prebuilt voice |
| `REDDIT_CLIENT_ID` | Reddit script app client id | only for Reddit topics |
| `REDDIT_CLIENT_SECRET` | Reddit script app client secret | only for Reddit topics |
| `REDDIT_USERNAME` | Reddit username | only for Reddit topics |
| `REDDIT_PASSWORD` | Reddit password | only for Reddit topics |
| `YOUTUBE_CLIENT_SECRETS` | path to the YouTube OAuth client file (default `client_secrets.json`) | only for upload |
| `YOUTUBE_TOKEN_FILE` | path to the saved OAuth token (default `token.json`) | only for upload |
| `YOUTUBE_PRIVACY_STATUS` | `public`, `private`, or `unlisted` (default `private`) | only for upload |
| `YOUTUBE_REDIRECT_PORT` | OAuth loopback port (default `8080`) | only for upload |
| `YOUTUBE_CATEGORY_ID` | video category id (default `27` = Education) | only for upload |
| `YOUTUBE_TAGS` | comma separated tags added to every upload | no |

Optional tuning knobs (each has a sane default, skip unless you want to change
behavior):

| Variable | Purpose |
|---|---|
| `TTS_MIN_WPS` / `TTS_MAX_WPS` | speaking-rate band for TTS pacing |
| `CAPTIONS_ENABLED` | enable or disable captions |
| `LOCAL_MODEL` | model override |
| `ASSET_*` | image fetching behavior (count, aspect ratio, text slop filter) |
| `STAGE_TIMEOUT_S` | per-stage timeout |
| `MAX_EVAL_ROUNDS`, `MAX_FACT_FIXES`, `MAX_SCENE_ATTEMPTS` | story retry budgets |

> The LLM fallback chain is: Groq, then Gemini, then Cloudflare. Only the first
> key you provide is read, the others are only used when the earlier ones fail.

## Step 4. The TTS model

The narrator uses Kyutai's Pocket-TTS. The model is **not** bundled in the
repo. It downloads from HuggingFace on the first TTS run and is cached under
`$HF_HOME/hub` (default `~/.cache/huggingface`, set `HF_HOME` in `.env` to
keep it inside the project, for example `HF_HOME=.models`).

There are two model variants:

| Variant | HuggingFace repo | Auth needed | Works with |
|---|---|---|---|
| Open | `kyutai/pocket-tts-without-voice-cloning` | no | a prebuilt `voices/narrator.safetensors` |
| Gated | `kyutai/pocket-tts` | yes | also lets you build a new voice from a `.wav` |

If no HuggingFace auth is available, the library silently downloads the open
variant and the prebuilt narrator voice still works. You only need the gated
model if you want to create a new voice from a reference audio file.

To enable the gated model:

```bash
# 1. open https://huggingface.co/kyutai/pocket-tts and click "Agree and access repository"
# 2. log in locally:
uvx hf auth login
#    or set HF_TOKEN=<your token> in .env
```

The download is a few GB and happens only once, on the first TTS run.

## Step 5. The narrator voice

The repo does **not** contain the voice. The `voices/` directory is gitignored,
so a fresh clone has no voice file. Pick one option.

### Option 1: use a prebuilt voice

Drop a narrator voice state file here:

```bash
voices/narrator.safetensors
```

You can copy it from an existing installation or a backup. This works with
either model variant.

### Option 2: build it from a reference wav

Put a 5 to 15 second clean narration `.wav` anywhere on disk and point to it in
`.env`:

```
POCKET_VOICE_REF=/absolute/path/to/your_sample.wav
```

On the first run, the pipeline builds `voices/narrator.safetensors` from it.
The reference wav can then be removed. This option requires the gated model
(see the TTS model section above).

Without one of these two options, the pipeline stops with a
`FileNotFoundError` when it reaches the TTS stage.

## Step 6. Optional integrations

### Reddit topics (optional)

Create a "script" app at https://www.reddit.com/prefs/apps and fill in
`REDDIT_CLIENT_ID`, `REDDIT_CLIENT_SECRET`, `REDDIT_USERNAME`, and
`REDDIT_PASSWORD` in `.env`.

### YouTube upload (optional)

Uploading is on by default for the main command. Skip it with `--no-upload`.

1. Create an OAuth client in Google Cloud Console and download it as
   `client_secrets.json` (or point `YOUTUBE_CLIENT_SECRETS` at it).
2. If the client type is "Web application", register the redirect URI
   `http://localhost:8080/` (and `http://127.0.0.1:8080/`). Desktop app
   clients accept any loopback port automatically.
3. Run the pipeline once. A browser opens, you authorize the app, and the token
   is saved to `token.json` (gitignored). Later runs reuse the token.

## Step 7. Verify the install

```bash
uv run python src/cli/main.py --list-voices
```

This loads the CLI, connects the TTS model path, and prints the registered
voices. Expect to see `narrator`. The first invocation downloads the TTS model
if it is not cached yet.

## Step 8. Run

Run from anywhere; the CLI bootstraps `sys.path` and loads `.env` itself.

```bash
# one video for a topic you supply
uv run python src/cli/main.py "The mystery of the Dyatlov Pass"

# same, but do not upload to YouTube
uv run python src/cli/main.py "Topic" --no-upload

# research topics and render a batch
uv run python src/cli/main.py --batch

# reuse an already saved topic batch
uv run python src/cli/main.py --batch --use-saved

# topics + scripts only, no assets/audio/video/upload
uv run python src/cli/main.py --script-only

# topic generation helper
uv run python src/cli/generate_topics.py --target 20 --save-only

# topic research helper
uv run python src/cli/research_topics.py --no-miner
```

The same commands work with `python -m src` instead of `python
src/cli/main.py`.

Main CLI flags:

| Flag | Meaning |
|---|---|
| `topic` (positional) | single topic to make a video for |
| `--batch` | generate topics and render all of them |
| `--use-saved` | use a saved topic batch instead of generating |
| `--upload` / `--no-upload` | upload (default) or skip YouTube upload |
| `--voice VOICE` | force a voice id |
| `--list-voices` | list registered voices and exit |
| `--script-only` | stop after topics + scripts |
| `--limit N` | posts per source when researching (default 100) |
| `--target N` | how many topics to research (default 24) |
| `--concurrency N` | render up to N videos at once in batch mode (moving assembly line; default 1) |

## Tests

The tests are standalone scripts, not pytest. Run each one directly:

```bash
# staged script pipeline (topics -> story -> script -> scene)
uv run python -u src/tests/script_lab_test.py

# channel mining
uv run python src/tests/test_channel_miner.py

# YouTube upload
uv run python src/tests/test_youtube_upload.py
```

## Layout

```
src/cli/            entry points (main, generate_topics, research_topics)
src/agents/         per-domain agents (topic, research, story, script, scene, asset, voice)
src/services/       TTS, DSP, providers/LLM, captions, video assembly, YouTube
src/pipeline/       orchestration (create_video, run_batch, lab)
src/utils/          shared file helpers
src/tests/          standalone tests
```

## First-run checklist

1. `ffmpeg`, `sox`, and `tesseract` are on `PATH`.
2. `.venv` exists and `requirements.txt` is installed.
3. `.env` exists with at least one LLM key and `PEXELS_API_KEY`.
4. A voice exists: `voices/narrator.safetensors` or `POCKET_VOICE_REF` set.
5. `output/`, `temp/`, and the HF cache dir are created on demand and gitignored.

## Troubleshooting

| Symptom | Cause and fix |
|---|---|
| `command not found: uv` | install uv: `curl -LsSf https://astral.sh/uv/install.sh | sh` |
| `[setup] ERROR: 'sox' is required...` | install the system package (see table above) |
| `No Pocket-TTS narrator voice state` | you have no voice; do Step 5 |
| `VOICE_CLONING_UNSUPPORTED` | you tried to build a voice without the gated model; do Step 4 |
| `[cf] CLOUDFLARE_API_TOKEN ... missing` | LLM chain ran out of keys; add a Groq key |
| `[groq] all N Groq keys failed` | all Groq keys failed; set `LLM_PROVIDER=ollama` or add valid keys |
| upload not working | check YouTube OAuth setup, Step 6 |
| first run very slow | the TTS model is downloading; it is a one time download |
| `where are the videos` | they are written to `output/<slug>.mp4` |