#!/usr/bin/env bash
# One-shot setup for the videoai pipeline.
# Usage:  bash setup.sh            (or:  ./setup.sh)
# Flags:  --no-playwright   skip the Chromium browser download (optional feature)
#         --no-voice-check  skip the narrator-voice availability check
set -euo pipefail

cd "$(dirname "$0")"

info()  { printf "\033[1;34m[setup]\033[0m %s\n" "$*"; }
fail()  { printf "\033[1;31m[setup] ERROR:\033[0m %s\n" "$*"; exit 1; }
ok()    { printf "\033[1;32m[setup]\033[0m %s\n" "$*"; }

SKIP_PLAYWRIGHT=0
SKIP_VOICE=0
for arg in "$@"; do
  case "$arg" in
    --no-playwright)  SKIP_PLAYWRIGHT=1 ;;
    --no-voice-check) SKIP_VOICE=1 ;;
    -h|--help)        sed -n '2,5p' "$0" | sed 's/^# //'; exit 0 ;;
    *) echo "Unknown flag: $arg (see: bash setup.sh --help)" >&2; exit 1 ;;
  esac
done

echo "────────────────────────────────────────────────────"
info "videoai setup"
echo "────────────────────────────────────────────────────"

# --- 1. Prerequisite binaries ---------------------------------------------
for bin in ffmpeg sox tesseract; do
  if command -v "$bin" >/dev/null 2>&1; then
    ok "found $bin: $(command -v "$bin")"
  else
    fail "'$bin' is required but not on PATH (install it, e.g. sudo apt install $bin)"
  fi
done

# --- 2. Python + package manager ------------------------------------------
PY="" 
if command -v uv >/dev/null 2>&1; then
  PY="uv"
elif command -v python3 >/dev/null 2>&1; then
  PY="pip"
else
  fail "neither 'uv' nor 'python3' found — install Python 3.13 (see setup.md)"
fi
ok "package tool: $PY"

# --- 3. Virtualenv + dependencies ------------------------------------------
if [ ! -d .venv ]; then
  info "creating .venv ..."
  if [ "$PY" = "uv" ]; then uv venv; else python3 -m venv .venv; fi
fi
info "installing requirements.txt ..."
if [ "$PY" = "uv" ]; then
  uv pip install -r requirements.txt
else
  ./.venv/bin/pip install -r requirements.txt
fi
ok "dependencies installed"

# --- 4. Playwright browser (optional) --------------------------------------
if [ "$SKIP_PLAYWRIGHT" -eq 0 ]; then
  info "installing Playwright Chromium (optional; ~150 MB, first download) ..."
  if [ "$PY" = "uv" ]; then uv run playwright install chromium; else ./.venv/bin/playwright install chromium; fi
  ok "playwright chromium installed"
else
  info "skipping playwright (topic generation falls back to plain requests)"
fi

# --- 5. .env from template ---------------------------------------------------
if [ ! -f .env ]; then
  [ -f .env.example ] || fail "missing .env.example — re-clone the repo"
  cp .env.example .env
  ok "created .env from .env.example — edit it and add your keys"
else
  ok ".env already exists"
fi

# --- 6. Narrator voice --------------------------------------------------------
if [ "$SKIP_VOICE" -eq 0 ]; then
  if [ -f voices/narrator.safetensors ]; then
    ok "narrator voice found: voices/narrator.safetensors"
  else
    if grep -q "^POCKET_VOICE_REF=" .env 2>/dev/null && ! grep -q "^POCKET_VOICE_REF=$" .env 2>/dev/null; then
      info "POCKET_VOICE_REF is set — the voice will be built on first run"
      info "(requires the gated pocket-tts model: accept terms on HF + 'uvx hf auth login')"
    else
      info "no narrator voice yet: add 'voices/narrator.safetensors' or set"
      info "POCKET_VOICE_REF=<path to a short .wav> in .env (see setup.md → Voice)"
    fi
  fi
fi

echo "────────────────────────────────────────────────────"
ok "Setup complete."
echo ""
echo "Next:"
echo "  1. Edit .env and add at least one LLM key + PEXELS_API_KEY."
echo "  2. Run:  uv run python src/cli/main.py --list-voices"
echo "           uv run python src/cli/main.py \"a topic\" --no-upload"
echo ""
echo "First run downloads the Pocket-TTS model into HF_HOME (a few GB)."
echo "If you build a new voice from a wav, also accept terms at"
echo "https://huggingface.co/kyutai/pocket-tts and run:  uvx hf auth login"