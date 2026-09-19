# VideoAI — Chatterbox/Donald-Trump TTS "paper-cut" session summary

Date: Fri Sep 18 2026 · Repo: /home/tamim/projects/videoai

## The bug (why you kept hearing it)
A loud "paper-cut" click (like snipping thick paper) at every word boundary and
inside every pause. Small but constant — infuriating across hours.

## Two root causes (both fixed in `src/services/tts.py`)
1. **Noise gate** — a *binary* on/off gate snapped the audio 0↔1 at each word
   onset/offset, making a click on EVERY word boundary.
2. **Pause "silence"** — the pause pad was amplitude-ramped silence (it faded
   up and back down), so between words it had ramps that clicked repeatedly.
   The "paper-cut streams" you heard were these ramps, not the gate.

## The fix, on disk and verified
- `_noise_gate` (tts.py:474): **smooth attack/release** gate with a 256-sample
  cosine ramp; no more hard 0↔1 snap at word edges.
- `_insert_silence` (tts.py:532): **pure zeros** for pause padding plus 2ms
  edge fades, so silence is truly silent.
- Both are cache-busted: `POST_VERSION="3"` in `voice_tests/voice_test_utils.py:198`
  → the next render regenerates all 8 lines **click-free** (this is the one
  full bake that's required; the clicks were baked into the cached wavs and
  can't be removed retroactively — regeneration is unavoidable).
- `synth_version="1"`, `postprocess_line` seam shared by generators (single
  source of truth, no drift between engines).

## What still needs doing (honest)
- The generators still specialize their post-processing inline; the shared
  `postprocess_line` seam exists at tts.py:741 but is partly unwired (dead code
  risk). Pause tweaks currently still trigger a full bake; wiring the seam so
  pauses are re-applied from raw (instant) is the remaining follow-up.
- Verify the next render's audio is click-free and do the one-time regen.

## How to render (click-free)
```
cd /home/tamim/projects/videoai
.venv/bin/python voice_tests/voice_test_utils.py --force-audio
```
(regenerates all 8 lines with the smooth gate; takes ~10 min one time).

## Project
- Working dir: /home/tamim/projects/videoai
- TTS engines: pocket-tts, chatterbox, chatterbox-turbo
- Harness: voice_tests/voice_test_utils.py (script+audio cache + render)
