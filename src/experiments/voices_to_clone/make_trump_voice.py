#!/usr/bin/env python3
"""Trump voice clone (English) via Chatterbox — zero-shot voice cloning.

Run:
    source ../../.venv/bin/activate
    python make_trump_voice.py "text for trump to say"
    python make_trump_voice.py                      # interactive prompt

Output: trump_<slug>.wav in this folder.
"""

import sys
import time
from datetime import datetime
from pathlib import Path

HERE = Path(__file__).resolve().parent
REF_AUDIO = str(HERE / "candidates" / "donald-trump" / "donald-trump_ref.wav")


def slugify(text: str) -> str:
    slug = "".join(c if c.isalnum() else "_" for c in text.lower())[:40].rstrip("_")
    return slug or "voice"


def main():
    if len(sys.argv) > 1:
        text = " ".join(sys.argv[1:])
    else:
        text = input("Enter text for Trump to speak: ").strip()

    if not text:
        raise SystemExit("No text given.")

    out_path = HERE / f"trump_{slugify(text)}.wav"

    from chatterbox import ChatterboxTTS

    print("Loading ChatterboxTTS (model is cached ~3.5 GB)...", flush=True)
    t0 = time.time()
    model = ChatterboxTTS.from_pretrained("cpu")
    print(f"Model loaded in {time.time()-t0:.0f}s", flush=True)

    print("Generating speech ...", flush=True)
    t1 = time.time()
    wav = model.generate(
        text=text,
        audio_prompt_path=REF_AUDIO,
        exaggeration=0.5,
        cfg_weight=0.5,
        temperature=0.8,
    )  # torch tensor, shape (1, N), 24 kHz
    print(f"Generated in {time.time()-t1:.0f}s", flush=True)

    import soundfile as sf

    sr = 24000
    sf.write(out_path, wav.squeeze(0).numpy(), sr)
    dur = wav.shape[1] / sr
    print(f"Saved {out_path}  ({dur:.1f}s @ {sr} Hz)", flush=True)


if __name__ == "__main__":
    main()