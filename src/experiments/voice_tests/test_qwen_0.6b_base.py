"""Qwen3-TTS 0.6B Base voice clone test.

Clones the current Kokoro narrator voice and synthesizes the "First up" line.
Run:  .venv-qwen/bin/python voice_tests/test_qwen_0.6b_base.py
"""
import os
import time
import torch
import soundfile as sf

from qwen_tts import Qwen3TTSModel

MODEL = "Qwen/Qwen3-TTS-12Hz-0.6B-Base"
REF = os.path.join(os.path.dirname(__file__), "chatterbox_ref.wav")
OUT = os.path.join(os.path.dirname(__file__), "test_output")
os.makedirs(OUT, exist_ok=True)

# Transcript of voice_tests/chatterbox_ref.wav (what Kokoro actually spoke)
REF_TEXT = (
    "First up, the Army poured a fortune into a secret balloon project. "
    "By one thousand nine hundred forty seven, the balloons had drifted "
    "into public view, and the truth was about to get weird."
)

SYN_TEXT = "First up, the Army poured a fortune into a secret balloon project."

print(f"Loading {MODEL} on CPU...")
t0 = time.time()
tts = Qwen3TTSModel.from_pretrained(
    MODEL,
    device_map="cpu",
    dtype=torch.float32,
)
print(f"  loaded in {time.time()-t0:.1f}s")

# Build the voice clone prompt once (reusable across lines)
print("Building voice clone prompt...")
prompt_items = tts.create_voice_clone_prompt(
    ref_audio=REF,
    ref_text=REF_TEXT,
    x_vector_only_mode=False,
)

lines = [
    "First up, the Army poured a fortune into a secret balloon project.",
    "By 1947, the balloons had drifted into public view, and the truth was about to get weird.",
    "The military kept the whole thing buried for decades.",
    "Ten seconds of footage, and a legend was born.",
]
for i, line in enumerate(lines):
    t0 = time.time()
    wavs, sr = tts.generate_voice_clone(
        text=line,
        language="English",
        voice_clone_prompt=prompt_items,
        max_new_tokens=2048,
    )
    path = os.path.join(OUT, f"qwen_0.6b_base_line{i}.wav")
    sf.write(path, wavs[0], sr)
    print(f"  line{i}: {time.time()-t0:.1f}s -> {path}")

print("Done.")