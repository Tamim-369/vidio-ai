"""Chatterbox TTS experiment.

Compare Chatterbox against the current Kokoro narrator voice.
Run:  .venv-chatterbox/bin/python test_chatterbox.py
"""
import time
import torchaudio as ta
import torch
from chatterbox.tts import ChatterboxTTS
from chatterbox.mtl_tts import ChatterboxMultilingualTTS

OUT = "test_output"

if torch.cuda.is_available():
    device = "cuda"
elif torch.backends.mps.is_available():
    device = "mps"
else:
    device = "cpu"
print(f"Using device: {device}")

TEXT = "First up, the Army poured a fortune into a secret balloon project."
REF = "temp/chatterbox_ref.wav"

print("\n--- English base model (default voice) ---")
t0 = time.time()
model = ChatterboxTTS.from_pretrained(device=device)
wav = model.generate(TEXT)
ta.save(f"{OUT}/chatterbox_default.wav", wav, model.sr)
print(f"  saved chatterbox_default.wav ({time.time()-t0:.1f}s)")

print("\n--- English base model (cloned from Kokoro ref) ---")
t0 = time.time()
wav = model.generate(TEXT, audio_prompt_path=REF)
ta.save(f"{OUT}/chatterbox_clone.wav", wav, model.sr)
print(f"  saved chatterbox_clone.wav ({time.time()-t0:.1f}s)")

print("\n--- Multilingual v3 (default voice) ---")
t0 = time.time()
m2 = ChatterboxMultilingualTTS.from_pretrained(device=device, t3_model="v3")
wav = m2.generate(TEXT, language_id="en")
ta.save(f"{OUT}/chatterbox_multi_v3.wav", wav, m2.sr)
print(f"  saved chatterbox_multi_v3.wav ({time.time()-t0:.1f}s)")

print("\nDone. Files in", OUT)