"""Pocket-TTS test.

Clones the current Kokoro narrator voice and synthesizes the "First up" line.
Run:  .venv/bin/python voice_tests/test_pocket_tts.py
"""
import os
import time
import scipy.io.wavfile
from pocket_tts import TTSModel

OUT = os.path.join(os.path.dirname(__file__), "test_output")
REF = os.path.join(os.path.dirname(__file__), "chatterbox_ref.wav")
os.makedirs(OUT, exist_ok=True)

LINES = [
    "First up, the Army poured a fortune into a secret balloon project.",
    "By 1947, the balloons had drifted into public view, and the truth was about to get weird.",
    "The military kept the whole thing buried for decades.",
    "Ten seconds of footage, and a legend was born.",
]

print("Loading Pocket-TTS model...")
t0 = time.time()
tts = TTSModel.load_model()
print(f"  loaded in {time.time()-t0:.1f}s, sample_rate={tts.sample_rate}")

# 1) Clone from the Kokoro narrator clip
print("\n--- Clone from Kokoro narrator (chatterbox_ref.wav) ---")
t0 = time.time()
voice_state = tts.get_state_for_audio_prompt(REF)
print(f"  voice prompt built in {time.time()-t0:.1f}s")

for i, line in enumerate(LINES):
    t0 = time.time()
    audio = tts.generate_audio(voice_state, line)
    path = os.path.join(OUT, f"pocket_clone_line{i}.wav")
    scipy.io.wavfile.write(path, tts.sample_rate, audio.numpy())
    print(f"  line{i}: {time.time()-t0:.1f}s -> {path}")

# 2) A couple of stock English voices for comparison
print("\n--- Stock voices ---")
for voice in ["peter_yearsley", "george"]:
    try:
        t0 = time.time()
        vs = tts.get_state_for_audio_prompt(voice)
        audio = tts.generate_audio(vs, LINES[0])
        path = os.path.join(OUT, f"pocket_{voice}_line0.wav")
        scipy.io.wavfile.write(path, tts.sample_rate, audio.numpy())
        print(f"  {voice}: {time.time()-t0:.1f}s -> {path}")
    except Exception as e:
        print(f"  {voice}: FAILED {e}")

print("\nDone.")