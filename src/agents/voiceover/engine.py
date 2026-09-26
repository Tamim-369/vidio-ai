import os

# Set HuggingFace cache dir before importing pocket_tts so models land in project folder
from dotenv import load_dotenv
load_dotenv()
hf_home = os.getenv("HF_HOME")
if hf_home:
    os.environ["HF_HOME"] = hf_home

import numpy as np
import soundfile as sf
TEMP_DIR = "temp"


TTS_RATE = float(os.getenv("TTS_RATE", "0.93"))

from src.agents.voiceover.normalize import _clean_text
from src.agents.voiceover.dsp import LOUD_GAIN, _de_shout, postprocess_line
from src.agents.voiceover.workers import (
    _effective_workers,
    _parallel_lines,
    _setup_worker_rng,
)
from src.agents.voiceover.models import (
    _chat_sr,
    _get_chatterbox_model,
    _get_pocket_model,
    _get_voice_state,
    _prep_chat_conds,
)

# --- Silence known harmless third-party warnings from Chatterbox's internals ---
# (all fired by chatterbox's own deps during model load / generation, not our code)
import warnings
warnings.filterwarnings("ignore",
    message=r"`LoRACompatibleLinear` is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings("ignore",
    message=r"`torch\.backends\.cuda\.sdp_kernel\(\)` is deprecated",
    category=FutureWarning,
)
warnings.filterwarnings("ignore",
    message=r"`output_attentions=True` is not supported with `attn_implementation`",
    category=UserWarning,
)
import logging
logging.getLogger("transformers.integrations.sdpa_attention").setLevel(logging.ERROR)

_model = None
_voice_state = None

def _apply_rate(samples: np.ndarray, sr: int) -> np.ndarray:
    """Stretch narration to TTS_RATE without a resampling pitch shift.

Chatterbox 0.1.7 has no rate control and its sampling knobs shift style rather
than tempo, so this is an ffmpeg atempo time-stretch (pitch preserved). Applied
to the raw array before the wav is written, so the measured actual_duration
already reflects it and card lengths stay correct."""
    if TTS_RATE == 1.0:
        return samples
    import subprocess
    import tempfile

    # atempo only accepts 0.5-2.0 per instance; chain factors outside that.
    factor = TTS_RATE
    chain = []
    while factor < 0.5:
        chain.append("0.5")
        factor /= 0.5
    while factor > 2.0:
        chain.append("2.0")
        factor /= 2.0
    chain.append(f"{factor:.6f}")
    filt = ",".join(f"atempo={c}" for c in chain)

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp_path = tmp.name
    try:
        sf.write(tmp_path, samples, sr)
        out = subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-i", tmp_path, "-filter:a", filt,
             "-ar", str(sr), "-ac", "1", "-f", "wav", "-"],
            capture_output=True, check=True).stdout
        import io
        stretched, _ = sf.read(io.BytesIO(out), dtype="float32")
        return np.asarray(stretched, dtype=np.float32)
    finally:
        try:
            os.remove(tmp_path)
        except OSError:
            pass


def _chat_loop(lines: list, voice: dict, audio_dir: str) -> list:
    """Generate voiceover with Chatterbox real-voice cloning (English)."""
    from chatterbox.tts import ChatterboxTTS  # noqa: F401  (type ref only)

    model = _get_chatterbox_model()
    params = voice.get("params", {})
    exaggeration = params.get("exaggeration", 0.5)
    cfg_weight = params.get("cfg_weight", 0.5)
    temperature = params.get("temperature", 0.8)
    repetition_penalty = params.get("repetition_penalty", 1.2)
    min_p = params.get("min_p", 0.05)
    top_p = params.get("top_p", 1.0)
    pitch_shift = params.get("pitch_shift", 0.0)

    _prep_chat_conds(model, voice)  # no-op if this process already prepared

    sr = _chat_sr
    for line in lines:
        line_id = line["id"]
        cleaned = _clean_text(line["text"])
        text = _de_shout(cleaned) if line.get("loud") else cleaned
        path = os.path.join(audio_dir, f"{line_id}.wav")

        print(f"  [tts] Line {line_id}: {text}", flush=True)

        wav = model.generate(
            text=text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
            repetition_penalty=repetition_penalty,
            min_p=min_p,
            top_p=top_p,
        )

        if hasattr(wav, 'numpy'):
            combined = wav.squeeze(0).numpy()
        else:
            combined = np.asarray(wav).squeeze()

        combined = combined.astype(np.float32)
        silence = np.zeros(int(0.06 * sr), dtype=np.float32)
        combined = np.concatenate([combined, silence])
        combined = np.concatenate([np.zeros(int(0.06 * sr), dtype=np.float32), combined])
        combined = _apply_rate(combined, sr)
        peak = float(np.abs(combined).max())
        if peak > 0.95:
            combined = combined * (0.95 / peak)
        sf.write(path, combined, sr)
        line["audio_path"] = path
        line["actual_duration"] = len(combined) / sr

    return lines





def _generate_pocket(lines: list, audio_dir: str) -> list:
    from pocket_tts import TTSModel  # noqa: F401  (validates import path for type hints)
    model = _get_pocket_model()
    voice_state = _get_voice_state()
    sr = model.sample_rate

    for line in lines:
        line_id = line["id"]
        cleaned = _clean_text(line["text"])
        text = _de_shout(cleaned) if line.get("loud") else cleaned
        path = os.path.join(audio_dir, f"{line_id}.wav")

        print(f"  [tts] Line {line_id}: {text}")

        max_tokens = max(200, int(len(text.split()) * 14) + 50)
        audio = model.generate_audio(voice_state, text, max_tokens=max_tokens)

        if hasattr(audio, 'numpy'):
            combined = audio.numpy()
        else:
            combined = np.asarray(audio)

        raw_audio = combined.astype(np.float32)
        gain = LOUD_GAIN if line.get("loud") else 1.0
        final, raw = postprocess_line(
            raw_audio, sr, "pocket", text, {"gain": gain}
        )
        # Preserve pre-finalize "raw" wav beside the final one (pause-only
        # re-bakes rebuild from this without touching the model).
        sf.write(os.path.join(audio_dir, f"{line_id}.raw.wav"), raw, sr)
        sf.write(path, final, sr)
        line["audio_path"] = path
        line["actual_duration"] = len(final) / sr

    return lines

def _worker_chatterbox(idx, chunk, voice, audio_dir, seed, threads, queue):
    try:
        _setup_worker_rng(seed, threads)
        out = _chat_loop(chunk, voice, audio_dir)  # model + conds inherited via fork
        queue.put((False, idx, out))
    except Exception as exc:
        queue.put((True, idx, f"{exc!r}"))


def _generate_chatterbox(lines: list, voice: dict, audio_dir: str) -> list:
    """Chatterbox voiceover with parallel line rendering (same model, same params).

    With workers > 1 the parent does NOT load the model itself; each worker loads
    its own copy so the two concurrent renders stay memory-safe (~8.5GB peak).
    """
    workers = _effective_workers(len(lines))
    if workers > 1:
        try:
            return _parallel_lines(workers, _worker_chatterbox, lines, voice, audio_dir)
        except Exception as exc:
            print(f"  [tts] Parallel render failed ({exc}); falling back to sequential.", flush=True)
    _prep_chat_conds(_get_chatterbox_model(), voice)  # sequential: load once here
    return _chat_loop(lines, voice, audio_dir)


def generate_audio(lines: list, voice: dict = None) -> list:
    """Generate voiceover for each line, written to temp/audio/<id>.wav.

    voice: registry entry from src/config/voices.py. If None (or engine
    "pocket") the legacy Pocket-TTS narrator is used.
    """
    audio_dir = os.path.join(TEMP_DIR, "audio")
    os.makedirs(audio_dir, exist_ok=True)

    engine = (voice or {}).get("engine", "pocket")
    if engine == "chatterbox":
        return _generate_chatterbox(lines, voice, audio_dir)
    return _generate_pocket(lines, audio_dir)