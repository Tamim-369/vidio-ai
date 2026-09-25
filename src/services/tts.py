import os
import threading
import multiprocessing
from concurrent.futures import ProcessPoolExecutor

# Set HuggingFace cache dir before importing pocket_tts so models land in project folder
from dotenv import load_dotenv
load_dotenv()
hf_home = os.getenv("HF_HOME")
if hf_home:
    os.environ["HF_HOME"] = hf_home

import numpy as np
import soundfile as sf
from src.utils.file_helpers import workspace_path

# Pocket-TTS narrator voice state. If the state file is missing, a reference
# wav can be used to build it (set POCKET_VOICE_REF to the path).
POCKET_VOICE_STATE = "voices/narrator.safetensors"
POCKET_VOICE_REF = os.getenv("POCKET_VOICE_REF", "")

# Pocket-TTS is NOT thread-safe (its docs say to run separate model instances
# for concurrent generation). Threaded instances give ~no speedup because the
# autoregressive loop is GIL-bound, but PROCESS-level instances parallelize on
# this box's idle cores (~1 core per job). Each worker owns one model + voice.
POCKET_TTS_INSTANCES = max(1, int(os.getenv("POCKET_TTS_INSTANCES", "3")))

# Normalization + DSP extracted into focused modules; re-exported here so callers
# keep working with the old `from src.services.tts import ...` imports.
from src.services.tts_text import _clean_text
from src.services.tts_dsp import (
    LOUD_GAIN,
    _de_shout,
    postprocess_line,
)

_tts_pool = None
_pool_lock = threading.Lock()


# ---------------------------------------------------------------------------
# Worker functions (top-level so ProcessPoolExecutor can pickle them; each
# child process loads ONE model + voice state in its initializer).
# ---------------------------------------------------------------------------

_worker_model = None
_worker_voice = None


def _tts_worker_init():
    global _worker_model, _worker_voice
    # Each worker already owns a core; pin it to one thread so 3 concurrent
    # workers don't oversubscribe and thrash (measured ~1.3x without this vs
    # ~1.56x with it).
    import torch
    torch.set_num_threads(1)
    from pocket_tts import TTSModel
    _worker_model = TTSModel.load_model()
    if os.path.exists(POCKET_VOICE_STATE):
        _worker_voice = _worker_model.get_state_for_audio_prompt(POCKET_VOICE_STATE)
    elif POCKET_VOICE_REF and os.path.exists(POCKET_VOICE_REF):
        _worker_voice = _worker_model.get_state_for_audio_prompt(POCKET_VOICE_REF)
    else:
        raise FileNotFoundError(
            f"No Pocket-TTS narrator voice state at {POCKET_VOICE_STATE} "
            "and no POCKET_VOICE_REF provided for a TTS worker."
        )


def _tts_worker_generate(text: str, loud: bool):
    """Generate audio for one cleaned line; returns (float32 mono, sample_rate)."""
    max_tokens = max(200, int(len(text.split()) * 14) + 50)
    audio = _worker_model.generate_audio(_worker_voice, text, max_tokens=max_tokens)
    arr = audio.numpy() if hasattr(audio, "numpy") else np.asarray(audio)
    return arr.astype(np.float32), _worker_model.sample_rate


# ---------------------------------------------------------------------------
# Pool management
# ---------------------------------------------------------------------------

def _ensure_voice_state_file():
    """Make sure voices/narrator.safetensors exists BEFORE workers spawn.

    Runs in the main process (only when the file is actually missing, i.e. the
    first run), so the parent never needs torch loaded when it forks workers.
    """
    if os.path.exists(POCKET_VOICE_STATE):
        return
    from pocket_tts import TTSModel, export_model_state
    model = TTSModel.load_model()
    if not POCKET_VOICE_REF or not os.path.exists(POCKET_VOICE_REF):
        raise FileNotFoundError(
            f"No Pocket-TTS narrator voice state at {POCKET_VOICE_STATE} "
            "and no POCKET_VOICE_REF provided."
        )
    voice_state = model.get_state_for_audio_prompt(POCKET_VOICE_REF)
    os.makedirs(os.path.dirname(POCKET_VOICE_STATE), exist_ok=True)
    export_model_state(voice_state, POCKET_VOICE_STATE)


def _get_tts_pool():
    """Get-or-create a process pool of Pocket-TTS workers.

    Each worker process loads the model + voice state once (in _tts_worker_init)
    and stays alive across lines/videos, so jobs only pay serialization, not a
    per-line model load. The pool is shared by all batch threads.
    """
    global _tts_pool
    if _tts_pool is None:
        with _pool_lock:
            if _tts_pool is None:
                _ensure_voice_state_file()
                ctx = multiprocessing.get_context("fork")
                _tts_pool = ProcessPoolExecutor(
                    max_workers=POCKET_TTS_INSTANCES,
                    mp_context=ctx,
                    initializer=_tts_worker_init,
                )
                print(f"  [tts] Pocket-TTS process pool ready: {POCKET_TTS_INSTANCES} worker(s)")
    return _tts_pool


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def _generate_pocket(lines: list, audio_dir: str) -> list:
    pool = _get_tts_pool()
    sr = None

    # Submit every line at once so ALL of a video's lines generate in parallel
    # across the pool's workers (plus sibling videos share the same pool).
    jobs = {}
    for line in lines:
        line_id = line["id"]
        cleaned = _clean_text(line["text"])
        text = _de_shout(cleaned) if line.get("loud") else cleaned
        loud = bool(line.get("loud"))
        print(f"  [tts] Line {line_id}: {text}")
        jobs[pool.submit(_tts_worker_generate, text, loud)] = (line, text)

    for fut, (line, text) in jobs.items():
        raw_audio, line_sr = fut.result()
        sr = line_sr
        line_id = line["id"]
        path = os.path.join(audio_dir, f"{line_id}.wav")

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


def generate_audio(lines: list, voice: dict = None, topic: str = "") -> list:
    """Generate voiceover for each line with Pocket-TTS, written to temp/<topic>/audio.

    voice: ignored — the pipeline always uses the Pocket-TTS narrator engine.
    """
    audio_dir = os.path.join(workspace_path(topic), "audio")
    os.makedirs(audio_dir, exist_ok=True)

    return _generate_pocket(lines, audio_dir)