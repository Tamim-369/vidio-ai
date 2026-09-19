import os
import multiprocessing as mp

# Set HuggingFace cache dir before importing pocket_tts so models land in project folder
from dotenv import load_dotenv
load_dotenv()
hf_home = os.getenv("HF_HOME")
if hf_home:
    os.environ["HF_HOME"] = hf_home

import numpy as np
import soundfile as sf
from src.config.settings import TEMP_DIR, POCKET_VOICE_STATE, POCKET_VOICE_REF, TTS_WORKERS, TTS_LEAD_BUFFER

# Normalization + DSP extracted into focused modules; re-exported here so callers
# keep working with the old `from src.services.tts import ...` imports.
from src.services.tts_text import _clean_text
from src.services.tts_dsp import (
    LOUD_GAIN,
    _de_shout,
    _enforce_pauses,
    _finalize,
    _normalize_pacing,
    _strip_lead_buffer,
    postprocess_line,
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

# Chatterbox engine state (lazy, cached across lines)
_chat_model = None
_chat_ref = None
_chat_sr = 24000


def _conds_cache_path(ref_audio: str, tag: str) -> str:
    """Persist prepared voice conditionals next to the reference so the costly
    voice-cloning embedding is computed once per audio, not every run."""
    import hashlib
    st = os.stat(ref_audio)
    key = hashlib.sha1(f"{ref_audio}:{st.st_size}:{st.st_mtime}".encode()).hexdigest()[:12]
    return os.path.join(os.path.dirname(ref_audio), f"{os.path.basename(ref_audio)}.{tag}.{key}.pts")


def _get_pocket_model():
    global _model
    if _model is None:
        from pocket_tts import TTSModel
        _model = TTSModel.load_model()
    return _model


def _get_voice_state():
    global _voice_state
    if _voice_state is None:
        from pocket_tts import export_model_state
        model = _get_pocket_model()
        if os.path.exists(POCKET_VOICE_STATE):
            _voice_state = model.get_state_for_audio_prompt(POCKET_VOICE_STATE)
        else:
            _voice_state = model.get_state_for_audio_prompt(POCKET_VOICE_REF)
            os.makedirs(os.path.dirname(POCKET_VOICE_STATE), exist_ok=True)
            export_model_state(_voice_state, POCKET_VOICE_STATE)
    return _voice_state
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


def _get_chatterbox_model():
    global _chat_model
    if _chat_model is None:
        from chatterbox import ChatterboxTTS
        print("  [tts] Loading Chatterbox model (cached)...", flush=True)
        _chat_model = ChatterboxTTS.from_pretrained("cpu")
    return _chat_model


def _prep_chat_conds(model, voice: dict) -> None:
    """Load/build Chatterbox voice conditionals in the calling process."""
    global _chat_ref
    ref_audio = voice["ref_audio"]
    if _chat_ref == ref_audio:
        return
    if not os.path.exists(ref_audio):
        raise FileNotFoundError(f"Voice reference audio missing: {ref_audio}")
    cache_path = _conds_cache_path(ref_audio, "chat")
    if os.path.exists(cache_path):
        from chatterbox.tts import Conditionals
        print(f"  [tts] Loading cached voice reference: {os.path.basename(ref_audio)}", flush=True)
        model.conds = Conditionals.load(cache_path)
    else:
        print(f"  [tts] Preparing voice reference: {ref_audio}", flush=True)
        model.prepare_conditionals(ref_audio, exaggeration=voice.get("params", {}).get("exaggeration", 0.5))
        try:
            model.conds.save(cache_path)
        except Exception:
            pass  # cache write is best-effort
    _chat_ref = ref_audio


def _chat_loop(lines: list, voice: dict, audio_dir: str) -> list:
    """Generate voiceover with Chatterbox real-voice cloning (English)."""
    from chatterbox.tts import ChatterboxTTS  # noqa: F401  (type ref only)

    model = _get_chatterbox_model()
    params = voice.get("params", {})
    exaggeration = params.get("exaggeration", 0.5)
    cfg_weight = params.get("cfg_weight", 0.5)
    temperature = params.get("temperature", 0.8)
    pitch_shift = params.get("pitch_shift", 0.0)

    _prep_chat_conds(model, voice)  # no-op if this process already prepared

    sr = _chat_sr
    for line in lines:
        line_id = line["id"]
        cleaned = _clean_text(line["text"])
        text = _de_shout(cleaned) if line.get("loud") else cleaned
        path = os.path.join(audio_dir, f"{line_id}.wav")

        # First line of the script gets a throwaway lead word ("Okay.") so the
        # model's weak-start phoneme lands on the buffer, NOT on the real first
        # word ("Listen"→"isten"). The buffer is stripped right after synthesis;
        # the real first word arrives mid-stream where onsets are fully voiced.
        synth_text = text
        if line.get("is_first") and TTS_LEAD_BUFFER:
            synth_text = f"{TTS_LEAD_BUFFER} {text}"

        print(f"  [tts] Line {line_id}: {synth_text}", flush=True)

        wav = model.generate(
            text=synth_text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )

        if hasattr(wav, 'numpy'):
            combined = wav.squeeze(0).numpy()
        else:
            combined = np.asarray(wav).squeeze()

        combined = _enforce_pauses(combined.astype(np.float32), sr, synth_text)
        if line.get("is_first") and TTS_LEAD_BUFFER:
            combined = _strip_lead_buffer(combined, sr, TTS_LEAD_BUFFER, synth_text)
        combined = _normalize_pacing(combined, sr, text, params)
        gain = LOUD_GAIN if line.get("loud") else params.get("gain", 1.0)
        # Every line keeps the same short 60ms head pad. Bigger lead-ins made a
        # ~1s dead-silence pause between sentences (huge audible gap every line).
        lead_in = 0.06
        combined = _finalize(combined.astype(np.float32), sr, pitch_shift=pitch_shift, gain=gain, eq=params.get("eq"), speed=params.get("speed", 1.0), attack_pitch=params.get("attack_pitch", 0.0), lead_in=lead_in)
        sf.write(path, combined, sr)
        line["audio_path"] = path
        line["actual_duration"] = len(combined) / sr

    return lines


def _effective_workers(n_lines: int) -> int:
    """Number of parallel render workers (settings.TTS_WORKERS, default 2).

    Measured on this machine: each worker holds ~4.3GB (own model copy) and a
    single worker already uses all memory bandwidth, so 2 workers give ~1.9x
    wall-clock speedup at identical quality; 3+ would exceed RAM on 14GB boxes.
    """
    if not n_lines or n_lines < 2:
        return 1
    return min(TTS_WORKERS, 2, n_lines)


def _split_chunks(lines: list, workers: int) -> list:
    """Balanced contiguous chunks preserving original line order."""
    n = len(lines)
    if n <= workers:
        return [[line] for line in lines]
    base, rem = divmod(n, workers)
    chunks, i = [], 0
    for w in range(workers):
        size = base + (1 if w < rem else 0)
        chunks.append(lines[i:i + size])
        i += size
    return chunks


def _setup_worker_rng(seed: int, threads: int) -> None:
    """Cap each worker's torch threads and re-seed RNG so concurrent lines
    get independent samples without oversubscribing the CPU."""
    import numpy as _np
    import torch as _torch
    _torch.set_num_threads(threads)
    _torch.manual_seed(seed)
    _np.random.seed(seed)


def _worker_chatterbox(idx, chunk, voice, audio_dir, seed, threads, queue):
    try:
        _setup_worker_rng(seed, threads)
        out = _chat_loop(chunk, voice, audio_dir)  # model + conds inherited via fork
        queue.put((False, idx, out))
    except Exception as exc:
        queue.put((True, idx, f"{exc!r}"))


def _parallel_lines(workers, worker_fn, lines, voice, audio_dir) -> list:
    """Render `workers` disjoint chunks of lines in parallel subprocesses.

    Uses spawn: each worker is a fresh interpreter that loads the model itself
    and shares nothing mutable with the parent. Torch is not fork-safe once its
    thread pools exist (forking the parent's loaded model caused hangs), and two
    full model copies still fit comfortably in RAM on this box.
    """
    ctx = mp.get_context("spawn")
    queue = ctx.SimpleQueue()
    procs = []
    threads = max(1, (os.cpu_count() or 4) // workers)
    for idx, chunk in enumerate(_split_chunks(lines, workers)):
        seed = ((os.getpid() << 16) ^ ((idx + 1) * 7919)) & 0xFFFFFFFF
        p = ctx.Process(target=worker_fn,
                        args=(idx, chunk, voice, audio_dir, seed, threads, queue))
        p.start()
        procs.append(p)
    for p in procs:
        p.join()
    errors, results = [], {}
    for _ in procs:
        err, idx, payload = queue.get()
        if err:
            errors.append(payload)
        else:
            results[idx] = payload
    if errors:
        raise RuntimeError("; ".join(str(e) for e in errors))
    out_lines = []
    for idx in sorted(results):
        out_lines.extend(results[idx])
    # Workers return deep copies (pickled across processes); fold the new
    # audio_path/actual_duration back into the original dict objects so callers
    # that rely on in-place mutation keep working.
    for orig, upd in zip(lines, out_lines):
        orig.update(upd)
    return lines


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