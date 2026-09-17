import os
import re
import shlex

# Set HuggingFace cache dir before importing pocket_tts so models land in project folder
from dotenv import load_dotenv
load_dotenv()
hf_home = os.getenv("HF_HOME")
if hf_home:
    os.environ["HF_HOME"] = hf_home

import numpy as np
import soundfile as sf
from src.config.settings import TEMP_DIR, POCKET_VOICE_STATE, POCKET_VOICE_REF

_model = None
_voice_state = None

# Chatterbox engine state (lazy, cached across lines)
_chat_model = None
_chat_ref = None
_chat_sr = 24000

# Turbo engine (15s conditioning window) — separate model + ref cache
_turbo_model = None
_turbo_ref = None


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


def _clean_text(text: str) -> str:
    """Normalize text for clean TTS output."""
    # Unicode normalization
    text = text.replace("\u2014", ", ").replace("\u2013", ", ")
    text = text.replace("\u2026", "...").replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')

    # Common abbreviations Pocket-TTS may mangle
    abbrevs = {
        r'\bAI\b': 'A I',
        r'\bDNA\b': 'D N A',
        r'\bUSA\b': 'U S A',
        r'\bUK\b': 'U K',
        r'\bUSSR\b': 'U S S R',
        r'\bNATO\b': 'N A T O',
        r'\bRAF\b': 'R A F',
        r'\bUSAF\b': 'U S A F',
        r'\be\.g\.\b': 'for example',
        r'\bi\.e\.\b': 'that is',
        r'\bvs\.\b': 'versus',
        r'\bvs\b': 'versus',
        r'\bMPH\b': 'miles per hour',
        r'\bmph\b': 'miles per hour',
        r'\bKPH\b': 'kilometers per hour',
        r'\bkph\b': 'kilometers per hour',
    }
    for pattern, replacement in abbrevs.items():
        text = re.sub(pattern, replacement, text)

    # Ensure proper spacing after punctuation
    text = re.sub(r'([?!])([^\s])', r'\1 \2', text)
    text = re.sub(r'([.])([A-Z])', r'\1 \2', text)  # Space after periods before capitals

    # Clean up multiple spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def _noise_gate(audio: np.ndarray, threshold: float = 0.008) -> np.ndarray:
    """Silence samples below threshold to remove background hiss between words."""
    kernel = np.ones(256) / 256
    envelope = np.convolve(np.abs(audio), kernel, mode='same')
    gate = envelope > threshold
    return audio * gate


def _attack_dip(audio: np.ndarray, sr: int, dip: float = -2.0, window: float = 0.09) -> np.ndarray:
    """Per-word/sentence onset pitch dip — the 'gruff attack' signature.

    Detects rising-energy onsets (word/sentence starts) and briefly drops pitch
    on the attack window via SoX (formant-preserving), crossfading boundaries to
    avoid clicks. dip < 0 makes word starts deeper, gliding back to normal.
    """
    if dip == 0.0:
        return audio
    import librosa
    import subprocess
    import tempfile

    # Onset detection on a 10ms envelope.
    hop = int(sr * 0.01)
    n = int(sr * 0.03)
    env = librosa.feature.rms(y=audio, frame_length=n, hop_length=hop)[0]
    env = env / (env.max() + 1e-9)
    base = float(np.percentile(env, 40))
    thr = base + (1.0 - base) * 0.25

    onsets = []
    above = env > thr
    min_run = int(0.04 / 0.01)  # 40ms sustained energy = a real onset
    min_gap = int(0.15 / 0.01)  # re-trigger only on distinct words/sentences
    prev_end = -10 ** 9
    i = 0
    while i < len(above) - 1:
        if above[i]:
            j = i
            while j < len(above) and above[j]:
                j += 1
            if (j - i) >= min_run and i - prev_end >= min_gap:
                onsets.append(i * hop)
                prev_end = j
            i = j
        else:
            i += 1

    if not onsets:
        return audio

    win_len = int(window * sr)
    cents = int(round(dip * 100))
    xf = int(0.008 * sr)  # 8ms crossfade at segment edges to prevent clicks

    # Gather all windows to process, then run sox per window.
    with tempfile.TemporaryDirectory() as td:
        for t0 in onsets:
            s0 = int(t0)
            e0 = min(s0 + win_len, len(audio))
            if e0 - s0 < int(0.03 * sr):
                continue
            seg = audio[s0:e0]
            tmp_in = os.path.join(td, "in.wav")
            tmp_out = os.path.join(td, "out.wav")
            sf.write(tmp_in, seg, sr)
            subprocess.run(
                ["sox", tmp_in, tmp_out, "pitch", str(cents)],
                check=True, capture_output=True,
            )
            seg_p, _ = sf.read(tmp_out, dtype="float32")
            if len(seg_p) != len(seg):
                seg_p = seg_p[:len(seg)]
            # Fade edges (attack inlet / release outlet) so the dip is a glide, not a click.
            f = np.ones(len(seg), dtype=np.float32)
            f[:xf] = np.linspace(0.0, 1.0, xf)
            f[-xf:] = np.linspace(1.0, 0.0, xf)
            seg = seg + (seg_p - seg) * f
            audio[s0:e0] = seg
    return audio


def _finalize(combined: np.ndarray, sr: int, pitch_shift: float = 0.0, gain: float = 1.0, eq: list = None, speed: float = 1.0, attack_pitch: float = 0.0) -> np.ndarray:
    """Shared post-processing: EQ + pitch/speed, attack dip, silence pad, noise gate, RMS normalize, clip.

    eq: list of sox filter args (e.g. ["highpass 90", "equalizer 3000 1 2.5"]) applied
    BEFORE normalization so loudness stays constant regardless of the EQ boost.
    speed: playback rate multiplier (<1.0 = slower, >1.0 = faster); pitch preserved.
    attack_pitch: per-onset pitch dip in semitones (negative = gruff word starts).
    """
    if eq or pitch_shift or speed != 1.0:
        # SoX formant-preserving pitch (no phase-vocoder smear like librosa), per-voice
        # timbre EQ and time-stretch. Round-trips through a temp wav since sox CLI works
        # when multiple effects chain.
        import subprocess
        import tempfile

        with tempfile.TemporaryDirectory() as td:
            tmp_in = os.path.join(td, "in.wav")
            tmp_out = os.path.join(td, "out.wav")
            sf.write(tmp_in, combined, sr)
            effects = []
            if eq:
                for fil in eq:
                    effects.extend(shlex.split(fil))
            if pitch_shift:
                cents = int(round(pitch_shift * 100))
                effects.extend(["pitch", str(cents)])
            if speed != 1.0:
                effects.extend(["tempo", f"{speed:.4f}"])
            subprocess.run(
                ["sox", tmp_in, tmp_out, *effects],
                check=True, capture_output=True,
            )
            combined, _ = sf.read(tmp_out, dtype="float32")

    if attack_pitch:
        combined = _attack_dip(combined, sr, dip=attack_pitch)

    silence = np.zeros(int(0.05 * sr), dtype=combined.dtype)  # 50ms gap
    combined = np.concatenate([combined, silence])
    combined = _noise_gate(combined)

    # RMS normalization — consistent loudness across all lines
    TARGET_RMS = 0.15
    rms = np.sqrt(np.mean(combined ** 2))
    if rms > 0:
        combined = combined * (TARGET_RMS / rms)
    if gain != 1.0:
        combined = combined * gain
    combined = np.clip(combined, -0.95, 0.95)
    return combined


def _generate_pocket(lines: list, audio_dir: str) -> list:
    from pocket_tts import TTSModel  # noqa: F401  (validates import path for type hints)
    model = _get_pocket_model()
    voice_state = _get_voice_state()
    sr = model.sample_rate

    for line in lines:
        line_id = line["id"]
        text = _clean_text(line["text"])
        path = os.path.join(audio_dir, f"{line_id}.wav")

        print(f"  [tts] Line {line_id}: {text}")

        max_tokens = max(200, int(len(text.split()) * 14) + 50)
        audio = model.generate_audio(voice_state, text, max_tokens=max_tokens)

        if hasattr(audio, 'numpy'):
            combined = audio.numpy()
        else:
            combined = np.asarray(audio)

        combined = _finalize(combined.astype(np.float32), sr)
        sf.write(path, combined, sr)
        line["audio_path"] = path
        line["actual_duration"] = len(combined) / sr

    return lines


def _get_chatterbox_model():
    global _chat_model
    if _chat_model is None:
        from chatterbox import ChatterboxTTS
        print("  [tts] Loading Chatterbox model (cached)...", flush=True)
        _chat_model = ChatterboxTTS.from_pretrained("cpu")
    return _chat_model


def _get_turbo_model():
    global _turbo_model
    if _turbo_model is None:
        from chatterbox.tts_turbo import ChatterboxTurboTTS
        print("  [tts] Loading Chatterbox-Turbo model (cached)...", flush=True)
        _turbo_model = ChatterboxTurboTTS.from_pretrained("cpu")
        # Patch S3 tokenizer mel_filters from double → float to match numpy float32 input.
        tok = _turbo_model.s3gen.tokenizer
        if hasattr(tok, '_mel_filters'):
            tok._mel_filters = tok._mel_filters.float()
    return _turbo_model


def _generate_chatterbox(lines: list, voice: dict, audio_dir: str) -> list:
    """Generate voiceover with Chatterbox real-voice cloning (English)."""
    from chatterbox.tts import ChatterboxTTS  # noqa: F401  (type ref only)
    global _chat_ref

    model = _get_chatterbox_model()
    ref_audio = voice["ref_audio"]
    params = voice.get("params", {})
    exaggeration = params.get("exaggeration", 0.5)
    cfg_weight = params.get("cfg_weight", 0.5)
    temperature = params.get("temperature", 0.8)
    pitch_shift = params.get("pitch_shift", 0.0)

    # Prepare reference conditionals once per voice; reuse across lines.
    if _chat_ref != ref_audio:
        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Voice reference audio missing: {ref_audio}")
        print(f"  [tts] Preparing voice reference: {ref_audio}", flush=True)
        model.prepare_conditionals(ref_audio, exaggeration=exaggeration)
        _chat_ref = ref_audio

    sr = _chat_sr
    for line in lines:
        line_id = line["id"]
        text = _clean_text(line["text"])
        path = os.path.join(audio_dir, f"{line_id}.wav")

        print(f"  [tts] Line {line_id}: {text}", flush=True)

        wav = model.generate(
            text=text,
            exaggeration=exaggeration,
            cfg_weight=cfg_weight,
            temperature=temperature,
        )

        if hasattr(wav, 'numpy'):
            combined = wav.squeeze(0).numpy()
        else:
            combined = np.asarray(wav).squeeze()

        combined = _finalize(combined.astype(np.float32), sr, pitch_shift=pitch_shift, gain=params.get("gain", 1.0), eq=params.get("eq"), speed=params.get("speed", 1.0), attack_pitch=params.get("attack_pitch", 0.0))
        sf.write(path, combined, sr)
        line["audio_path"] = path
        line["actual_duration"] = len(combined) / sr

    return lines


def _generate_turbo(lines: list, voice: dict, audio_dir: str) -> list:
    """Voiceover with Chatterbox-Turbo: 15s conditioning window (2.5x more reference).

    Turbo is non-CFG — cfg_weight and exaggeration are ignored by the model.
    """
    global _turbo_ref

    model = _get_turbo_model()
    ref_audio = voice["ref_audio"]
    params = voice.get("params", {})
    temperature = params.get("temperature", 0.8)
    pitch_shift = params.get("pitch_shift", 0.0)

    # Reference must exceed 5s (turbo asserts this); reuse across lines.
    if _turbo_ref != ref_audio:
        if not os.path.exists(ref_audio):
            raise FileNotFoundError(f"Voice reference audio missing: {ref_audio}")
        print(f"  [tts] Preparing turbo voice reference: {ref_audio}", flush=True)
        import tempfile
        import librosa as _librosa
        # turbo's norm_loudness uses pyloudnorm which coerces to float64 and breaks
        # the S3 tokenizer's float32 mel filters, so we normalize + cast ourselves.
        wav, _sr = _librosa.load(ref_audio, sr=24000)
        with tempfile.TemporaryDirectory() as td:
            tmp = os.path.join(td, "ref.wav")
            sf.write(tmp, wav.astype(np.float32), 24000)
            model.prepare_conditionals(tmp, norm_loudness=False)
        _turbo_ref = ref_audio

    sr = _chat_sr
    for line in lines:
        line_id = line["id"]
        text = _clean_text(line["text"])
        path = os.path.join(audio_dir, f"{line_id}.wav")

        print(f"  [tts] Line {line_id}: {text}", flush=True)

        wav = model.generate(
            text=text,
            temperature=temperature,
        )

        if hasattr(wav, 'numpy'):
            combined = wav.squeeze(0).numpy()
        else:
            combined = np.asarray(wav).squeeze()

        combined = _finalize(combined.astype(np.float32), sr, pitch_shift=pitch_shift, gain=params.get("gain", 1.0), eq=params.get("eq"), speed=params.get("speed", 1.0), attack_pitch=params.get("attack_pitch", 0.0))
        sf.write(path, combined, sr)
        line["audio_path"] = path
        line["actual_duration"] = len(combined) / sr

    return lines


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
    if engine == "turbo":
        return _generate_turbo(lines, voice, audio_dir)
    return _generate_pocket(lines, audio_dir)