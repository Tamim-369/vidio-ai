"""Model loading and voice-conditioning caches for both TTS engines.

Pocket and Chatterbox each keep a lazily-built model in a module global, so a
run loads weights once and reuses them for every line. The Chatterbox voice
conditionals are cached next to the reference audio by content hash.
"""

import os

POCKET_VOICE_STATE = "voices/narrator.safetensors"
POCKET_VOICE_REF = "src/experiments/voice_tests/chatterbox_ref.wav"

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
