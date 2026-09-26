import os
from dotenv import load_dotenv

load_dotenv()

# Cloudflare Workers AI (last-resort LLM)
CLOUDFLARE_API_TOKEN = os.getenv("CLOUDFLARE_API_TOKEN", "")
CLOUDFLARE_ACCOUNT_ID = os.getenv("CLOUDFLARE_ACCOUNT_ID", "")
# Primary TEXT model (script gen / JSON structuring). llama-3.3-70b-instruct
# is FREE on Workers AI, ~2x faster than gemma-4-26b, produces no empty-content
# retries (gemma's reasoning mode frequently ate the token budget and blanked),
# and its viral-shorts output is stronger (measured: 8.2s vs 16.1s+ on the same
# Arnold-style prompt).
CLOUDFLARE_MODEL = os.getenv("CLOUDFLARE_MODEL", "@cf/meta/llama-3.3-70b-instruct-fp8-fast")

# Groq settings (secondary) — 3 keys: rotate on failure, fall back to Gemini
# when all three are exhausted.
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY_SECOND = os.getenv("GROQ_API_KEY_SECOND")
GROQ_API_KEY_THIRD = os.getenv("GROQ_API_KEY_THIRD")
GROQ_API_KEY_BACKUP = os.getenv("GROQ_API_KEY_BACKUP")  # Legacy alias
# TEXT model (script/research): gpt-oss-20b is a fast, standard service model.
# gpt-oss-120b returns empty completions on this Groq org, so it is avoided
# (the retry chain now treats empty content as a failure and rotates keys).
GROQ_MODEL = "openai/gpt-oss-20b"

# Quote generation (src/services/quote_agent.py) runs on its own model so it
# can be changed without touching the script/research model above. Quotes are
# Groq-only by requirement: the agent passes allow_fallback=False to call_groq
# so a dead key raises instead of silently producing Gemini output.
# Override with GROQ_QUOTE_MODEL in .env.
GROQ_QUOTE_MODEL = os.getenv("GROQ_QUOTE_MODEL", "qwen/qwen3.8-27b")

# Gemini (topic generation + final fallback after the 3 Groq keys).
# Flash is the free-tier workhorse (~15 RPM, ~1500 RPD) — plenty for a
# handful of brainstorm calls per day. Rotated in key order ONE..FIVE.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")
GEMINI_KEYS = []
for _k in ("GEMINI_API_KEY_ONE", "GEMINI_API_KEY_TWO", "GEMINI_API_KEY_THREE",
           "GEMINI_API_KEY_FOUR", "GEMINI_API_KEY_FIVE"):
    if os.getenv(_k):
        GEMINI_KEYS.append(os.getenv(_k))
if GEMINI_API_KEY and GEMINI_API_KEY not in GEMINI_KEYS:
    GEMINI_KEYS.append(GEMINI_API_KEY)

VIDEO_FORMAT = "9:16"         # "9:16" for Shorts/Reels, "16:9" for YouTube

OUTPUT_DIR = "output"
TEMP_DIR = "temp"

# --- Background music ---
# Mixed under the narration after assembly (src/services/music.py), so the
# TTS/assembler behaviour is untouched. MUSIC_ENABLED=0 renders voice-only.
MUSIC_ENABLED = os.getenv("MUSIC_ENABLED", "1") == "1"
MUSIC_PATH = os.getenv("MUSIC_PATH", "src/music/oogway.mp3")
# The dead air at the head of the track was removed from the FILE itself (the
# first 5s are already cut, see git history), so there is nothing left to skip
# here. Keep this at 0 unless you swap in a different music file that still has
# a silent or abrupt opening — it is applied on top of the file's own start.
MUSIC_SKIP_S = float(os.getenv("MUSIC_SKIP_S", "0"))
# The bed is loudness-normalised to this LUFS before the trim below, so its
# level in the mix does not depend on the track's own loudness. Chatterbox
# narration lands near -18 dB mean, so -24 puts the music roughly 8 dB under
# the voice: clearly audible on phone and laptop speakers, still behind the
# punchline. -28 is subtler; -20 is nearly level with the narration.
MUSIC_TARGET_LUFS = float(os.getenv("MUSIC_TARGET_LUFS", "-24"))
# Trim applied AFTER normalisation, so it is an absolute, predictable offset
# rather than one relative to the track's own (wildly varying) loudness.
# 0 is neutral; negative pulls the bed further under the voice.
MUSIC_GAIN_DB = float(os.getenv("MUSIC_GAIN_DB", "0"))
# Fade in/out so the clip does not start or end on a hard music edge.
# The fade-IN is deliberately tiny: the bed should already be audible on frame
# one, establishing the mood before the first word, not swelling in from
# nothing over a second and a half. The fade-OUT stays long so the ending
# resolves instead of stopping dead.
MUSIC_FADE_IN_S = float(os.getenv("MUSIC_FADE_IN_S", "0.10"))
MUSIC_FADE_OUT_S = float(os.getenv("MUSIC_FADE_OUT_S", "1.5"))

# Narration rate. Chatterbox 0.1.7 has no rate knob, so this is an ffmpeg
# atempo time-stretch applied to each synthesised line (pitch preserved).
# 1.0 is the model's natural pace; 0.93 is a touch slower, which suits a
# reflective channel better than a rushed one.
TTS_RATE = float(os.getenv("TTS_RATE", "0.93"))

# --- Quote card typography ---
# Playfair Display, from the repo's Fonts/ folder (SIL OFL 1.1, see
# Fonts/Playfair_Display/OFL.txt). A serif suits the "wisdom" register far
# better than the UI sans the documentary captions used.
# NOTE: Fonts/ is not committed to git, so the renderer must fall back to
# FONT_PATH when these files are absent (fresh clone, CI, another machine).
FONT_DIR = os.getenv("FONT_DIR", "Fonts/Playfair_Display/static")
QUOTE_FONT_PATH = os.getenv(
    "QUOTE_FONT_PATH", f"{FONT_DIR}/PlayfairDisplay-Bold.ttf")
# The book/work line under the author reads better in the italic cut.
QUOTE_FONT_ITALIC_PATH = os.getenv(
    "QUOTE_FONT_ITALIC_PATH", f"{FONT_DIR}/PlayfairDisplay-Italic.ttf")

# --- Quote card pacing ---
# Cards butt up against each other by default, which makes a two-quote video
# feel like one long breath. These pad each card's audio with silence so the
# beat lands. The card stays on screen through its own tail, and the music bed
# runs underneath the silence, so a gap or the ending hold is music-only rather
# than a dead cut to black-and-quiet.
# Gap after every quote except the last.
QUOTE_GAP_S = float(os.getenv("QUOTE_GAP_S", "2.0"))
# Hold on the final card after the last word, so the video does not cut on it.
QUOTE_END_TAIL_S = float(os.getenv("QUOTE_END_TAIL_S", "4.0"))

# Pocket-TTS voice (cloned from the Kokoro narrator ref in src/experiments/voice_tests/chatterbox_ref.wav)
POCKET_VOICE_STATE = "voices/narrator.safetensors"
POCKET_VOICE_REF = "src/experiments/voice_tests/chatterbox_ref.wav"

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# --- TTS ---
# Parallel chatterbox render workers (1-2). Each worker holds its own ~4.3GB
# model copy; 2 gives ~1.9x wall-clock speedup and 3+ risks RAM on 14GB boxes.
TTS_WORKERS = int(os.getenv("TTS_WORKERS", "2"))
# Speaking-rate band (words per second), measured over SPEECH-ONLY time
# (silence gaps subtracted) so deliberate pauses/dramatic lines are not
# punished. Only lines whose actual speech bursts fall outside the band are
# corrected (whole-line uniform tempo): faster than TTS_MAX_WPS → slowed to it,
# slower than TTS_MIN_WPS → picked up to it. Tuned to natural energized
# narration (~3.0-3.4 wps of burst speech): a floor of 2.6 left lots of lines
# audibly draggy, and forcing a 1.05x stretch felt robotic, so the floor sits at
# normal narration (+1) and the ceiling just reins in the rare rush; 0 disables.
TTS_MIN_WPS = float(os.getenv("TTS_MIN_WPS", "3.0"))
TTS_MAX_WPS = float(os.getenv("TTS_MAX_WPS", "3.4"))
# Throwaway word(s) prepended to the FIRST line's TTS prompt and then stripped
# from the audio. Chatterbox voices the first phoneme of a fresh synthesis
# weakly ("Listen"→"isten"); the buffer absorbs that artifact and the real
# first word is then synthesized mid-stream where onsets are full. 0/empty = off.
#
# Default is OFF now: scripts are written to open with a strong hook word
# ("Hey" for Arnold, a promise-hook like "I am about to tell you..." for
# others), so the model's first phoneme is a vowel/strong onset and no buffer
# is needed. An audible buffer word would leak "okay".
TTS_LEAD_BUFFER = os.getenv("TTS_LEAD_BUFFER", "")

# Image resolution based on format
VIDEO_RESOLUTIONS = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
}

# --- YouTube upload settings ---
# Set AUTO_PUBLISH=1 in .env to publish videos after rendering.
# Requires client_secrets.json (OAuth client) in the project root.
AUTO_PUBLISH = os.getenv("AUTO_PUBLISH", "0") == "1"

YOUTUBE_CLIENT_SECRETS = os.getenv("YOUTUBE_CLIENT_SECRETS", "client_secrets.json")
YOUTUBE_TOKEN_FILE = os.getenv("YOUTUBE_TOKEN_FILE", "token.json")

# Fixed localhost port for the OAuth redirect. If your client is a
# "Web application" type, register this exact URI in Google Cloud Console:
#   http://localhost:8080/  and  http://127.0.0.1:8080/
# Desktop-app clients accept any loopback port automatically.
YOUTUBE_REDIRECT_PORT = int(os.getenv("YOUTUBE_REDIRECT_PORT", "8080"))

# public | private | unlisted
YOUTUBE_PRIVACY_STATUS = os.getenv("YOUTUBE_PRIVACY_STATUS", "private")

# YouTube video category id (22 = People & Blogs, 27 = Education)
YOUTUBE_CATEGORY_ID = os.getenv("YOUTUBE_CATEGORY_ID", "27")

# Default tags appended to every upload (e.g. "shorts", "faceless")
YOUTUBE_TAGS = os.getenv("YOUTUBE_TAGS", "")

YOUTUBE_SCOPES = [
    "https://www.googleapis.com/auth/youtube.upload",
    "https://www.googleapis.com/auth/youtube",
]
