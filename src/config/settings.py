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
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
# TEXT model (script/research): gpt-oss-20b is a fast, standard service model.
# gpt-oss-120b returns empty completions on this Groq org, so it is avoided
# (the retry chain now treats empty content as a failure and rotates keys).
GROQ_MODEL = "openai/gpt-oss-20b"

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "minimax-m3:cloud")

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
VIDEO_STYLE = "attraction"    # educational | motivational | ad | storytelling | attraction
VOICE = "M3"                # from voices.py

OUTPUT_DIR = "output"
TEMP_DIR = "temp"

# Pocket-TTS voice (cloned from the Kokoro narrator ref in voice_tests/chatterbox_ref.wav)
POCKET_VOICE_STATE = "voices/narrator.safetensors"
POCKET_VOICE_REF = "voice_tests/chatterbox_ref.wav"

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

# --- Asset fetching ---
ASSET_MAX_PARALLEL_WORKERS = int(os.getenv("ASSET_MAX_PARALLEL_WORKERS", "8"))
ASSET_IMAGES_PER_LINE = int(os.getenv("ASSET_IMAGES_PER_LINE", "3"))
ASSET_MAX_REFINE_ATTEMPTS = int(os.getenv("ASSET_MAX_REFINE_ATTEMPTS", "4"))
ASSET_MAX_ASPECT_RATIO = float(os.getenv("ASSET_MAX_ASPECT_RATIO", "1.5"))
# Topic-first asset strategy: per-line search queries from the query agent, keep
# at least MIN OCR-clean images, then assign them to lines (multi per line ok).
ASSET_TARGET_IMAGES = int(os.getenv("ASSET_TARGET_IMAGES", "6"))
ASSET_MIN_IMAGES = int(os.getenv("ASSET_MIN_IMAGES", "4"))
# Deterministic text-overlay slop filter (pytesseract OCR, no LLM): reject any
# image whose readable text covers more than ASSET_MAX_TEXT_AREA fraction of its
# area (photos with captions/memes/watermark blocks - not tiny credit marks).
ASSET_REJECT_TEXT_OVERLAY = os.getenv("ASSET_REJECT_TEXT_OVERLAY", "1") == "1"
ASSET_MAX_TEXT_AREA = float(os.getenv("ASSET_MAX_TEXT_AREA", "0.04"))
ASSET_TEXT_MIN_CONF = int(os.getenv("ASSET_TEXT_MIN_CONF", "50"))

# --- Captions / subtitles ---
# Styled word-by-word "karaoke" captions burned into the frames (matches the
# video vibe via VIDEO_STYLE accent color). Disable to render caption-free.
CAPTIONS_ENABLED = os.getenv("CAPTIONS_ENABLED", "1") == "1"
# Optional accent override (hex, e.g. "#FFC94D"). Empty = auto from VIDEO_STYLE.
CAPTION_ACCENT = os.getenv("CAPTION_ACCENT", "")

# Image resolution based on format
VIDEO_RESOLUTIONS = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
}

# --- Reddit OAuth (topic generator) ---
# Create a "script" app at https://www.reddit.com/prefs/apps and fill these in .env.
# Permanent auth: no cookie refresh needed, 100 requests/min free.
REDDIT_CLIENT_ID = os.getenv("REDDIT_CLIENT_ID", "")
REDDIT_CLIENT_SECRET = os.getenv("REDDIT_CLIENT_SECRET", "")
REDDIT_USERNAME = os.getenv("REDDIT_USERNAME", "")
REDDIT_PASSWORD = os.getenv("REDDIT_PASSWORD", "")

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
