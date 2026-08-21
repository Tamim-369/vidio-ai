import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_API_KEY_BACKUP = os.getenv("GROQ_API_KEY_BACKUP")  # Backup key for rate limits
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_VISION_MODEL = "qwen/qwen3.6-27b"

# Ollama settings
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "minimax-m3:cloud")

# Gemini (topic generation). Flash is the free-tier workhorse (~15 RPM,
# ~1500 RPD) — plenty for a handful of brainstorm calls per day.
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.6-flash")

VIDEO_FORMAT = "9:16"         # "9:16" for Shorts/Reels, "16:9" for YouTube
VIDEO_STYLE = "attraction"    # educational | motivational | ad | storytelling | attraction
VOICE = "M3"                # from voices.py

OUTPUT_DIR = "output"
TEMP_DIR = "temp"

# Pocket-TTS voice (cloned from the Kokoro narrator ref in voice_tests/chatterbox_ref.wav)
POCKET_VOICE_STATE = "voices/narrator.safetensors"
POCKET_VOICE_REF = "voice_tests/chatterbox_ref.wav"

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

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
