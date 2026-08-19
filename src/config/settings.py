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

VIDEO_FORMAT = "9:16"         # "9:16" for Shorts/Reels, "16:9" for YouTube
VIDEO_STYLE = "attraction"    # educational | motivational | ad | storytelling | attraction
VOICE = "M3"                # from voices.py

OUTPUT_DIR = "output"
TEMP_DIR = "temp"

FONT_PATH = "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"

# Image resolution based on format
VIDEO_RESOLUTIONS = {
    "9:16": (1080, 1920),
    "16:9": (1920, 1080),
}
