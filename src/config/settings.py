import os
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
PEXELS_API_KEY = os.getenv("PEXELS_API_KEY")
GROQ_MODEL = "openai/gpt-oss-120b"

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
