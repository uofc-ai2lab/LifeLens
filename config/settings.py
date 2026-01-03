from pathlib import Path
import os, platform
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# -------------------------
# Base paths
# -------------------------
BASE_DIR = Path(os.getcwd()).resolve()

SRC_DIR = BASE_DIR / "src"
DATA_DIR = BASE_DIR / "data"

AUDIO_DIR = DATA_DIR / "audio_files"
TRANSCRIPT_DIR = DATA_DIR / "transcript_files"
MEANING_DIR = DATA_DIR / "meaning_files"
TEST_DATA_DIR = DATA_DIR / "test_data"


# -------------------------
# Environment
# -------------------------
ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
DEBUG = os.getenv("DEBUG", "false").lower() == "true"
IS_JETSON = platform.machine() == "aarch64"

# -------------------------
# Audio / Transcription
# -------------------------
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN", "")
DEVICE = os.getenv("DEVICE", "cuda")
MODEL_SIZE = os.getenv("MODEL_SIZE", "base")
if IS_JETSON:
    MODEL_SIZE = os.getenv("MODEL_SIZE_TRT", "base.en")  # WhisperTRT models: tiny.en, base.en, small.en, medium.en

PYANNOTE_CACHE_DIR = os.getenv("PYANNOTE_CACHE_DIR", None)  # Optional custom cache path
MODEL_CACHE_PATH = os.getenv("WHISPER_TRT_CACHE", None)  # Optional custom cache path
USE_OFFLINE_MODELS = int(os.getenv("USE_OFFLINE_MODELS", "0"))
OUTPUT_DIR = os.getenv("OUTPUT_DIR","")
if OUTPUT_DIR == "":
    OUTPUT_DIR = TRANSCRIPT_DIR


_raw_audio_files = os.getenv("AUDIO_FILES")
if _raw_audio_files:
    # Explicit list provided via env var
    AUDIO_FILES_LIST = [AUDIO_DIR / f.strip() for f in _raw_audio_files.split(",") if f.strip()]
else:
    # Default: take ALL files in audio_files directory
    AUDIO_FILES_LIST = [f for f in AUDIO_DIR.iterdir() if f.is_file()]
    
# -------------------------
# NLP / Meaning extraction
# -------------------------
MIN_MEANING_CONFIDENCE = float(
    os.getenv("MIN_MEANING_CONFIDENCE", 0.6)
)

ENABLE_SEMANTIC_FILTERING = (
    os.getenv("ENABLE_SEMANTIC_FILTERING", "true").lower() == "true"
)

# -------------------------
# Encryption
# -------------------------
# ENCRYPTION_KEY = os.getenv("ENCRYPTION_KEY")

# if not ENCRYPTION_KEY and ENVIRONMENT != "test":
#     raise RuntimeError("ENCRYPTION_KEY must be set in .env")

# -------------------------
# Logging
# -------------------------
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
