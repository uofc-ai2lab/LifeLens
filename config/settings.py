from pathlib import Path
import os, platform
from dotenv import load_dotenv
from google import genai
import spacy

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

METADATA_FILENAME = "audio_pipeline_metadata.json"
METADATA_JSON_PATH = DATA_DIR / METADATA_FILENAME

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

_raw_audio_files = os.getenv("AUDIO_FILES")
if _raw_audio_files:
    # Explicit list provided via env var
    AUDIO_FILES_LIST = [AUDIO_DIR / f.strip() for f in _raw_audio_files.split(",") if f.strip()]
else:
    # Default: take ALL files in audio_files directory
    AUDIO_FILES_LIST = [f for f in AUDIO_DIR.iterdir() if f.is_file()]
   
TRANSCRIPT_DIR_NEW = os.getenv("TRANSCRIPT_DIR","")
if TRANSCRIPT_DIR_NEW != "":
    TRANSCRIPT_DIR = TRANSCRIPT_DIR_NEW
    
# -------------------------
# NLP / Meaning extraction
# -------------------------
MEDCAT_DATA_DIR = Path("data/data_p3.2")
MODEL_PACK_PATH = MEDCAT_DATA_DIR / "medmen_wstatus_2021_oct.zip"

try:
    from medcat.cat import CAT
except:
    print("ERROR: MedCAT not installed or environment broken.")
    exit()
    
MODEL_PACK = CAT.load_model_pack(MODEL_PACK_PATH)
NLP = spacy.load("en_core_web_sm")

_raw_transcript_files = os.getenv("TRANSCRIPT_FILES")
if _raw_transcript_files:
    # Explicit list provided via env var
    TRANSCRIPT_FILES_LIST = [TRANSCRIPT_DIR / f.strip() for f in _raw_transcript_files.split(",") if f.strip()]
else:
    # Default: take ALL files in audio_files directory
    TRANSCRIPT_FILES_LIST = [f for f in TRANSCRIPT_DIR.iterdir() if f.is_file()]
   
MEANING_DIR_NEW = os.getenv("MEANING_DIR","")
if MEANING_DIR_NEW != "":
    MEANING_DIR = MEANING_DIR_NEW

ENABLE_SEMANTIC_FILTERING = int(os.getenv("ENABLE_SEMANTIC_FILTERING", "0"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
GENAI_MODEL = None
GENAI_CLIENT = None

if ENABLE_SEMANTIC_FILTERING:
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables.")
    GENAI_MODEL = "gemini-flash-latest"
    GENAI_CLIENT = genai.Client()

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
