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

SRC_DIR = BASE_DIR / "src_audio"
DATA_DIR = BASE_DIR / "data"
AUDIO_DATA_DIR = DATA_DIR / "audio"

AUDIO_DIR = AUDIO_DATA_DIR / "audio_files"
os.makedirs(AUDIO_DIR, exist_ok=True)

PROCESSED_AUDIO_DIR = AUDIO_DATA_DIR / "processed_audio"
os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)

METADATA_FILENAME = "audio_pipeline_metadata.json"
METADATA_JSON_PATH = AUDIO_DATA_DIR / METADATA_FILENAME

# -------------------------
# Audio / Transcription
# -------------------------
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN", "")
DEVICE = os.getenv("DEVICE", "cpu") # Won't run if defaults to cuda and machine doesn't have Nvidia GPU, but will run on most machines if set to cpu by default.
MODEL_SIZE = os.getenv("MODEL_SIZE", "base")
IS_JETSON = platform.machine() == "aarch64"
if IS_JETSON:
    MODEL_SIZE = os.getenv("MODEL_SIZE_TRT", "base.en")  # WhisperTRT models: tiny.en, base.en, small.en, medium.en

PYANNOTE_CACHE_DIR = os.getenv("PYANNOTE_CACHE_DIR", None)  # Optional custom cache path
MODEL_CACHE_PATH = os.getenv("WHISPER_TRT_CACHE", None)  # Optional custom cache path
USE_OFFLINE_MODELS = int(os.getenv("USE_OFFLINE_MODELS", "0"))

_raw_audio_files = os.getenv("AUDIO_FILES")
if _raw_audio_files:
    # Explicit list provided via env var
    _audio_files_list = [AUDIO_DIR / f.strip() for f in _raw_audio_files.split(",") if f.strip()]
else:
    # Default: take ALL files in audio_files directory
    _audio_files_list = [f for f in AUDIO_DIR.iterdir() if f.is_file()]

AUDIO_FILES_DICT = {}
for parent_audio in _audio_files_list:
    parent_audio_dir = PROCESSED_AUDIO_DIR / Path(parent_audio).stem
    parent_audio_dir.mkdir(parents=True, exist_ok=True)
    parent_audio_dir_with_ext = parent_audio_dir.with_suffix(Path(parent_audio).suffix)
    AUDIO_FILES_DICT[parent_audio_dir_with_ext] = []

# -------------------------
# NLP / Meaning extraction
# -------------------------
MEDCAT_DATA_DIR = DATA_DIR / "data_p3.2"
MODEL_PACK_PATH = MEDCAT_DATA_DIR / "medmen_wstatus_2021_oct.zip"
ENABLE_MEDCAT = int(os.getenv("ENABLE_MEDCAT", "0"))

if ENABLE_MEDCAT:
    try:
        from medcat.cat import CAT
    except:
        print("ERROR1: MedCAT not installed or environment broken.")
        exit()

    MODEL_PACK = CAT.load_model_pack(MODEL_PACK_PATH)
    NLP = spacy.load("en_core_web_sm")

else:
    print("MEDCAT DISABLED. Please enable prior to running intervention extraction.")
    MODEL_PACK = None
    NLP = None

_raw_transcript_files = os.getenv("TRANSCRIPT_FILES")
_unknown_audio_dir = PROCESSED_AUDIO_DIR / "unknown_audio"
_unknown_audio_dir.mkdir(parents=True, exist_ok=True)

if _raw_transcript_files:
    # Explicit list provided via env var
    TRANSCRIPT_FILES_LIST = [_unknown_audio_dir / f.strip() for f in _raw_transcript_files.split(",") if f.strip()]
else:
    # Default: take ALL files in transcript directory
    TRANSCRIPT_FILES_LIST = [f for f in _unknown_audio_dir.iterdir() if f.is_file()]

ENABLE_SEMANTIC_FILTERING = int(os.getenv("ENABLE_SEMANTIC_FILTERING", "0"))
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", None)
GENAI_MODEL = None
GENAI_CLIENT = None

if ENABLE_SEMANTIC_FILTERING:
    if not GOOGLE_API_KEY:
        print("Error: GOOGLE_API_KEY or GEMINI_API_KEY not found in environment variables.")
    GENAI_MODEL = "gemini-flash-latest"
    GENAI_CLIENT = genai.Client()
