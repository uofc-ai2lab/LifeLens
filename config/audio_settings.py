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
AUDIO_DIR = DATA_DIR / "audio/audio_files"
AUDIO_CHUNKS_DIR = DATA_DIR / "audio/audio_chunks"
PROCESSED_AUDIO_DIR = DATA_DIR / "audio/processed_audio"

os.makedirs(AUDIO_DIR, exist_ok=True)
os.makedirs(AUDIO_CHUNKS_DIR, exist_ok=True)
os.makedirs(PROCESSED_AUDIO_DIR, exist_ok=True)

STARTUP_SCRIPT_PATH = BASE_DIR / "scripts" / "run_jetson_startup_tasks.sh"
USAGE_FILE_PATH = DATA_DIR / "resource_usage.csv"

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
