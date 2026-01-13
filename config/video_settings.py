from __future__ import annotations
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables from .env (if present). In VS Code, this is often
# driven by python.envFile.
load_dotenv()


# -------------------------
# Base paths (video)
# -------------------------
# Anchor to repo root (directory containing `config/`) rather than the current
# working directory. This keeps data paths stable no matter how the script is run
# (VS Code Run, `python src_video/main.py`, etc.).
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"

VIDEO_DIR = DATA_DIR / "video"
VIDEO_SOURCE_DIR = VIDEO_DIR / "source_files"
VIDEO_OUTPUT_DIR = VIDEO_DIR / "output_files"


def _env_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    raw = raw.strip().lower()
    if raw in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if raw in {"0", "false", "f", "no", "n", "off"}:
        return False
    return default


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_list(name: str, default: list[str]) -> list[str]:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return [item.strip() for item in raw.split(",") if item.strip()]


def load_video_pipeline_settings() -> dict:
    pipeline_root = os.getenv("PIPELINE_ROOT", str(VIDEO_OUTPUT_DIR)).replace("\\", "/")
    detection_output = os.getenv("PIPELINE_DETECTION_OUTPUT", f"{pipeline_root}/DetectionOutput")
    classification_output = os.getenv(
        "PIPELINE_CLASSIFICATION_OUTPUT",
        f"{pipeline_root}/ClassificationOutput",
    )

    return {
        "DETECTION_SOURCE": os.getenv("PIPELINE_DETECTION_SOURCE", str(VIDEO_SOURCE_DIR)).replace("\\", "/"),
        "PIPELINE_ROOT": pipeline_root,
        "DETECTION_OUTPUT": detection_output,
        "CLASSIFICATION_OUTPUT": classification_output,
        "DETECTION_MODEL": os.getenv("PIPELINE_DETECTION_MODEL", "MnLgt/yolo-human-parse"),
        "MAX_IMAGES": _env_int("PIPELINE_MAX_IMAGES", 200),
        "ADD_HEAD": _env_bool("PIPELINE_ADD_HEAD", True),
        "ALPHA_PNG": _env_bool("PIPELINE_ALPHA_PNG", False),
        "MIN_AREA": _env_int("PIPELINE_MIN_AREA", 250),
        "MARGIN": _env_float("PIPELINE_MARGIN", 0.10),
        "CLASSES": _env_list(
            "PIPELINE_CLASSES",
            ["face", "arm", "hand", "leg", "foot", "neck", "torso", "head"],
        ),
        "DEVICE": os.getenv("PIPELINE_DEVICE", None),
        "DEBUG": _env_bool("PIPELINE_DEBUG", False),
        "INJURY_CHECKPOINT_PATH": os.getenv(
            "PIPELINE_INJURY_CHECKPOINT",
            "checkpoints/classificationModel/injury/best_swin_tiny_patch4_window7_224.pt",
        ),
        "INJURY_IMG_SIZE": _env_int("PIPELINE_INJURY_IMG_SIZE", 224),
        "INJURY_BATCH_SIZE": _env_int("PIPELINE_INJURY_BATCH_SIZE", 32),
        "INJURY_NUM_WORKERS": _env_int("PIPELINE_INJURY_NUM_WORKERS", 0),
        "INJURY_REPORT_JSON": os.getenv(
            "PIPELINE_INJURY_REPORT_JSON",
            f"{classification_output}/injury_predictions.json",
        ),
        "INJURY_REPORT_CSV": os.getenv(
            "PIPELINE_INJURY_REPORT_CSV",
            f"{classification_output}/injury_predictions_summary.csv",
        ),
        # Crop filename parsing
        # Crops are named like: <origstem>_<body_part>_<idx>.jpg
        # This controls which token is interpreted as body-part label.
        "BODY_PART_LABEL_POSITION": _env_int("PIPELINE_BODY_PART_LABEL_POSITION", -2),
    }
