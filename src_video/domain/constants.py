"""Constants (facts about the code that never change)."""
from __future__ import annotations
import os
import re

# Default crop filename format: <stem>_<part>_<idx>.jpg
FILENAME_DELIMITER = "_"

# Common image extensions used across the video pipeline.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

# -------------------------
# Detection / body-part labeling
# -------------------------

# Default parts we attempt to crop from the segmentation model.
DETECTION_PART_DEFAULT = ["face", "arm", "hand", "leg", "foot", "neck", "torso", "head"]

# Parts that can occur twice and benefit from side-like disambiguation.
# NOTE: This is a *camera/image* heuristic, not anatomical left/right.
SIDEABLE_PARTS = {"arm", "hand", "leg", "foot"}

# Parts used to estimate the body's approximate midline.
MIDLINE_PARTS = {"torso", "head", "face", "neck"}

# -------------------------
# Body-part normalization (ranking / reporting)
# -------------------------

# Canonical side label format used across the pipeline.
# IMPORTANT: do NOT use an underscore here (e.g. "arm_1") because crops are named
# like: <stem>_<body_part>_<idx>.jpg and classification parses body_part by
# splitting on '_' (see BODY_PART_LABEL_POSITION).
BODY_PART_SIDE_LABEL_SEPARATOR = ""  # e.g. "arm1", "arm2"


def format_sideable_part_label(part: str, side_index: int | str) -> str:
	return f"{part}{BODY_PART_SIDE_LABEL_SEPARATOR}{side_index}"


# Accept variants like: arm1, arm_1, arm 1, arm-1
# NOTE: This is normalization only; the pipeline *emits* canonical labels via
# format_sideable_part_label() to keep filenames parseable.
BODY_PART_SIDE_SUFFIX_PATTERN = r"^([a-zA-Z_ -]+?)[ _-]*([12])$"

BODY_PART_SIDE_SUFFIX_RE = re.compile(BODY_PART_SIDE_SUFFIX_PATTERN)

# Color settings for annotations
COLOR_OUTLINE = (0, 255, 0)
COLOR_CORNERS = (255, 0, 0)
COLOR_CENTER = (0, 0, 255)
COLOR_TEXT = (0, 255, 255)
COLOR_ID_TEXT = (0, 255, 0)
COLOR_DISTANCE_TEXT = (255, 0, 0)


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


# Camera Settings (needed - passed to gstreamer_pipeline)
# Low-memory defaults for Jetson stability. Override via env if needed.
CAPTURE_WIDTH = _env_int("VIDEO_CAPTURE_WIDTH", 1920)
CAPTURE_HEIGHT = _env_int("VIDEO_CAPTURE_HEIGHT", 1080)
DISPLAY_WIDTH = _env_int("VIDEO_DISPLAY_WIDTH", 960)
DISPLAY_HEIGHT = _env_int("VIDEO_DISPLAY_HEIGHT", 540)
FRAME_RATE = _env_int("VIDEO_FRAME_RATE", 20)
FLIP_METHOD = _env_int("VIDEO_FLIP_METHOD", 4)  # 0=none, 1=counterclockwise, 2=180, 3=clockwise, 4=horizontal mirror, 5=upper-right diagonal, 6=vertical flip, 7=upper-left diagonal
# GStreamer videobalance brightness in [-1.0, 1.0]. Positive values brighten.
CAMERA_BRIGHTNESS = _env_float("VIDEO_BRIGHTNESS", 0.12)

# Camera Calibration - NEEDED FOR ACCURATE DISTANCE
# we don't need this to be super accurate for our use case, so these are approximate values
CAMERA_FX=1225.0
CAMERA_FY=1225.0
CAMERA_CX=960.0
CAMERA_CY=540.0

CAMERA_PARAMS = (CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY)

# ── YOLO person detector ──────────────────────────────────────────────────
YOLO_INPUT_SIZE       = 640          
YOLO_CONF_THRESH      = 0.45         
YOLO_NMS_IOU_THRESH   = 0.45         
YOLO_PERSON_CLASS     = 0            
PERSON_MIN_AREA_PX    = 3000         
                                     
# ── ResNet50 body ReID ────────────────────────────────────────────────────
REID_INPUT_SIZE = (128, 256)          

# ── REID Matching thresholds ───────────────────────────────────────────────────
BODY_SIMILARITY_THRESH      = 0.80   # primary path — YOLO bbox → body crop
BODY_HIGH_CONF_THRESH       = 0.86   # above this → high confidence
BODY_FALLBACK_THRESH        = 0.87   # fallback path (no YOLO detections)

# ── Enrollment ────────────────────────────────────────────────────────────
ENROLL_N_FRAMES  = 8
REID_COOLDOWN_S  = 5.0