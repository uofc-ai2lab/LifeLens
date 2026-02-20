"""Constants (facts about the code that never change)."""

import os

# Default crop filename format: <stem>_<part>_<idx>.jpg
FILENAME_DELIMITER = "_"

# Common image extensions used across the video pipeline.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}

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


# Camera Settings (needed - passed to gstreamer_pipeline)
# Low-memory defaults for Jetson stability. Override via env if needed.
CAPTURE_WIDTH = _env_int("VIDEO_CAPTURE_WIDTH", 1280)
CAPTURE_HEIGHT = _env_int("VIDEO_CAPTURE_HEIGHT", 720)
DISPLAY_WIDTH = _env_int("VIDEO_DISPLAY_WIDTH", 640)
DISPLAY_HEIGHT = _env_int("VIDEO_DISPLAY_HEIGHT", 360)
FRAME_RATE = _env_int("VIDEO_FRAME_RATE", 20)
FLIP_METHOD = _env_int("VIDEO_FLIP_METHOD", 0)  # 0=none, 1=counterclockwise, 2=180, 3=clockwise, 4=horizontal flip, 5=vertical flip, 6=upper right diag, 7=upper left diag

# Camera Calibration - NEEDED FOR ACCURATE DISTANCE
# we don't need this to be super accurate for our use case, so these are approximate values
CAMERA_FX=1225.0
CAMERA_FY=1225.0
CAMERA_CX=960.0
CAMERA_CY=540.0

CAMERA_PARAMS = (CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY)
