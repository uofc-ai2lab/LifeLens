"""Constants (facts about the code that never change)."""

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


# Camera Settings (NEEDED - passed to gstreamer_pipeline)
CAPTURE_WIDTH=1920
CAPTURE_HEIGHT=1080
DISPLAY_WIDTH=960
DISPLAY_HEIGHT=540

# Camera Calibration - NEEDED FOR ACCURATE DISTANCE
# we don't need this to be super accurate for our use case, so these are approximate values
CAMERA_FX=1225.0
CAMERA_FY=1225.0
CAMERA_CX=960.0
CAMERA_CY=540.0

CAMERA_PARAMS = (CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY)
