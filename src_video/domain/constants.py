"""Constants (facts about the code that never change)."""

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

# Color settings for annotations
COLOR_OUTLINE = (0, 255, 0)
COLOR_CORNERS = (255, 0, 0)
COLOR_CENTER = (0, 0, 255)
COLOR_TEXT = (0, 255, 255)
COLOR_ID_TEXT = (0, 255, 0)
COLOR_DISTANCE_TEXT = (255, 0, 0)


# Camera Settings (NEEDED - passed to gstreamer_pipeline)
CAPTURE_WIDTH=1920
CAPTURE_HEIGHT=720
DISPLAY_WIDTH=960
DISPLAY_HEIGHT=540
FRAME_RATE=30
FLIP_METHOD=0  # 0=none, 1=counterclockwise, 2=180, 3=clockwise, 4=horizontal flip, 5=vertical flip, 6=upper right diag, 7=upper left diag

# Camera Calibration - NEEDED FOR ACCURATE DISTANCE
# we don't need this to be super accurate for our use case, so these are approximate values
CAMERA_FX=1225.0
CAMERA_FY=1225.0
CAMERA_CX=960.0
CAMERA_CY=540.0

CAMERA_PARAMS = (CAMERA_FX, CAMERA_FY, CAMERA_CX, CAMERA_CY)
