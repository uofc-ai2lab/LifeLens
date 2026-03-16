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
CAPTURE_WIDTH=1280
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

# ── YOLO person detector ──────────────────────────────────────────────────
YOLO_INPUT_SIZE       = 640          
YOLO_CONF_THRESH      = 0.45         
YOLO_NMS_IOU_THRESH   = 0.45         
YOLO_PERSON_CLASS     = 0            
PERSON_MIN_AREA_PX    = 3000         
                                     
# ── ResNet50 body ReID ────────────────────────────────────────────────────
REID_INPUT_SIZE = (128, 256)          

# ── REID Matching thresholds ───────────────────────────────────────────────────
BODY_SIMILARITY_THRESH      = 0.72   # primary path — YOLO bbox → body crop
BODY_HIGH_CONF_THRESH       = 0.76   # above this → high confidence
BODY_FALLBACK_THRESH        = 0.77   # fallback path (no YOLO detections)

# ── Enrollment ────────────────────────────────────────────────────────────
ENROLL_N_FRAMES  = 8
REID_COOLDOWN_S  = 5.0
