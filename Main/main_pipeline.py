"""Main pipeline entrypoint.

Press Run in VS Code to:
1. Run body-part detection on the internal Priv_personpart image set (sample subset recommended).
2. Export cropped parts into an ImageFolder-style directory for classification.
3. Train Swin-Tiny classifier on the exported crops.

Adjust the CONFIG block below as needed.
"""
from pathlib import Path
import sys

# Ensure root (parent of this 'Main' folder) is on sys.path so sibling 'VisualProcessing' resolves
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from VisualProcessing.ObjectDetection.detect_body_parts import run_detection, detect_and_train_classification

# ----------------------- CONFIG (edit as needed) -----------------------
# Source images directory (use the provided ImageSamples subset for quick iteration)
DETECTION_SOURCE = "VisualProcessing/ObjectDetection/Priv_personpart/ImageSamples"
# Central pipeline output root
PIPELINE_ROOT = "Main/PipelineOutputs"
# Subfolders for detection and classification stage outputs
DETECTION_OUTPUT = f"{PIPELINE_ROOT}/DetectionOutput"
CLASSIFICATION_OUTPUT = f"{PIPELINE_ROOT}/ClassificationOutput"
CLASSIFICATION_EXPORT = f"{CLASSIFICATION_OUTPUT}/parts_dataset"
# Detection parameters
DETECTION_MODEL = "MnLgt/yolo-human-parse"  # or local .pt
MAX_IMAGES = 200            # set <=0 to use all
ADD_HEAD = True             # include composite head
ALPHA_PNG = False           # set True if RGBA crops also desired
MIN_AREA = 250              # filter tiny masks
MARGIN = 0.10               # expand crop bbox
CLASSES = ["face", "arm", "hand", "leg", "foot", "neck", "torso", "head"]
DEVICE = None               # e.g. '0' for first CUDA GPU, or 'cpu'
DEBUG = False

# Classification parameters
CLS_EPOCHS = 5
CLS_BATCH_SIZE = 32
CLS_IMG_SIZE = 224
CLS_VAL_RATIO = 0.2
CLS_TEST_RATIO = 0.0
CLS_LR = 3e-4
CLS_SPLIT_SEED = 42  # reproducible split
CLS_FREEZE_BACKBONE = False
# ----------------------------------------------------------------------


def main():
    print("[pipeline] Starting detection phase...")
    # Ensure pipeline folders exist
    Path(PIPELINE_ROOT).mkdir(parents=True, exist_ok=True)
    Path(DETECTION_OUTPUT).mkdir(parents=True, exist_ok=True)
    Path(CLASSIFICATION_OUTPUT).mkdir(parents=True, exist_ok=True)

    detection_summary = run_detection(
        model=DETECTION_MODEL,
        source=DETECTION_SOURCE,
        output=DETECTION_OUTPUT,
        classes=CLASSES,
        margin=MARGIN,
        min_area=MIN_AREA,
        device=DEVICE,
        add_head=ADD_HEAD,
        debug=DEBUG,
        alpha_png=ALPHA_PNG,
        max_images=MAX_IMAGES,
        classification_export_dir=CLASSIFICATION_EXPORT,
    )
    export_dir = detection_summary.get("classification_export_dir")
    if not export_dir:
        print("[pipeline] No classification export directory produced; skipping classification stage.")
        return

    # Ensure export dir exists (run_detection should have created it)
    Path(export_dir).mkdir(parents=True, exist_ok=True)

    print("[pipeline] Detection complete. Starting classification phase...")
    cls_result = detect_and_train_classification(
        detection_export_dir=export_dir,
        classification_epochs=CLS_EPOCHS,
        classification_batch_size=CLS_BATCH_SIZE,
        classification_img_size=CLS_IMG_SIZE,
        classification_val_ratio=CLS_VAL_RATIO,
        classification_test_ratio=CLS_TEST_RATIO,
        classification_lr=CLS_LR,
        classification_split_seed=CLS_SPLIT_SEED,
        freeze_backbone=CLS_FREEZE_BACKBONE,
    )

    print("[pipeline] Classification complete.")
    print("[pipeline] Summary:")
    print({
        "pipeline_root": PIPELINE_ROOT,
        "detection_output": detection_summary["output"],
        "classification_output_root": CLASSIFICATION_OUTPUT,
        "classification_export_dataset": export_dir,
        "best_val_accuracy": cls_result.get("best_val_accuracy"),
        "val_metrics": cls_result.get("val_metrics"),
        "checkpoint_path": cls_result.get("checkpoint_path"),
    })


if __name__ == "__main__":
    main()
