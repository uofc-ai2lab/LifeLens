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

from VisualProcessing.ObjectDetection.detect_body_parts import run_detection
from VisualProcessing.Classification.ClassificationModels.simple_train_swin_tiny import train_swin_tiny

# ----------------------- CONFIG (edit as needed) -----------------------
# Source images directory (use the provided ImageSamples subset for quick iteration)
DETECTION_SOURCE = "VisualProcessing/ObjectDetection/Priv_personpart/ImageSamples"
# Central pipeline output root
PIPELINE_ROOT = "Main/PipelineOutputs"
# Subfolders for detection and classification stage outputs
DETECTION_OUTPUT = f"{PIPELINE_ROOT}/DetectionOutput"
CLASSIFICATION_OUTPUT = f"{PIPELINE_ROOT}/ClassificationOutput"
CLASSIFICATION_EXPORT = f"{CLASSIFICATION_OUTPUT}/parts_dataset"  # legacy ImageFolder export (not needed when parsing filenames)
USE_DETECTION_CROPS_FOR_TRAINING = True  # train directly from detection crops filenames
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
        classification_export_dir=(None if USE_DETECTION_CROPS_FOR_TRAINING else CLASSIFICATION_EXPORT),
    )
    crops_root = Path(DETECTION_OUTPUT) / "crops"
    if not crops_root.exists():
        print(f"[pipeline] No crops directory found at {crops_root}; aborting classification.")
        return
    # Count crops
    crop_count = sum(1 for _ in crops_root.rglob("*.jpg"))
    if crop_count == 0:
        print(f"[pipeline] No crop images found under {crops_root}; cannot train.")
        return

    print("[pipeline] Detection complete. Starting classification phase from detection crops...")
    cls_result = train_swin_tiny(
        data_dir=None,
        from_detection_crops=True,
        detection_crops_root=str(crops_root),
        detection_label_position=-2,  # part token in <stem>_<part>_<idx>
        epochs=CLS_EPOCHS,
        batch_size=CLS_BATCH_SIZE,
        img_size=CLS_IMG_SIZE,
        val_ratio=CLS_VAL_RATIO,
        test_ratio=CLS_TEST_RATIO,
        lr=CLS_LR,
        split_seed=CLS_SPLIT_SEED,
        freeze_backbone=CLS_FREEZE_BACKBONE,
        save_root=str(Path("experiments/checkpoints/parts_from_detection")),
    )

    print("[pipeline] Classification complete.")
    print("[pipeline] Summary:")
    print({
        "pipeline_root": PIPELINE_ROOT,
        "detection_output": detection_summary["output"],
        "crops_used": str(crops_root),
        "crop_count": crop_count,
        "best_val_accuracy": cls_result.get("best_val_accuracy"),
        "val_metrics": cls_result.get("val_metrics"),
        "checkpoint_path": cls_result.get("checkpoint_path"),
    })


if __name__ == "__main__":
    main()
