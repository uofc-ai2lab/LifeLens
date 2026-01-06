"""Main pipeline entrypoint.

Press Run in VS Code to:
1. Run body-part detection on the internal Priv_personpart image set (sample subset recommended).
2. Produce body-part crops.
3. Run an injury classifier checkpoint on those crops.

Configuration is loaded from `.env.template`.
"""
from pathlib import Path
import sys

# Ensure root (parent of this 'Main' folder) is on sys.path so sibling 'VisualProcessing' resolves
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Main.detect_body_parts import run_detection
from Main.infer_injuries_on_crops import predict_injuries_on_detection_crops
from Main.pipeline_config import load_pipeline_config


def main():
    config = load_pipeline_config()
    print("[pipeline] Starting detection phase...")
    # Ensure pipeline folders exist
    Path(config["PIPELINE_ROOT"]).mkdir(parents=True, exist_ok=True)
    Path(config["DETECTION_OUTPUT"]).mkdir(parents=True, exist_ok=True)
    Path(config["CLASSIFICATION_OUTPUT"]).mkdir(parents=True, exist_ok=True)

    detection_summary = run_detection(
        model=config["DETECTION_MODEL"],
        source=config["DETECTION_SOURCE"],
        output=config["DETECTION_OUTPUT"],
        classes=config["CLASSES"],
        margin=config["MARGIN"],
        min_area=config["MIN_AREA"],
        device=config["DEVICE"],
        add_head=config["ADD_HEAD"],
        debug=config["DEBUG"],
        alpha_png=config["ALPHA_PNG"],
        max_images=config["MAX_IMAGES"],
        classification_export_dir=(None if config["USE_DETECTION_CROPS_FOR_TRAINING"] else config["CLASSIFICATION_EXPORT"]),
    )

    # call blur function on the source of detection
    # call blur on faces of the detection output images
    
    crops_root = Path(config["DETECTION_OUTPUT"]) / "crops"
    if not crops_root.exists():
        print(f"[pipeline] No crops directory found at {crops_root}; aborting classification.")
        return
    # Count crops
    crop_count = sum(1 for _ in crops_root.rglob("*.jpg"))
    if crop_count == 0:
        print(f"[pipeline] No crop images found under {crops_root}; cannot train.")
        return

    print("[pipeline] Detection complete. Starting injury classification on body-part crops...")
    infer_summary = predict_injuries_on_detection_crops(
        crops_root=str(crops_root),
        checkpoint_path=config["INJURY_CHECKPOINT_PATH"],
        out_json_path=config["INJURY_REPORT_JSON"],
        out_csv_path=config["INJURY_REPORT_CSV"],
        image_size=config["INJURY_IMG_SIZE"],
        batch_size=config["INJURY_BATCH_SIZE"],
        num_workers=config["INJURY_NUM_WORKERS"],
    )

    print("[pipeline] Injury classification complete.")
    print("[pipeline] Summary:")
    print({
        "pipeline_root": config["PIPELINE_ROOT"],
        "detection_output": detection_summary["output"],
        "crops_used": str(crops_root),
        "crop_count": crop_count,
        "injury_checkpoint": config["INJURY_CHECKPOINT_PATH"],
        "injury_report_json": infer_summary.get("out_json"),
        "injury_report_csv": infer_summary.get("out_csv"),
    })


if __name__ == "__main__":
    main()
