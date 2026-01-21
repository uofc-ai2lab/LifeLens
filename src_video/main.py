"""Video pipeline entrypoint.

This file is meant to be runnable directly (e.g. press **Run** in VS Code):
    python src_video/main.py

It executes the full video pipeline using `config/video_settings.py`.
"""
<<<<<<< HEAD

from __future__ import annotations
from src_video.services.camera_capture_service.capture_img import run_show_camera
=======
>>>>>>> 023d10a (Scrum 186 add image capture to directory (#23))
from pathlib import Path
import sys
from __future__ import annotations
from config.video_settings import load_video_pipeline_settings
from src_video.services.camera_capture_service.capture_img import run_show_camera
from src_video.services.detection_service.detect_body_parts import run_detection
from src_video.services.classification_service.infer_injuries_on_crops import (predict_injuries_on_detection_crops,)
from src_video.services.deidentification_service.deidentify import run_deidentification
<<<<<<< HEAD
from src_video.services.camera_capture_service.capture_img import run_show_camera
=======
>>>>>>> 023d10a (Scrum 186 add image capture to directory (#23))

def _as_posix(path: str) -> str:
    return str(path).replace("\\", "/")


def main() -> int:
    settings = load_video_pipeline_settings()
    run_show_camera()

    detection_output = Path(settings["DETECTION_OUTPUT"])
    crops_root = Path(settings["CROPS_ROOT"])

    try:
        print("[video] Starting detection...\n")
        run_detection(
            model=settings["DETECTION_MODEL"],
            source=settings["DETECTION_SOURCE"],
            output=_as_posix(settings["DETECTION_OUTPUT"]),
            classes=settings["CLASSES"],
            margin=float(settings["MARGIN"]),
            min_area=int(settings["MIN_AREA"]),
            device=settings.get("DEVICE"),
            add_head=bool(settings["ADD_HEAD"]),
            debug=bool(settings["DEBUG"]),
            alpha_png=bool(settings["ALPHA_PNG"]),
            max_images=int(settings["MAX_IMAGES"]),
            classification_export_dir=None,
        )
        print("\n[video] Detection finished.\n")

        # Make it very obvious where outputs landed.
        annotated_dir = Path(settings["ANNOTATED_DIR"])
        vis_dir = Path(settings["VIS_DIR"])
        annotated_count = sum(1 for _ in annotated_dir.rglob("*.jpg")) if annotated_dir.exists() else 0
        crop_count_now = sum(1 for _ in crops_root.rglob("*.jpg")) if crops_root.exists() else 0
        vis_count = sum(1 for _ in vis_dir.rglob("*.jpg")) if vis_dir.exists() else 0
        print(
            {
                "detection_output": str(detection_output),
                "annotated_jpg": int(annotated_count),
                "crop_jpg": int(crop_count_now),
                "vis_jpg": int(vis_count),
            }
        )
    except Exception as e:
        print("[video] Detection failed:", e)
        return 1

    crop_count = 0
    try:
        print("[video] Starting injury inference...\n")
        if not crops_root.exists():
            raise FileNotFoundError(
                f"No crops directory found at {crops_root}. Run detection first or set PIPELINE_DETECTION_OUTPUT."
            )

        crop_count = sum(1 for _ in crops_root.rglob("*.jpg"))
        if crop_count == 0:
            raise RuntimeError(f"No crop images found under {crops_root}")

        infer_summary = predict_injuries_on_detection_crops(
            crops_root=_as_posix(str(crops_root)),
            checkpoint_path=str(settings["INJURY_CHECKPOINT_PATH"]),
            out_json_path=str(settings["INJURY_REPORT_JSON"]),
            out_csv_path=str(settings["INJURY_REPORT_CSV"]),
            image_size=int(settings["INJURY_IMG_SIZE"]),
            batch_size=int(settings["INJURY_BATCH_SIZE"]),
            num_workers=int(settings["INJURY_NUM_WORKERS"]),
            device=None,
            filename_delimiter="_",
            body_part_label_position=int(settings["BODY_PART_LABEL_POSITION"]),
        )
        print("\n[video] Injury inference finished.\n")
    except Exception as e:
        print("[video] Injury inference failed:", e)
        return 1

    # Placeholder only. Intentionally disabled by default.
    try:
        print("[video] De-identification step (placeholder)...\n")
        run_deidentification(
            input_dir=_as_posix(settings["DETECTION_OUTPUT"]),
            output_dir=_as_posix(str(Path(settings["DETECTION_OUTPUT"]) / "deidentified")),
            enabled=False,
        )
        print("[video] De-identification skipped (disabled).\n")
    except Exception as e:
        print("[video] De-identification failed:", e)
        return 1

    print("[video] Summary:")
    print(
        {
            "pipeline_root": settings["PIPELINE_ROOT"],
            "detection_output": str(detection_output),
            "crops_used": str(crops_root),
            "crop_count": int(crop_count),
            "injury_checkpoint": str(settings["INJURY_CHECKPOINT_PATH"]),
            "injury_report_json": str(infer_summary.get("out_json")),
            "injury_report_csv": str(infer_summary.get("out_csv")),
        }
    )

    print("[video] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
