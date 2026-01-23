from __future__ import annotations
import time
"""Video pipeline entrypoint.

This file is meant to be runnable directly (e.g. press **Run** in VS Code):
    python src_video/main.py

It executes the full video pipeline using `config/video_settings.py`.
"""
import argparse
from pathlib import Path
import asyncio
import cv2
from config.video_settings import load_video_pipeline_settings, SNAPSHOT_COUNT, SNAPSHOT_INTERVAL, COLOR_TEXT
from src_video.services.detection_service.detect_body_parts import run_detection
from src_video.services.classification_service.infer_injuries_on_crops import (predict_injuries_on_detection_crops,)
from src_video.services.deidentification_service.deidentify import run_deidentification
from src_video.services.detect_marker_service.detect_marker import detect_apriltags
from src_video.services.camera_capture_service.capture_img import gstreamer_pipeline, capture_images

def _as_posix(path: str) -> str:
    return str(path).replace("\\", "/")

IMAGE_COUNT = 10
INTERVAL = 2

async def main() -> int:
    parser = argparse.ArgumentParser(description="Run microservices")
    parser.add_argument(
        "service",
        nargs="?",
        type=str,
        choices=["camera", "detect_marker"],
        default=None
    )

    args = parser.parse_args()
    settings = load_video_pipeline_settings()

    window_title = "CSI Camera"

    # To flip the image, modify the flip_method parameter (0 and 2 are the most common)
    video_capture = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)
    if video_capture.isOpened():

        # State variables
        frame_count = 0
        start_time = time.time()
        fps = 0.0
        show_visualization = True
        print_info = True
        DETECTED_TAG = False
        num_initial_snaps = 0
        last_snap_time = -1


        print("Camera started successfully! Detecting markers...\n")
        try:
            cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)
            while True:
                ret_val, frame = video_capture.read()
                if not ret_val:
                    print("ERROR: Failed to grab frame")
                    break
                # Check to see if the user closed the window
                keyCode = cv2.waitKey(10) & 0xFF
                # Stop the program on the ESC key or 'q'
                if keyCode == 27 or keyCode == ord('q'):
                    break

                # Calculate FPS
                frame_count += 1
                elapsed = time.time() - start_time
                if elapsed > 2.0:
                    fps = frame_count / elapsed
                    if print_info:
                        print(f"\n[PERFORMANCE] FPS: {fps:.1f}")
                    frame_count = 0
                    start_time = time.time()

                # Draw FPS and tag count on frame
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
                cv2.putText(frame, f"Viz: {'ON' if show_visualization else 'OFF'}", (10, 90),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

                # Display frame
                if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                    cv2.imshow(window_title, frame)
                else:
                    break

                #  Detect AprilTags
                if not DETECTED_TAG:
                    DETECTED_TAG = detect_apriltags(video_capture, show_visualization=show_visualization, print_info=print_info)
                    print(f"[INFO] Tags detected? {DETECTED_TAG}. Continuing to monitor...\n")                       
                # if we detect tags then we run capture image service
                if DETECTED_TAG:
                    if num_initial_snaps < SNAPSHOT_COUNT and (last_snap_time == -1 or time.time() - last_snap_time >= SNAPSHOT_INTERVAL):
                        capture_images(video_capture)
                        num_initial_snaps += 1
                        last_snap_time = time.time()
                    elif num_initial_snaps == SNAPSHOT_COUNT:
                        print(f"\n[INFO] Captured {SNAPSHOT_COUNT} images. Exiting camera service.\n")
                        break
        finally:
            video_capture.release()
            cv2.destroyAllWindows()
    else:
        print("Error: Unable to open camera")

    # else run full pipeline
    print("running full pipeline...")

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
    raise SystemExit(asyncio.run(main()))
