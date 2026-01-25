from __future__ import annotations
import time
import asyncio
from pathlib import Path
from typing import Dict, Any
import threading

import cv2

from config.video_settings import (
    load_video_pipeline_settings,
    SNAPSHOT_COUNT,
    SNAPSHOT_INTERVAL,
)
from src_video.services.detection_service.detect_body_parts import run_detection
from src_video.services.classification_service.infer_injuries_on_crops import (
    predict_injuries_on_detection_crops,
)
from src_video.services.deidentification_service.deidentify import run_deidentification
from src_video.services.detect_marker_service.detect_marker import detect_apriltags
from src_video.services.camera_capture_service.capture_img import (
    gstreamer_pipeline,
    capture_images,
)
from src_video.domain.constants import COLOR_TEXT


def _as_posix(path: str) -> str:
    """Convert path to POSIX format."""
    return str(path).replace("\\", "/")


def initialize_camera(flip_method: int = 0) -> cv2.VideoCapture:
    """Initialize and return the camera capture object."""
    video_capture = cv2.VideoCapture(
        gstreamer_pipeline(flip_method=flip_method), cv2.CAP_GSTREAMER
    )
    if not video_capture.isOpened():
        raise RuntimeError("Error: Unable to open camera")
    print("Camera started successfully! Detecting markers...\n")
    return video_capture


def update_fps(frame_count: int, start_time: float, print_info: bool) -> tuple[int, float, float]:
    """Calculate and return updated FPS metrics."""
    elapsed = time.time() - start_time
    if elapsed > 2.0:
        fps = frame_count / elapsed
        if print_info:
            print(f"\n[PERFORMANCE] FPS: {fps:.1f}")
        return 0, time.time(), fps
    return frame_count, start_time, 0.0


def draw_overlay(frame, fps: float, show_visualization: bool, processing: bool = False) -> None:
    """Draw FPS and visualization status on the frame."""
    cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2
    )
    cv2.putText(
        frame,
        f"Viz: {'ON' if show_visualization else 'OFF'}",
        (10, 60),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        COLOR_TEXT,
        2,
    )
    # puts processing text on the frame when detection is running
    if processing:
        cv2.putText(
            frame,
            "PROCESSING...",
            (10, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),  # Orange
            2,
        )


def run_detection_pipeline(settings: Dict[str, Any]) -> bool:
    """
    Run the body part detection pipeline.
    
    Returns:
        True if detection succeeded, False otherwise.
    """
    print("[video] Starting detection...\n")
    detection_output = Path(settings["DETECTION_OUTPUT"])

    try:
        run_detection(
            model=settings["DETECTION_MODEL"],
            source=settings["IMAGE_SAVE_DIR"],
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

        # Log output statistics
        annotated_dir = Path(settings["ANNOTATED_DIR"])
        vis_dir = Path(settings["VIS_DIR"])
        crops_root = Path(settings["CROPS_ROOT"])

        annotated_count = sum(1 for _ in annotated_dir.rglob("*.jpg")) if annotated_dir.exists() else 0
        crop_count = sum(1 for _ in crops_root.rglob("*.jpg")) if crops_root.exists() else 0
        vis_count = sum(1 for _ in vis_dir.rglob("*.jpg")) if vis_dir.exists() else 0

        print({
            "detection_output": str(detection_output),
            "annotated_jpg": int(annotated_count),
            "crop_jpg": int(crop_count),
            "vis_jpg": int(vis_count),
        })
        return True

    except Exception as e:
        print(f"[video] Detection failed: {e}")
        return False


def run_injury_inference(settings: Dict[str, Any]) -> tuple[bool, int, Dict[str, Any]]:
    """
    Run injury classification inference on detected crops.
    
    Returns:
        Tuple of (success, crop_count, inference_summary).
    """
    print("[video] Starting injury inference...\n")
    crops_root = Path(settings["CROPS_ROOT"])

    try:
        if not crops_root.exists():
            raise FileNotFoundError(
                f"No crops directory found at {crops_root}. Run detection first."
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
        return True, crop_count, infer_summary

    except Exception as e:
        print(f"[video] Injury inference failed: {e}")
        return False, 0, {}


def run_deidentification_pipeline(settings: Dict[str, Any]) -> bool:
    """
    Run de-identification pipeline (currently disabled by default).
    
    Returns:
        True if succeeded (or skipped), False on error.
    """
    print("[video] De-identification step (placeholder)...\n")

    try:
        run_deidentification(
            input_dir=_as_posix(settings["DETECTION_OUTPUT"]),
            output_dir=_as_posix(str(Path(settings["DETECTION_OUTPUT"]) / "deidentified")),
            enabled=False,
        )
        print("[video] De-identification skipped (disabled).\n")
        return True

    except Exception as e:
        print(f"[video] De-identification failed: {e}")
        return False


def print_summary(settings: Dict[str, Any], crop_count: int, infer_summary: Dict[str, Any]) -> None:
    """Print final pipeline summary."""
    print("[video] Summary:")
    print({
        "pipeline_root": settings["PIPELINE_ROOT"],
        "detection_output": str(Path(settings["DETECTION_OUTPUT"])),
        "crops_used": str(Path(settings["CROPS_ROOT"])),
        "crop_count": int(crop_count),
        "injury_checkpoint": str(settings["INJURY_CHECKPOINT_PATH"]),
        "injury_report_json": str(infer_summary.get("out_json")),
        "injury_report_csv": str(infer_summary.get("out_csv")),
    })


def process_single_image(settings: Dict[str, Any]) -> bool:
    """
    Process a single captured image through the full pipeline.
    
    Returns:
        True if processing succeeded, False otherwise.
    """
    # Run detection
    if not run_detection_pipeline(settings):
        print("Error: Detection pipeline failed for this image.")
        return False

    # Run injury inference
    success, crop_count, infer_summary = run_injury_inference(settings)
    if not success:
        print("Error: Injury inference failed for this image.")
        return False

    # Run de-identification
    if not run_deidentification_pipeline(settings):
        print("Error: De-identification failed for this image.")
        return False

    # Print summary for this image
    print_summary(settings, crop_count, infer_summary)
    print("[video] Image processing complete.\n")
    
    return True


async def main() -> int:

    settings = load_video_pipeline_settings()
    
    window_title = "CSI Camera"
    video_capture = None

    try:
        video_capture = initialize_camera(flip_method=0)
        cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

        # State variables
        frame_count = 0
        start_time = time.time()
        fps = 0.0
        show_visualization = True
        print_info = True
        detected_tag = False
        last_snap_time = -1
        total_images_processed = 0
        is_processing = False

        print("Starting camera monitoring...\n")

        while True:
            ret_val, frame = video_capture.read()
            if not ret_val:
                print("ERROR: Failed to grab frame")
                return 1

            # Check for user exit
            keyCode = cv2.waitKey(10) & 0xFF
            if keyCode == 27 or keyCode == ord("q"):
                print("\n[INFO] User requested exit.\n")
                break

            # Update FPS
            frame_count += 1
            frame_count, start_time, new_fps = update_fps(frame_count, start_time, print_info)
            if new_fps > 0:
                fps = new_fps

            # Draw overlay and display frame
            draw_overlay(frame, fps, show_visualization, is_processing)
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                cv2.imshow(window_title, frame)
            else:
                break

            # Detect AprilTag 
            detected_tag = detect_apriltags(video_capture, show_visualization=show_visualization, print_info=print_info)

            # If tag detected and not currently processing
            if detected_tag and not is_processing:
                current_time = time.time()
                should_capture = (last_snap_time == -1 or 
                                current_time - last_snap_time >= SNAPSHOT_INTERVAL)

                if should_capture:
                    print(f"\n[INFO] Tag detected! Capturing and processing image #{total_images_processed + 1}...\n")
                    
                    # Capture the image
                    if not capture_images(video_capture):
                        print("ERROR: Failed to capture image")
                        return 1
                    
                    # Set processing flag to prevent overlapping captures
                    is_processing = True
                    
                    # Process the captured image through full pipeline
                    if process_single_image(settings):
                        total_images_processed += 1
                        print(f"[INFO] Successfully processed {total_images_processed} images total.\n")
                    else:
                        print("[WARNING] Image processing failed, continuing monitoring...\n")
                    
                    # Reset processing flag
                    is_processing = False
                    last_snap_time = current_time

            elif not detected_tag and total_images_processed > 0:
                print("[INFO] Tag lost. Continuing to monitor...\n")

        print(f"\n[INFO] Session complete. Processed {total_images_processed} images total.\n")
        return 0

    except Exception as e:
        print(f"Camera error: {e}")
        return 1

    finally:
        if video_capture is not None:
            video_capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))