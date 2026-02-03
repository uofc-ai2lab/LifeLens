from __future__ import annotations
import os
import time
import asyncio
from pathlib import Path
from typing import Dict, Any
import threading
import shutil
import cv2

from config.video_settings import (
    load_video_pipeline_settings,
    SNAPSHOT_COUNT,
    SNAPSHOT_INTERVAL,
    IMAGE_SAVE_DIR,
)
from src_video.services.detection_service.detect_body_parts import run_detection
from src_video.services.classification_service.infer_injuries_on_crops import predict_injuries_on_detection_crops
from src_video.services.deidentification_service.deidentify import run_deidentification
from src_video.services.detect_marker_service.detect_marker import detect_apriltags
from src_video.services.camera_capture_service.capture_img import (gstreamer_pipeline, capture_images,)
from src_video.services.person_reid_service.tracker import PersonReIDEngine
from src_video.services.person_reid_service.integration import run_person_reid_on_detections, generate_reid_report
from src_video.domain.constants import COLOR_TEXT
from src_video.domain.entities import create_body_parts


def _as_posix(path: str) -> str:
    """Convert path to POSIX format."""
    return str(path).replace("\\", "/")


def initialize_camera(flip_method: int = 0) -> cv2.VideoCapture:
    """Initialize and return the camera capture object (Jetson CSI via nvarguscamerasrc).

    Notes:
    - OpenCV/GStreamer can report "opened" even if Argus fails to create a CaptureSession.
      We do a short warmup read loop so we only claim success once frames are actually available.
    - Set `VIDEO_CAMERA_DEBUG=1` to print the GStreamer pipeline string.
    """

    pipeline = gstreamer_pipeline(flip_method=flip_method)
    debug = os.getenv("VIDEO_CAMERA_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
    if debug:
        print(f"[video][camera] gstreamer pipeline: {pipeline}")

    video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    if not video_capture.isOpened():
        raise RuntimeError("Error: Unable to open camera (GStreamer pipeline did not open)")

    # Warm up and validate we can actually grab frames.
    warmup_reads = 20
    warmup_sleep_s = 0.05
    last_err = None
    for _ in range(warmup_reads):
        ret_val, frame = video_capture.read()
        if ret_val and frame is not None:
            print("Camera started successfully! Detecting markers...\n")
            return video_capture
        last_err = "read() returned no frame"
        time.sleep(warmup_sleep_s)

    video_capture.release()
    raise RuntimeError(
        "Error: Camera opened but no frames received. "
        "If you see 'Failed to create CaptureSession', restart `nvargus-daemon`, "
        "ensure no other process is using the CSI camera, and verify the camera ribbon/port. "
        f"({last_err})"
    )


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


def print_summary(settings: Dict[str, Any], infer_summary: Dict[str, Any]) -> None:
    """Print final pipeline summary."""
    print("[video] Summary:")
    print({
        "pipeline_root": settings["PIPELINE_ROOT"],
        "detection_output": str(settings["DETECTION_OUTPUT"]),
        "crops_used": str(settings["CROPS_ROOT"]),
        "injury_checkpoint": str(settings["INJURY_CHECKPOINT_PATH"]),
        "injury_report_json": str(infer_summary.get("out_json")),
        "injury_report_csv": str(infer_summary.get("out_csv")),
    })

def body_ranking(settings: Dict[str, Any]) -> bool:
    """
    Track best injury predictions per body part across multiple captures.
    Creates and updates visual_output.json and visual_output.csv with 
    the highest confidence prediction for each body part.
    
    Returns:
        True if successful, False otherwise.
    """
    import json
    import csv
    
    classification_output = Path(settings["CLASSIFICATION_OUTPUT"])
    prediction_json = classification_output / "injury_predictions.json"
    template_json = classification_output / "visual_output.json"
    output_csv = classification_output / "visual_output.csv"
    
    
    try:
        if not prediction_json.exists():
            print(f"[WARNING] Prediction file not found: {prediction_json}")
            return False
        
        # predictions of classified injuries of body parts 
        with open(prediction_json, 'r') as f:
            predictions_data = json.load(f)
        
        # Load existing or create a visual output file 
        if template_json.exists():
            with open(template_json, 'r') as f:
                best_results = json.load(f)
        else:
            print(f"[INFO] Creating new visual_output file: {template_json}")
            best_results = create_body_parts()
        
        # Update best results with new predictions
        updated_count = 0
        for pred in predictions_data.get("predictions", []):
            body_part = pred.get("body_part")
            injury_pred = pred.get("injury_pred")
            injury_prob = pred.get("injury_prob", 0.0)
            image_id = pred.get("image_id")
            
            # Skip if body part not in template
            #TODO: should this be a predefined file or continuously add to it if our model gets a new body part? 
            if body_part not in best_results:
                continue
            
            current_accuracy = best_results[body_part].get("accuracy")
            
            # Update if no existing prediction or new prediction has higher accuracy
            if current_accuracy is None or injury_prob > current_accuracy:
                best_results[body_part] = {
                    "image_id": image_id,
                    "injury_pred": injury_pred,
                    "accuracy": injury_prob
                }
                updated_count += 1
                print(f"[UPDATE] {body_part}: {injury_pred} ({injury_prob:.3f})")
        
        # Save updated JSON
        with open(template_json, 'w') as f:
            json.dump(best_results, f, indent=2)
        
        # Generate CSV
        with open(output_csv, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(["body_part", "image_id", "injury_pred", "accuracy"])
            
            for body_part, data in best_results.items():
                writer.writerow([
                    body_part,
                    data.get("image_id") or "",
                    data.get("injury_pred") or "",
                    data.get("accuracy") or ""
                ])
        
        print(f"\n[INFO] Body ranking complete. Updated {updated_count} body parts.")
        print(f"[INFO] JSON output: {template_json}")
        print(f"[INFO] CSV output: {output_csv}\n")
        
        return True
        
    except Exception as e:
        print(f"[ERROR] Body ranking failed: {e}")
        return False


def process_single_image(settings: Dict[str, Any], reid_engine: PersonReIDEngine, image_count: int) -> bool:
    """
    Process a single captured image through the full pipeline.
    
    Args:
        settings: Pipeline configuration settings
        reid_engine: PersonReIDEngine instance for person re-identification
        image_count: Current image count (1-indexed)
    
    Returns:
        True if processing succeeded, False otherwise.
    """

    # Object detection
    try:
        run_detection(
            model=settings["DETECTION_MODEL"],
            source=_as_posix(IMAGE_SAVE_DIR),
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

    except Exception as e:
        print(f"[video] Object Detection failed: {e}")
    
    # Run person re-identification on detected body parts
    crops_root = Path(settings["CROPS_ROOT"])
    is_gallery_registration = (image_count == 1)
    
    try:
        print("\n[video] Running person re-identification...\n")
        reid_result = run_person_reid_on_detections(
            crops_root=crops_root,
            image_id=f"image_{image_count:03d}",
            reid_engine=reid_engine,
            is_gallery_registration=is_gallery_registration,
        )
        
        if reid_result["success"]:
            if is_gallery_registration:
                print(f"[video] Gallery registration: {reid_result['detections_registered']} body parts")
                print(f"[video] Person ID: {reid_result['person_id']}")
            else:
                summary = reid_result.get("summary", {})
                print(f"[video] Re-identification: {summary.get('identified_count')}/{summary.get('total_detections')} identified")
                print(f"[video] ReID rate: {summary.get('reid_rate', 0):.1%}")

                # Print detected person IDs for this captured image.
                reid_rows = reid_result.get("reid_results", [])
                matched_person_ids = sorted({
                    row.get("person_id")
                    for row in reid_rows
                    if row.get("is_identified") and row.get("person_id")
                })
                if matched_person_ids:
                    print(f"[video] Detected person ID(s) in this image: {matched_person_ids}")
                else:
                    print("[video] Detected person ID(s) in this image: [] (no gallery match)")
        else:
            print(f"[video] Person ReID failed: {reid_result.get('error')}")
    
    except Exception as e:
        print(f"[video] Person ReID error: {e}")
    
    # Run injury inference
    try:

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

    except Exception as e:
        print(f"[video] Injury inference failed: {e}")

    # # Update body part rankings with best predictions 
    # if not body_ranking(settings):
    #     print("Warning: Body ranking update failed, but continuing...")


    # Run de-identification

    try:
        run_deidentification(
            input_dir=_as_posix(settings["DETECTION_OUTPUT"]),
            output_dir=_as_posix(str(Path(settings["DETECTION_OUTPUT"]) / "deidentified")),
            enabled=False,
        )
        print("[video] De-identification skipped (disabled).\n")

    except Exception as e:
        print(f"[video] De-identification failed: {e}")

    # Print summary for this image
    print_summary(settings, infer_summary)

    # deleting the crops folder to prevent accumulation
    # crops_root = Path(settings["CROPS_ROOT"])

    # if crops_root.exists():
    #     try:
    #         shutil.rmtree(crops_root)
    #         crops_root.mkdir(parents=True, exist_ok=True)
    #         print(f"[INFO] Cleaned crops folder: {crops_root}")
    #     except Exception as e:
    #         print(f"[WARNING] Failed to clean crops folder: {e}")
    
    # print("[video] Image processing complete.\n")
    
    return True


async def main(video_capture) -> int:

    settings = load_video_pipeline_settings()
    
    # Initialize person re-identification engine
    reid_dir = Path(settings["DETECTION_OUTPUT"]) / "person_reid"
    reid_engine = PersonReIDEngine(storage_dir=str(reid_dir))
    
    window_title = "CSI Camera"

    try:
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
            if total_images_processed >= SNAPSHOT_COUNT:
                print(f"\n[INFO] Reached SNAPSHOT_COUNT={SNAPSHOT_COUNT}. Ending video session.\n")
                break

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

            # Detect AprilTag (reuse current frame; avoids double-grabbing)
            detected_tag = detect_apriltags(frame, show_visualization=show_visualization, print_info=print_info)

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
                    if process_single_image(settings, reid_engine, total_images_processed + 1):
                        total_images_processed += 1
                        print(f"[INFO] Successfully processed {total_images_processed} images total.\n")
                    else:
                        print("[WARNING] Image processing failed, continuing monitoring...\n")
                    
                    # Reset processing flag
                    is_processing = False
                    last_snap_time = current_time

            elif not detected_tag and total_images_processed > 0:
                print("[INFO] Tag lost. Continuing to monitor...\n")

        # Generate final re-identification report
        if total_images_processed > 0:
            reid_report_path = Path(settings["DETECTION_OUTPUT"]) / "person_reid" / "reid_report.json"
            generate_reid_report(reid_engine, reid_report_path, saved_images_dir=Path(IMAGE_SAVE_DIR))

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
    video_capture = initialize_camera(flip_method=0)
    raise SystemExit(asyncio.run(main(video_capture)))