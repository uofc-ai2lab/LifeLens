from __future__ import annotations

import os
import time
import threading
import shutil
import cv2
import argparse

from pathlib import Path
from typing import Dict, Any
from queue import Queue, Empty

from config.logger import video_logger as log
from config.video_settings import (
    load_video_pipeline_settings,
    SNAPSHOT_INTERVAL,
    IMAGE_SAVE_DIR,
)

from src_video.services.detection_service.detect_body_parts import run_detection
from src_video.services.body_ranking.body_injury_ranking import body_ranking
from src_video.services.classification_service.infer_injuries_on_crops import predict_injuries_on_detection_crops
from src_video.services.deidentification_service.deidentify import run_deidentification
from src_video.services.detect_marker_service.detect_marker import detect_apriltags
from src_video.services.camera_capture_service.capture_img import (
    initialize_camera,
    draw_overlay
)

def _as_posix(path: str) -> str:
    return str(path).replace("\\", "/")


def put_latest(queue: Queue, item):
    """
    Drop old item if queue is full, keep newest.
    """
    if queue.full():
        try:
            queue.get_nowait()
        except Empty:
            pass
    queue.put(item)


def read_frame_from_pipeline(pipeline):
    """Read a frame from either VideoCapture or GStreamerVideoPipeline."""
    if hasattr(pipeline, "read_frame"):
        return pipeline.read_frame()
    return pipeline.read()


def release_pipeline(pipeline):
    """Release any pipeline type safely."""
    if hasattr(pipeline, "cleanup"):
        pipeline.cleanup()
    elif hasattr(pipeline, "stop"):
        pipeline.stop()
    elif hasattr(pipeline, "release"):
        pipeline.release()


def save_frame(frame) -> bool:
    timestamp = cv2.getTickCount()
    filename = os.path.join(IMAGE_SAVE_DIR, f"captured_img_{timestamp}.jpg")
    return cv2.imwrite(filename, frame)


def capture_frame_from_pipeline(frame, image_save_dir: str) -> bool:
    """
    Saves a single frame to disk from the video pipeline.
    """
    if frame is None:
        log.error("No frame to save")
        return False
    timestamp = cv2.getTickCount()
    filename = os.path.join(image_save_dir, f"captured_img_{timestamp}.jpg")

    if not cv2.imwrite(filename, frame):
        log.error(f"Failed to save frame to {filename}")
        return False
    
    log.info(f"Frame saved to {filename}")
    return True

def process_single_image(settings: Dict[str, Any]) -> bool:
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

        log.success("Detection done")

    except Exception as e:
        log.error(f"Detection failed: {e}")
        return False

    try:
        crops_root = Path(settings["CROPS_ROOT"])
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

        log.success("Classification done")

    except Exception as e:
        log.error(f"Classification failed: {e}")

    if not body_ranking(settings):
        log.warning("Ranking failed")

    try:
        deidentify_result = run_deidentification(
            input_dir=_as_posix(IMAGE_SAVE_DIR),
            output_dir=_as_posix(str(Path(settings["DETECTION_OUTPUT"]) / "deidentified")),
            enabled=True,
            threshold=0.2,
            replacewith="blur",
            mask_scale=1.3,
            ellipse=True,
            draw_scores=False,
        )
        if deidentify_result.get("success"):
            log.success(f"De-identification complete: {deidentify_result['processed_count']} images")
        else:
            log.warning(f"De-identification issue: {deidentify_result.get('note', deidentify_result.get('error'))}")

    except Exception as e:
        log.error(f"De-identification failed: {e}")

    try:
        crops_root = Path(settings["CROPS_ROOT"])
        if crops_root.exists():
            shutil.rmtree(crops_root)
            crops_root.mkdir(parents=True, exist_ok=True)

    except Exception as e:
        log.warning(f"Cleanup failed: {e}")

    log.info("Image processed")
    return True


def processing_worker(queue: Queue, settings: Dict[str, Any]):
    BATCH_SIZE = 2
    BATCH_TIMEOUT = 2.0
    batch = []
    last_flush = time.time()

    log.info("Processing worker started")

    while True:
        try:
            job = queue.get(timeout=0.5)

        except Empty:
            job = None

        # Shutdown
        if job is None and batch:
            pass
        elif job is None:
            continue

        if job == "STOP":
            queue.task_done()
            break

        batch.append(job)
        queue.task_done()

        now = time.time()

        if (
            len(batch) >= BATCH_SIZE
            or (now - last_flush) >= BATCH_TIMEOUT
        ):
            log.info(f"Processing batch ({len(batch)} jobs)")
            try:
                process_single_image(settings)
            except Exception as e:
                log.error(f"Batch processing error: {e}")

            batch.clear()
            last_flush = now

    log.info("Processing worker stopped")


def main() -> int:
    """
    Main video processing pipeline.
    
    Args:
        video_pipeline: Optional GStreamer video pipeline object. If None, initializes own pipeline.
                       When called from orchestrator, passes pre-initialized pipeline.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    
    settings = load_video_pipeline_settings()

    if args.dev:
        log.header("DEV Mode")
        process_single_image(settings)
        return 0

    # Use provided pipeline or initialize new one
    if video_pipeline is None:
        video_pipeline = initialize_camera()
        owns_pipeline = True
    else:
        # Use the pipeline passed from orchestrator (already initialized)
        owns_pipeline = False

    image_queue = Queue(maxsize=3)

    worker = threading.Thread(
        target=processing_worker,
        args=(image_queue, settings),
        daemon=True,
    )

    worker.start()
    window = "CSI Camera"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)

    last_snap = 0
    frame_count = 0
    start_time = time.time()
    fps = 0.0

    processing = False
    log.header("Video Pipeline Started")

    try:
        while True:
            ok, frame = read_frame_from_pipeline(video_pipeline)
            if not ok or frame is None:
                log.error("Camera read failed")
                break

            # Check if frame is valid
            if frame.size == 0:
                log.error("Empty frame received")
                continue

            # FPS
            frame_count += 1
            elapsed = time.time() - start_time

            if elapsed >= 2.0:
                fps = frame_count / elapsed
                frame_count = 0
                start_time = time.time()

            # Marker detection
            detected = detect_apriltags(frame)
            now = time.time()

            # Capture
            if detected and (now - last_snap) >= SNAPSHOT_INTERVAL:

                if capture_frame_from_pipeline(frame, IMAGE_SAVE_DIR):

                    job = {
                        "time": now,
                        "id": int(now * 1000),
                    }

                    put_latest(image_queue, job)
                    processing = True
                    last_snap = now

                    log.success("Job queued")


            draw_overlay(frame, fps, processing)
            cv2.imshow(window, frame)
            
            # Single waitKey with proper ESC and 'q' handling
            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                log.info("Exit key pressed")
                break


    except KeyboardInterrupt:
        log.info("Interrupted")

    finally:
        log.info("Shutting down")
        image_queue.put("STOP")
        worker.join(timeout=5)

        # Only release pipeline if we created it (not if passed from orchestrator)
        if owns_pipeline and video_pipeline is not None:
            release_pipeline(video_pipeline)
        
        cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    raise SystemExit(main())

