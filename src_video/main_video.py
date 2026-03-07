from __future__ import annotations

import time
import threading
import shutil
import cv2
import numpy as np
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from queue import Queue, Empty

from config.jetson_startup import run_jetson_startup_tasks
from config.resource_usage import start_monitoring, stop_monitoring
from config.audio_settings import USAGE_FILE_PATH
from config.logger import video_logger as log
from config.video_settings import (
    load_video_pipeline_settings,
    SNAPSHOT_INTERVAL,
    IMAGE_SAVE_DIR,
)

from src_video.services.camera_capture_service.gstreamer_video_pipeline import (
    GStreamerVideoPipeline,
    draw_overlay,
    capture_frame_from_pipeline,
)
from src_video.services.detection_service.detect_body_parts import run_detection
from src_video.services.body_ranking.body_injury_ranking import body_ranking
from src_video.services.classification_service.infer_injuries_on_crops import predict_injuries_on_detection_crops
from src_video.services.deidentification_service.deidentify import run_deidentification
from src_video.services.detect_marker_service.detect_marker import detect_apriltags

# ── NEW: ReID service ──────────────────────────────────────────────────────
from src_video.services.person_reid_service.reid_service import build_reid_service, ReIDEvent
# ──────────────────────────────────────────────────────────────────────────


def _as_posix(path: str) -> str:
    return str(path).replace("\\", "/")


def put_latest(queue: Queue, item):
    """Drop old item if queue is full, keep newest."""
    if queue.full():
        try:
            queue.get_nowait()
        except Empty:
            pass
    queue.put(item)


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
        predict_injuries_on_detection_crops(
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
    BATCH_SIZE    = 2
    BATCH_TIMEOUT = 2.0
    batch         = []
    last_flush    = time.time()

    log.info("Processing worker started")

    while True:
        try:
            job = queue.get(timeout=0.5)
        except Empty:
            job = None

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
        if len(batch) >= BATCH_SIZE or (now - last_flush) >= BATCH_TIMEOUT:
            log.info(f"Processing batch ({len(batch)} jobs)")
            try:
                process_single_image(settings)
            except Exception as e:
                log.error(f"Batch processing error: {e}")
            batch.clear()
            last_flush = now

    log.info("Processing worker stopped")


def main(video_pipeline: Optional[GStreamerVideoPipeline] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    settings = load_video_pipeline_settings()
    DEV_MODE  = args.dev

    if video_pipeline is None and not DEV_MODE:
        log.error("VIDEO pipeline failed to initialize camera")
        return 1

    if DEV_MODE:
        log.header("DEV Mode")
        start_monitoring(interval=1.0, log_file=USAGE_FILE_PATH, show_stderr_line=True)
        process_single_image(settings)
        stop_monitoring()
        return 0

    log.header("Video Pipeline Starting")

    # ── NEW: initialise ReID service ─────────────────────────────────────
    image_queue = Queue(maxsize=3)   # shared with reid callback below

    def _on_reid(event: ReIDEvent):
        """Callback invoked on the video-loop thread when primary person found."""
        log.success(f"ReID triggered  sim={event.similarity:.3f}")
        if capture_frame_from_pipeline(event.frame, IMAGE_SAVE_DIR):
            job = {"time": event.timestamp, "id": int(event.timestamp * 1000), "source": "reid"}
            put_latest(image_queue, job)

    reid = build_reid_service(
        use_trt           = bool(settings.get("REID_USE_TRT",   False)),
        similarity_thresh = float(settings.get("REID_THRESHOLD", 0.55)),
        on_reid_callback  = _on_reid,
    )
    log.success("ReID service ready")
    # ─────────────────────────────────────────────────────────────────────

    worker = threading.Thread(
        target=processing_worker,
        args=(image_queue, settings),
        daemon=True,
    )
    worker.start()

    window = "CSI Camera"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)

    last_snap     = 0.0
    frame_count   = 0
    start_time    = time.time()
    fps           = 0.0
    processing    = False
    enrolled      = False   # NEW: track enrolment state for overlay

    log.header("Video Pipeline Started")

    try:
        while True:
            ok, frame = video_pipeline.read_frame()
            if not ok or frame is None:
                log.error("Camera read failed")
                break
            if frame.size == 0:
                log.error("Empty frame received")
                continue

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 2.0:
                fps        = frame_count / elapsed
                frame_count = 0
                start_time  = time.time()

            # ── AprilTag detection ────────────────────────────────────────
            detected = detect_apriltags(frame)
            now      = time.time()

            # ── CHANGED: on tag detection, enroll OR capture ──────────────
            if detected:
                if not enrolled:
                    # Enrolment phase: buffer face embeddings
                    if reid.enroll_from_frame(frame):
                        enrolled   = True
                        last_snap  = now
                        log.success("Primary person enrolled – beginning ReID tracking")

                elif (now - last_snap) >= SNAPSHOT_INTERVAL:
                    # Tag still visible after enrolment → direct snapshot
                    if capture_frame_from_pipeline(frame, IMAGE_SAVE_DIR):
                        job = {"time": now, "id": int(now * 1000), "source": "tag"}
                        put_latest(image_queue, job)
                        processing = True
                        last_snap  = now
                        log.success("Job queued (tag)")

            # ── NEW: continuous ReID matching (only after enrolment) ──────
            if enrolled:
                reid_event = reid.process_frame(frame)
                if reid_event:
                    processing = True
            # ─────────────────────────────────────────────────────────────

            draw_overlay(frame, fps, processing)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                log.info("Exit key pressed")
                break
            # Optional: press 'r' to reset enrolment mid-session
            if key == ord('r'):
                reid.reset_enrollment()
                enrolled = False
                log.info("Enrolment reset via keypress")

    except KeyboardInterrupt:
        log.info("Interrupted")

    finally:
        log.info("Shutting down")
        image_queue.put("STOP")
        worker.join(timeout=5)
        video_pipeline.cleanup()
        cv2.destroyAllWindows()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    if args.dev:
        raise SystemExit(main())
    else:
        log.info("Running startup tasks...")
        run_jetson_startup_tasks()
        start_monitoring(interval=1.0, log_file=USAGE_FILE_PATH, show_stderr_line=True)
        video_pipeline = GStreamerVideoPipeline(flip_method=0)
        if not video_pipeline.start():
            log.error("Failed to initialize camera")
            raise SystemExit(1)
        try:
            raise SystemExit(main(video_pipeline))
        finally:
            video_pipeline.cleanup()
            stop_monitoring()
