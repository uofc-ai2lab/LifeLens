from __future__ import annotations

import time
import threading, asyncio
import shutil
import gc
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
from config.gpu_guard import gpu_exclusive
from config.memory_cleanup import cleanup_memory, clear_jtop_cache
from config.video_settings import (
    load_video_pipeline_settings,
    SNAPSHOT_INTERVAL,
    IMAGE_SAVE_DIR,
    PROCESSED_IMAGE_DIR,
)

from src_video.services.camera_capture_service.gstreamer_video_pipeline import (
    GStreamerVideoPipeline,
    draw_overlay,
    capture_frame_from_pipeline,
)
from src_video.services.detection_service.detect_body_parts import run_detection
from src_video.services.body_ranking.body_injury_ranking import body_ranking
from src_video.services.classification_service.infer_injuries_on_crops import predict_injuries_on_detection_crops
# from src_video.services.deidentification_service.deidentify import run_deidentification
from src_video.services.detect_marker_service.detect_marker import detect_apriltags
from src_video.services.person_reid_service.reid_service import (
    build_reid_service,
    ReIDEvent,
)

# Set False to skip snapshots from center-crop fallback (low-confidence).
# Recommended: keep False unless YOLO frequently misses the enrolled person.
SNAPSHOT_ON_LOW_CONFIDENCE = False

# Run AprilTag detection every N frames while enrolling to reduce continuous
# detector pressure and memory churn on long camera sessions.
APRILTAG_DETECT_EVERY_N_FRAMES = 3

# After first AprilTag hit, pause AprilTag calls and collect enrollment samples
# directly to avoid running both heavy paths repeatedly.
ENROLL_SAMPLE_EVERY_N_FRAMES = 2
ENROLLMENT_TIMEOUT_S = 12.0

def _as_posix(path: str) -> str:
    return str(path).replace("\\", "/")


def _is_cuda_detection_failure(exc: Exception) -> bool:
    msg = str(exc).lower()
    markers = (
        "cudacachingallocator",
        "cuda",
        "nvml",
        "out of memory",
        "nvmamemallocinternaltagged",
        "cudnn",
    )
    return any(marker in msg for marker in markers)


def _clear_cuda_cache_if_available() -> None:
    """Best-effort CUDA + heap cleanup for video models."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Best-effort CUDA cleanup only.
        pass

    # Also trim host heap where possible.
    cleanup_memory()
    clear_jtop_cache()


def _run_detection_with_cpu_fallback(settings: Dict[str, Any]) -> None:
    detection_kwargs = dict(
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
        auto_rotate_subject=bool(settings.get("AUTO_ROTATE_SUBJECT", False)),
        classification_export_dir=None,
        face_multicrop=bool(settings.get("FACE_MULTICROP", False)),
        face_multicrop_parts=settings.get("FACE_MULTICROP_PARTS", ["face"]),
        face_multicrop_scales=settings.get("FACE_MULTICROP_SCALES", [0.85, 0.70]),
    )

    try:
        try:
            run_detection(**detection_kwargs)
        except Exception as e:
            if not _is_cuda_detection_failure(e):
                raise

            log.warning(
                f"Detection hit CUDA/NVML runtime failure ({e}). "
                f"Retrying once on CPU..."
            )
            _clear_cuda_cache_if_available()

            detection_kwargs["device"] = "cpu"
            run_detection(**detection_kwargs)
            log.success("Detection retry on CPU completed")
    finally:
        # Always clear jtop cache after detection completes (success or failure).
        clear_jtop_cache()


def _run_classification_with_cpu_fallback(settings: Dict[str, Any]) -> None:
    classification_kwargs = dict(
        crops_root=_as_posix(str(Path(settings["CROPS_ROOT"]))),
        checkpoint_path=str(settings["INJURY_CHECKPOINT_PATH"]),
        out_json_path=str(settings["INJURY_REPORT_JSON"]),
        out_csv_path=str(settings["INJURY_REPORT_CSV"]),
        image_size=int(settings["INJURY_IMG_SIZE"]),
        batch_size=int(settings["INJURY_BATCH_SIZE"]),
        num_workers=int(settings["INJURY_NUM_WORKERS"]),
        device=None,
        filename_delimiter="_",
        body_part_label_position=int(settings["BODY_PART_LABEL_POSITION"]),
        use_max_non_no_injury_aggregation=bool(
            settings.get("INJURY_AGG_MAX_NON_NO_INJURY", False)
        ) and bool(settings.get("FACE_MULTICROP", False)),
        no_injury_label=str(settings.get("NO_INJURY_LABEL", "no_injury")),
    )

    try:
        try:
            predict_injuries_on_detection_crops(**classification_kwargs)
        except Exception as e:
            if not _is_cuda_detection_failure(e):
                raise

            log.warning(
                f"Classification hit CUDA/NVML runtime failure ({e}). "
                f"Retrying once on CPU..."
            )
            _clear_cuda_cache_if_available()

            import torch

            classification_kwargs["device"] = torch.device("cpu")
            predict_injuries_on_detection_crops(**classification_kwargs)
            log.success("Classification retry on CPU completed")
    finally:
        # Always clear jtop cache after classification completes (success or failure).
        clear_jtop_cache()


# ═══════════════════════════════════════════════════════════════════════════
# POST-CAMERA PIPELINE  (unchanged)
# ═══════════════════════════════════════════════════════════════════════════

def move_images_to_processed() -> None:
    """Move all images from IMAGE_SAVE_DIR into PROCESSED_IMAGE_DIR."""
    inbox_dir = Path(IMAGE_SAVE_DIR)
    processed_dir = Path(PROCESSED_IMAGE_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    moved = 0
    try:
        for path in inbox_dir.glob("*"):
            if not path.is_file():
                continue
            dest = processed_dir / path.name
            path.replace(dest)
            moved += 1
    except Exception as e:
        log.warning(f"Failed moving images to processed dir: {e}")
    else:
        if moved:
            log.info(f"Moved {moved} image(s) to processed dir: {processed_dir}")


def run_post_camera_pipeline(settings: Dict[str, Any], snapshot_count: int) -> bool:
    if snapshot_count == 0:
        log.warning("No snapshots captured — skipping pipeline")
        return False

    log.header(f"Post-camera pipeline starting ({snapshot_count} snapshots)")

    try:
        with gpu_exclusive("video:detection+classification", logger=log):
            _run_detection_with_cpu_fallback(settings)
            log.success("Detection done")

            try:
                _run_classification_with_cpu_fallback(settings)
                log.success("Classification done")
            except Exception as e:
                log.error(f"Classification failed: {e}")
    except Exception as e:
        log.error(f"Detection failed: {e}")
        return False

    if not body_ranking(settings):
        log.warning("Ranking failed")
    else:
        log.success("Ranking done")

    # try:
    #     deidentify_result = run_deidentification(
    #         input_dir=_as_posix(IMAGE_SAVE_DIR),
    #         output_dir=_as_posix(str(Path(settings["DETECTION_OUTPUT"]) / "deidentified")),
    #         enabled=True,
    #         threshold=0.2,
    #         replacewith="blur",
    #         mask_scale=1.3,
    #         ellipse=True,
    #         draw_scores=False,
    #     )
    #     if deidentify_result.get("success"):
    #         log.success(
    #             f"De-identification complete: "
    #             f"{deidentify_result['processed_count']} images"
    #         )
    #     else:
    #         log.warning(
    #             f"De-identification issue: "
    #             f"{deidentify_result.get('note', deidentify_result.get('error'))}"
    #         )
    # except Exception as e:
    #     log.error(f"De-identification failed: {e}")

    try:
        crops_root = Path(settings["CROPS_ROOT"])
        if crops_root.exists():
            if bool(settings.get("KEEP_CROPS", True)):
                log.info(f"Keeping crops for debugging: {crops_root}")
            else:
                shutil.rmtree(crops_root)
                log.info(f"Removed crops directory: {crops_root}")
    except Exception as e:
        log.warning(f"Cleanup failed: {e}")

    log.success("Post-camera pipeline complete")
    return True


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(video_pipeline: Optional[GStreamerVideoPipeline] = None, external_stop_event: Optional[threading.Event] = None) -> int:
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
        run_post_camera_pipeline(settings, snapshot_count=1)
        stop_monitoring()
        return 0

    log.header("Video Pipeline Starting")

    snapshot_count = 0

    # ── ReID callback ─────────────────────────────────────────────────────
    def _on_reid(event: ReIDEvent):
        nonlocal snapshot_count

        if event.confidence == "low" and not SNAPSHOT_ON_LOW_CONFIDENCE:
            log.info(
                f"Skipping low-confidence match  "
                f"sim={event.similarity:.3f}  type={event.match_type}"
            )
            return

        log.success(
            f"ReID triggered  sim={event.similarity:.3f}  "
            f"[{event.match_type}]  confidence={event.confidence}"
        )
        if capture_frame_from_pipeline(event.frame, IMAGE_SAVE_DIR):
            snapshot_count += 1
            log.info(f"Snapshot saved ({snapshot_count} total)")

    # ── Build ReID service ────────────────────────────────────────────────
    body_thresh = float(
        settings.get(
            "REID_BODY_THRESHOLD",
            settings.get("REID_THRESHOLD", 0.80),
        )
    )

    reid = build_reid_service(
        yolo_path         = settings.get(
            "YOLO_MODEL_PATH",
            "src_video/services/person_reid_service/yolov8n.onnx",
        ),
        embedder_path     = settings.get(
            "EMBEDDER_ONNX_PATH",
            "src_video/services/person_reid_service/resnet50_market1501_aicity156.onnx",
        ),
        use_trt           = bool(settings.get("REID_USE_TRT", False)),
        similarity_thresh = body_thresh,
        on_reid_callback  = _on_reid,
    )
    log.success("ReID service ready")

    window = "CSI Camera"
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)

    frame_count = 0
    start_time  = time.time()
    fps         = 0.0
    enrolled    = False
    loop_count  = 0
    marker_locked = False
    enroll_started_at: Optional[float] = None

    log.header("Camera session started  —  press Q / ESC to end")

    try:
        while True:
            # Check for external stop event (e.g., power button)
            if external_stop_event is not None and external_stop_event.is_set():
                log.info("External stop requested")
                break

            ok, frame = video_pipeline.read_frame()
            if not ok or frame is None:
                log.error("Camera read failed")
                break
            if frame.size == 0:
                continue

            loop_count += 1

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 2.0:
                fps         = frame_count / elapsed
                frame_count = 0
                start_time  = time.time()

            detected = False
            if not enrolled and not marker_locked and (loop_count % APRILTAG_DETECT_EVERY_N_FRAMES == 0):
                detected = detect_apriltags(
                    frame,
                    show_visualization=False,
                    print_info=False,
                )

            if detected and not marker_locked:
                marker_locked = True
                enroll_started_at = time.time()
                log.info("AprilTag acquired — starting enrollment")

            # ── Enrollment ────────────────────────────────────────────────
            if marker_locked and not enrolled and (loop_count % ENROLL_SAMPLE_EVERY_N_FRAMES == 0):
                if reid.enroll_from_frame(frame):
                    enrolled = True
                    marker_locked = False
                    enroll_started_at = None
                    log.success("Primary person enrolled — ReID tracking active")

                    if capture_frame_from_pipeline(frame, IMAGE_SAVE_DIR):
                        snapshot_count += 1
                        log.info(f"Enrollment snapshot saved ({snapshot_count} total)")

            if marker_locked and not enrolled and enroll_started_at is not None:
                if (time.time() - enroll_started_at) > ENROLLMENT_TIMEOUT_S:
                    marker_locked = False
                    enroll_started_at = None
                    reid.reset_enrollment()
                    log.warning("Enrollment timed out — waiting for AprilTag reacquire")

            # ── Tracking ──────────────────────────────────────────────────
            # process_frame() runs YOLO internally each call.
            # Alternatively, pass person_bboxes from run_detection if you
            # want to reuse detections across the pipeline.
            if enrolled:
                reid.process_frame(frame)

            reid.draw_debug(frame, person_bboxes=None)
            draw_overlay(frame, fps, processing=enrolled)
            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                log.info("Camera session ended by user")
                break
            if key == ord('r'):
                reid.reset_enrollment()
                enrolled       = False
                marker_locked  = False
                enroll_started_at = None
                snapshot_count = 0
                log.info("Enrollment reset — snapshots cleared")

    except KeyboardInterrupt:
        log.info("Interrupted")

    finally:
        video_pipeline.cleanup()
        cv2.destroyAllWindows()

    log.info("Releasing ReID service...")
    cleanup_memory(reid)
    clear_jtop_cache()

    log.header("Camera closed — starting post-camera pipeline")
    run_post_camera_pipeline(settings, snapshot_count)

    move_images_to_processed()
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
        video_pipeline = GStreamerVideoPipeline()
        if not video_pipeline.start():
            log.error("Failed to initialize camera")
            raise SystemExit(1)
        try:
            raise SystemExit(main(video_pipeline))
        finally:
            video_pipeline.cleanup()
            stop_monitoring()