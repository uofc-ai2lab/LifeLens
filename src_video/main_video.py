from __future__ import annotations

import time
import datetime
import threading, asyncio
import shutil
import gc
import os
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
from src_audio.services.transcription_service.transcription_whispertrt import unload_whisper_model
from src_video.services.detect_marker_service.detect_marker import detect_apriltags
from src_video.services.person_reid_service.reid_service import (
    build_reid_service,
    ReIDEvent,
)


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default

# Temporary switch to measure RAM impact with REID disabled.
REID_ENABLED = False

# Set False to skip snapshots from center-crop fallback (low-confidence).
# Recommended: keep False unless YOLO frequently misses the enrolled person.
SNAPSHOT_ON_LOW_CONFIDENCE = False

# Run AprilTag detection every N frames while enrolling to reduce continuous
# detector pressure and memory churn on long camera sessions.
APRILTAG_DETECT_EVERY_N_FRAMES = 3
APRILTAG_CONTINUOUS_SNAPSHOT_INTERVAL_S = 5.0

# Sample enrollment frames at a lower cadence to reduce model pressure.
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
    """Best-effort CUDA + RAM cleanup for video models."""
    drop_os_cache = os.getenv("LIFELENS_DROP_OS_PAGE_CACHE", "0").strip().lower() in {
        "1", "true", "t", "yes", "y", "on"
    }
    cleanup_memory(
        clear_cuda_cache=True,
        clear_jtop=True,
        drop_linux_page_cache=drop_os_cache,
    )


def _run_detection_with_cpu_fallback(settings: Dict[str, Any]) -> None:
    requested_device = settings.get("DEVICE")
    log.info(
        "Detection configured device="
        f"{requested_device if requested_device is not None else 'auto'}"
    )

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
        crops_namespace=settings.get("CROPS_NAMESPACE"),
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
    clear_jtop_cache()

    # Unload Whisper before loading detection model — both are large and on Jetson
    # unified RAM they cannot comfortably coexist with the 71M-param YOLO model.
    try:
        unload_whisper_model()
    except Exception as e:
        log.warning(f"Could not unload Whisper before post-camera pipeline: {e}")

    # Isolate each run's crops so classification does not traverse historical data.
    run_id = datetime.datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_settings = dict(settings)
    run_settings["CROPS_NAMESPACE"] = run_id
    run_settings["CROPS_ROOT"] = str(Path(settings["DETECTION_OUTPUT"]) / "crops" / run_id)

    try:
        with gpu_exclusive("video:detection+classification", logger=log):
            _run_detection_with_cpu_fallback(run_settings)
            log.success("Detection done")

            # Flush CUDA allocations and trim heap between the two heavy models.
            _clear_cuda_cache_if_available()

            try:
                _run_classification_with_cpu_fallback(run_settings)
                log.success("Classification done")
            except Exception as e:
                log.error(f"Classification failed: {e}")
    except Exception as e:
        log.error(f"Detection failed: {e}")
        return False

    if not body_ranking(run_settings):
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
        crops_root = Path(run_settings["CROPS_ROOT"])
        if crops_root.exists():
            if bool(run_settings.get("KEEP_CROPS", True)):
                log.info(f"Keeping crops for debugging: {crops_root}")
            else:
                shutil.rmtree(crops_root)
                log.info(f"Removed crops directory: {crops_root}")
    except Exception as e:
        log.warning(f"Cleanup failed: {e}")

    log.success("Post-camera pipeline complete")
    return True


def _reset_video_directory_on_startup() -> None:
    """Delete and recreate the video data directory at run start."""
    video_root = Path(IMAGE_SAVE_DIR).parent

    try:
        if video_root.exists():
            shutil.rmtree(video_root)
            log.info(f"Deleted startup video directory: {video_root}")

        (video_root / "source_files").mkdir(parents=True, exist_ok=True)
        Path(IMAGE_SAVE_DIR).mkdir(parents=True, exist_ok=True)
        Path(PROCESSED_IMAGE_DIR).mkdir(parents=True, exist_ok=True)
        log.info(f"Recreated startup video directories under: {video_root}")
    except Exception as e:
        log.warning(f"Startup video directory reset failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main(video_pipeline: Optional[GStreamerVideoPipeline] = None, external_stop_event: Optional[threading.Event] = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    _reset_video_directory_on_startup()

    settings = load_video_pipeline_settings()
    DEV_MODE  = args.dev
    

    if video_pipeline is None and not DEV_MODE:
        log.error("VIDEO pipeline failed to initialize camera")
        return 1

    if DEV_MODE:
        log.header("DEV Mode")
        start_monitoring(interval=1.0, log_file=USAGE_FILE_PATH, show_stderr_line=True)
        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}
        try:
            snapshot_count = sum(
                1
                for p in Path(IMAGE_SAVE_DIR).rglob("*")
                if p.is_file() and p.suffix.lower() in exts
            )
        except Exception:
            snapshot_count = 0
        run_post_camera_pipeline(settings, snapshot_count=snapshot_count)
        stop_monitoring()
        return 0
     

    log.header("Video Pipeline Starting")
    log.info(
        "Video runtime config: "
        f"APRILTAG_DETECT_EVERY_N_FRAMES={APRILTAG_DETECT_EVERY_N_FRAMES}, "
        f"REID_ENABLED={REID_ENABLED}, "
        f"SNAPSHOT_INTERVAL={SNAPSHOT_INTERVAL}"
    )

    snapshot_count = 0
    capture_flash_until = 0.0
    capture_flash_duration_s = 0.18

    # ── ReID callback ─────────────────────────────────────────────────────
    def _on_reid(event: ReIDEvent):
        nonlocal snapshot_count, capture_flash_until

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
            capture_flash_until = time.time() + capture_flash_duration_s
            log.info(f"Snapshot saved ({snapshot_count} total)")

    # ── Build ReID service ────────────────────────────────────────────────
    reid = None
    if REID_ENABLED:
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
    else:
        log.warning("REID disabled for RAM profiling")

    window = "CSI Camera"
    log.info("Creating OpenCV window")
    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(window, 960, 540)
    log.info("OpenCV window ready")

    frame_count = 0
    start_time  = time.time()
    fps         = 0.0
    enrolled    = False
    loop_count  = 0
    marker_locked = False
    enroll_started_at: Optional[float] = None
    first_frame_logged = False
    last_loop_heartbeat = time.time()
    apriltag_visible = False
    apriltag_has_been_seen = False
    next_apriltag_snapshot_at: Optional[float] = None

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

            if not first_frame_logged:
                log.success(
                    f"First frame received: shape={frame.shape}, dtype={frame.dtype}"
                )
                first_frame_logged = True

            loop_count += 1

            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed >= 2.0:
                fps         = frame_count / elapsed
                frame_count = 0
                start_time  = time.time()

            now = time.time()
            if now - last_loop_heartbeat >= 5.0:
                log.info(
                    "Camera loop heartbeat: "
                    f"fps={fps:.1f}, loop_count={loop_count}, "
                    f"marker_locked={marker_locked}, enrolled={enrolled}, "
                    f"apriltag_visible={apriltag_visible}, snapshots={snapshot_count}"
                )
                last_loop_heartbeat = now

            detected = False
            if (
                not enrolled
                and (loop_count % APRILTAG_DETECT_EVERY_N_FRAMES == 0)
            ):
                detected = detect_apriltags(
                    frame,
                    show_visualization=False,
                    print_info=False,
                )
                if detected != apriltag_visible:
                    apriltag_visible = detected
                    if apriltag_visible:
                        log.info("AprilTag visible")
                    else:
                        log.info("AprilTag no longer visible")

                if detected and not apriltag_has_been_seen:
                    apriltag_has_been_seen = True
                    next_apriltag_snapshot_at = now + APRILTAG_CONTINUOUS_SNAPSHOT_INTERVAL_S

            if detected and not marker_locked:
                marker_locked = True
                enroll_started_at = time.time()
                log.info("AprilTag acquired — starting enrollment")
                if capture_frame_from_pipeline(frame, IMAGE_SAVE_DIR):
                    snapshot_count += 1
                    capture_flash_until = time.time() + capture_flash_duration_s
                    log.info(f"AprilTag snapshot saved ({snapshot_count} total)")

            if (
                apriltag_has_been_seen
                and apriltag_visible
                and next_apriltag_snapshot_at is not None
                and now >= next_apriltag_snapshot_at
            ):
                if capture_frame_from_pipeline(frame, IMAGE_SAVE_DIR):
                    snapshot_count += 1
                    capture_flash_until = time.time() + capture_flash_duration_s
                    log.info(
                        "AprilTag continuous snapshot saved "
                        f"({snapshot_count} total)"
                    )
                next_apriltag_snapshot_at = now + APRILTAG_CONTINUOUS_SNAPSHOT_INTERVAL_S

            # ── Enrollment ────────────────────────────────────────────────
            if (
                reid is not None
                and REID_ENABLED
                and
                marker_locked
                and not enrolled
                and (loop_count % ENROLL_SAMPLE_EVERY_N_FRAMES == 0)
            ):
                if reid.enroll_from_frame(frame):
                    enrolled = True
                    marker_locked = False
                    enroll_started_at = None
                    log.success("Primary person enrolled — ReID tracking active")

                    if capture_frame_from_pipeline(frame, IMAGE_SAVE_DIR):
                        snapshot_count += 1
                        capture_flash_until = time.time() + capture_flash_duration_s
                        log.info(f"Enrollment snapshot saved ({snapshot_count} total)")

            if marker_locked and not enrolled and enroll_started_at is not None:
                if (time.time() - enroll_started_at) > ENROLLMENT_TIMEOUT_S:
                    marker_locked = False
                    enroll_started_at = None
                    if reid is not None:
                        reid.reset_enrollment()
                    log.warning("Enrollment timed out — waiting for AprilTag reacquire")

            # ── Tracking ──────────────────────────────────────────────────
            # process_frame() runs YOLO internally each call.
            # Alternatively, pass person_bboxes from run_detection if you
            # want to reuse detections across the pipeline.
            if reid is not None and REID_ENABLED and enrolled:
                reid.process_frame(frame)

            if reid is not None:
                reid.draw_debug(frame, person_bboxes=None)
            draw_overlay(frame, fps, processing=enrolled)

            tag_label = "APRILTAG: DETECTED" if apriltag_visible else "APRILTAG: NOT DETECTED"
            tag_color = (0, 220, 0) if apriltag_visible else (0, 0, 220)
            tag_w = cv2.getTextSize(tag_label, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)[0][0]
            cv2.putText(
                frame,
                tag_label,
                (max(16, frame.shape[1] - tag_w - 16), 34),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.75,
                tag_color,
                2,
                cv2.LINE_AA,
            )

            if time.time() < capture_flash_until:
                h, w = frame.shape[:2]
                cv2.rectangle(frame, (2, 2), (w - 3, h - 3), (0, 255, 0), 2)

            cv2.imshow(window, frame)

            key = cv2.waitKey(1) & 0xFF
            if key == 27 or key == ord('q'):
                log.info("Camera session ended by user")
                break
            if key == ord('r'):
                if reid is not None:
                    reid.reset_enrollment()
                enrolled       = False
                marker_locked  = False
                enroll_started_at = None
                snapshot_count = 0
                apriltag_visible = False
                apriltag_has_been_seen = False
                next_apriltag_snapshot_at = None
                log.info("Enrollment reset — snapshots cleared")

    except KeyboardInterrupt:
        log.info("Interrupted")

    finally:
        video_pipeline.cleanup()
        cv2.destroyAllWindows()

    if reid is not None:
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