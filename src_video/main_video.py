from __future__ import annotations

import time
import threading, asyncio
import shutil
import cv2
import argparse
from pathlib import Path
from typing import Dict, Any, Optional
from queue import Queue, Empty
from ultralytics import YOLO
from boxmot import OcSort
import numpy as np

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
from src_video.services.person_reid_service import ResNet50ReIDExtractor

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


def main(video_pipeline: Optional[GStreamerVideoPipeline] = None) -> int:
    """
    Main video processing pipeline.
    
    Args:
        video_pipeline: Pre-initialized GStreamer video pipeline object from orchestrator.
    """

    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    
    settings = load_video_pipeline_settings()
    DEV_MODE = args.dev
    
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
    log.info("Running startup tasks...")
    run_jetson_startup_tasks()
    start_monitoring(interval=1.0, log_file=USAGE_FILE_PATH, show_stderr_line=True)

    image_queue = Queue(maxsize=3)

    worker = threading.Thread(
        target=processing_worker,
        args=(image_queue, settings),
        daemon=True,
    )
    tracker = OcSort(
        conf_thres=0.3,
        iou_thres=0.1,  # Lower for better re-identification
        max_age=300  # 10 seconds at 30fps (was 30 = 1 second)
    )

    # Initialize ReID extractor for person re-identification
    reid_extractor = ResNet50ReIDExtractor(
        model_path="src_video/resnet50_market1501_aicity156.onnx",
        device="cpu"
    )
    
    # Persistent person tracking
    track_to_persistent_id = {}  # Maps OcSort track_id -> persistent_person_id
    persistent_id_counter = 0  # Counter for assigning persistent IDs
    persistent_features = {}  # Maps persistent_person_id -> feature vector
    
    # Primary person tracking
    primary_person_id = None  # Persistent ID of person with marker
    primary_person_feature = None  # Stored feature for re-identification
    primary_person_assigned = False

    patient_id = None
    person_model = YOLO("yolov8n.pt")

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
            ok, frame = video_pipeline.read_frame()
            if not ok or frame is None:
                log.error("Camera read failed")
                break

            # Check if frame is valid
            if frame.size == 0:
                log.error("Empty frame received")
                continue
            results = person_model.predict(frame, classes=[0], verbose=False)

            detections = []

            for r in results:
                boxes = r.boxes
                for box in boxes:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    conf = box.conf[0].cpu().numpy()
                    detections.append([x1, y1, x2, y2, conf, 0])  
            
            detections_np = np.array(detections, dtype=np.float32) if detections else np.array([])
            tracks = tracker.update(detections_np, frame)
            
            # Extract ReID features for improved person re-identification
            if len(tracks) > 0 and len(detections) > 0:
                person_features = reid_extractor.extract_features(frame, detections)
                
                # Assign persistent IDs to tracks
                for i, track in enumerate(tracks):
                    track_id = int(track[4])
                    
                    # If track already has a persistent ID, skip
                    if track_id in track_to_persistent_id:
                        continue
                    
                    # Try to match with primary person using appearance features
                    if primary_person_feature is not None and i < len(person_features) and len(person_features[i]) > 0:
                        current_feature = person_features[i]
                        distance = 1 - np.dot(current_feature, primary_person_feature) / (
                            np.linalg.norm(current_feature) * np.linalg.norm(primary_person_feature) + 1e-5
                        )
                        
                        # If feature distance is low, it's the same person
                        if distance < 0.5:  # Cosine distance threshold
                            track_to_persistent_id[track_id] = primary_person_id
                            log.success(f"Re-identified primary person with new track {track_id} (distance: {distance:.3f})")
                            continue
                    
                    # Assign new persistent ID for new person
                    persistent_id_counter += 1
                    track_to_persistent_id[track_id] = persistent_id_counter
                    
                    if i < len(person_features) and len(person_features[i]) > 0:
                        persistent_features[persistent_id_counter] = person_features[i]
                    
                    log.info(f"New person {persistent_id_counter} detected (track {track_id})")


            # detections = np.array(detections)

            # if not ok or frame is None:
            #     log.error("Camera read failed")
            #     break

            # # Check if frame is valid
            # if frame.size == 0:
            #     log.error("Empty frame received")
            #     continue

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
            
            # Assign primary person when AprilTag detected
            if detected and not primary_person_assigned and tracks is not None and len(tracks) > 0:
                # Find which track has the AprilTag
                for i, tag in enumerate(detected if isinstance(detected, list) else [detected]):
                    if not hasattr(tag, 'center_x'):
                        continue
                    tag_x = float(tag.center_x)
                    tag_y = float(tag.center_y)
                    
                    for j, trk in enumerate(tracks):
                        x1, y1, x2, y2 = trk[0], trk[1], trk[2], trk[3]
                        track_id = int(trk[4])
                        
                        # Check if AprilTag is inside bounding box
                        if x1 <= tag_x <= x2 and y1 <= tag_y <= y2:
                            # Get or create persistent ID for this person
                            if track_id not in track_to_persistent_id:
                                persistent_id_counter += 1
                                track_to_persistent_id[track_id] = persistent_id_counter
                            
                            primary_person_id = track_to_persistent_id[track_id]
                            
                            # Store primary person's feature
                            if j < len(person_features) and len(person_features[j]) > 0:
                                primary_person_feature = person_features[j].copy()
                                persistent_features[primary_person_id] = primary_person_feature
                            
                            primary_person_assigned = True
                            patient_id = primary_person_id
                            
                            log.success(f"Primary person assigned: persistent ID {primary_person_id} (track {track_id})")
                            break
                    
                    if primary_person_assigned:
                        break

            # Capture when primary person is in view or AprilTag is detected
            should_capture = False

            if detected:
                should_capture = True

            elif primary_person_assigned and primary_person_id is not None:
                # Or capture when primary person is in view
                for trk in (tracks if tracks is not None else []):
                    track_id = int(trk[4])
                    if track_id in track_to_persistent_id:
                        if track_to_persistent_id[track_id] == primary_person_id:
                            should_capture = True
                            break

            # if primary_in_view and (now - last_snap) >= SNAPSHOT_INTERVAL:
            if should_capture and (now - last_snap) >= SNAPSHOT_INTERVAL:

                if capture_frame_from_pipeline(frame, IMAGE_SAVE_DIR):

                    job = {
                        "time": now,
                        "id": int(now * 1000),
                    }

                    put_latest(image_queue, job)
                    processing = True
                    last_snap = now

                    log.success("Job queued")


            draw_overlay(frame, fps, processing, tracks)
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

        video_pipeline.cleanup()
        
        cv2.destroyAllWindows()
        
        stop_monitoring()

    return 0

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    
    if args.dev:
        raise SystemExit(main())
    else:
        # Standalone camera mode
        video_pipeline = GStreamerVideoPipeline(flip_method=0)
        if not video_pipeline.start():
            log.error("Failed to initialize camera")
            raise SystemExit(1)
        
        try:
            raise SystemExit(main(video_pipeline))
        finally:
            video_pipeline.cleanup()
