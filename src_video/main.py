from __future__ import annotations

import os
import time
import threading
import shutil
import cv2
import argparse

from pathlib import Path
from typing import Dict, Any, Optional
from queue import Queue, Empty


from config.video_settings import (
    load_video_pipeline_settings,
    SNAPSHOT_INTERVAL,
    IMAGE_SAVE_DIR,
)

from src_video.services.detection_service.detect_body_parts import run_detection
from src_video.services.body_ranking.body_injury_ranking import body_ranking
from src_video.services.classification_service.infer_injuries_on_crops import (
    predict_injuries_on_detection_crops,
)
from src_video.services.deidentification_service.deidentify import run_deidentification
from src_video.services.detect_marker_service.detect_marker import detect_apriltags
from src_video.services.camera_capture_service.capture_img import draw_overlay
from src_video.services.camera_capture_service.gstreamer_video_pipeline import GStreamerVideoPipeline



def _as_posix(path: str) -> str:
    return str(path).replace("\\", "/")


def put_latest(queue: Queue, item):
    """
    Drop old item if queue is full, keep newest.
    """
    if queue.full():
        try:
            queue.get_nowait()
        except:
            pass

    queue.put(item)

def save_frame(frame) -> bool:
    timestamp = cv2.getTickCount()
    filename = os.path.join(IMAGE_SAVE_DIR, f"captured_img_{timestamp}.jpg")
    return cv2.imwrite(filename, frame)

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

        print("[video] Detection done")

    except Exception as e:
        print(f"[video object] Detection failed: {e}")
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

        print("[PIPELINE] Inference done")

    except Exception as e:
        print(f"[ERROR] Inference failed: {e}")



    if not body_ranking(settings):
        print("[WARN] Ranking failed")


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
            print(f"[video] De-identification complete: {deidentify_result['processed_count']} images processed.\n")
        else:
            print(f"[video] De-identification warning: {deidentify_result.get('note', deidentify_result.get('error'))}\n")

    except Exception as e:
        print(f"[video] De-identification failed: {e}")

    try:

        crops_root = Path(settings["CROPS_ROOT"])

        if crops_root.exists():
            shutil.rmtree(crops_root)
            crops_root.mkdir(parents=True, exist_ok=True)

    except Exception as e:
        print(f"[WARN] Cleanup failed: {e}")


    print("[PIPELINE] Image processed\n")

    return True


def processing_worker(queue: Queue, settings: Dict[str, Any]):

    BATCH_SIZE = 2
    BATCH_TIMEOUT = 2.0

    batch = []
    last_flush = time.time()

    print("[WORKER] Batching with Queue Started")

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

            print(f"[WORKER] Processing batch ({len(batch)})")

            try:
                process_single_image(settings)

            except Exception as e:
                print(f"[WORKER ERROR] {e}")

            batch.clear()
            last_flush = now


    print("[WORKER] Stopped")


def main(
    video_pipeline: Optional[GStreamerVideoPipeline] = None,
    video_ready: Optional[threading.Event] = None,
    video_failed: Optional[threading.Event] = None,
) -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    DEV_MODE = args.dev

    settings = load_video_pipeline_settings()

    if DEV_MODE:
        print("[MODE] DEV")
        process_single_image(settings)
        return 0

    # Use provided pipeline or initialize new one
    owns_pipeline = False
    if video_pipeline is None:
        video_pipeline = GStreamerVideoPipeline(flip_method=0)
        owns_pipeline = True
        
        if not video_pipeline.start():
            print("[VIDEO MAIN] ERROR: Failed to start video pipeline")
            if video_failed is not None:
                video_failed.set()
            return 1

    if video_ready is not None:
        video_ready.set()

    image_queue = Queue(maxsize=3)

    worker = threading.Thread(
        target=processing_worker,
        args=(image_queue, settings),
        daemon=True,
    )

    worker.start()

    window = "CSI Camera"

    cv2.namedWindow(window, cv2.WINDOW_AUTOSIZE)


    last_snap = 0
    frame_count = 0
    start_time = time.time()
    fps = 0.0

    processing = False

    print("[MAIN VIDEO PIPELINE] Started\n")


    try:
        while True:

            ok, frame = video_pipeline.read_frame()

            if not ok:
                print("[ERROR] Camera read failed")
                break


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

            keyCode = cv2.waitKey(10) & 0xFF
            # Stop the program on the ESC key or 'q'
            if keyCode == 27 or keyCode == ord('q'):
                break


            # Capture
            if detected and (now - last_snap) >= SNAPSHOT_INTERVAL:
                if save_frame(frame):

                    job = {
                        "time": now,
                        "id": int(now * 1000),
                    }

                    put_latest(image_queue, job)

                    processing = True
                    last_snap = now

                    print("[MAIN] Job queued")


            draw_overlay(frame, fps, processing)

            cv2.imshow(window, frame)
            key = cv2.waitKey(1) & 0xFF

            if key in (27, ord("q")):
                break


    except KeyboardInterrupt:

        print("\n[VIDEO MAIN] Interrupted")


    finally:

        print("[VIDEO MAIN] Shutting down")

        image_queue.put("STOP")
        worker.join(timeout=5)

        # Clean up pipeline if we created it
        if owns_pipeline and video_pipeline is not None:
            video_pipeline.stop()
            video_pipeline.cleanup()
        
        cv2.destroyAllWindows()


    return 0

if __name__ == "__main__":

    raise SystemExit(main())

