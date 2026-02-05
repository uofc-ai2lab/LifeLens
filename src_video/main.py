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
from src_video.services.camera_capture_service.capture_img import (
    gstreamer_pipeline,
    capture_images,
    initialize_camera,
    draw_overlay
)

from src_video.services.person_reid_service.session import ReIDSession
from src_video.services.person_reid_service.tracker import PersonReIDEngine



def _as_posix(path: str) -> str:
    return str(path).replace("\\", "/")


def put_latest(queue: Queue, item):
    """
    Drop old item if queue is full, keep newest.
    """
    if queue.full():
        try:
            queue.get_nowait()
            queue.task_done()
        except:
            pass

    queue.put(item)



def process_single_image(
    settings: Dict[str, Any],
    image_paths: list[str],
    reid_session: ReIDSession | None = None,
) -> bool:

    detection_output_dir = Path(settings["DETECTION_OUTPUT"])
    crops_root = Path(settings["CROPS_ROOT"])  # .../DetectionOutput/crops

    for image_path in image_paths:
        image_path_p = Path(image_path)
        if not image_path_p.exists():
            print(f"[video] Skipping missing image: {image_path}")
            continue

        image_stem = image_path_p.stem

        try:
            run_detection(
                model=settings["DETECTION_MODEL"],
                source=_as_posix(str(image_path_p)),
                output=_as_posix(settings["DETECTION_OUTPUT"]),
                classes=settings["CLASSES"],
                margin=float(settings["MARGIN"]),
                min_area=int(settings["MIN_AREA"]),
                device=settings.get("DEVICE"),
                add_head=bool(settings["ADD_HEAD"]),
                debug=bool(settings["DEBUG"]),
                alpha_png=bool(settings["ALPHA_PNG"]),
                max_images=1,
                classification_export_dir=None,
            )
            print("[video] Detection done")
        except Exception as e:
            print(f"[video object] Detection failed: {e}")
            return False

        try:
            per_image_crops_root = crops_root / image_stem
            infer_summary = predict_injuries_on_detection_crops(
                crops_root=_as_posix(str(per_image_crops_root)),
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

        if reid_session is not None:
            try:
                reid_session.process_image(image_stem=image_stem)
            except Exception as e:
                print(f"[reid] ReID failed: {e}")

    try:
        deidentify_result = run_deidentification(
            input_dir=_as_posix(IMAGE_SAVE_DIR),
            output_dir=_as_posix(str(detection_output_dir / "deidentified")),
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

    print("[PIPELINE] Batch processed\n")
    return True


def processing_worker(queue: Queue, settings: Dict[str, Any]):

    BATCH_SIZE = 2
    BATCH_TIMEOUT = 2.0

    batch = []
    last_flush = time.time()

    detection_output_dir = Path(settings["DETECTION_OUTPUT"])
    reid_dir = detection_output_dir / "person_reid"
    reid_engine = PersonReIDEngine(storage_dir=str(reid_dir))
    reid_session = ReIDSession(
        reid_engine=reid_engine,
        detection_output_dir=detection_output_dir,
        report_path=reid_dir / "reid_report.json",
        saved_images_dir=Path(IMAGE_SAVE_DIR),
    )

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
                image_paths = [j.get("image_path") for j in batch if isinstance(j, dict) and j.get("image_path")]
                process_single_image(settings, image_paths=image_paths, reid_session=reid_session)

            except Exception as e:
                print(f"[WORKER ERROR] {e}")

            batch.clear()
            last_flush = now


    print("[WORKER] Stopped")


def main() -> int:

    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    DEV_MODE = args.dev

    settings = load_video_pipeline_settings()

    if DEV_MODE:
        print("[MODE] DEV")
        try:
            saved = sorted(Path(IMAGE_SAVE_DIR).glob("*.jpg"), key=lambda p: p.stat().st_mtime)
            if not saved:
                print(f"[DEV] No images found in {IMAGE_SAVE_DIR}")
                return 0

            detection_output_dir = Path(settings["DETECTION_OUTPUT"])
            reid_dir = detection_output_dir / "person_reid"
            reid_engine = PersonReIDEngine(storage_dir=str(reid_dir))
            reid_session = ReIDSession(
                reid_engine=reid_engine,
                detection_output_dir=detection_output_dir,
                report_path=reid_dir / "reid_report.json",
                saved_images_dir=Path(IMAGE_SAVE_DIR),
            )
            process_single_image(settings, image_paths=[str(saved[-1])], reid_session=reid_session)
        except Exception as e:
            print(f"[DEV] Failed: {e}")
        return 0

    video_capture = initialize_camera()

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

            ok, frame = video_capture.read()

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

                saved_path = capture_images(video_capture)

                if saved_path:

                    job = {
                        "time": now,
                        "id": int(now * 1000),
                        "image_path": str(saved_path),
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

        video_capture.release()
        cv2.destroyAllWindows()


    return 0

if __name__ == "__main__":

    raise SystemExit(main())
