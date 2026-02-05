from __future__ import annotations

import argparse
import time
import threading
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty

from config.audio_settings import AUDIO_CHUNKS_DIR, PROCESSED_AUDIO_DIR
from src_audio.services.transcription_service.transcription_whispertrt import run_transcription
from src_audio.services.anonymization_service.transcript_anonymization import run_anonymization
from src_audio.services.medication_extraction_service.medication_extraction import run_medication_extraction
from src_audio.services.intervention_extraction_service.intervention_extraction import run_intervention_extraction


def put_latest(queue: Queue, item):
    """Drop old signal if queue is full, keep newest."""
    if queue.full():
        try:
            queue.get_nowait()
        except:
            pass
    queue.put(item)

def move_chunk_to_processed(chunk_path: Path) -> Path:
    """
    Move chunk file from AUDIO_CHUNKS_DIR into: PROCESSED_AUDIO_DIR/<chunk_stem>/<chunk_filename>
    Returns new path.
    """
    chunk_stem = chunk_path.stem  # recording_1323243_chunk_0
    dest_dir = PROCESSED_AUDIO_DIR / chunk_stem
    dest_dir.mkdir(parents=True, exist_ok=True)

    dest_path = dest_dir / chunk_path.name
    chunk_path.replace(dest_path)
    return dest_path

def process_audio_chunk() -> bool:
    """
    Processes whatever is currently in AUDIO_CHUNKS_DIR.
    """
    try:
        inbox = Path(AUDIO_CHUNKS_DIR)
        files = sorted([p for p in inbox.glob("*") if p.is_file()])
        if not files:
            print("[audio] No chunks to process")
            return False

        # Process the first chunk in folder (FIFO)
        latest = files[0]
        print(f"[audio] Found latest chunk in inbox: {latest.name}")

        # MOVE into processed folder (this also prevents reprocessing)
        chunk_path = move_chunk_to_processed(latest)
        print(f"[audio] Moved to processed dir: {chunk_path}")

        transcript_path = run_transcription(str(chunk_path))
        print(f"[audio] Transcription complete for {chunk_path.name}\n")
        run_anonymization(str(chunk_path), transcript_path)
        print(f"[audio] Anonymization complete for {chunk_path.name}\n")
        run_medication_extraction(str(chunk_path), transcript_path)
        print(f"[audio] Med Extraction complete for {chunk_path.name}\n")
        run_intervention_extraction(str(chunk_path), transcript_path)
        print(f"[audio] Intervention Extraction complete for {chunk_path.name}\n")

        print(f"[audio] {chunk_path.name} processed\n")
        return True

    except Exception as e:
        print(f"[audio ERROR] Processing failed: {e}")
        return False


def processing_worker(queue: Queue):
    """
    Worker thread: waits for signals, processes audio directory.
    """
    print("[AUDIO WORKER] Started")

    while True:
        try:
            job = queue.get(timeout=0.5)
        except Empty:
            continue

        if job == "STOP":
            queue.task_done()
            break

        try:
            process_audio_chunk()
        except Exception as e:
            print(f"[AUDIO WORKER ERROR] {e}")
        finally:
            queue.task_done()

    print("[AUDIO WORKER] Stopped")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    start_time = datetime.now()

    if args.dev:
        print("[AUDIO MODE] DEV")
        process_audio_chunk()
        return 0

    from src_audio.services.recording_audio_service.record_audio import record_one_chunk
    
    audio_queue = Queue(maxsize=2)

    worker = threading.Thread(
        target=processing_worker,
        args=(audio_queue,),
        daemon=True,
    )
    worker.start()

    stop_event = threading.Event()

    def wait_for_enter():
        input("\nPress ENTER to stop recording...\n")
        stop_event.set()
        print("[AUDIO MAIN] Stop requested")

    threading.Thread(target=wait_for_enter, daemon=True).start()

    print("[MAIN AUDIO PIPELINE] Started\n")

    try:
        while True:
            # - record ~3 min or until stop_event
            # - save chunk into AUDIO_CHUNKS_DIR
            # - return True if a chunk was written
            chunk_written = record_one_chunk(
                output_dir=AUDIO_CHUNKS_DIR,
                stop_event=stop_event,
            )

            if chunk_written:
                put_latest(audio_queue, {"time": time.time()})
                print("[MAIN AUDIO] Chunk queued")

            if stop_event.is_set():
                break

    except KeyboardInterrupt:
        print("\n[AUDIO MAIN] Interrupted")

    finally:
        print("[AUDIO MAIN] Recording stopped. Waiting for processing to finish...")
        
        # Wait until worker finishes all queued work
        audio_queue.join()

        print("[AUDIO MAIN] Processing finished. Shutting down worker...")
        audio_queue.put("STOP")
        worker.join()

    elapsed = datetime.now() - start_time
    total_seconds = int(elapsed.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    print(f"Complete AUDIO pipeline time: {hours}h {minutes}m {seconds}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
