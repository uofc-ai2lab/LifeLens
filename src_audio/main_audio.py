from __future__ import annotations

import argparse
import time
import threading
import os
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty
from typing import Optional
import signal
import psutil

from config.audio_settings import AUDIO_CHUNKS_DIR, PROCESSED_AUDIO_DIR
from src_audio.domain.constants import AUDIT_COLUMNS
from src_audio.services.transcription_service.transcription_whispertrt import run_transcription
from src_audio.services.anonymization_service.transcript_anonymization import run_anonymization
from src_audio.services.medication_extraction_service.medication_extraction import run_medication_extraction, MedicationStateTracker
from src_audio.services.intervention_extraction_service.intervention_extraction import run_intervention_extraction
from src_audio.services.recording_audio_service.gstreamer_audio_pipeline import record_one_chunk
from config.jetson_startup import run_jetson_startup_tasks
from config.audio_settings import USAGE_FILE_PATH
from config.resource_usage import start_monitoring, stop_monitoring
from config.memory_cleanup import cleanup_memory, clear_jtop_cache
from config.logger import audio_logger as log
from src_audio.utils.export_to_csv import export_to_csv

def put_latest(queue: Queue, item):
    """Drop old signal if queue is full, keep newest."""
    if queue.full():
        try:
            queue.get_nowait()
        except:
            pass
    queue.put(item)

# creating global tracker and audit log object instance to maintain state across chunks (for medication extraction)
medication_tracker = MedicationStateTracker()
audit_log = []


def _clear_cuda_cache_if_available() -> None:
    """Best-effort CUDA + heap cleanup for audio models."""
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        # Best-effort only; ignore CUDA issues here.
        pass

    # Also trim host heap where possible.
    cleanup_memory()
    clear_jtop_cache()

def move_chunk_to_processed(chunk_path: Path) -> Path:
    """
    Move chunk file from AUDIO_CHUNKS_DIR into: PROCESSED_AUDIO_DIR/<chunk_stem>/<chunk_filename>
    Returns new path.
    """
    chunk_stem = chunk_path.stem  # recording_YYYYMMDD_HHMMSS
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
        # Prevent heavy NLP/transcription work when system memory is already high.
        # This avoids Argus/NvMap OOM spirals under sustained load.
        mem_limit = float(os.getenv("LIFELENS_AUDIO_PROCESS_MEM_MAX_PCT", "82"))
        mem_now = psutil.virtual_memory().percent
        if mem_now >= mem_limit:
            log.warning(
                f"Skipping chunk processing: system memory {mem_now:.1f}% "
                f">= limit {mem_limit:.1f}%"
            )
            return False

        inbox = Path(AUDIO_CHUNKS_DIR)
        files = sorted([p for p in inbox.glob("*") if p.is_file()])
        if not files:
            log.info("No chunks to process")
            return False

        # Process the first chunk in folder (FIFO)
        latest = files[0]
        log.info(f"Found chunk: {latest.name}")

        # MOVE into processed folder (this also prevents reprocessing)
        chunk_path = move_chunk_to_processed(latest)
        log.info(f"Moved to processed dir: {chunk_path}")

        transcript_path = run_transcription(str(chunk_path))
        clear_jtop_cache()
        if transcript_path is None:
            log.error("Transcription failed; skipping anonymization and extraction for this chunk.")
            _clear_cuda_cache_if_available()
            return False

        run_anonymization(str(chunk_path), transcript_path)
        clear_jtop_cache()

        run_medication_extraction(str(chunk_path), transcript_path, medication_tracker, audit_log)
        clear_jtop_cache()

        run_intervention_extraction(str(chunk_path), transcript_path)
        clear_jtop_cache()
        log.success(f"{chunk_path.name} processed")
        _clear_cuda_cache_if_available()
        return True

    except Exception as e:
        log.error(f"Processing failed: {e}")
        _clear_cuda_cache_if_available()
        return False


def processing_worker(queue: Queue):
    """
    Worker thread: waits for signals, processes audio directory.
    """
    log.info("Processing worker started")

    while True:
        try:
            job = queue.get(timeout=0.5)
        except Empty:
            continue

        if job == "STOP":
            queue.task_done()
            break

        try:
            # Drain all pending chunks, not just one — prevents accumulation
            # when the memory gate previously caused skips.
            while process_audio_chunk():
                pass
        except Exception as e:
            log.error(f"Worker error: {e}")
        finally:
            queue.task_done()

    log.info("Processing worker stopped")


def main(
    register_signal_handlers: bool = True,
    external_stop_event: Optional[threading.Event] = None,
    enable_enter: bool = True,
) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()

    start_time = datetime.now()

    if args.dev:
        log.header("DEV Mode")
        start_monitoring(interval=1.0, log_file=USAGE_FILE_PATH, show_stderr_line=True)
        process_audio_chunk()
        stop_monitoring()
        # Export audit log entries produced (if any)
        if audit_log:
            export_to_csv(
                data=audit_log,
                audio_chunk_path=Path(PROCESSED_AUDIO_DIR),
                service="medX_audit",
                columns=AUDIT_COLUMNS,
                empty_ok=False,
            )
        return 0
    
    log.header("Audio Pipeline Starting")
    audio_queue = Queue(maxsize=2)

    worker = threading.Thread(
        target=processing_worker,
        args=(audio_queue,),
        daemon=True,
    )
    worker.start()

    stop_event = external_stop_event if external_stop_event is not None else threading.Event()
    
    if enable_enter:
        def wait_for_enter():
            input("\nPress ENTER to stop recording...\n")
            stop_event.set()
            log.info("Stop requested")

        threading.Thread(target=wait_for_enter, daemon=True).start()

    def _handle_stop(signum, frame):
        log.info(f"Stop signal received ({signum})")
        stop_event.set()

    if register_signal_handlers and threading.current_thread() is threading.main_thread():
        signal.signal(signal.SIGTERM, _handle_stop)
        signal.signal(signal.SIGINT, _handle_stop)
    elif register_signal_handlers:
        log.info("Signal handler registration skipped (non-main thread)")

    log.header("Audio Pipeline Started")

    try:
        while True:
            if stop_event.is_set():
                break

            chunk_written = record_one_chunk(
                output_dir=AUDIO_CHUNKS_DIR,
                stop_event=stop_event,
            )

            if chunk_written:
                put_latest(audio_queue, {"time": time.time()})
                log.success("Chunk queued")

            if stop_event.is_set():
                break

    except KeyboardInterrupt:
        log.info("Interrupted")
        stop_event.set()

    finally:
        log.info("Recording stopped, waiting for processing...")
        audio_queue.join()

         # Export audit log entries produced (if any)
        if audit_log:
            export_to_csv(
                data=audit_log,
                audio_chunk_path=Path(PROCESSED_AUDIO_DIR),
                service="medX_audit",
                columns=AUDIT_COLUMNS,
                empty_ok=False,
            )

        log.info("Processing finished, shutting down worker...")
        audio_queue.put("STOP")
        worker.join()

    elapsed = datetime.now() - start_time
    total_seconds = int(elapsed.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    log.success(f"Complete pipeline time: {hours}h {minutes}m {seconds}s")
    return 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    
    if args.dev:
        raise SystemExit(main())
    else:
        # Standalone mic mode
        log.info("Running startup tasks...")
        run_jetson_startup_tasks()
        start_monitoring(interval=1.0, log_file=USAGE_FILE_PATH, show_stderr_line=True)
        try:
            raise SystemExit(main())
        finally:
            stop_monitoring()
    
