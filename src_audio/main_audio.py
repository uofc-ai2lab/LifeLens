from __future__ import annotations

import argparse
import time
import threading
import os
from datetime import datetime
from pathlib import Path
from queue import Queue, Empty, Full
from typing import Optional
import signal
import psutil

from config.audio_settings import AUDIO_CHUNKS_DIR, PROCESSED_AUDIO_DIR
from src_audio.domain.constants import AUDIT_COLUMNS
from src_audio.services.transcription_service.transcription_whispertrt import run_transcription
from src_audio.services.anonymization_service.transcript_anonymization import run_anonymization
from src_audio.services.medication_extraction_service.medication_extraction import run_medication_extraction, MedicationStateTracker
from src_audio.services.medication_extraction_service.extractor import unload_medication_extractor
from src_audio.services.anonymization_service.anonymizer import unload_transcript_anonymizer
from src_audio.services.transcription_service.transcription_whispertrt import unload_whisper_model
from src_audio.services.intervention_extraction_service.intervention_extraction import run_intervention_extraction
from src_audio.services.intervention_extraction_service.intervention_extraction import unload_intervention_resources
from src_audio.services.recording_audio_service.gstreamer_audio_pipeline import record_one_chunk
from config.jetson_startup import run_jetson_startup_tasks
from config.audio_settings import USAGE_FILE_PATH
from config.resource_usage import start_monitoring, stop_monitoring
from config.memory_cleanup import cleanup_memory, clear_jtop_cache
from config.logger import audio_logger as log
from src_audio.utils.export_to_csv import export_to_csv
from data_transfer.sender_global import get

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


def _count_pending_chunk_files() -> int:
    """Return number of unprocessed audio files currently in AUDIO_CHUNKS_DIR."""
    inbox = Path(AUDIO_CHUNKS_DIR)
    if not inbox.exists():
        return 0
    return len([p for p in inbox.glob("*") if p.is_file()])


def _drain_audio_before_shutdown(audio_queue: Queue, timeout_s: float = 180.0) -> bool:
    """Wait for queued jobs and chunk files to be drained before stopping worker."""
    deadline = time.monotonic() + max(0.0, timeout_s)
    signaled_drain = False

    while True:
        pending_files = _count_pending_chunk_files()
        pending_jobs = audio_queue.unfinished_tasks

        if pending_files == 0 and pending_jobs == 0:
            return True

        # If files remain but queue is idle, trigger one more worker pass.
        if pending_files > 0 and pending_jobs == 0 and not signaled_drain:
            try:
                put_latest(audio_queue, {"time": time.time(), "reason": "shutdown-drain"})
                signaled_drain = True
                log.info("Shutdown drain signal queued")
            except Exception as e:
                log.warning(f"Unable to queue shutdown drain signal: {e}")

        if time.monotonic() >= deadline:
            log.warning(
                f"Timed out waiting for audio drain: pending_files={pending_files}, pending_jobs={pending_jobs}"
            )
            return False

        if pending_files == 0 or pending_jobs > 0:
            signaled_drain = False

        time.sleep(0.25)


def _wait_for_memory_headroom() -> None:
    """Apply bounded backpressure when RAM is high, but never skip queued chunks."""
    try:
        mem_limit = float(os.getenv("LIFELENS_AUDIO_PROCESS_MEM_MAX_PCT", "82"))
    except Exception:
        mem_limit = 82.0

    try:
        wait_interval_s = float(os.getenv("LIFELENS_AUDIO_MEM_WAIT_INTERVAL_S", "1.0"))
    except Exception:
        wait_interval_s = 1.0

    try:
        max_wait_s = float(os.getenv("LIFELENS_AUDIO_MEM_MAX_WAIT_S", "20.0"))
    except Exception:
        max_wait_s = 20.0

    waited_s = 0.0
    first_warning_emitted = False

    while True:
        mem_now = psutil.virtual_memory().percent
        if mem_now < mem_limit:
            return

        if not first_warning_emitted:
            log.warning(
                f"System memory high ({mem_now:.1f}% >= {mem_limit:.1f}%). "
                f"Applying backpressure for up to {max_wait_s:.1f}s before forced processing."
            )
            first_warning_emitted = True

        # Try to reclaim memory while waiting for headroom.
        _release_models_on_memory_pressure(stage="pre-processing")

        if waited_s >= max_wait_s:
            log.warning(
                f"Memory still high after {waited_s:.1f}s ({mem_now:.1f}%). "
                "Proceeding anyway so queued chunks are not skipped."
            )
            return

        time.sleep(wait_interval_s)
        waited_s += wait_interval_s


def _release_models_on_memory_pressure(stage: str = "") -> None:
    """Unload heavy NLP/ASR models when RAM usage crosses threshold."""
    try:
        unload_threshold = float(os.getenv("LIFELENS_AUDIO_UNLOAD_MEM_PCT", "80"))
    except Exception:
        unload_threshold = 80.0

    mem_now = psutil.virtual_memory().percent
    if mem_now < unload_threshold:
        return

    log.warning(
        f"Memory pressure after {stage or 'audio stage'}: {mem_now:.1f}% >= {unload_threshold:.1f}%. "
        "Unloading heavy model singletons."
    )

    if stage == "transcription":
        try:
            unload_whisper_model()
        except Exception as e:
            log.debug(f"Whisper unload skipped: {e}")
    elif stage == "anonymization":
        try:
            unload_transcript_anonymizer()
        except Exception as e:
            log.debug(f"Anonymizer unload skipped: {e}")
    elif stage == "medication":
        try:
            unload_medication_extractor()
        except Exception as e:
            log.debug(f"Medication extractor unload skipped: {e}")
    elif stage == "intervention":
        try:
            unload_intervention_resources()
        except Exception as e:
            log.debug(f"Intervention resource unload skipped: {e}")
    else:
        try:
            unload_whisper_model()
        except Exception:
            pass
        try:
            unload_transcript_anonymizer()
        except Exception:
            pass
        try:
            unload_medication_extractor()
        except Exception:
            pass
        try:
            unload_intervention_resources()
        except Exception:
            pass

    _clear_cuda_cache_if_available()


def _clear_cuda_cache_if_available() -> None:
    """Best-effort CUDA + RAM cleanup for audio models."""
    drop_os_cache = os.getenv("LIFELENS_DROP_OS_PAGE_CACHE", "0").strip().lower() in {
        "1", "true", "t", "yes", "y", "on"
    }
    cleanup_memory(
        clear_cuda_cache=True,
        clear_jtop=True,
        drop_linux_page_cache=drop_os_cache,
    )

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
        # Never skip queued chunks: wait briefly for headroom, then continue.
        _wait_for_memory_headroom()

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

        # get data sender
        data_sender = get()

        transcript_path = run_transcription(str(chunk_path))
        _release_models_on_memory_pressure(stage="transcription")
        clear_jtop_cache()
        if transcript_path is None:
            log.error("Transcription failed; skipping anonymization and extraction for this chunk.")
            _clear_cuda_cache_if_available()
            return False

        anonymization_path =run_anonymization(str(chunk_path), transcript_path)
        _release_models_on_memory_pressure(stage="anonymization")
        clear_jtop_cache()

        medication_path = run_medication_extraction(str(chunk_path), transcript_path, medication_tracker, audit_log)
        _release_models_on_memory_pressure(stage="medication")
        clear_jtop_cache()

        intervention_path = run_intervention_extraction(str(chunk_path), transcript_path)
        _release_models_on_memory_pressure(stage="intervention")
        clear_jtop_cache()

        # Send data to server
        if data_sender is not None:
            data_sender.send_batch(
            pipeline="audio", files=[
                (str(anonymization_path), "anonymization"),
                (str(medication_path), "medx"),
                (str(intervention_path), "intervention")     
            ])
        log.success(f"{chunk_path.name} processed")
        _clear_cuda_cache_if_available()
        return True

    except Exception as e:
        log.error(f"Processing failed: {e}")
        _clear_cuda_cache_if_available()
        return False


def processing_worker(
    queue: Queue,
    processing_enabled_event: Optional[threading.Event] = None,
):
    """
    Worker thread: waits for signals, then processes audio directory.

    If processing_enabled_event is provided and not set, worker will defer
    heavy processing until the event is set.
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
            if processing_enabled_event is not None:
                while not processing_enabled_event.is_set():
                    time.sleep(0.2)

            # Drain all pending chunks, not just one.
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
    processing_enabled_event: Optional[threading.Event] = None,
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
        args=(audio_queue, processing_enabled_event),
        daemon=False,
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
        if processing_enabled_event is not None:
            # Ensure shutdown cannot deadlock waiting on a disabled processing gate.
            processing_enabled_event.set()

        # Stop the resource monitor first so Ctrl+C does not keep printing status lines.
        stop_monitoring()
        log.info("Recording stopped, requesting processing worker shutdown...")

         # Export audit log entries produced (if any)
        if audit_log:
            export_to_csv(
                data=audit_log,
                audio_chunk_path=Path(PROCESSED_AUDIO_DIR),
                service="medX_audit",
                columns=AUDIT_COLUMNS,
                empty_ok=False,
            )

        try:
            shutdown_timeout_s = float(os.getenv("LIFELENS_AUDIO_SHUTDOWN_DRAIN_TIMEOUT_S", "210"))
        except Exception:
            shutdown_timeout_s = 180.0

        log.info("Waiting for pending audio chunks to finish processing...")
        drained_ok = _drain_audio_before_shutdown(audio_queue, timeout_s=shutdown_timeout_s)
        if drained_ok:
            log.info("Audio chunks drained; stopping processing worker")

        log.info("Shutting down processing worker...")
        try:
            audio_queue.put("STOP", timeout=2.0)
        except Full:
            log.warning("Processing queue still full during shutdown; retrying STOP with blocking put.")
            audio_queue.put("STOP")

        worker.join(timeout=shutdown_timeout_s + 2.0)
        if worker.is_alive():
            log.warning("Worker still running after shutdown timeout; forcing process exit path.")

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
    
