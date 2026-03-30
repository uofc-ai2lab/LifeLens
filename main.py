import sys
import threading
import time
import os
import traceback

from src_audio.main_audio import main as audio_main
from src_video.main_video import main as video_main
from src_video.services.camera_capture_service.gstreamer_video_pipeline import GStreamerVideoPipeline
from config.jetson_startup import run_jetson_startup_tasks
from config.logger import root_logger as log
from config.audio_settings import USAGE_FILE_PATH
from config.resource_usage import start_monitoring, stop_monitoring
from data_transfer.sender_global import init
from data_transfer.domain.constants import DEVICE_ID
"""
Dual GStreamer Pipeline Architecture
=====================================

Two separate GStreamer pipelines run concurrently in a single process using Python threading:

1. AUDIO PIPELINE (async loop in dedicated thread)
   - Uses GStreamer alsasrc + wavenc
   - Captures multi-channel USB audio
   - Runs full audio processing pipeline

2. VIDEO PIPELINE (async loop in dedicated thread)
   - Uses GStreamer nvarguscamerasrc (NVIDIA Jetson hardware acceleration)
   - Captures CSI camera frames
   - Runs full video processing pipeline

Threading Model:
- Both pipelines run asynchronously in their own event loops
- Each pipeline has its own GStreamer pipeline instance
- Coordinated startup via threading.Event for camera readiness
- Proper cleanup on shutdown or error

GStreamer handles the heavy lifting:
- Hardware acceleration for video (nvarguscamerasrc, nvvidconv)
- Audio format conversion and encoding
- Multi-threading within GStreamer plugin chains
- Automatic buffer management and synchronization
"""


def _start_stdin_stop_listener(stop_event: threading.Event):
    """Listens for STOP commands on stdin and sets a shared stop event."""

    def _listen():
        try:
            for raw in sys.stdin:
                cmd = raw.strip().upper()
                if cmd in {"STOP", "EXIT", "Q", "QUIT"}:
                    log.info(f"Stop command received on stdin: {cmd}")
                    stop_event.set()
                    break
        except Exception as e:
            log.warning(f"stdin stop listener ended: {e}")

    threading.Thread(target=_listen, name="StdinStopListener", daemon=True).start()


def run_audio_pipeline(
    stop_event: threading.Event,
    processing_enabled_event: threading.Event | None = None,
):
    """Runs audio async pipeline in its own event loop with GStreamer."""
    gate_state = (
        "enabled"
        if (processing_enabled_event is not None and processing_enabled_event.is_set())
        else "paused"
    )
    log.info(f"Starting AUDIO pipeline thread (processing gate: {gate_state})")
    try:
        audio_main(
            register_signal_handlers=False,
            external_stop_event=stop_event,
            enable_enter=False,
            processing_enabled_event=processing_enabled_event,
        )
    except Exception as e:
        log.error(f"AUDIO pipeline error: {e}")
        log.error(traceback.format_exc())
    log.info("AUDIO pipeline finished")


def run_video_pipeline(
    video_ready: threading.Event,
    video_failed: threading.Event,
    stop_event: threading.Event,
):
    """Runs video async pipeline in its own event loop with GStreamer.

    Important: Open and release the CSI camera in the SAME thread to avoid
    Argus/GStreamer sessions lingering between runs.
    """
    log.info("Starting VIDEO pipeline thread")
    video_pipeline = None
    try:
        # Initialize GStreamer video pipeline
        log.info("VIDEO thread: creating GStreamerVideoPipeline()")
        video_pipeline = GStreamerVideoPipeline()
        log.info("VIDEO thread: calling video_pipeline.start()")
        if not video_pipeline.start():
            video_failed.set()
            log.error("VIDEO pipeline failed to initialize camera")
            return

        log.success("VIDEO thread: camera initialized successfully")
        video_ready.set()
        log.info("VIDEO thread: readiness event set; entering video_main()")
        
        # video_main is synchronous; running it directly avoids asyncio.run()
        # raising "a coroutine was expected" on shutdown.
        video_main(video_pipeline, external_stop_event=stop_event)
        log.info("VIDEO thread: video_main() returned")
        
    except Exception as e:
        video_failed.set()
        log.error(f"VIDEO pipeline error: {e}")
        log.error(traceback.format_exc())
    finally:
        log.info("VIDEO thread: cleanup begin")
        if video_pipeline:
            video_pipeline.cleanup()
        log.info("VIDEO thread: cleanup complete")
    
    log.info("VIDEO pipeline finished")

def main():
    """
    Main orchestrator for dual GStreamer pipeline execution.
    
    Manages:
    - Jetson startup tasks (camera reset)
    - Startup synchronization (video before audio)
    - Proper threading with daemon=False for graceful shutdown
    - Error handling and cleanup
    - Timing statistics
    """

    log.header("LifeLens Dual Pipeline System Starting")
    log.info("Running startup tasks...")
    run_jetson_startup_tasks()
    start_monitoring(interval=1.0, log_file=USAGE_FILE_PATH, show_stderr_line=True)

    disable_audio = os.getenv("LIFELENS_DISABLE_AUDIO", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    audio_start_after_video = os.getenv("LIFELENS_AUDIO_START_AFTER_VIDEO", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    audio_process_during_video = os.getenv("LIFELENS_AUDIO_PROCESS_DURING_VIDEO", "0").strip().lower() in {
        "1", "true", "yes", "on"
    }
    try:
        startup_settle_s = float(os.getenv("LIFELENS_AUDIO_START_DELAY", "2.0"))
    except ValueError:
        startup_settle_s = 2.0

    log.info(
        "Startup config: "
        f"disable_audio={disable_audio}, "
        f"audio_start_after_video={audio_start_after_video}, "
        f"audio_process_during_video={audio_process_during_video}, "
        f"audio_start_delay_s={startup_settle_s:.1f}"
    )
    
    start_time = time.time()

    # Instantiate MQTT singleton object for data transfer and start connection.
    log.info(f"Initializing MQTT data transfer with device ID: {DEVICE_ID}")
    data_sender = init(DEVICE_ID)
    data_sender.connect()
    data_sender.start_session()
    data_sender.start_heartbeat()  

    # Synchronization events
    video_ready = threading.Event()
    video_failed = threading.Event()
    stop_event = threading.Event()
    audio_processing_enabled = threading.Event()

    if audio_process_during_video:
        audio_processing_enabled.set()

    _start_stdin_stop_listener(stop_event)
    log.info("stdin stop listener started")

    # Start video thread (must initialize camera first)
    video_thread = threading.Thread(
        target=run_video_pipeline,
        name="VideoThread_GStreamer",
        args=(video_ready, video_failed, stop_event),
        daemon=False,
    )

    audio_thread = None
    if not disable_audio:
        # Start audio thread (will wait for video ready)
        audio_thread = threading.Thread(
            target=run_audio_pipeline,
            name="AudioThread_GStreamer",
            args=(stop_event, audio_processing_enabled),
            daemon=False,
        )

    # Start video first; only start audio once camera is confirmed ready
    log.info("Initializing GStreamer pipelines...")
    video_thread.start()
    log.info("VIDEO thread started; waiting for readiness event")

    # Wait briefly for video initialization to complete
    init_timeout_s = 15.0
    deadline = time.time() + init_timeout_s
    last_wait_log = 0.0
    while time.time() < deadline and not (video_ready.is_set() or video_failed.is_set()):
        now = time.time()
        if now - last_wait_log >= 2.0:
            remaining = max(0.0, deadline - now)
            log.info(
                "Waiting for VIDEO readiness... "
                f"remaining={remaining:.1f}s, "
                f"video_ready={video_ready.is_set()}, video_failed={video_failed.is_set()}"
            )
            last_wait_log = now
        time.sleep(0.1)

    log.info(
        "VIDEO readiness wait finished: "
        f"video_ready={video_ready.is_set()}, video_failed={video_failed.is_set()}, "
        f"thread_alive={video_thread.is_alive()}"
    )

    if video_failed.is_set():
        log.error("VIDEO pipeline failed to initialize. Aborting audio pipeline.")
    elif not video_ready.is_set():
        log.error("VIDEO pipeline readiness timed out without explicit failure event")
    else:
        if disable_audio:
            log.warning("AUDIO pipeline disabled via LIFELENS_DISABLE_AUDIO")
        elif not audio_start_after_video:
            log.success("VIDEO pipeline ready. Starting AUDIO pipeline...")
            log.info(f"Settling {startup_settle_s:.1f}s before AUDIO startup to reduce memory pressure")
            time.sleep(startup_settle_s)
            audio_thread.start()
            if not audio_process_during_video:
                log.warning(
                    "AUDIO processing paused while VIDEO is active "
                    "(recording continues). Set LIFELENS_AUDIO_PROCESS_DURING_VIDEO=1 to run full audio processing concurrently."
                )
        else:
            log.warning(
                "AUDIO deferred until VIDEO completes "
                "(LIFELENS_AUDIO_START_AFTER_VIDEO=1)"
            )
    
    # Wait for both threads to complete
    log.info("Both pipelines started, waiting for completion")
    video_thread.join()
    log.info("VIDEO thread joined")

    # Enable/defer heavy audio processing once video is done to protect camera memory.
    audio_processing_enabled.set()

    if audio_thread is not None:
        if audio_start_after_video and not disable_audio and not audio_thread.is_alive():
            log.info("VIDEO finished. Starting deferred AUDIO pipeline now...")
            audio_thread.start()

        if audio_thread.is_alive():
            log.info("Waiting for AUDIO thread to finish")
            audio_thread.join()
            log.info("AUDIO thread joined")
    
    # Properly end MQTT session and disconnect
    data_sender.end_session()
    data_sender.disconnect()

    stop_monitoring()
    
    elapsed = time.time() - start_time
    log.success(f"All pipelines completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
