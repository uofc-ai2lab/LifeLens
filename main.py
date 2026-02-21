import asyncio
import threading
import time

from src_audio.main_audio import main as audio_main
from src_video.main_video import main as video_main
from src_video.services.camera_capture_service.gstreamer_video_pipeline import GStreamerVideoPipeline
from config.jetson_startup import run_jetson_startup_tasks
from config.logger import root_logger as log
from config.audio_settings import USAGE_FILE_PATH
from config.resource_usage import start_monitoring, stop_monitoring
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


def run_audio_pipeline():
    """Runs audio async pipeline in its own event loop with GStreamer."""
    log.info("Starting AUDIO pipeline thread")
    try:
        asyncio.run(audio_main())
    except Exception as e:
        log.error(f"AUDIO pipeline error: {e}")
    log.info("AUDIO pipeline finished")


def run_video_pipeline(video_ready: threading.Event, video_failed: threading.Event):
    """Runs video async pipeline in its own event loop with GStreamer.

    Important: Open and release the CSI camera in the SAME thread to avoid
    Argus/GStreamer sessions lingering between runs.
    """
    log.info("Starting VIDEO pipeline thread")
    video_pipeline = None
    try:
        # Initialize GStreamer video pipeline
        video_pipeline = GStreamerVideoPipeline(flip_method=0)
        if not video_pipeline.start():
            video_failed.set()
            log.error("VIDEO pipeline failed to initialize camera")
            return
        
        video_ready.set()
        
        # Run video processing pipeline with initialized camera
        asyncio.run(video_main(video_pipeline))
        
    except Exception as e:
        video_failed.set()
        log.error(f"VIDEO pipeline error: {e}")
    finally:
        if video_pipeline:
            video_pipeline.cleanup()
    
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
    #run_jetson_startup_tasks()
    start_monitoring(interval=1.0, log_file=USAGE_FILE_PATH, show_stderr_line=True)
    
    start_time = time.time()

    # Synchronization events
    video_ready = threading.Event()
    video_failed = threading.Event()

    # Start video thread (must initialize camera first)
    video_thread = threading.Thread(
        target=run_video_pipeline,
        name="VideoThread_GStreamer",
        args=(video_ready, video_failed),
        daemon=False,
    )

    # Start audio thread (will wait for video ready)
    audio_thread = threading.Thread(
        target=run_audio_pipeline,
        name="AudioThread_GStreamer",
        daemon=False,
    )

    # Start video first; only start audio once camera is confirmed ready
    log.info("Initializing GStreamer pipelines...")
    video_thread.start()

    # Wait briefly for video initialization to complete
    init_timeout_s = 15.0
    deadline = time.time() + init_timeout_s
    while time.time() < deadline and not (video_ready.is_set() or video_failed.is_set()):
        time.sleep(0.1)

    if video_failed.is_set():
        log.error("VIDEO pipeline failed to initialize. Aborting audio pipeline.")
    else:
        log.success("VIDEO pipeline ready. Starting AUDIO pipeline...")
        audio_thread.start()
    
    # Wait for both threads to complete
    log.info("Both pipelines started, waiting for completion")
    video_thread.join()
    if audio_thread.is_alive():
        audio_thread.join()
    
    stop_monitoring()
    
    elapsed = time.time() - start_time
    log.success(f"All pipelines completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()
