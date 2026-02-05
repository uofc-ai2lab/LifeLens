import threading
import time
import subprocess
from pathlib import Path

from src_audio.main import main as audio_main
from src_video.main import main as video_main
from config.audio_settings import IS_JETSON

"""
Dual GStreamer Pipeline Architecture:
- Video thread: Hardware-accelerated CSI camera capture (NVIDIA Jetson)
- Audio thread: Multi-channel USB audio recording
- Both run concurrently in separate threads with synchronized startup
"""

def run_audio_pipeline(video_ready: threading.Event, video_failed: threading.Event):
    """Runs audio GStreamer pipeline in its own thread.
    
    Waits for video pipeline to be ready before starting audio to prevent
    resource contention.
    """
    print("[root] AUDIO thread: Waiting for video pipeline readiness signal")
    
    # Wait for video to initialize (with timeout)
    init_timeout = 15.0
    deadline = time.time() + init_timeout
    while time.time() < deadline and not (video_ready.is_set() or video_failed.is_set()):
        time.sleep(0.1)
    
    if video_failed.is_set():
        print("[root] AUDIO thread: Video pipeline failed, not starting audio")
        return
    
    if not video_ready.is_set():
        print("[root] AUDIO thread: Video initialization timeout, aborting audio")
        return
    
    print("[root] AUDIO thread: Video ready, starting audio pipeline")

    try:
        # Run audio processing services (chunked recording uses GStreamer internally)
        audio_main()
    except Exception as e:
        print(f"[root] AUDIO pipeline error: {e}")

    print("[root] AUDIO pipeline finished")


def run_video_pipeline(video_ready: threading.Event, video_failed: threading.Event):
    """Runs video GStreamer pipeline in its own thread.

    Important: Camera must be opened and closed in the SAME thread to avoid
    Argus/GStreamer sessions lingering between runs.
    """
    print("[root] VIDEO thread: Starting video pipeline")
    
    try:
        # Run video processing services (pipeline created and managed inside)
        video_main(video_ready=video_ready, video_failed=video_failed)
    except Exception as e:
        video_failed.set()
        print(f"[root] VIDEO pipeline error: {e}")

    print("[root] VIDEO pipeline finished")


def run_jetson_startup_tasks() -> None:
    """Run Jetson-specific startup tasks via script."""
    print("[root] Jetson detected. Running startup tasks...")

    script_path = Path(__file__).resolve().parent / "scripts" / "run_jetson_startup_tasks.sh"
    if not script_path.exists():
        print(f"[root] WARNING: Startup script not found: {script_path}")
        return

    try:
        subprocess.run([
            "bash",
            str(script_path),
        ], check=True, text=True)
    except subprocess.CalledProcessError as e:
        print(f"[root] WARNING: Startup script failed (exit {e.returncode}).")
    except Exception as e:
        print(f"[root] WARNING: Startup script failed: {e}")


def main():
    """
    Orchestrate dual GStreamer pipelines for concurrent audio and video processing.
    
    Execution flow:
    1. Start video thread - initializes camera via GStreamer
    2. Wait for video to be ready (with timeout)
    3. Start audio thread - records audio via GStreamer
    4. Both threads run concurrently, processing their respective streams
    5. Graceful shutdown on completion or interrupt
    """
    start_time = time.time()

    # Jetson-specific startup tasks
    if IS_JETSON:
        run_jetson_startup_tasks()
    
    # Synchronization events
    video_ready = threading.Event()
    video_failed = threading.Event()
    
    # Create threads for both pipelines
    video_thread = threading.Thread(
        target=run_video_pipeline,
        name="VideoThread",
        args=(video_ready, video_failed),
        daemon=False,
    )
    
    audio_thread = threading.Thread(
        target=run_audio_pipeline,
        name="AudioThread",
        args=(video_ready, video_failed),
        daemon=False,
    )

    # Start video thread first (must initialize first to avoid resource contention)
    print("[root] Starting dual GStreamer pipeline orchestration")
    print(f"[root] Video: GStreamer pipeline (hardware-accelerated CSI camera)")
    print(f"[root] Audio: GStreamer pipeline (6-channel USB audio)")
    video_thread.start()
    
    # Wait briefly for video initialization outcome
    # Audio thread checks video_ready/video_failed and doesn't start if video fails
    # This prevents resource contention and allows graceful failure handling
    init_timeout_s = 15.0
    deadline = time.time() + init_timeout_s
    while time.time() < deadline and not (video_ready.is_set() or video_failed.is_set()):
        time.sleep(0.1)
    
    # Start audio thread (will wait for video_ready signal in its thread)
    audio_thread.start()
    
    # Wait for both threads to complete
    print("[root] Both pipelines started, waiting for completion")
    video_thread.join()
    if audio_thread.is_alive():
        audio_thread.join()

    if IS_JETSON:
        pid_file = Path("/tmp/tegrastats.pid")
        if pid_file.exists():
            try:
                pid = int(pid_file.read_text().strip())
                subprocess.run(["sudo", "kill", "-TERM", str(pid)], check=False)
            except Exception:
                pass
    
    elapsed = time.time() - start_time
    print(f"\n[root] ✓ All pipelines completed successfully in {elapsed:.2f}s")
    print(f"[root] Dual GStreamer pipeline orchestration finished")


if __name__ == "__main__":
    main()
