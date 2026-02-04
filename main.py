import threading
import asyncio
import time

from src_audio.main import main as audio_main
from src_video.main import main as video_main, initialize_camera
"""
In Python, we're not actually threading, we are switching b/w tasks and using the downtimes 
of the different tasks to execute the other tasks
"""

def run_audio_pipeline():
    """Runs audio async pipeline in its own event loop"""
    print("[root] Starting AUDIO pipeline thread")
    asyncio.run(audio_main())
    print("[root] AUDIO pipeline finished")

def run_video_pipeline(video_ready: threading.Event, video_failed: threading.Event):
    """Runs video async pipeline in its own event loop.

    Important: Plz open and release the CSI camera in the SAME thread to avoid
    Argus/GStreamer sessions lingering between runs.
    """
    print("[root] Starting VIDEO pipeline thread")
    video_capture = None
    try:
        video_capture = initialize_camera(flip_method=0)
        video_ready.set()
        asyncio.run(video_main(video_capture))
    except Exception as e:
        video_failed.set()
        print(f"[root] VIDEO pipeline failed to start: {e}")
    finally:
        if video_capture is not None:
            try:
                video_capture.release()
            except Exception:
                pass
    print("[root] VIDEO pipeline finished")

def main():
    start_time = time.time()

    video_ready = threading.Event()
    video_failed = threading.Event()

    video_thread = threading.Thread(
        target=run_video_pipeline,
        name="VideoThread",
        args=(video_ready, video_failed),
        daemon=False,
    )

    audio_thread = threading.Thread(
        target=run_audio_pipeline,
        name="AudioThread",
        daemon=False,
    )

    # Start video first; only start audio once camera is confirmed ready.
    video_thread.start()

    # Wait briefly for video init outcome so we don't start recording audio if video can't run.
    init_timeout_s = 10.0
    deadline = time.time() + init_timeout_s
    while time.time() < deadline and not (video_ready.is_set() or video_failed.is_set()):
        time.sleep(0.05)

    audio_thread.start()
    
    # Wait for both to finish
    video_thread.join()
    audio_thread.join()

    elapsed = time.time() - start_time
    print(f"[root] All pipelines completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()