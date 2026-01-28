import threading
import asyncio
import time

from src_audio.main import main as audio_main
from src_video.main import main as video_main
"""
In Python, we're not actually threading, we are switching b/w tasks and using the downtimes 
of the different tasks to execute the other tasks
"""

def run_audio_pipeline():
    """Runs audio async pipeline in its own event loop"""
    print("[root] Starting AUDIO pipeline thread")
    asyncio.run(audio_main())
    print("[root] AUDIO pipeline finished")

def run_video_pipeline():
    """Runs video async pipeline in its own event loop"""
    print("[root] Starting VIDEO pipeline thread")
    asyncio.run(video_main())
    print("[root] VIDEO pipeline finished")

def main():
    start_time = time.time()
    
    video_thread = threading.Thread(
        target=run_video_pipeline,
        name="VideoThread",
        daemon=False,
    )

    audio_thread = threading.Thread(
        target=run_audio_pipeline,
        name="AudioThread",
        daemon=False,
    )

    
    """
    daemon means running in the bg and as soon as the main thread and 
    the other important threads have finished, we can quit the script
    we don't need to wait for the sections within the functions to finish, it will just quit right away
    We have daemon as False because we don't want that.
    """

    # Start both pipelines
    audio_thread.start()
    video_thread.start()

    # Wait for both to finish
    audio_thread.join()
    video_thread.join()

    elapsed = time.time() - start_time
    print(f"[root] All pipelines completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()