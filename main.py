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

def run_video_pipeline(video_capture):
    """Runs video async pipeline in its own event loop"""
    print("[root] Starting VIDEO pipeline thread")
    asyncio.run(video_main(video_capture))
    print("[root] VIDEO pipeline finished")

def main():
    start_time = time.time()
    
    video_capture = initialize_camera(flip_method=0)
    
    video_thread = threading.Thread(
        target=run_video_pipeline,
        name="VideoThread",
        args=(video_capture,),
        daemon=False,
    )

    audio_thread = threading.Thread(
        target=run_audio_pipeline,
        name="AudioThread",
        daemon=False,
    )

    # Start both pipelines
    video_thread.start()
    audio_thread.start()
    
    # Wait for both to finish
    video_thread.join()
    audio_thread.join()

    elapsed = time.time() - start_time
    print(f"[root] All pipelines completed in {elapsed:.2f}s")


if __name__ == "__main__":
    main()