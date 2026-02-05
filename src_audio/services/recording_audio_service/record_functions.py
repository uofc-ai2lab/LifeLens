import os
import time
import shutil
import signal
import asyncio
from config.audio_settings import AUDIO_DIR, create_parent_audio_dir
from src_audio.domain.constants import MAX_RECORD_SECONDS, RECORDING_DIR, SIGNAL_FILE
from src_audio.services.recording_audio_service.gstreamer_audio_pipeline import GStreamerAudioPipeline


def start_recording() -> tuple:
    """
    Start recording using GStreamer pipeline.
    
    Returns:
        tuple: (GStreamerAudioPipeline, str) - pipeline object and output file path
    """
    os.makedirs(RECORDING_DIR, exist_ok=True)
    if os.path.exists(SIGNAL_FILE):
        os.remove(SIGNAL_FILE)

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(RECORDING_DIR, f"recording_{timestamp}.wav")

    print(f"[Recording] Starting audio recording to {output_file}")
    
    pipeline = GStreamerAudioPipeline(output_file)
    if not pipeline.start():
        raise RuntimeError("Failed to start GStreamer audio pipeline")

    return pipeline, output_file


def wait_for_recording(output_file: str):
    """
    Wait for recording completion and copy files to target directory.
    
    Args:
        output_file (str): Specific file to copy
    """
    # Wait for signal file
    max_wait_time = 60  # Wait up to 60 seconds
    elapsed = 0
    while not os.path.exists(SIGNAL_FILE) and elapsed < max_wait_time:
        print("[Recording] Waiting for recording to finish...")
        time.sleep(5)
        elapsed += 5

    print("[Recording] Recording complete. Processing the file...")

    # Give the file system a moment to finalize the file
    time.sleep(1)

    # Handle specific file
    if output_file is not None:
        if os.path.isfile(output_file):
            print(f"[Recording] Copying recorded file: {output_file}")
            copy_to_target(output_file)
        else:
            print(f"[Recording] WARNING: Output file does not exist: {output_file}")
        return

    # Fallback: Get all WAV files in recording directory
    wav_files = [
        os.path.join(RECORDING_DIR, f)
        for f in os.listdir(RECORDING_DIR)
        if f.lower().endswith(".wav") and os.path.isfile(os.path.join(RECORDING_DIR, f))
    ]
    
    if not wav_files:
        print("[Recording] No .wav files found in recording directory.")
        return

    for path in wav_files:
        print(f"[Recording] Copying recording: {path}")
        copy_to_target(path)


def copy_to_target(recording_path: str):
    """
    Copy recorded file to target directory for next service.
    
    Args:
        recording_path (str): Path to recording file
    """
    os.makedirs(AUDIO_DIR, exist_ok=True)

    target_path = os.path.join(AUDIO_DIR, os.path.basename(recording_path))
    print(f"[Recording] Copying {recording_path} to {target_path}...")

    try:
        shutil.copy2(recording_path, target_path)
        print(f"[Recording] File successfully copied to {target_path}")
        create_parent_audio_dir(target_path)
    except Exception as e:
        print(f"[Recording] ERROR copying file: {e}")


async def run_recording_service():
    """
    Start recording with manual stop (ENTER) or 5-minute timeout.
    
    Uses GStreamer pipeline for audio capture.
    Recording is automatically saved and pipeline continues.
    """
    pipeline = None
    output_file = None
    
    try:
        pipeline, output_file = start_recording()

        print("\n[Recording] Recording started.")
        print("→ Press ENTER to stop recording and continue pipeline")
        print(f"→ Auto-stop after {MAX_RECORD_SECONDS} seconds")

        loop = asyncio.get_running_loop()
        enter_future = loop.run_in_executor(None, input)
        timeout_future = asyncio.sleep(MAX_RECORD_SECONDS)

        done, pending = await asyncio.wait(
            [enter_future, timeout_future],
            return_when=asyncio.FIRST_COMPLETED
        )

        # Cancel pending tasks
        for task in pending:
            task.cancel()

        print("→ Manual stop requested." if enter_future in done else "→ Auto-stop reached.")

        print("[Recording] Stopping recording...")
        pipeline.stop()
        
        # Write signal file to indicate recording completion
        os.makedirs(RECORDING_DIR, exist_ok=True)
        with open(SIGNAL_FILE, "w", encoding="utf-8") as flag:
            flag.write(f"done:{time.time()}")

        wait_for_recording(output_file)
        
    except Exception as e:
        print(f"[Recording] ERROR: {e}")
    finally:
        if pipeline:
            pipeline.cleanup()
