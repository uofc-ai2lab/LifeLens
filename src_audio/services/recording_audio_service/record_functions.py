import asyncio
import subprocess
import os
import time
import shutil
import signal
from typing import Optional
from config.audio_settings import AUDIO_DIR, create_parent_audio_dir
from src_audio.domain.constants import MAX_RECORD_SECONDS, RECORDING_DIR, SIGNAL_FILE, ARECORD_DEVICE, CHUNK_SECONDS

def _make_output_path() -> str:
    os.makedirs(RECORDING_DIR, exist_ok=True) # Ensure the recording directory exists.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(RECORDING_DIR, f"recording_{timestamp}.wav")

def _copy_to_target(recording_path: str) -> str:
    """
    Copy recorded file to target directory for next service.
    
    Args:
        recording_path (str): Path to recording file
    """
    os.makedirs(AUDIO_DIR, exist_ok=True)
    target_path = os.path.join(AUDIO_DIR, os.path.basename(recording_path))
    print(f"Copying {recording_path} to {target_path}...")
    try:
        shutil.copy2(recording_path, target_path)
        print(f"File successfully copied to {target_path}")
        create_parent_audio_dir(target_path)
    except Exception as e:
        print(f"Error copying file: {e}")
    return target_path

async def record_one_chunk(stop_event: asyncio.Event) -> Optional[str]:
    """
    Records up to `chunk_seconds` seconds OR until stop_event is set.
    Returns the copied target path, or None if nothing recorded.
    """
    
    if os.path.exists(SIGNAL_FILE): # Remove stale signal file to prevent false positives.
        os.remove(SIGNAL_FILE)

    output_file = _make_output_path()
    print(f"[audio] Recording chunk -> {output_file}")
    
    process = await asyncio.create_subprocess_exec(
        "arecord",
        "-D", ARECORD_DEVICE,
        "-f", "S16_LE",
        "-r", "16000",
        "-c", "6",
        output_file,
        stdout=asyncio.subprocess.DEVNULL,
        stderr=asyncio.subprocess.PIPE
    )
    
    try:
        # Wait until either stop_event triggers or chunk duration passes
        stop_task = asyncio.create_task(stop_event.wait())
        sleep_task = asyncio.create_task(asyncio.sleep(CHUNK_SECONDS))

        done, pending = await asyncio.wait(
            {stop_task, sleep_task},
            return_when=asyncio.FIRST_COMPLETED,
        )
        
        for t in pending:
            t.cancel()
        
        # Stop recording cleanly
        process.send_signal(signal.SIGINT)
        await process.wait()
        
    except Exception:
        # If anything goes wrong, try to terminate process
        try:
            process.terminate()
        except Exception:
            pass
        raise
    
    # If file exists and is non-empty, copy it forward
    if os.path.exists(output_file) and os.path.getsize(output_file) > 0:
        target_path = _copy_to_target(output_file)
        print(f"[audio] Chunk saved -> {target_path}")
        return target_path

    print("[audio] Chunk file missing/empty; skipping")
    return None

    """
    Start recording with manual stop (ENTER) or 5-minute timeout.
    
    Recording is automatically saved and pipeline continues.
    """
    process, output_file = start_recording()

    print("\nRecording started.")
    print("→ Press ENTER to stop recording and continue pipeline")
    print("→ Auto-stop after 5 minutes")

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

    print("Manual stop requested." if enter_future in done else "Auto-stop reached (5 minutes).")

    print("Stopping recording...")
    process.send_signal(signal.SIGINT)
    process.wait()
    
    # Write signal file to indicate recording completion.
    os.makedirs(RECORDING_DIR, exist_ok=True)
    with open(SIGNAL_FILE, "w", encoding="utf-8") as flag:
        flag.write(f"done:{time.time()}")

    wait_for_recording(output_file)