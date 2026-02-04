import asyncio
import subprocess
import os
import time
import shutil
import signal
from pathlib import Path
from typing import Optional
from config.audio_settings import AUDIO_DIR
from src_audio.utils.make_directories import create_parent_audio_dir
from src_audio.domain.constants import MAX_RECORD_SECONDS, RECORDING_DIR, SIGNAL_FILE, ARECORD_DEVICE, CHUNK_SECONDS

def _copy_to_target(recording_path: str) -> str:
    """
    Copy raw recorded file into AUDIO_DIR chunk directory.
    """
    
    recording_path = Path(recording_path)
    
    chunk_dir = Path(AUDIO_DIR) / recording_path.stem
    chunk_dir.mkdir(parents=True, exist_ok=True)
    
    target_path = chunk_dir / recording_path.name

    print(f"Copying {recording_path} -> {target_path}")
    shutil.move(recording_path, target_path)

    create_parent_audio_dir(target_path)
    return target_path

async def record_one_chunk(
    output_file: Path, 
    stop_event: asyncio.Event
    ) -> Optional[str]:
    """
    Records up to `chunk_seconds` seconds OR until stop_event is set.
    Records audio to output_file (raw), copies it into AUDIO_DIR (chunked), returns the TARGET path used by services.
    """
    
    if os.path.exists(SIGNAL_FILE): # Remove stale signal file to prevent false positives.
        os.remove(SIGNAL_FILE)
    
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
    if output_file.exists() and output_file.stat().st_size > 0:
        target_path = _copy_to_target(output_file)
        print(f"[audio] Chunk saved -> {target_path}")
        return str(target_path)

    print("[audio] Chunk file missing/empty; skipping")
    return None