import argparse
import asyncio
from datetime import datetime
from typing import Dict, Any
import os, shutil
import time
from pathlib import Path
from config.audio_settings import AUDIO_DIR
from src_audio.services.transcription_service.transcription_whispertrt import run_transcription
from src_audio.services.medication_extraction_service.medication_extraction import run_medication_extraction
from src_audio.services.intervention_extraction_service.intervention_extraction import run_intervention_extraction
from src_audio.services.anonymization_service.transcript_anonymization import run_anonymization_service
from src_audio.services.recording_audio_service.record_audio import record_one_chunk
from src_audio.utils.make_directories import create_chunk_dir
from src_audio.domain.constants import RECORDING_DIR, AUDIO_EXTS

def _dev_wrap_audio_into_chunk(parent_audio_file: str, chunk_idx: int = 0) -> Path:
    """
    DEV helper:
    Takes an existing wav file and wraps it into the same
    chunk directory structure used by recording function.
    """
    parent_audio_file = Path(parent_audio_file)

    # skip junk
    if parent_audio_file.suffix.lower() not in AUDIO_EXTS:
        raise ValueError(f"Unsupported audio ext: {parent_audio_file.suffix}")

    chunk_filename = (
        f"{parent_audio_file.stem}_chunk_{chunk_idx}"
        f"{parent_audio_file.suffix}"
    )

    chunk_audio_dir = parent_audio_file.with_suffix("") / Path(chunk_filename).stem
    chunk_audio_dir.mkdir(parents=True, exist_ok=True)

    chunk_audio_path = chunk_audio_dir / chunk_filename

    # Move (or copy) the file into the chunk directory
    if not chunk_audio_path.exists():
        shutil.move(parent_audio_file, chunk_audio_path)

    return chunk_audio_path


def _make_output_path() -> str:
    os.makedirs(RECORDING_DIR, exist_ok=True) # Ensure the recording directory exists.
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    return os.path.join(RECORDING_DIR, f"recording_{timestamp}.wav")

async def process_audio_chunk(chunk_path: str) -> None:
    try:
        print("[audio] Starting transcription...")
        transcript_path = await run_transcription(chunk_path)
        print("[audio] Transcription finished.")
        
        print("[audio] Starting anonymization...")
        await run_anonymization_service(chunk_path, transcript_path)
        print("[audio] Anonymization finished.")
        
        print("[audio] Starting medication extraction...")
        await run_medication_extraction(chunk_path, transcript_path)
        print("[audio] Medication extraction finished.")
        
        print("[audio] Starting intervention extraction...")
        await run_intervention_extraction(chunk_path, transcript_path)
        print("[audio] Intervention extraction finished.")
        
    except Exception as e:
        print("[audio] Service failed:", e)

async def producer(queue: asyncio.Queue, stop_event: asyncio.Event) -> None:
    """
    Records chunks forever until stop_event is set.
    Puts chunk paths into queue. Uses a sentinel None at the end.
    """
    
    chunk_idx = 0
    parent_audio_file = _make_output_path()
    
    try:
        while True:
            chunk_dir = create_chunk_dir(parent_audio_file, chunk_idx)
            audio_chunk_path = chunk_dir / f"{chunk_dir.name}.wav"
            
            target_path = await record_one_chunk(audio_chunk_path, stop_event)
            
            if target_path:
                if queue.full():
                    try:
                        _ = queue.get_nowait()
                        queue.task_done()
                    except Exception:
                        pass

                await queue.put(target_path)
                
            chunk_idx += 1
            
            if stop_event.is_set():
                break
    finally:
        await queue.put(None)  # sentinel


async def consumer(queue: asyncio.Queue) -> None:
    """ Processes chunks until stop is received """
    while True:
        item = await queue.get()
        if item is None:
            queue.task_done()
            break
        await process_audio_chunk(item)
        queue.task_done()
      
        
async def wait_for_enter(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, input, "\nPress ENTER to stop recording...\n")
    stop_event.set()
        
        
async def main() -> int:
    """
    Main function to run microservices found in their respective folders using the command line.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dev", action="store_true")
    args = parser.parse_args()
    
    start_time = datetime.now()
    end_time = start_time
    
    DEV_MODE = args.dev
    
    if DEV_MODE:
        print("[AUDIO MODE] DEV")

        audio_dir = Path(AUDIO_DIR)
        files = [
            p for p in audio_dir.rglob("*")
            if p.is_file() and p.suffix.lower() in AUDIO_EXTS
        ]
        
        if not files:
            print(f"[dev] No audio files found in {audio_dir} with {AUDIO_EXTS}")
            return 0

        for f in files:
            chunked_path = _dev_wrap_audio_into_chunk(f, chunk_idx=0)
            print(f"[dev] Processing {chunked_path}")

            await process_audio_chunk(str(chunked_path))

        return 0

    
    stop_event = asyncio.Event()
    queue = asyncio.Queue(maxsize=1) # maxsize=1 gives the latest only
    
    await asyncio.gather(
        producer(queue, stop_event),
        consumer(queue),
        wait_for_enter(stop_event),
    )
    
    end_time = datetime.now()
    total_time = end_time - start_time
    
    total_seconds = int(total_time.total_seconds())
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)

    print(f"Complete AUDIO pipeline time: {hours} hours, {minutes} minutes, and {seconds} seconds")
    
    return 0
    
if __name__ == "__main__":
    asyncio.run(main())
