import argparse
import asyncio
from datetime import datetime
from typing import Dict, Any
# from queue import Queue, Empty
from typing import Optional
from src_audio.services.transcription_service.transcription_whispertrt import run_transcription
from src_audio.services.medication_extraction_service.medication_extraction import run_medication_extraction
from src_audio.services.intervention_extraction_service.intervention_extraction import run_intervention_extraction
from src_audio.services.semantic_filtering_service.semantic_filtering import run_semantic_filtering
from src_audio.services.anonymization_service.transcript_anonymization import run_anonymization_service
from src_audio.services.recording_audio_service.record_functions import record_one_chunk
from src_audio.services.audio_chunking_service.trim_audio import run_audio_trimming
from src_audio.utils.metadata import setup_metadata, finalize_metadata

def put_latest(queue: Queue, item):
        """
            most likely, each audio chunk is queued every 3 minutes so we ideally don't need a queue until the last audio chunk 
        """
        
        if queue.full():
            try:
                queue.get_nowait()
                queue.task_done()
            except:
                pass
        queue.put(item)
    
    
async def process_audio_chunk(chunk_path: str) -> None:
    try:
        print("[audio] Starting transcription...")
        await run_transcription(chunk_path)
        print("[audio] Transcription finished.")
        
        # print("[audio] Starting anonymization...")
        # await run_anonymization_service()
        # print("[audio] Anonymization finished.")
        
        # print("[audio] Starting medication extraction...")
        # await run_medication_extraction()
        # print("[audio] Medication extraction finished.")
        
        # print("[audio] Starting intervention extraction...")
        # await run_intervention_extraction()
        # print("[audio] Intervention extraction finished.")
        
    except Exception as e:
        print("[audio] Service failed:", e)

async def producer(queue: asyncio.Queue, stop_event: asyncio.Event) -> None:
    """
    Records chunks forever until stop_event is set.
    Puts chunk paths into queue. Uses a sentinel None at the end.
    """
    try:
        while True:
            chunk_path = await record_one_chunk(stop_event)
            if chunk_path:
                # “Keep latest” behavior (maxsize=1 queue):
                if queue.full():
                    try:
                        _ = queue.get_nowait()
                        queue.task_done()
                    except Exception:
                        pass
                await queue.put(chunk_path)
            if stop_event.is_set():
                break
    finally:
        await queue.put(None)  # sentinel


async def consumer(queue: asyncio.Queue) -> None:
    """ Processes chunks until stop is received """
    setup_metadata()
    try:
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            await process_audio_chunk(item)
            queue.task_done()
    finally:
        finalize_metadata()

async def wait_for_enter(stop_event: asyncio.Event) -> None:
    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, input, "\nPress ENTER to stop recording...\n")
    stop_event.set()
        
async def main():
    """
    Main function to run microservices found in their respective folders using the command line.
    """
    start_time = datetime.now()
    end_time = start_time
    
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
    
if __name__ == "__main__":
    asyncio.run(main())
