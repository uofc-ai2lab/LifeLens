import time
from pathlib import Path
from src_audio.domain.constants import CHUNK_SECONDS
from src_audio.services.recording_audio_service.gstreamer_audio_pipeline import GStreamerAudioPipeline
from config.logger import Logger

log = Logger("[audio][microphone]")

def record_one_chunk(output_dir: str | Path, stop_event) -> bool:
    """
    Records ONE audio chunk using GStreamer pipeline.
    Records for CHUNK_SECONDS or until stop_event is set.
    Saves the chunk as WAV file in output_dir.
    
    Args:
        output_dir: Directory to save the chunk file
        stop_event: threading.Event to signal early stop
    
    Returns:
        True if a chunk file was successfully written, False otherwise
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"recording_{ts}.wav"

    log.info(f"Recording chunk -> {output_path.name}")

    # Create GStreamer audio pipeline for this chunk
    pipeline = GStreamerAudioPipeline(str(output_path))

    try:
        # Start recording
        if not pipeline.start():
            log.error(f"Failed to start recording pipeline")
            return False

        # Record for CHUNK_SECONDS or until stop_event is set
        start_time = time.time()
        while True:
            # Check if stop event was set (user pressed ENTER)
            if stop_event.is_set():
                log.info("Stop event received, stopping chunk recording")
                break
            
            # Check if chunk duration exceeded
            elapsed = time.time() - start_time
            if elapsed >= CHUNK_SECONDS:
                log.info(f"Chunk duration ({elapsed:.1f}s) reached, saving chunk")
                break
            
            time.sleep(0.1)

        pipeline.stop()
        pipeline.cleanup()

    except Exception as e:
        log.error(f"Error during chunk recording: {e}")
        try:
            pipeline.stop()
            pipeline.cleanup()
        except:
            pass
        return False

    ok = output_path.exists() and output_path.stat().st_size > 0
    if ok:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        log.success(f"Chunk written -> {output_path.name} ({file_size_mb:.2f}MB)")
    else:
        log.warning("Chunk missing or empty; skipping")

    return ok

