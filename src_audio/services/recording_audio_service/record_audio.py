import time
from pathlib import Path
from src_audio.domain.constants import CHUNK_SECONDS
from src_audio.services.recording_audio_service.gstreamer_audio_pipeline import GStreamerAudioPipeline

def record_one_chunk(output_dir: str | Path, stop_event) -> bool:
    """
    Records ONE audio chunk using GStreamer pipeline.
    
    Records for CHUNK_SECONDS (180s) or until stop_event is set.
    Saves the chunk as WAV file in output_dir.
    
    Args:
        output_dir: Directory to save the chunk file
        stop_event: threading.Event to signal early stop
    
    Returns:
        True if a chunk file was successfully written, False otherwise
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate timestamped filename
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = output_dir / f"recording_{ts}.wav"

    print(f"[audio] Recording chunk -> {out_path}")

    # Create GStreamer audio pipeline for this chunk
    pipeline = GStreamerAudioPipeline(str(out_path))

    try:
        # Start recording
        if not pipeline.start():
            print(f"[audio] ERROR: Failed to start recording pipeline for {out_path}")
            return False

        # Record for CHUNK_SECONDS or until stop_event is set
        start_time = time.time()
        while True:
            # Check if stop event was set (user pressed ENTER)
            if stop_event.is_set():
                print("[audio] Stop event received, stopping chunk recording")
                break
            
            # Check if chunk duration exceeded
            elapsed = time.time() - start_time
            if elapsed >= CHUNK_SECONDS:
                print(f"[audio] Chunk duration ({elapsed:.1f}s) reached, saving chunk")
                break
            
            # Poll frequently to respond quickly to stop_event
            time.sleep(0.1)

        # Stop recording
        pipeline.stop()
        pipeline.cleanup()

    except Exception as e:
        print(f"[audio] ERROR during chunk recording: {e}")
        try:
            pipeline.stop()
            pipeline.cleanup()
        except:
            pass
        return False

    # Verify chunk was written successfully
    ok = out_path.exists() and out_path.stat().st_size > 0
    if ok:
        file_size_mb = out_path.stat().st_size / (1024 * 1024)
        print(f"[audio] ✓ Chunk written -> {out_path} ({file_size_mb:.2f}MB)")
    else:
        print("[audio] ✗ Chunk missing or empty; skipping")

    return ok

