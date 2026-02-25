import os
from pathlib import Path
from datetime import datetime
import torch
import gc
import psutil
from transformers import pipeline

from src_audio.utils.export_to_csv import export_to_csv
from config.logger import Logger
from config.audio_settings import IS_JETSON, MODEL_CACHE_PATH

log = Logger("[audio][transcription]")

# Persistent global storage
_WHISPER_PIPE = None  # Your fine-tuned model (HF Pipeline)
_WHISPER_FALLBACK = None  # Original OpenAI Whisper


def normalize_whisper_segments(segments, base_datetime: datetime = None):
    """
    Convert Whisper segments to real-time clock timestamps.
    base_datetime: The actual wall-clock time when the audio recording STARTED.
    """
    if base_datetime is None:
        # Fallback to 'now' if no start time is provided
        base_datetime = datetime.now()
        log.warning(
            "No base_datetime provided. Timestamps will be relative to current execution time."
        )

    normalized = []
    for seg in segments:
        # 1. Get relative offsets from Whisper
        rel_start = (
            seg.get("start")
            if seg.get("start") is not None
            else seg.get("timestamp", [0, 0])[0]
        )
        rel_end = (
            seg.get("end")
            if seg.get("end") is not None
            else seg.get("timestamp", [0, 0])[1]
        )

        # 2. Add relative seconds to the base datetime
        # This handles the "Simple Math" you mentioned in a way that scales
        import datetime as dt

        real_start_dt = base_datetime + dt.timedelta(seconds=rel_start)
        real_end_dt = base_datetime + dt.timedelta(seconds=rel_end)

        # 3. Format as string (e.g., 14:30:05)
        str_start = real_start_dt.strftime("%H:%M:%S")
        str_end = real_end_dt.strftime("%H:%M:%S")

        normalized.append(
            {
                "start_time": str_start,  # Now real-world time
                "end_time": str_end,  # Now real-world time
                "text": seg.get("text", "").strip(),
                "speaker": "UNKNOWN",
                "rel_start": rel_start,  # Keeping raw offset just in case
            }
        )
    return normalized


def load_fine_tuned_whisper(model_path: str):
    """Loads the fine-tuned Whisper model via HF Pipeline."""
    global _WHISPER_PIPE
    if _WHISPER_PIPE is not None:
        return _WHISPER_PIPE

    log.info(f"Loading Fine-Tuned Whisper model (FP16 optimized) from: {model_path}")
    try:
        device = 0 if torch.cuda.is_available() else -1
        # Similar to your Parakeet setup, we force FP16 for memory efficiency
        _WHISPER_PIPE = pipeline(
            "automatic-speech-recognition",
            model=model_path,
            chunk_length_s=30,
            device=device,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        log.success("Fine-tuned Whisper pipeline loaded successfully in FP16 mode")
        return _WHISPER_PIPE
    except Exception as e:
        log.error(f"Error loading Fine-Tuned Whisper: {e}")
        return None


def load_whisper_fallback(model_size="base"):
    """Lazy-loads original OpenAI Whisper as a safety net."""
    global _WHISPER_FALLBACK
    if _WHISPER_FALLBACK is not None:
        return _WHISPER_FALLBACK

    log.info(f"Loading Whisper {model_size.upper()} as fallback model")
    try:
        import whisper
        # Use MODEL_CACHE_PATH from your config
        _WHISPER_FALLBACK = whisper.load_model(
            model_size, download_root=MODEL_CACHE_PATH
        )
        log.success(f"Fallback model {model_size} loaded successfully")
        return _WHISPER_FALLBACK
    except Exception as e:
        log.error(f"Failed to load fallback model: {e}")
        return None


def verify_audio_file_exists(audio_file: str) -> bool:
    log.info(f"Verifying input file: {Path(audio_file).name}")
    
    if not os.path.exists(audio_file):
        log.error(f"Audio file not found: {audio_file}")
        return False
    
    file_size = os.path.getsize(audio_file)
    log.info(f"File size: {file_size / (1024*1024):.2f} MB")
    
    if file_size == 0:
        log.error("Audio file is empty!")
        return False
    
    return True

def verify_transcription_output(result: dict):
    log.info("Verifying transcription output")

    if not isinstance(result, dict):
        log.error(f"Result is not a dictionary! Got: {type(result)}")
        return None

    full_text = result.get('text', '')
    log.debug(f"Full text length: {len(full_text)} characters")
    if not full_text.strip():
        log.warning("Transcription text is EMPTY!")

    segments = result.get('segments', [])
    log.debug(f"Number of segments: {len(segments)}")

    if segments:
        log.debug(f"First segment: {segments[0].get('text', '')[:50]}...")
        return segments
    else:
        log.warning("No segments found")
        return []

def normalize_whisper_segments(segments):
    """Convert keys to pipeline-standard keys for CSV export."""
    normalized = []
    for seg in segments:
        # Compatibility check for HF 'timestamp' vs OpenAI 'start'/'end'
        start = (
            seg.get("start")
            if seg.get("start") is not None
            else seg.get("timestamp", [0, 0])[0]
        )
        end = (
            seg.get("end")
            if seg.get("end") is not None
            else seg.get("timestamp", [0, 0])[1]
        )

        normalized.append(
            {
                "start_time": start,
                "end_time": end,
                "text": seg.get("text", "").strip(),
                "speaker": "UNKNOWN",
            }
        )
    return normalized


def transcribe_audio(audio_file: str, model_obj, model_type: str):
    log.info(f"Starting {model_type} transcription pass")
    try:
        with torch.no_grad():
            if model_type == "fine_tuned":
                # HF Pipeline inference
                raw_output = model_obj(str(audio_file), return_timestamps=True)
                result = {
                    "text": raw_output["text"],
                    "segments": raw_output.get("chunks", []),
                }
            else:
                # OpenAI Whisper inference
                result = model_obj.transcribe(str(audio_file))

        # IMPORTANT: Explicitly free VRAM after every successful pass
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()
        return result

    except Exception as e:
        error_str = str(e).lower()
        if "out of memory" in error_str:
            log.error(f"CUDA Out of Memory during {model_type} pass. Clearing VRAM.")
        else:
            log.error(f"Error during {model_type} transcription: {e}")

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

        # Fallback recursive logic
        if model_type == "fine_tuned":
            log.warning("Falling back to original Whisper due to error.")
            fallback = load_whisper_fallback()
            if fallback:
                return transcribe_audio(audio_file, fallback, "whisper_fallback")

        raise e


def run_transcription(audio_chunk_file, model_path="./whisper-medical-final"):
    log.info("Starting Transcription Pipeline")

    # Memory Check
    available_gb = psutil.virtual_memory().available / (1024**3)
    if available_gb < 1.5:
        log.warning(f"Low memory detected ({available_gb:.2f}GB). Forcing cache flush.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    file_stat = os.stat(audio_chunk_file)
    recording_start_time = datetime.fromtimestamp(file_stat.st_ctime)
    log.info(
        f"Recording anchored at: {recording_start_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # ==================== STEP 1: LOAD MODEL ====================
    model = load_fine_tuned_whisper(model_path)
    model_type = "fine_tuned"

    if model is None:
        log.warning("Primary Fine-Tuned model failed to load. Switching to Fallback.")
        model = load_whisper_fallback()
        model_type = "whisper_fallback"

    if model is None:
        log.error("CRITICAL: No transcription models could be loaded. Stopping.")
        return None

    log.info(f"Current audio file: {Path(audio_chunk_file).name}")

    # ==================== STEP 2: CHECK INPUT ====================
    if verify_audio_file_exists(audio_chunk_file) is False:
        log.error("File does not exist or is invalid. Stopping transcription.")
        return None

    total_start = datetime.now()

    # ==================== STEP 3: TRANSCRIBE ====================
    transcribe_start = datetime.now()
    try:
        result = transcribe_audio(audio_chunk_file, model, model_type)
    except Exception as e:
        log.error(f"TRANSCRIPTION FAILED after fallback: {e}")
        return None
    transcribe_end = datetime.now()

    # ==================== STEP 4: CHECK & NORMALIZE ====================
    verified_segments = verify_transcription_output(result)
    if verified_segments is None:
        log.error("TRANSCRIPTION VERIFICATION FAILED - STOPPING PIPELINE")
        return None

    normalized_result = normalize_whisper_segments(verified_segments)

    # ==================== STEP 5: EXPORT ====================
    export_start = datetime.now()
    log.info("Exporting results to CSV")

    transcript_path = export_to_csv(
        data=normalized_result,
        audio_chunk_path=Path(audio_chunk_file),
        service="transcript",
        columns=["start_time", "end_time", "text", "speaker"],
    )
    export_end = datetime.now()

    # Timing Summary
    time_for_transcription = transcribe_end - transcribe_start
    time_for_export = export_end - export_start
    time_total = export_end - total_start

    log.info(f"Total time: {time_total.seconds // 60}m {time_total.seconds % 60}s")
    log.debug(f"  Transcription: {time_for_transcription.seconds // 60}m {time_for_transcription.seconds % 60}s")
    log.debug(f"  Export: {time_for_export.seconds // 60}m {time_for_export.seconds % 60}s")
    log.success("Transcription completed successfully")

    return transcript_path
