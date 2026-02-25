import os
from pathlib import Path
from datetime import datetime
import tempfile
import numpy as np
import torch
import gc
import psutil

import librosa
from src_audio.utils.export_to_csv import export_to_csv
from config.logger import Logger
from config.audio_settings import (
    IS_JETSON,
    MODEL_SIZE,
    MODEL_CACHE_PATH,
)

import soundfile as sf

# https://uofc-my.sharepoint.com/:u:/g/personal/aryan_karadia_ucalgary_ca/IQD7uvrZXqe0RZbAgrqjmynVAXgqUb17MkHi7KRPsbVsvRk?e=Yio2iB&nav=eyJyZWZlcnJhbEluZm8iOnsicmVmZXJyYWxBcHAiOiJTdHJlYW1XZWJBcHAiLCJyZWZlcnJhbFZpZXciOiJTaGFyZURpYWxvZy1MaW5rIiwicmVmZXJyYWxBcHBQbGF0Zm9ybSI6IldlYiIsInJlZmVycmFsTW9kZSI6InZpZXcifX0%3D


log = Logger("[audio][transcription]")


def prepare_parakeet_audio(audio_file: str) -> str:
    """
    Ensure audio is 16kHz mono WAV for Parakeet.
    Creates a temporary normalized file and returns its path.
    """

    original_path = Path(audio_file)

    fd, temp_path = tempfile.mkstemp(
        suffix=".wav",
        prefix=f"parakeet_temp_{original_path.stem}_",
        dir=original_path.parent
    )
    os.close(fd)

    log.info(f"[Parakeet] Preparing audio: {original_path.name}")
    log.info(f"[Parakeet] Temp file created: {temp_path}")

    try:
        y, sr = librosa.load(audio_file, sr=16000, mono=False)

        # Manual mono conversion if stereo
        if y.ndim > 1:
            log.info("Multi-channel detected. Converting to mono.")
            y = np.mean(y, axis=0)

        # Squeeze ensures the shape is (N,) and not (N, 1) or (1, N)
        y = np.squeeze(y)

        # Write as standard 16-bit PCM for compatibility
        sf.write(temp_path, y.astype(np.float32), 16000, subtype="PCM_16")
        return temp_path
    except Exception as e:
        log.error(f"[Parakeet] Error preparing audio: {e}")

        if os.path.exists(temp_path):
            os.remove(temp_path)
            log.info(f"[Parakeet] Deleted temp file due to failure: {temp_path}")

        raise

_PARAKEET_MODEL = None  # Persistent global storage

def load_parakeet_model():
    global _PARAKEET_MODEL
    # Only load if the cache is empty
    if _PARAKEET_MODEL is not None:
        return _PARAKEET_MODEL

    log.info("Loading Parakeet ASR model (FP16 optimized)")
    try:
        model = ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
        model = model.to("cuda")

        # Converts 32-bit floats to 16-bit. Essential for 8GB RAM devices.
        model.half()
        model.eval()

        _PARAKEET_MODEL = model
        log.info("Parakeet model loaded successfully in FP16 mode")
        return _PARAKEET_MODEL
    except Exception as e:
        log.error(f"Error loading Parakeet model: {e}")
        raise


def map_parakeet_to_whisper(hypotheses):
    """
    Transforms NeMo Parakeet output into a Whisper-compatible dictionary format.
    """
    if not hypotheses:
        return {"text": "", "segments": []}

    # Parakeet returns a list of Hypotheses; we process the first one
    hyp = hypotheses[0]

    # Initialize the Whisper-style structure
    whisper_style_result = {"text": getattr(hyp, "text", ""), "segments": []}

    # NeMo stores segment data in hyp.timestamp['segment'] when timestamps=True
    if hasattr(hyp, "timestamp") and "segment" in hyp.timestamp:
        for i, p_seg in enumerate(hyp.timestamp["segment"]):
            whisper_style_result["segments"].append(
                {
                    "id": i,
                    "start": p_seg.get("start", 0.0),
                    "end": p_seg.get("end", 0.0),
                    "text": p_seg.get("segment", ""),
                    # Placeholder keys to satisfy Whisper-specific pipeline checks
                    "avg_logprob": 0.0,
                    "no_speech_prob": 0.0,
                    "speaker": "UNKNOWN",
                }
            )

    return whisper_style_result


def load_whisper_model(model_size: str, model_cache_path: str = None):
    """Transcribe audio using WhisperTRT or fallback to original Whisper"""
    log.info(f"Loading {model_size.upper()} model")
    model = None

    # Determine if we should use WhisperTRT or original Whisper
    use_whispertrt = IS_JETSON and model_size in ["tiny.en", "base.en"]

    if use_whispertrt:
        try:
            from whisper_trt import load_trt_model
            log.info(f"Using WhisperTRT for {model_size} (TensorRT accelerated)")
            log.info("Note: First run will build TensorRT engine (takes 2-5 minutes)")

            if model_cache_path:
                log.debug(f"Using custom cache path: {model_cache_path}")
                model_file_path = os.path.join(model_cache_path, f"{model_size}.pth")
                model = load_trt_model(model_size, path=model_file_path)
            else:
                log.debug(f"Using default cache: ~/.cache/whisper_trt/")
                model = load_trt_model(model_size)

            log.success(f"Model loaded successfully (type: {type(model).__name__})")

        except ImportError:
            log.warning(f"WhisperTRT not installed. Falling back to original Whisper...")
            use_whispertrt = False
        except Exception as e:
            log.error(f"Error loading WhisperTRT model: {e}")
            log.warning(f"Falling back to original Whisper...")
            use_whispertrt = False

    if not use_whispertrt:
        log.info(f"Using original Whisper for {model_size}")
        log.info("Note: Slower than TensorRT but more memory efficient")

        try:
            import whisper

            # Use download_root parameter if cache path is specified
            if model_cache_path:
                model = whisper.load_model(model_size, download_root=model_cache_path)
            else:
                model = whisper.load_model(model_size)

            log.success(f"Model loaded successfully (type: {type(model).__name__})")

        except Exception as e:
            log.error(f"Error loading model: {e}")
            raise

    return model

_WHISPER_FALLBACK = None

def load_whisper_fallback(model_size: str, model_cache_path: str = None):
    """Lazy-loads Whisper Tiny only if needed to save initial RAM."""
    global _WHISPER_FALLBACK
    if _WHISPER_FALLBACK is not None:
        return _WHISPER_FALLBACK

    log.info("Loading Whisper Tiny as fallback model")
    try:
        import whisper
        # Use download_root parameter if cache path is specified
            if model_cache_path:
                _WHISPER_FALLBACK = whisper.load_model(model_size, download_root=model_cache_path)
            else:
                _WHISPER_FALLBACK = whisper.load_model(model_size)

            log.success(f"Model loaded successfully (type: {type(_WHISPER_FALLBACK).__name__})")
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
        
    # Check full text
    full_text = result.get('text', '')
    log.debug(f"Full text length: {len(full_text)} characters")
    if not full_text.strip:
        log.warning("Transcription text is EMPTY!")
    
    # Check segments
    segments = result.get('segments', [])
    log.debug(f"Number of segments: {len(segments)}")
    if segments:
        log.debug(f"First segment: {segments[0].get('text', '')[:50]}...")
        return segments
    else:
        log.warning("No segments found")
        return full_text

def normalize_whisper_segments(segments):
    """
    Convert Whisper segment keys to pipeline-standard keys.
    """
    normalized = []
    for seg in segments:
        normalized.append({
            "start_time": seg["start"],
            "end_time": seg["end"],
            "text": seg.get("text", ""),
            "speaker": seg.get("speaker", "UNKNOWN")
        })
    return normalized


def transcribe_audio(audio_file: str, model, model_type: str):
    log.info(f"Starting {model_type} transcription pass")
    try:
        with torch.no_grad():  # Saves massive memory by not tracking gradients
            if model_type == "parakeet":
                raw_output = model.transcribe([audio_file], timestamps=True)
                result = map_parakeet_to_whisper(raw_output)
            else:
                result = model.transcribe(str(audio_file))

        # IMPORTANT: Explicitly free VRAM after every successful chunk
        torch.cuda.empty_cache()
        gc.collect()
        return result

    except RuntimeError as e:
        # Catch the "Out of Memory" error specifically
        if "out of memory" in str(e).lower():
            log.error("CUDA Out of Memory. Clearing VRAM and falling back to Whisper.")
            torch.cuda.empty_cache()
            gc.collect()

            if model_type == "parakeet":
                fallback = load_whisper_fallback()
                return transcribe_audio(audio_file, fallback, "whisper_fallback")
        raise e


def run_transcription(audio_chunk_file):
    log.info("Starting Transcription Pipeline")

    available_gb = psutil.virtual_memory().available / (1024**3)
    if available_gb < 1.5:
        log.warning(f"Low memory detected ({available_gb:.2f}GB). Forcing cache flush.")
        torch.cuda.empty_cache()
        gc.collect()

    # ==================== STEP 1: LOAD MODEL (Singleton) ====================
    # Only loads once; stays in VRAM for performance
    model = load_parakeet_model()
    model_type = "parakeet"

    log.info(f"Current audio file: {Path(audio_chunk_file).name}")

    # ==================== STEP 2: CHECK & PREPARE INPUT ====================
    if verify_audio_file_exists(audio_chunk_file) is False:
        log.error("File does not exist. Stopping transcription")
        return None

    # Standardize to 16kHz Mono for NeMo
    prepared_audio = prepare_parakeet_audio(audio_chunk_file)

    # Track total time
    total_start = datetime.now()

    # ==================== STEP 3: TRANSCRIBE ====================
    transcribe_start = datetime.now()
    # This now handles the Parakeet -> Whisper fallback internally
    result = transcribe_audio(prepared_audio, model, model_type)

    # ==================== STEP 4: CHECK OUTPUT ====================
    verified_result = verify_transcription_output(result)
    transcribe_end = datetime.now()

    if verified_result is None:
        log.error("TRANSCRIPTION FAILED - STOPPING PIPELINE")
        # Ensure cleanup even on failure
        if os.path.exists(prepared_audio):
            os.remove(prepared_audio)
        return None

    normalized_result = normalize_whisper_segments(verified_result)

    # ==================== STEP 5: EXPORT ====================
    export_start = datetime.now()
    columns = ["start_time", "end_time", "text", "speaker"]
    log.info("Exporting results to CSV")

    # Note: We pass the original filename to the exporter so the CSV
    # doesn't get named "parakeet_temp_..."
    transcript_path = export_to_csv(
        data=normalized_result,
        audio_chunk_path=Path(audio_chunk_file),
        service="transcript",
        columns=columns,
    )

    # ==================== STEP 6: CLEANUP ====================
    # Explicitly delete the prepared temp file
    if os.path.exists(prepared_audio):
        try:
            os.remove(prepared_audio)
            log.info(f"Deleted temp file: {Path(prepared_audio).name}")
        except Exception as e:
            log.warning(f"Could not delete temp file: {e}")

    # Fallback cleanup: remove any dangling parakeet temp files in the directory
    temp_pattern = f"parakeet_temp_{Path(audio_chunk_file).stem}_*.wav"
    dangling_files = list(Path(audio_chunk_file).parent.glob(temp_pattern))
    for df in dangling_files:
        try:
            os.remove(df)
        except:
            pass

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
