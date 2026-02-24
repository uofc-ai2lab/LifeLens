import os
from pathlib import Path
from datetime import datetime
import tempfile
import numpy as np

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
        # Load WITHOUT forcing mono so we can inspect channels
        y, sr = librosa.load(audio_file, sr=16000, mono=False)

        log.info(f"[Parakeet] Loaded audio shape: {y.shape}")
        log.info(f"[Parakeet] Loaded sample rate: {sr}")

        if sr != 16000:
            log.error(f"[Parakeet] Unexpected sample rate after resample: {sr}")

        # If multi-channel, collapse manually
        if y.ndim > 1:
            log.info(f"[Parakeet] Multi-channel detected ({y.shape}). Converting to mono.")
            y = np.mean(y, axis=0)
            log.info(f"[Parakeet] After mono conversion shape: {y.shape}")

        # Ensure strictly 1D
        y = np.squeeze(y)

        if y.ndim != 1:
            log.error(f"[Parakeet] Audio is NOT 1D after processing. Shape: {y.shape}")
            raise ValueError(f"Audio must be 1D. Found shape {y.shape}")

        log.info(f"[Parakeet] Final waveform shape before write: {y.shape}")
        log.info(f"[Parakeet] Waveform dtype before write: {y.dtype}")

        # Write explicitly as PCM 16-bit mono
        sf.write(
            temp_path,
            y.astype(np.float32),
            16000,
            subtype="PCM_16"
        )

        # Verify written file
        data, sr_check = sf.read(temp_path)

        log.info(f"[Parakeet] Written file shape: {data.shape}")
        log.info(f"[Parakeet] Written file sample rate: {sr_check}")

        if data.ndim != 1:
            log.error(f"[Parakeet] Written file is NOT mono. Shape: {data.shape}")
            raise ValueError(f"Written file must be mono. Found shape {data.shape}")

        if sr_check != 16000:
            log.error(f"[Parakeet] Written file sample rate incorrect: {sr_check}")
            raise ValueError(f"Written file must be 16kHz. Found {sr_check}")

        log.info("[Parakeet] Audio successfully prepared for Parakeet.")

        return temp_path

    except Exception as e:
        log.error(f"[Parakeet] Error preparing audio: {e}")

        if os.path.exists(temp_path):
            os.remove(temp_path)
            log.info(f"[Parakeet] Deleted temp file due to failure: {temp_path}")

        raise

def load_parakeet_model():
    """Load NVIDIA NeMo Parakeet ASR model for transcription (alternative to Whisper)"""
    log.info("Loading Parakeet ASR model")
    try:
        from nemo.collections.asr.models import ASRModel
        model = ASRModel.from_pretrained(model_name="nvidia/parakeet-tdt-0.6b-v3")
        log.success("Parakeet model loaded successfully")
        return model
    except ImportError:
        log.error("NeMo toolkit not installed. Please install with 'pip install nemo_toolkit[asr]'")
        raise
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

def transcribe_audio(audio_file: str, model, modelType: str):
    log.info("Running transcription")
    transcribe_start = datetime.now()
    try:
        if modelType == "parakeet":
            log.info("Using Parakeet ASR model for transcription")

            # 1. Standardize audio: NeMo Parakeet requires 16kHz Mono
            # Ensure your audio_file meets these requirements to avoid TypeError

            # 2. Transcribe using NeMo's native call
            # Parakeet-TDT provides native accurate word-level timestamps
            raw_output = model.transcribe([audio_file], timestamps=True)

            # 3. Apply the mapping function to "Whisper-ify" the output
            result = map_parakeet_to_whisper(raw_output)

        elif modelType == "whispertrt":
            log.info("Using WhisperTRT model")
            result = model.transcribe(str(audio_file))
        else:
            log.info("Using original Whisper model")
            result = model.transcribe(str(audio_file))

        log.success(f"Transcription completed in {datetime.now() - transcribe_start}")
        return result

    except Exception as e:
        log.error(f"Error during transcription: {e}")
        raise
def run_transcription(audio_chunk_file):
    log.header("Starting Transcription...")

    """Main runner function for WhisperTRT (or Whisper) transcription """
    # ==================== STEP 1: LOAD MODEL ====================
    # model = load_whisper_model(MODEL_SIZE, MODEL_CACHE_PATH)
    # modelType = "whispertrt" if IS_JETSON and MODEL_SIZE in ["tiny.en", "base.en"] else "whisper"
    model = load_parakeet_model()
    modelType = "parakeet"

    log.info(f"Current audio file: {Path(audio_chunk_file).name}")

    # ==================== STEP 2: CHECK INPUT FILE ====================
    if verify_audio_file_exists(audio_chunk_file) is False:
        log.error(f"File does not exist. Stopping transcription")
        return
    
    if modelType == "parakeet":
        audio_chunk_file = prepare_parakeet_audio(audio_chunk_file)

    # Track total time
    total_start = datetime.now()

    # ==================== STEP 3: TRANSCRIBE ====================
    transcribe_start = datetime.now()
    result = transcribe_audio(audio_chunk_file, model, modelType)

    # ==================== STEP 3.1: Diarize ====================
    # print_formatting("heading","STEP 3.1: Diarizing with pyannote...")
    # diarize_start = datetime.now()
    # result = await assign_speakers(device, audio_file, result, use_offline_models, hugging_face_token)
    # diarize_end = datetime.now()

    # ==================== STEP 4: CHECK OUTPUT ====================
    verified_result = verify_transcription_output(result)
    transcribe_end = datetime.now()

    # Check if transcription failed
    if verified_result is None:
        log.error("TRANSCRIPTION FAILED - STOPPING PIPELINE")
        return

    normalized_result = normalize_whisper_segments(verified_result)

    # ==================== STEP 5: EXPORT ====================
    export_start = datetime.now()
    columns=["start_time", "end_time", "text", "speaker"]
    log.info("Exporting results")

    transcript_path = export_to_csv(
        data=normalized_result,
        audio_chunk_path=Path(audio_chunk_file),
        service="transcript",
        columns=columns,
    )

    # delete temp file if it exists
    temp_files = list(Path(audio_chunk_file).parent.glob(f"parakeet_temp_{Path(audio_chunk_file).stem}_*.wav"))
    for temp_file in temp_files:
        try:
            os.remove(temp_file)
            log.debug(f"Deleted temp file: {temp_file.name}")
        except Exception as e:
            log.warning(f"Could not delete temp file {temp_file.name}: {e}")

    export_end = datetime.now()

    # Print timing summary
    time_for_transcription = transcribe_end - transcribe_start
    time_for_export = export_end - export_start
    time_total = export_end - total_start

    log.info(f"Total time: {time_total.seconds // 60}m {time_total.seconds % 60}s")
    log.debug(f"  Transcription: {time_for_transcription.seconds // 60}m {time_for_transcription.seconds % 60}s")
    log.debug(f"  Export: {time_for_export.seconds // 60}m {time_for_export.seconds % 60}s")
    log.success("Transcription completed successfully!")

    return transcript_path
