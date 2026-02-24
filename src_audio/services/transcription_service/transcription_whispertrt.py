import os
from pathlib import Path
from datetime import datetime
from src_audio.utils.export_to_csv import export_to_csv
from config.logger import Logger
from config.audio_settings import (
    IS_JETSON,
    MODEL_SIZE,
    MODEL_CACHE_PATH,
)

log = Logger("[audio][transcription]")

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
        # if parakeet model, use its transcribe method and format input correctly. Also set imtestamps to true
        if modelType == "parakeet":
            log.info("Using Parakeet ASR model for transcription")
            result = model.transcribe([audio_file], timestamps=True)
        elif modelType == "whispertrt":
            log.info("Using WhisperTRT model for transcription")
            result = model.transcribe(str(audio_file))
        else:
            log.info("Using original Whisper model for transcription")
            result = model.transcribe(str(audio_file))
        transcribe_end = datetime.now()
        log.success(f"Transcription completed in {transcribe_end - transcribe_start}")
        return result
    except Exception as e:
        log.error(f"Error during transcription: {e}")
        import traceback
        traceback.print_exc()
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