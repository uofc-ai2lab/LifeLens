import os
import re
import wave
from pathlib import Path
from datetime import datetime, timedelta
import torch
import gc
import psutil
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

from src_audio.utils.export_to_csv import export_to_csv
from src_audio.domain.constants import CHUNK_SECONDS
from config.logger import Logger
from config.audio_settings import MODEL_CACHE_PATH
from config.gpu_guard import gpu_exclusive

log = Logger("[audio][transcription]")

# Persistent global storage
_WHISPER_PIPE = None  # fine-tuned model (HF Pipeline)
_WHISPER_FALLBACK = None  # Original OpenAI Whisper
_WHISPER_PIPE_MODEL_PATH = None

_WHISPER_BASE_MODEL_MAP = {
    "tiny": "openai/whisper-tiny",
    "base": "openai/whisper-base",
    "small": "openai/whisper-small",
    "medium": "openai/whisper-medium",
    "large": "openai/whisper-large-v3",
}

# At the top of transcription_service.py
_SERVICE_DIR = Path(__file__).resolve().parent  # always points to transcription_service/
_STARTING_TIME_SECONDS = 1
_CURRENT_RECORDING_KEY = None
_MIN_FREE_CUDA_GB = float(os.getenv("LIFELENS_MIN_FREE_CUDA_GB", "2.5"))
_MIN_FREE_SYSTEM_GB = float(os.getenv("LIFELENS_MIN_FREE_SYSTEM_GB", "2.0"))

_RECORDING_FILENAME_PATTERN = re.compile(
    r"^recording_(?P<date>\d{8})_(?P<time>\d{6})\.[^.]+$"
)


def normalize_whisper_segments(segments, base_datetime: datetime = None):
    if base_datetime is None:
        base_datetime = datetime.now()
        log.warning("No base_datetime provided. Timestamps will be relative to current execution time.")

    log.debug(f"Normalizing {len(segments)} segments. Sample raw segment: {segments[0] if segments else 'N/A'}")

    normalized = []
    for i, seg in enumerate(segments):
        import datetime as dt

        rel_start = (
            seg.get("start")
            if seg.get("start") is not None
            else seg.get("timestamp", [0, 0])[0]
        )
        rel_end = (
            seg.get("end")
            if seg.get("end") is not None
            else seg.get("timestamp", [None, None])[1]
        )

        # 2. Add relative seconds to the base datetime
        real_start_dt = base_datetime + timedelta(seconds=rel_start)
        real_end_dt = base_datetime + timedelta(seconds=rel_end)

        # 3. Format as string (e.g., 14:30:05)
        str_start = real_start_dt.strftime("%H:%M:%S")
        str_end = real_end_dt.strftime("%H:%M:%S")

        normalized.append(
            {
                "start_time": str_start,  # Now real-world time
                "end_time": str_end,  # Now real-world time
                "text": seg.get("text", "").strip(),
                "speaker": "UNKNOWN",
            }
        )
    return normalized


def parse_recording_base_datetime(audio_file: str):
    """Parse base datetime from recording_YYYYMMDD_HHMMSS.*"""
    filename = Path(audio_file).name
    match = _RECORDING_FILENAME_PATTERN.match(filename)
    if not match:
        return None, None

    recording_key = f"{match.group('date')}_{match.group('time')}"
    try:
        base_datetime = datetime.strptime(recording_key, "%Y%m%d_%H%M%S")
        return base_datetime, recording_key
    except ValueError:
        return None, None


def get_chunk_advance_seconds(audio_file: str) -> float:
    """Advance by CHUNK_SECONDS, unless actual duration is shorter (manual stop/final chunk)."""
    advance_seconds = float(CHUNK_SECONDS)
    try:
        with wave.open(str(audio_file), "rb") as wav_file:
            frame_rate = wav_file.getframerate()
            frame_count = wav_file.getnframes()
            if frame_rate > 0:
                duration_seconds = frame_count / frame_rate
                if duration_seconds < advance_seconds:
                    advance_seconds = duration_seconds
    except Exception as e:
        log.warning(
            f"Could not read WAV duration for chunk advance ({Path(audio_file).name}): {e}. "
            f"Using CHUNK_SECONDS={CHUNK_SECONDS}."
        )

    return max(advance_seconds, 0.0)


def load_fine_tuned_whisper(model_path: str):
    """Loads the fine-tuned Whisper LoRA model via HF Pipeline."""
    global _WHISPER_PIPE, _WHISPER_PIPE_MODEL_PATH

    resolved_model_path = str(Path(model_path).resolve())
    if _WHISPER_PIPE is not None and _WHISPER_PIPE_MODEL_PATH == resolved_model_path:
        return _WHISPER_PIPE

    log.info(f"Loading Fine-Tuned Whisper model (FP16 optimized) from: {model_path}")
    try:
        cuda_available = torch.cuda.is_available()
        use_cuda = cuda_available

        if cuda_available:
            try:
                free_cuda_gb = torch.cuda.mem_get_info()[0] / 1e9
            except Exception:
                free_cuda_gb = 0.0

            free_system_gb = psutil.virtual_memory().available / (1024**3)

            if free_cuda_gb < _MIN_FREE_CUDA_GB or free_system_gb < _MIN_FREE_SYSTEM_GB:
                use_cuda = False
                log.warning(
                    f"Insufficient memory headroom for GPU Whisper "
                    f"(CUDA free {free_cuda_gb:.2f}GB, system free {free_system_gb:.2f}GB). "
                    f"Using CPU for this pass."
                )

        dtype = torch.float16 if use_cuda else torch.float32

        # Processor always lives in root adapter dir, not inside checkpoint subfolders
        processor_path = str(Path(resolved_model_path).parent) \
            if "checkpoint-" in resolved_model_path else resolved_model_path

        log.info(f"Adapter path:   {resolved_model_path}")
        log.info(f"Processor path: {processor_path}")

        # Infer base model size from path
        model_path_lower = resolved_model_path.lower()
        base_model_name = next(
            (hf for key, hf in _WHISPER_BASE_MODEL_MAP.items() if key in model_path_lower),
            _WHISPER_BASE_MODEL_MAP["base"]
        )
        log.info(f"Inferred base model: {base_model_name}")

        # Step 1: Load base model — no device_map, keeps accelerate out entirely
        base_model = WhisperForConditionalGeneration.from_pretrained(
            base_model_name,
            torch_dtype=dtype,
            low_cpu_mem_usage=True,
        )

        # Step 2: Merge LoRA weights on CPU
        model = PeftModel.from_pretrained(base_model, resolved_model_path)
        model = model.merge_and_unload()
        model.eval()

        # Step 3: Fix generation config to suppress warnings and lock to English
        model.generation_config.forced_decoder_ids = None
        model.generation_config.language = "en"
        model.generation_config.task = "transcribe"

        # Step 4: Move merged model to GPU after merge is complete
        if use_cuda:
            import ctypes
            gc.collect()
            gc.collect()
            ctypes.CDLL("libc.so.6").malloc_trim(0)
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

            free_mem = torch.cuda.mem_get_info()[0] / 1e9
            log.info(f"Free CUDA memory before GPU move: {free_mem:.2f}GB")

            model = model.to("cuda", dtype=dtype)

        log.info(f"Model device: {next(model.parameters()).device}")

        processor = WhisperProcessor.from_pretrained(processor_path)

        # Step 5: Build pipeline — device=0 safe since accelerate was never involved
        _WHISPER_PIPE = pipeline(
            "automatic-speech-recognition",
            model=model,
            feature_extractor=processor.feature_extractor,
            tokenizer=processor.tokenizer,
            chunk_length_s=30,
            stride_length_s=5,         # reduce chunk overlap to prevent segment duplication
            dtype=dtype,
            device=0 if use_cuda else -1,
            ignore_warning=True,
            generate_kwargs={
                "language": "en",
                "task": "transcribe",
                "no_repeat_ngram_size": 3,   # prevents "so, so, so..." loops
                "repetition_penalty": 1.05,    # penalises repeated tokens
                "temperature": 0.0,           # greedy decode, no randomness
            }
        )
        _WHISPER_PIPE_MODEL_PATH = resolved_model_path
        if use_cuda:
            log.info("Fine-tuned Whisper pipeline loaded successfully on GPU")
        else:
            log.info("Fine-tuned Whisper pipeline loaded successfully on CPU")
        return _WHISPER_PIPE

    except Exception as e:
        _WHISPER_PIPE = None
        _WHISPER_PIPE_MODEL_PATH = None
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


def run_transcription(audio_chunk_file, model_path=None):
    if model_path is None:
        model_path = str(_SERVICE_DIR / "models/whisper-base-medical-lora/checkpoint-3000")

    global _STARTING_TIME_SECONDS
    global _CURRENT_RECORDING_KEY

    log.info(f"Resolved model path: {model_path}") 

    # Memory Check
    available_gb = psutil.virtual_memory().available / (1024**3)
    if available_gb < 1.5:
        log.warning(f"Low memory detected ({available_gb:.2f}GB). Forcing cache flush.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    parsed_base_time, recording_key = parse_recording_base_datetime(
        audio_chunk_file
    )
    if parsed_base_time is not None:
        if _CURRENT_RECORDING_KEY != recording_key:
            _CURRENT_RECORDING_KEY = recording_key
            _STARTING_TIME_SECONDS = 1
            log.info(
                f"Detected new recording session {recording_key}. Reset global starting offset to 1s."
            )
        recording_start_time = parsed_base_time
    else:
        # if parsing filename fails, log error and fall back to current time
        log.error(
            f"Failed to parse recording base datetime from filename: {Path(audio_chunk_file).name}. "
            f"Ensure it follows the pattern 'recording_YYYYMMDD_HHMMSS.*'. Using current time as base."
        )
        recording_start_time = datetime.now()
        _CURRENT_RECORDING_KEY = None
        _STARTING_TIME_SECONDS = 1

    effective_chunk_base_time = recording_start_time + timedelta(
        seconds=_STARTING_TIME_SECONDS
    )
    log.info(
        f"Recording anchored at: {recording_start_time.strftime('%Y-%m-%d %H:%M:%S')} | "
        f"chunk base offset: {_STARTING_TIME_SECONDS:.2f}s | "
        f"effective base: {effective_chunk_base_time.strftime('%Y-%m-%d %H:%M:%S')}"
    )

    # ==================== STEP 1: CHECK INPUT ====================
    if verify_audio_file_exists(audio_chunk_file) is False:
        log.error("File does not exist or is invalid. Stopping transcription.")
        return None

    total_start = datetime.now()

    # ==================== STEP 2+3: LOAD + TRANSCRIBE (GPU-serialized) ====================
    with gpu_exclusive("audio:transcription", logger=log):
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

    normalized_result = normalize_whisper_segments(
        verified_segments,
        base_datetime=effective_chunk_base_time,
    )

    advance_seconds = get_chunk_advance_seconds(audio_chunk_file)
    _STARTING_TIME_SECONDS += advance_seconds
    log.debug(
        f"Updated global starting offset by {advance_seconds:.2f}s. "
        f"Next chunk offset: {_STARTING_TIME_SECONDS:.2f}s"
    )

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
