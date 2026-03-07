import os
from pathlib import Path
from datetime import datetime
import re
import torch
import gc
import psutil
from transformers import pipeline, WhisperForConditionalGeneration, WhisperProcessor
from peft import PeftModel

from src_audio.utils.export_to_csv import export_to_csv
from config.logger import Logger
from config.audio_settings import IS_JETSON, MODEL_CACHE_PATH

log = Logger("[audio][transcription]")

# Persistent global storage
_WHISPER_PIPE = None  # Your fine-tuned model (HF Pipeline)
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

        # Guard: if end is None (Whisper cut off mid-word), estimate from next segment
        # or fall back to start + 30s (max Whisper chunk)
        if rel_end is None:
            if i + 1 < len(segments):
                next_start = segments[i + 1].get("timestamp", [None])[0]
                rel_end = next_start if next_start is not None else rel_start + 30.0
            else:
                rel_end = rel_start + 30.0  # last segment fallback
            log.warning(f"Segment {i} missing end timestamp — estimated as {rel_end:.1f}s")

        # Guard: if start is also None, skip the segment entirely
        if rel_start is None:
            log.warning(f"Segment {i} has no start timestamp — skipping")
            continue

        real_start_dt = base_datetime + dt.timedelta(seconds=rel_start)
        real_end_dt   = base_datetime + dt.timedelta(seconds=rel_end)

        normalized.append({
            "start_time": real_start_dt.strftime("%H:%M:%S"),
            "end_time":   real_end_dt.strftime("%H:%M:%S"),
            "text":       seg.get("text", "").strip(),
            "speaker":    "UNKNOWN",
            "rel_start":  rel_start,
        })

    return normalized

def get_chunk_start_datetime(audio_chunk_file: str) -> datetime:
    """
    Resolve chunk start wall-clock time from filename.
    Expected filename format: recording_YYYYMMDD_HHMMSS.wav
    Falls back to file ctime if parsing fails.
    """
    chunk_name = Path(audio_chunk_file).name
    match = re.search(r"recording_(\d{8}_\d{6})", chunk_name)
    if match:
        try:
            parsed_start = datetime.strptime(match.group(1), "%Y%m%d_%H%M%S")
            log.info(
                f"Chunk timestamp parsed from filename: {parsed_start.strftime('%Y-%m-%d %H:%M:%S')}"
            )
            return parsed_start
        except ValueError:
            log.warning(
                f"Could not parse timestamp token in chunk filename: {chunk_name}"
            )

    file_stat = os.stat(audio_chunk_file)
    fallback_start = datetime.fromtimestamp(file_stat.st_ctime)
    log.warning(
        "Falling back to filesystem ctime for chunk start anchoring; consider using recording_YYYYMMDD_HHMMSS.wav naming."
    )
    return fallback_start


def load_fine_tuned_whisper(model_path: str):
    """Loads the fine-tuned Whisper LoRA model via HF Pipeline."""
    global _WHISPER_PIPE, _WHISPER_PIPE_MODEL_PATH

    resolved_model_path = str(Path(model_path).resolve())
    if _WHISPER_PIPE is not None and _WHISPER_PIPE_MODEL_PATH == resolved_model_path:
        return _WHISPER_PIPE

    log.info(f"Loading Fine-Tuned Whisper model (FP16 optimized) from: {model_path}")
    try:
        cuda_available = torch.cuda.is_available()
        dtype = torch.float16 if cuda_available else torch.float32

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
        if cuda_available:
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
            device=0 if cuda_available else -1,
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
        log.info("Fine-tuned Whisper pipeline loaded successfully on GPU")
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
    
    log.info(f"Resolved model path: {model_path}")  # add this so you can verify

    # Memory Check
    available_gb = psutil.virtual_memory().available / (1024**3)
    if available_gb < 1.5:
        log.warning(f"Low memory detected ({available_gb:.2f}GB). Forcing cache flush.")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        gc.collect()

    recording_start_time = get_chunk_start_datetime(audio_chunk_file)
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

    normalized_result = normalize_whisper_segments(
        verified_segments,
        base_datetime=recording_start_time,
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
