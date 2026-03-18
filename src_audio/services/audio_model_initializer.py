from datetime import datetime

from config.logger import audio_logger as log
from config.audio_settings import MODEL_PACK
from src_audio.services.transcription_service import transcription_whispertrt as transcription_svc
from src_audio.services.anonymization_service.transcript_anonymization import (
    get_transcript_anonymizer,
)
from src_audio.services.medication_extraction_service.medication_extraction import (
    get_medication_extractor,
)


def initialize_audio_models() -> None:
    """Eagerly load and time all heavy audio models before first chunk."""
    log.header("Initializing audio models...")
    start_overall = datetime.now()

    # Fine-tuned Whisper LoRA (HF pipeline)
    whisper_start = datetime.now()
    try:
        transcription_svc.load_fine_tuned_whisper(
            transcription_svc._DEFAULT_LORA_MODEL_PATH
        )
        whisper_ok = True
    except Exception as e:
        whisper_ok = False
        log.error(f"Failed to initialize Fine-Tuned Whisper model: {e}")
    whisper_elapsed = (datetime.now() - whisper_start).total_seconds()
    log.info(f"Whisper fine-tuned init time: {whisper_elapsed:.1f}s (ok={whisper_ok})")

    # Medication extractor (Med7 spaCy model)
    med7_start = datetime.now()
    try:
        get_medication_extractor()
        med7_ok = True
    except Exception as e:
        med7_ok = False
        log.error(f"Failed to initialize Med7 medication extractor: {e}")
    med7_elapsed = (datetime.now() - med7_start).total_seconds()
    log.info(f"Med7 medication NER init time: {med7_elapsed:.1f}s (ok={med7_ok})")

    # Presidio transcript anonymizer
    anon_start = datetime.now()
    try:
        get_transcript_anonymizer()
        anon_ok = True
    except Exception as e:
        anon_ok = False
        log.error(f"Failed to initialize transcript anonymizer: {e}")
    anon_elapsed = (datetime.now() - anon_start).total_seconds()
    log.info(f"Presidio anonymizer init time: {anon_elapsed:.1f}s (ok={anon_ok})")

    # MedCAT MODEL_PACK (used by intervention extraction)
    if MODEL_PACK is not None:
        log.info("MedCAT MODEL_PACK already loaded and available for interventions")
    else:
        log.warning("MedCAT MODEL_PACK is None; intervention extraction may not run")

    total_elapsed = (datetime.now() - start_overall).total_seconds()
    log.success(f"Audio model initialization completed in {total_elapsed:.1f}s")
