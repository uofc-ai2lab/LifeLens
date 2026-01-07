from datetime import datetime
from config.settings import TRANSCRIPT_FILES_LIST, TRANSCRIPT_DIR, METADATA_JSON_PATH, AUDIO_FILES_LIST
from src.entities import AUDIO_PIPELINE_METADATA, AudioFileMetaData
from pathlib import Path
import json
from dataclasses import asdict

def _write_metadata_json():
    METADATA_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(m) for m in AUDIO_PIPELINE_METADATA],
            f,
            indent=2
        )
        
def setup_metadata():
    """
    Initialize metadata from existing audio + transcript files.
    Always creates a NEW metadata JSON file.
    """
    global AUDIO_PIPELINE_METADATA
    AUDIO_PIPELINE_METADATA.clear()

    # 1. Seed audio files
    for audio in AUDIO_FILES_LIST:
        AUDIO_PIPELINE_METADATA.append(
            AudioFileMetaData(
                audio_file=Path(audio).stem,
                transcript_filename=None,
                medication_filename=None,
                intervention_filename=None,
                semantic_filename=None,
            )
        )

    # 2. Seed transcript files (may not have audio)
    for transcript in TRANSCRIPT_FILES_LIST:
        stem = Path(transcript).stem
        if not search_metadata("transcript_filename", stem):
            AUDIO_PIPELINE_METADATA.append(
                AudioFileMetaData(
                    audio_file=None,
                    transcript_filename=stem,
                    medication_filename=None,
                    intervention_filename=None,
                    semantic_filename=None,
                )
            )

    _write_metadata_json()

def search_metadata(field_name, input_file):
    """
    Look up metadata by matching the transcript_input_file of the transcript filename.
    """
    
    existing = next((m for m in AUDIO_PIPELINE_METADATA if getattr(m, field_name, None) == input_file), None)
    if existing:
        return existing
    else:
        return None

def create_and_update_metadata(input_filename, service, output_filename):
    """
    Add or update metadata for an audio file in the global AUDIO_PIPELINE_METADATA list.

    Args:
        input_filename (str): The stem of the audio file (e.g., "trauma_simulation")
        service (str): One of the audio microservices
    """
    
    global AUDIO_PIPELINE_METADATA
    
    existing_audio = search_metadata("audio_file",input_filename)
    existing_transc = None
    
    if service == "transcript":
        # audio exists and we are adding transcription file (likely shouldn't happen)
        if existing_audio:
            existing_audio.transcript_filename=Path(output_filename).stem
            TRANSCRIPT_FILES_LIST.append(TRANSCRIPT_DIR / Path(output_filename))
        # audio doesn't exist, new entry created
        elif existing_audio is None:
            AUDIO_PIPELINE_METADATA.append(
                AudioFileMetaData(
                    audio_file=Path(input_filename).stem, 
                    transcript_filename=Path(output_filename).stem))
            TRANSCRIPT_FILES_LIST.append(TRANSCRIPT_DIR / Path(output_filename))
    
    else: # if not a transcription service, check for transcript filename 
        existing_transc = search_metadata("transcript_filename",input_filename)

        if existing_transc:
            if service == "medX":
                existing_transc.medication_filename=Path(output_filename).stem
            elif service == "intervention":
                existing_transc.intervention_filename=Path(output_filename).stem
            elif service == "semantic":
                existing_transc.semantic_filename=Path(output_filename).stem
            else:
                raise ValueError(f"1: Unknown service type: {service}")
        
        else:
            # Transcript not registered yet -> create new entry
            print(f"Warning: No existing transcript found for '{input_filename}', creating new entry for {service}")
            AUDIO_PIPELINE_METADATA.append(
                AudioFileMetaData(
                    audio_file=None,
                    transcript_filename=Path(input_filename).stem,
                    medication_filename=Path(output_filename).stem if service == "medX" else None,
                    intervention_filename=Path(output_filename).stem if service == "intervention" else None,
                    semantic_filename=Path(output_filename).stem if service == "semantic" else None,
                )
            )
    
    _write_metadata_json()
 