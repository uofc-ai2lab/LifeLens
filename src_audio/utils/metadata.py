import json
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
import pandas as pd
from src_audio.domain.constants import INTER_COLUMNS, MED_COLUMNS
from config.audio_settings import TRANSCRIPT_FILES_LIST, TRANSCRIPT_DIR, METADATA_JSON_PATH, AUDIO_FILES_LIST, OUTPUT_DIR, MEANING_DIR
from src_audio.domain.entities import AUDIO_PIPELINE_METADATA, AudioFileMetaData

def _write_metadata_json():
    """
    Persist the in-memory pipeline metadata to disk as JSON (the single write-point)
    """
    
    METADATA_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [asdict(m) for m in AUDIO_PIPELINE_METADATA],
            f,
            indent=2
        )
        
def setup_metadata():
    """
    Initialize pipeline metadata from existing audio and transcript files.
    Clears in-memory metadata and creates a new metadata JSON file based on files found at startup.
    """
    global AUDIO_PIPELINE_METADATA
    AUDIO_PIPELINE_METADATA.clear()

    # 1. Seed audio files
    for audio in AUDIO_FILES_LIST:
        AUDIO_PIPELINE_METADATA.append(
            AudioFileMetaData(
                audio_file=Path(audio).name,
                transcript_filename=None,
                medication_filename=None,
                intervention_filename=None,
                semantic_filename=None,
            )
        )

    # 2. Seed transcript files (may not have audio)
    for transcript in TRANSCRIPT_FILES_LIST:
        name = Path(transcript).name
        if not search_metadata("transcript_filename", name):
            AUDIO_PIPELINE_METADATA.append(
                AudioFileMetaData(
                    audio_file=None,
                    transcript_filename=name,
                    medication_filename=None,
                    intervention_filename=None,
                    semantic_filename=None,
                )
            )

    _write_metadata_json()

def search_metadata(field_name, input_file):
    """
    Find and return a metadata entry by matching a specific field value.
    Returns None if no match is found.
    """
    
    existing = next((m for m in AUDIO_PIPELINE_METADATA if getattr(m, field_name, None) == input_file), None)
    if existing:
        return existing
    else:
        return None

def create_update_metadata(input_filename, service, output_filename):
    """
    Create or update metadata when a service generates an output file.

    Updates in-memory pipeline state and persists changes to the
    metadata JSON file.
    """
    
    global AUDIO_PIPELINE_METADATA
    
    existing_audio = search_metadata("audio_file",input_filename)
    existing_transc = None
    
    # audio exists and we are adding transcription file (likely shouldn't happen) or output file
    if service == "transcript":
        if existing_audio:
            existing_audio.transcript_filename=Path(output_filename).name
            TRANSCRIPT_FILES_LIST.append(TRANSCRIPT_DIR / Path(output_filename))
        # audio doesn't exist, new entry created
        elif existing_audio is None:
            AUDIO_PIPELINE_METADATA.append(
                AudioFileMetaData(
                    audio_file=Path(input_filename).name, 
                    transcript_filename=Path(output_filename).name))
            TRANSCRIPT_FILES_LIST.append(TRANSCRIPT_DIR / Path(output_filename))
    
    else: # if not a transcription service, check for transcript filename 
        existing_transc = search_metadata("transcript_filename",input_filename)

        if existing_transc:
            if service == "medX":
                existing_transc.medication_filename=Path(output_filename).name
            elif service == "intervention":
                existing_transc.intervention_filename=Path(output_filename).name
            elif service == "semantic":
                existing_transc.semantic_filename=Path(output_filename).name
            else:
                raise ValueError(f"Unknown service type: {service}")
        
        else:
            # Transcript not registered yet -> create new entry
            print(f"Warning: No existing transcript found for '{input_filename}', creating new entry for {service}")
            AUDIO_PIPELINE_METADATA.append(
                AudioFileMetaData(
                    audio_file=None,
                    transcript_filename=Path(input_filename).name,
                    medication_filename=Path(output_filename).name if service == "medX" else None,
                    intervention_filename=Path(output_filename).name if service == "intervention" else None,
                    semantic_filename=Path(output_filename).name if service == "semantic" else None
                )
            )
    
    _write_metadata_json()
 
def finalize_metadata():
    """
    Finalize pipeline outputs by combining medication and intervention
    results into a single CSV per audio/transcript.
    """
    
    for meta in AUDIO_PIPELINE_METADATA:
        if not meta.medication_filename or not meta.intervention_filename:
            continue
        
        med_path = MEANING_DIR / Path(meta.medication_filename)
        inter_path = MEANING_DIR / Path(meta.intervention_filename)

        if not med_path.exists() or not inter_path.exists():
            continue
            
        try:
            med_df = pd.read_csv(med_path)
        except Exception as e:
            print(f"Failed to read Medications CSV at {med_path}: {e}")
            
        try:
            inter_df = pd.read_csv(inter_path)
        except Exception as e:
            print(f"Failed to read Interventions CSV at {inter_path}: {e}")
        
        all_columns = []
        for col in INTER_COLUMNS + MED_COLUMNS:
            if col not in all_columns:
                all_columns.append(col)
                if col not in med_df:
                    med_df[col] = None
                if col not in inter_df:
                    inter_df[col] = None

        
        combined_df = pd.concat(
            [med_df[all_columns], inter_df[all_columns]],
            ignore_index=True
        )

        combined_df.sort_values("start_time", inplace=True, ignore_index=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = meta.audio_file or meta.transcript_filename
        full_output_filename = f"{timestamp}_final_{Path(base_name).stem}.csv"
        full_output_path = OUTPUT_DIR / full_output_filename
        full_output_path.parent.mkdir(parents=True, exist_ok=True)

        combined_df.to_csv(full_output_path, index=False)

        meta.output_file = full_output_filename

    _write_metadata_json()