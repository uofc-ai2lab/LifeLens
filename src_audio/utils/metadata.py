import json
from datetime import datetime
from pathlib import Path
from dataclasses import asdict
import pandas as pd
from src_audio.domain.constants import INTER_COLUMNS, MED_COLUMNS
from config.audio_settings import (
    PROCESSED_AUDIO_DIR, 
    TRANSCRIPT_FILES_LIST, 
    METADATA_JSON_PATH,
    AUDIO_FILES_DICT
)
from src_audio.domain.entities import AUDIO_PIPELINE_METADATA, AudioFileMetaData

def _write_metadata_json():
    """
    Persist the in-memory pipeline metadata to disk as JSON
    """
    def serialize(obj):
        if isinstance(obj, Path):
            return str(obj)
        return obj

    METADATA_JSON_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(METADATA_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [
                {k: serialize(v) for k, v in asdict(m).items()}
                for m in AUDIO_PIPELINE_METADATA
            ],
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
    for parent_audio_path, chunk_files in AUDIO_FILES_DICT.items():
        for chunk_path in chunk_files:
            AUDIO_PIPELINE_METADATA.append(
                AudioFileMetaData(
                    parent_audio_path=parent_audio_path,
                    chunk_audio_path=chunk_path,
                    transcript_path=None,
                    medication_path=None,
                    intervention_path=None,
                    semantic_path=None,
                )
            )

    # 2. Seed transcript files (may not have audio)
    for transcript in TRANSCRIPT_FILES_LIST:
        if not search_metadata("transcript_path", transcript):
            AUDIO_PIPELINE_METADATA.append(
                AudioFileMetaData(
                    parent_audio_path=None,
                    chunk_audio_path=None,
                    transcript_path=transcript,
                    medication_path=None,
                    intervention_path=None,
                    semantic_path=None,
                )
            )

    _write_metadata_json()

def search_metadata(field_name, value):
    """Find metadata entry by field match"""
    return next(
        (m for m in AUDIO_PIPELINE_METADATA if getattr(m, field_name, None) == value), 
        None,
        )

def create_update_metadata(input_file_path, service, output_file):
    """
    Create or update metadata when a service generates an output file.

    Updates in-memory pipeline state and persists changes to the
    metadata JSON file.
    """
    input_file_path = Path(input_file_path)
    output_file = Path(output_file)
    
    existing_audio_meta = search_metadata("chunk_audio_path",input_file_path)
    
    # ────────────────────────── TRANSCRIPTION ──────────────────────────
    if service == "transcript":
        if existing_audio_meta:
            existing_audio_meta.transcript_path=output_file
            
            parent_dir = (
                existing_audio_meta.chunk_audio_path.parent
                if existing_audio_meta.chunk_audio_path
                else PROCESSED_AUDIO_DIR / "unknown_audio"
            )
            
            TRANSCRIPT_FILES_LIST.append(parent_dir / output_file)
       
        else:
            # transcript without known audio
            AUDIO_PIPELINE_METADATA.append(
                AudioFileMetaData(
                    parent_audio_path=None,
                    chunk_audio_path=None,
                    transcript_path=output_file))
            
            # Save transcript in chunk folder if exists, else unknown_audio
            parent_dir = (
                input_file_path.parent
                if input_file_path.parent.exists()
                else PROCESSED_AUDIO_DIR / "unknown_audio"
            )
            
            TRANSCRIPT_FILES_LIST.append(parent_dir / output_file)
# ────────────────────────── DE-NOISING ──────────────────────────
    elif service == "denoise":
    # match on chunk_audio_path
        existing_audio_meta = search_metadata("chunk_audio_path", input_file_path)
        if existing_audio_meta:
            existing_audio_meta.denoised_audio_path = output_file
        else:
            AUDIO_PIPELINE_METADATA.append(
                AudioFileMetaData(
                    parent_audio_path=None,
                    chunk_audio_path=input_file_path,
                    denoised_audio_path=output_file
                )
            )
        _write_metadata_json()
        return
    # ────────────────────────── OTHER SERVICES ──────────────────────────
    else: # if not a transcription service, check for transcript filename 
        existing_transc = search_metadata("transcript_path",input_file_path)

        if existing_transc:
            if service == "medX":
                existing_transc.medication_path=output_file
            elif service == "intervention":
                existing_transc.intervention_path=output_file
            elif service == "semantic":
                existing_transc.semantic_path=output_file
            elif service == "anonymization":
                existing_transc.anonymization_path=output_file
            else:
                raise ValueError(f"Unknown service type: {service}")
        
        else:
            # Transcript not registered yet -> create new entry
            print(f"Warning: No existing transcript found for '{input_file_path}', creating new entry for {service}")
            AUDIO_PIPELINE_METADATA.append(
                AudioFileMetaData(
                    transcript_path=input_file_path,
                    medication_path=output_file if service == "medX" else None,
                    intervention_path=output_file if service == "intervention" else None,
                    semantic_path=output_file if service == "semantic" else None,
                    anonymization_path=output_file if service == "anonymization" else None,
                )
            )
    
    _write_metadata_json()
 
def finalize_metadata():
    """
    Finalize pipeline outputs by combining medication and intervention
    results into a single CSV per audio/transcript.
    """
    
    for meta in AUDIO_PIPELINE_METADATA:
        if not meta.medication_path or not meta.intervention_path:
            continue
        
        med_path = meta.medication_path
        inter_path = meta.intervention_path

        if not med_path.exists() or not inter_path.exists():
            continue
        
        med_df = pd.DataFrame()
        inter_df = pd.DataFrame()
        try:
            med_df = pd.read_csv(med_path)
            if med_df.empty:
                print(f"Warning: Medications CSV is empty → {med_path}")
        except Exception as e:
            print(f"ERROR: Failed to read Medications CSV at {med_path}: {e}")
            
        try:
            inter_df = pd.read_csv(inter_path)
            if inter_df.empty:
                print(f"Warning: Interventions CSV is empty → {inter_path}")
        except Exception as e:
            print(f"ERROR: Failed to read Interventions CSV at {inter_path}: {e}")

        all_columns = []
        for col in INTER_COLUMNS + MED_COLUMNS:
            if col not in all_columns:
                all_columns.append(col)
        
        for df in [med_df, inter_df]:
            for col in all_columns:
                if col not in df.columns:
                    df[col] = None  # Fill missing columns with None

        # Now this is safe:
        combined_df = pd.concat(
            [med_df[all_columns], inter_df[all_columns]],
            ignore_index=True
        )
        
        combined_df.sort_values("start_time", inplace=True, ignore_index=True)

        base_name = meta.chunk_audio_path or meta.transcript_path
        output_name = f"final_{base_name.stem}.csv"
        output_path = base_name.parent / output_name
        output_path.parent.mkdir(parents=True, exist_ok=True)

        combined_df.to_csv(output_path, index=False)

        meta.output_path = output_path

    _write_metadata_json()