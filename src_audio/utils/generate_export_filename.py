from pathlib import Path
from datetime import datetime
from src_audio.utils.metadata import create_update_metadata, search_metadata

def generate_export_filename(
    input_file_path: Path,
    service: str = "",
    ) -> str:
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Default output filename (fallback)
    output_filename = f"{service}_{input_file_path.parent.stem}.csv"
    output_file_path = input_file_path.parent / output_filename

    # Decide what string to search our metadata on based on service type
    if service == "denoise":
        metadata_key = "audio_chunk_path" # For de-noising service, we need to look for a chunk audio file
    elif service == "transcript":
        metadata_key = "denoised_audio_path" # For transcription service, we we need to look for a denoised audio file
    else:
        metadata_key = "transcript_path" # For other services, we need to look for a transcript file
    
    existing_metadata = search_metadata(metadata_key, input_file_path)
        
    # NOT SURE WHAT IS HAPPENING HERE: ASK ISHA
    # # If metadata exists and has an associated chunk audio, rename output accordingly
    # if existing_metadata.:
    #     print(f"EXIST - {existing_metadata}")
    #     if existing_metadata.chunk_audio_path: 
    #         output_filename = f"{timestamp}_{service}_{existing_metadata.chunk_audio_path.parent.stem}.csv"

    # Update metadata
    create_update_metadata(input_file_path, service, output_file_path)

    # Re-fetch metadata to ensure update succeeded
    
    if search_metadata(metadata_key, input_file_path) is None:
        raise RuntimeError("Metadata update failed")

    return output_file_path