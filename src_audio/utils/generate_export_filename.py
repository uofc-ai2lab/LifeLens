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
    
    # For non-transcript services, we are given a transcript file
    existing_metadata = None
    if service == "denoise":
        existing_metadata = search_metadata(
            "audio_chunk_path",
            input_file_path
        )
    # For de-noising service, we are given a denoised audio file
    elif service == "transcript": 
        existing_metadata = search_metadata(
            "denoised_audio_path",
            input_file_path
        )
    else:
        existing_metadata = search_metadata("transcript_filename", input_file_path)
        
    # If metadata exists and has an associated chunk audio, rename output accordingly
    if existing_metadata:
        print(f"EXIST - {existing_metadata}")
        if existing_metadata.chunk_audio_path: 
            output_filename = f"{timestamp}_{service}_{existing_metadata.chunk_audio_path.parent.stem}.csv"

    # Update metadata
    create_update_metadata(input_file_path, service, output_file_path)

    # Re-fetch metadata to ensure update succeeded
    check_existing_metadata = None
    if service == "denoise":
        check_existing_metadata = search_metadata(
            "audio_chunk_path",
            input_file_path
        )
    # For de-noising service, we are given a denoised audio file
    elif service == "transcript": 
        check_existing_metadata = search_metadata(
            "denoised_audio_path",
            input_file_path
        )
    else:
        check_existing_metadata = search_metadata("transcript_path", input_file_path)
    
    if check_existing_metadata is None:
        raise RuntimeError("Metadata update failed")

    return output_file_path