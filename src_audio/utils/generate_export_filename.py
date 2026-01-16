from pathlib import Path
from datetime import datetime
from src_audio.utils.metadata import create_update_metadata, search_metadata

def generate_export_filename(
    input_filename: str,
    service: str = "",
    ) -> str:
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    full_output_filename = f"{timestamp}_{service}_{Path(input_filename).stem}.csv"
    
    existing_metadata = None
    if service != "transcript":
        existing_metadata=search_metadata("transcript_filename",input_filename) # check if transcript's metadata exists
    
    if existing_metadata:
        if existing_metadata.audio_file is not None: #if it exists, rename the output file
            full_output_filename = f"{timestamp}_{service}_{Path(existing_metadata.audio_file).stem}.csv"

    create_update_metadata(input_filename, service, full_output_filename)
    
    # re-fetch to ensure updated object
    metadata = search_metadata(
        "transcript_filename" if service != "transcript" else "audio_file",
        input_filename
    )

    if metadata is None:
        raise RuntimeError("Metadata update failed")

    return full_output_filename
