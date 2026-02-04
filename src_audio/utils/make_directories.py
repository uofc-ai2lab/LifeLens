from pathlib import Path
from config.audio_settings import AUDIO_DIR

def create_parent_audio_dir(parent_audio):
    parent_audio_dir = AUDIO_DIR / Path(parent_audio).stem
    parent_audio_dir.mkdir(parents=True, exist_ok=True)

    
def create_chunk_dir(parent_audio_file: str | Path, chunk_idx: int) -> Path:
    parent_audio_file = Path(parent_audio_file)

    chunk_filename = (
        f"{parent_audio_file.stem}_chunk_{chunk_idx}"
        f"{parent_audio_file.suffix}"
    )

    chunk_audio_dir = (
        parent_audio_file.with_suffix("") / Path(chunk_filename).stem
    )

    chunk_audio_dir.mkdir(parents=True, exist_ok=True)

    return chunk_audio_dir
