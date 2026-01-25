import librosa
import IPython.display as ipd
import soundfile as sf
from pathlib import Path

from config.audio_settings import (
    PROCESSED_AUDIO_DIR,
    AUDIO_FILES_DICT,
    AUDIO_DIR
)

AUDIO_EXTS = {".wav", ".mp3", ".m4a", ".flac"}

def trim_audio(parent_audio_file):
    parent_audio_location = AUDIO_DIR / Path(parent_audio_file).name # find the parent audio file in audio_files dir
    
    # skip junk files (.DS_Store, etc.)
    if parent_audio_location.suffix.lower() not in AUDIO_EXTS:
        return

    audio, sr = librosa.load(parent_audio_location, sr=None)
    chunk_duration = 180  # seconds (3 minutes)
    chunk_size = chunk_duration * sr

    start = 0
    chunk_idx = 0

    while start < len(audio):  # EOF condition
        end = start + chunk_size
        chunk = audio[start:end]  # last chunk may be shorter — that's fine

        chunk_filename = f"{Path(parent_audio_file).stem}_chunk_{chunk_idx}{Path(parent_audio_file).suffix}"
        chunk_audio_dir = parent_audio_file.with_suffix("") / Path(chunk_filename).stem
        chunk_audio_dir.mkdir(parents=True, exist_ok=True)
        chunk_audio_path = chunk_audio_dir / chunk_filename
        
        sf.write(chunk_audio_path, chunk, sr)
        AUDIO_FILES_DICT[parent_audio_file].append(chunk_audio_path)

        start = end
        chunk_idx += 1


async def run_audio_trimming():
    """Async wrapper to run the audio trimming."""
    for parent_audio in AUDIO_FILES_DICT.keys(): trim_audio(parent_audio)
        
        