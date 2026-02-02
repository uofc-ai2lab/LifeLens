import noisereduce as nr
from scipy.io import wavfile
import numpy as np
import os
from pathlib import Path
from config.audio_settings import AUDIO_FILES_DICT, AUDIO_DIR, create_parent_audio_dir
from src_audio.utils.metadata import create_update_metadata


def load_audio(input_audio_path: Path):
    """Load and prepare audio file for noise reduction"""
    
    if not os.path.exists(input_audio_path):
        print(f"File {input_audio_path} not found!")
        return
    
    # skip junk files (.DS_Store, etc.)
    if input_audio_path.suffix.lower() != ".wav":
        print(f"Unsupported file format: {input_audio_path.suffix}")
        return
    # Read the audio file
    sample_rate, audio_data = wavfile.read(str(input_audio_path))
    
    # Convert stereo to mono by averaging channels
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Normalize audio to prevent distortion
    audio_data = audio_data.astype(np.float32)
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    print(f"Loaded {len(audio_data)/sample_rate:.1f} seconds of audio at {sample_rate}Hz")
    return sample_rate, audio_data

def denoise(input_audio_path: Path, output_audio_path: Path): 
   """Remove noise from audio file and save the cleaned version"""
   
   # Load the original audio
   sample_rate, audio_data = load_audio(input_audio_path)
   
   # Apply noise reduction
   reduced_noise = nr.reduce_noise(
       y=audio_data,
       sr=sample_rate,
       prop_decrease=0.8,    # Remove 80% of detected noise
       use_tqdm=True
   )
   
   # Convert back to integer format for saving
   reduced_noise = np.int16(reduced_noise * 32767)
   
   # Save the cleaned audio
   wavfile.write(str(output_audio_path), sample_rate, reduced_noise)
   print(f"Cleaned audio saved as {output_audio_path}\n")
   
async def run_denoise_service():
    """Main function to run the de-noising service"""
    print(f"Starting de-noising service for {list(AUDIO_FILES_DICT.keys())}..\n")
    for parent, chunks in AUDIO_FILES_DICT.items():
        for i, chunk in enumerate(list(chunks)):
            chunk = Path(chunk)
            out = chunk.parent / f"denoised_{chunk.name}"
            try:
                denoise(chunk, out)
                # replace the entry so transcription uses denoised file
                AUDIO_FILES_DICT[parent][i] = out
                # optionally update metadata:
                create_update_metadata(chunk, "denoise", out)
            except Exception as e:
                print(f"Failed denoising {chunk}: {e}")