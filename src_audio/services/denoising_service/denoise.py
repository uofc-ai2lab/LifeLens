import noisereduce as nr
from scipy.io import wavfile
import numpy as np
import os
from pathlib import Path
from config.audio_settings import AUDIO_FILES_DICT, AUDIO_DIR, create_parent_audio_dir


def load_audio(file_path: str):
    """Load and prepare audio file for noise reduction"""
    audio_location = AUDIO_DIR / Path(file_path).name # find the parent audio file in audio_files dir
    
    if not os.path.exists(audio_location):
        print(f"File {audio_location} not found!")
        return
    
    # skip junk files (.DS_Store, etc.)
    if audio_location.suffix.lower() != ".wav":
        return
    # Read the audio file
    sample_rate, audio_data = wavfile.read(audio_location)
    
    # Convert stereo to mono by averaging channels
    if len(audio_data.shape) > 1:
        audio_data = np.mean(audio_data, axis=1)
    
    # Normalize audio to prevent distortion
    audio_data = audio_data.astype(np.float32)
    if np.max(np.abs(audio_data)) > 0:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    print(f"Loaded {len(audio_data)/sample_rate:.1f} seconds of audio at {sample_rate}Hz")
    return sample_rate, audio_data

def denoise(input_file, output_file): 
   """Remove noise from audio file and save the cleaned version"""
   
   # Load the original audio
   sample_rate, audio_data = load_audio(input_file)
   
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
   create_parent_audio_dir(output_file)
   wavfile.write(output_file, sample_rate, reduced_noise)
   print(f"Cleaned audio saved as {output_file}")
   
async def run_denoise_service():
    """Main function to run the de-noising service"""
    print(f"Starting de-noising service for {AUDIO_FILES_DICT}..\n")
    for parent_audio in AUDIO_FILES_DICT.keys(): 
        cleaned_file = f"cleaned_{Path(AUDIO_DIR / parent_audio).name}"
        denoise(parent_audio, cleaned_file)
