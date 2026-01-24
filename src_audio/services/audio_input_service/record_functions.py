import subprocess
from config.audio_settings import AUDIO_DIR    
import os
import time
import shutil

# Path to the directory where the recording is saved
recording_dir = "/home/capstone/recordings"
signal_file = os.path.join(recording_dir, "recording_done.flag")

# Target directory where the next service expects the file
target_dir = AUDIO_DIR

def wait_for_recording():
    # Path to the script
    script_path = '/home/capstone/LifeLens/src_audio/services/audio_input_service/create_audio.sh'

    # Run the script
    subprocess.run([script_path])

    # Wait for the signal file to appear (indicating the recording is done)
    while not os.path.exists(signal_file):
        print("Waiting for recording to finish...")
        time.sleep(5)  # Check every 5 seconds

    # Signal file exists, recording is done
    print("Recording complete. Processing the file...")

    # Copy all .wav files from recording_dir (if multiple, copy them all; if one, proceed as before)
    wav_files = [f for f in os.listdir(recording_dir)
                 if f.lower().endswith('.wav') and os.path.isfile(os.path.join(recording_dir, f))]

    if not wav_files:
        print("No .wav files found in recording_dir.")
        return

    if len(wav_files) == 1:
        recording_path = os.path.join(recording_dir, wav_files[0])
    else:
        for fname in wav_files:
            path = os.path.join(recording_dir, fname)
            print(f"Copying recording: {path}")
            copy_to_target(path)
        return

    print(f"Using recording: {recording_path}")

    # Copy the file to the target directory for the next service
    copy_to_target(recording_path)


def copy_to_target(recording_path):
    """
    Copy the recorded file to the next service's expected directory.
    """
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)  # Create the target directory if it doesn't exist

    # Define the path where the file will be copied
    target_path = os.path.join(target_dir, os.path.basename(recording_path))

    try:
        # Copy the file to the target directory
        shutil.copy2(recording_path, target_path)
        print(f"File successfully copied to {target_path}")
    except Exception as e:
        print(f"Error copying file: {e}")
