import subprocess
from config.audio_settings import AUDIO_DIR, create_parent_audio_dir
import os
import time
import shutil
import signal
import asyncio
from src_audio.domain.constants import MAX_RECORD_SECONDS

# Path to the directory where the recording is saved
recording_dir = "/home/capstone/recordings"
signal_file = os.path.join(recording_dir, "recording_done.flag")


def _ensure_recording_dir():
    """Make sure the recording directory exists."""
    os.makedirs(recording_dir, exist_ok=True)


def _write_signal_file():
    """Create the signal file to unblock downstream waiters."""
    _ensure_recording_dir()

    with open(signal_file, "w", encoding="utf-8") as flag:
        flag.write(f"done:{time.time()}")


def _clear_stale_signal_file():
    """Remove an old signal file if present to avoid false positives."""
    if os.path.exists(signal_file):
        os.remove(signal_file)

# Target directory where the next service expects the file
target_dir = AUDIO_DIR
def start_recording():
    """
    Start the recording using `arecord` command. It will record until stopped.
    """
    _ensure_recording_dir()
    _clear_stale_signal_file()

    # Define output file name with timestamp
    output_file = os.path.join(recording_dir, f"recording_{time.strftime('%Y%m%d_%H%M%S')}.wav")

    # Start the arecord process
    print(f"Starting recording to {output_file}")
    process = subprocess.Popen(
        ["arecord", "-D", "hw:CARD=ArrayUAC10,DEV=0", "-f", "S16_LE", "-r", "16000", "-c", "6", output_file],
        stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )

    return process, output_file

def stop_recording(process):
    """
    Stop the recording process.
    """
    print("Stopping recording...")
    # Send a SIGINT to stop the arecord process (equivalent to pressing Ctrl+C)
    process.send_signal(signal.SIGINT)

def wait_for_recording(output_file=None):
    # Wait for the signal file to appear (indicating the recording is done)
    while not os.path.exists(signal_file):
        print("Waiting for recording to finish...")
        time.sleep(5)

    print("Recording complete. Processing the file...")

    # Case 1: output_file was explicitly provided
    if output_file is not None:
        if os.path.isfile(output_file):
            print(f"Copying specified recording: {output_file}")
            copy_to_target(output_file)
        else:
            print(f"Specified output file does not exist: {output_file}")
        return

    # Case 2: output_file is None → copy all .wav files in recording_dir
    wav_files = [
        os.path.join(recording_dir, f)
        for f in os.listdir(recording_dir)
        if f.lower().endswith(".wav")
        and os.path.isfile(os.path.join(recording_dir, f))
    ]

    if not wav_files:
        print("No .wav files found in recording_dir.")
        return

    for path in wav_files:
        print(f"Copying recording: {path}")
        copy_to_target(path)
    

def copy_to_target(recording_path):
    """
    Copy the recorded file to the next service's expected directory.
    """
    # Ensure the target directory exists
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)  # Create the target directory if it doesn't exist

    # Define the path where the file will be copied
    target_path = os.path.join(target_dir, os.path.basename(recording_path))
    print(f"Copying {recording_path} to {target_path}...")

    try:
        # Copy the file to the target directory
        shutil.copy2(recording_path, target_path)
        print(f"File successfully copied to {target_path}")
        create_parent_audio_dir(target_path)  # Update AUDIO_FILES_DICT with the new file

    except Exception as e:
        print(f"Error copying file: {e}")

async def run_recording_service():
    """
    Starts recording and stops on:
    - ENTER keypress
    - 5 minute timeout
    Recording is saved and pipeline continues.

    Args:
        None

    Returns:
        None
    """
    process, output_file = start_recording()

    print("\nRecording started.")
    print("→ Press ENTER to stop recording and continue pipeline")
    print("→ Auto-stop after 5 minutes")

    loop = asyncio.get_running_loop()

    # Future that completes when ENTER is pressed
    enter_future = loop.run_in_executor(None, input)

    # Future that completes after timeout
    timeout_future = asyncio.sleep(MAX_RECORD_SECONDS)

    # Wait for whichever happens first
    done, pending = await asyncio.wait(
        [enter_future, timeout_future],
        return_when=asyncio.FIRST_COMPLETED
    )

    # Cancel whichever future didn't fire
    for task in pending:
        task.cancel()

    if enter_future in done:
        print("Manual stop requested.")
    else:
        print("Auto-stop reached (5 minutes).")

    stop_recording(process)

    # Ensure arecord exits cleanly
    process.wait()

    # Signal downstream waiters that recording has completed
    _write_signal_file()

    wait_for_recording(output_file)