import subprocess
from config.audio_settings import AUDIO_DIR, create_parent_audio_dir
import os
import time
import shutil
import signal
import asyncio
from src_audio.domain.constants import MAX_RECORD_SECONDS, RECORDING_DIR, SIGNAL_FILE, ARECORD_DEVICE

def _ensure_recording_dir():
    """Ensure the recording directory exists."""
    os.makedirs(RECORDING_DIR, exist_ok=True)


def _write_signal_file():
    """Write signal file to indicate recording completion."""
    _ensure_recording_dir()
    with open(SIGNAL_FILE, "w", encoding="utf-8") as flag:
        flag.write(f"done:{time.time()}")


def _clear_stale_signal_file():
    """Remove stale signal file to prevent false positives."""
    if os.path.exists(SIGNAL_FILE):
        os.remove(SIGNAL_FILE)


def start_recording():
    """
    Start recording using arecord command.
    
    Returns:
        tuple: (subprocess.Popen, str) - process and output file path
    """
    _ensure_recording_dir()
    _clear_stale_signal_file()

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    output_file = os.path.join(RECORDING_DIR, f"recording_{timestamp}.wav")

    print(f"Starting recording to {output_file}")
    process = subprocess.Popen(
        [
            "arecord",
            "-D", ARECORD_DEVICE,
            "-f", "S16_LE",
            "-r", "16000",
            "-c", "6",
            output_file
        ],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE
    )

    return process, output_file


def stop_recording(process):
    """Stop the recording process with SIGINT."""
    print("Stopping recording...")
    process.send_signal(signal.SIGINT)


def _get_wav_files():
    """
    Get all WAV files in recording directory.
    
    Returns:
        list: Paths to all .wav files
    """
    return [
        os.path.join(RECORDING_DIR, f)
        for f in os.listdir(RECORDING_DIR)
        if f.lower().endswith(".wav") and os.path.isfile(os.path.join(RECORDING_DIR, f))
    ]


def wait_for_recording(output_file=None):
    """
    Wait for recording completion and copy files to target directory.
    
    Args:
        output_file (str, optional): Specific file to copy. If None, copies all WAV files.
    """
    # Wait for signal file
    while not os.path.exists(SIGNAL_FILE):
        print("Waiting for recording to finish...")
        time.sleep(5)

    print("Recording complete. Processing the file...")

    # Handle specific file
    if output_file is not None:
        if os.path.isfile(output_file):
            print(f"Copying specified recording: {output_file}")
            copy_to_target(output_file)
        else:
            print(f"Specified output file does not exist: {output_file}")
        return

    # Handle all WAV files
    wav_files = _get_wav_files()
    if not wav_files:
        print("No .wav files found in recording directory.")
        return

    for path in wav_files:
        print(f"Copying recording: {path}")
        copy_to_target(path)


def copy_to_target(recording_path):
    """
    Copy recorded file to target directory for next service.
    
    Args:
        recording_path (str): Path to recording file
    """
    os.makedirs(AUDIO_DIR, exist_ok=True)

    target_path = os.path.join(AUDIO_DIR, os.path.basename(recording_path))
    print(f"Copying {recording_path} to {target_path}...")

    try:
        shutil.copy2(recording_path, target_path)
        print(f"File successfully copied to {target_path}")
        create_parent_audio_dir(target_path)
    except Exception as e:
        print(f"Error copying file: {e}")


async def run_recording_service():
    """
    Start recording with manual stop (ENTER) or 5-minute timeout.
    
    Recording is automatically saved and pipeline continues.
    """
    process, output_file = start_recording()

    print("\nRecording started.")
    print("→ Press ENTER to stop recording and continue pipeline")
    print("→ Auto-stop after 5 minutes")

    loop = asyncio.get_running_loop()
    enter_future = loop.run_in_executor(None, input)
    timeout_future = asyncio.sleep(MAX_RECORD_SECONDS)

    done, pending = await asyncio.wait(
        [enter_future, timeout_future],
        return_when=asyncio.FIRST_COMPLETED
    )

    # Cancel pending tasks
    for task in pending:
        task.cancel()

    print("Manual stop requested." if enter_future in done else "Auto-stop reached (5 minutes).")

    stop_recording(process)
    process.wait()
    _write_signal_file()
    wait_for_recording(output_file)