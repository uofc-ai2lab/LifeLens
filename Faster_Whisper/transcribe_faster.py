from faster_whisper import WhisperModel
import os
from dotenv import load_dotenv
from datetime import datetime
from pyannote.audio import Pipeline
import pandas as pd
import torch

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'

def load_env() -> tuple:
    """Load environment variables from .env file"""
    load_dotenv()
    
    device = os.getenv("DEVICE", "cuda")
    audio_from_env = os.getenv("AUDIO_FILE_PATH")
    audio_file_list = audio_from_env.split("/") if audio_from_env else []
    audio_file = os.path.join(*audio_file_list)
    
    model_size = os.getenv("MODEL_SIZE", "base")
    compute_type = os.getenv("COMPUTE_TYPE", "float16")  # float16 or int8
    output_dir = os.getenv("OUTPUT_DIR", "./output")
    
    use_offline_models = int(os.getenv("USE_OFFLINE_MODELS", "0"))
    hugging_face_token = os.getenv("HUGGING_FACE_TOKEN", "")
    
    print(bcolors.HEADER + 
        f"Using device {device} with compute type {compute_type} for faster-whisper.\n" +
        f"Model: {model_size}\n\n" + bcolors.ENDC
    )
    
    print(bcolors.OKGREEN + f"Transcribing {audio_file}...\n" + bcolors.ENDC)
    
    return device, audio_file, model_size, compute_type, output_dir, use_offline_models, hugging_face_token

async def transcribe_audio(device: str, audio_file: str, model_size: str, compute_type: str):
    """Transcribe audio using faster-whisper"""
    print(bcolors.OKGREEN + f"Loading {model_size} model...\n" + bcolors.ENDC)
    
    # Load model
    model = WhisperModel(
        model_size,
        device=device,
        compute_type=compute_type
    )
    
    print(bcolors.OKGREEN + "Transcribing audio...\n" + bcolors.ENDC)
    transcribe_start = datetime.now()
    
    # Transcribe (returns generator)
    segments, info = model.transcribe(
        audio_file,
        beam_size=5,
        language="en",  # or None for auto-detect
        vad_filter=True,  # Voice Activity Detection
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    # Convert generator to list and format as dict
    result = {
        "segments": [],
        "language": info.language
    }
    
    for segment in segments:
        result["segments"].append({
            "start": segment.start,
            "end": segment.end,
            "text": segment.text.strip()
        })
    
    transcribe_end = datetime.now()
    print(bcolors.OKGREEN + f"Transcription completed, took {transcribe_end - transcribe_start}.\n" + bcolors.ENDC)
    
    return result

async def assign_speakers(device: str, audio_file: str, result: dict, use_offline_models: int, hugging_face_token: str):
    """Assign speakers using pyannote"""
    print(bcolors.OKBLUE + "Running diarization...\n" + bcolors.ENDC)
    diarize_start = datetime.now()
    
    if use_offline_models:
        print(bcolors.OKBLUE + "Using offline pyannote models for diarization.\n" + bcolors.ENDC)
        pyannote_cache_dir = os.getenv("PYANNOTE_CACHE_DIR", "./pyannote_models")
        os.environ['HF_HUB_OFFLINE'] = '1'
        
        diarize_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            cache_dir=pyannote_cache_dir
        )
    else:
        print(bcolors.OKBLUE + "Using online pyannote models for diarization.\n" + bcolors.ENDC)
        if not hugging_face_token:
            raise ValueError("HUGGING_FACE_TOKEN environment variable not set.")
        
        diarize_model = Pipeline.from_pretrained(
            "pyannote/speaker-diarization-3.1",
            use_auth_token=hugging_face_token
        )
    
    # Move to GPU if available
    if device == "cuda":
        diarize_model.to(torch.device("cuda"))
    
    # Run diarization
    diarization = diarize_model(audio_file)
    
    # Convert pyannote output to segments
    diarize_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        diarize_segments.append({
            'start': turn.start,
            'end': turn.end,
            'speaker': speaker
        })
    
    # Assign speakers to transcription segments
    for seg in result["segments"]:
        seg_mid = (seg["start"] + seg["end"]) / 2
        seg["speaker"] = "UNKNOWN"
        
        for diar in diarize_segments:
            if diar["start"] <= seg_mid <= diar["end"]:
                seg["speaker"] = diar["speaker"]
                break
    
    diarize_end = datetime.now()
    print(bcolors.OKGREEN + f"Diarization completed, took {diarize_end - diarize_start}.\n" + bcolors.ENDC)
    
    return result

def export_results(result: dict, output_dir: str = "output", filename: str = "transcript"):
    """Export results to CSV format with start/end timestamps, text, and speaker labels"""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = f"{output_dir}/{filename}.csv"
    df_data = []
    
    for seg in result["segments"]:
        speaker = seg.get("speaker", "UNKNOWN")
        start_time = format_timestamp(seg["start"])
        end_time = format_timestamp(seg["end"])
        df_data.append({
            "start": start_time,
            "end": end_time,
            "text": seg["text"],
            "speaker": speaker,
        })
    
    pd.DataFrame(df_data).to_csv(csv_path, index=False)
    
    print(bcolors.OKCYAN + f"\nResults exported to '{output_dir}/' directory:" + bcolors.ENDC)
    print(bcolors.OKGREEN + f"   ✓ {filename}.csv (timestamps + text)" + bcolors.ENDC)

def format_timestamp(seconds: float) -> str:
    """Convert seconds to timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

async def run_faster_whisper():
    """Main runner function for faster-whisper transcription with diarization"""
    # Load environment variables
    device, audio_file, model_size, compute_type, output_dir, use_offline_models, hugging_face_token = load_env()
    
    # Track total time
    total_start = datetime.now()
    
    # Transcribe
    transcribe_start = datetime.now()
    result = await transcribe_audio(device, audio_file, model_size, compute_type)
    transcribe_end = datetime.now()
    
    # Diarization
    print(bcolors.OKGREEN + "\nDiarizing with pyannote...\n" + bcolors.ENDC)
    diarize_start = datetime.now()
    result = await assign_speakers(device, audio_file, result, use_offline_models, hugging_face_token)
    diarize_end = datetime.now()
    
    # Export
    print(bcolors.OKGREEN + "\nFinished diarization, exporting results...\n" + bcolors.ENDC)
    export_start = datetime.now()
    export_results(result, output_dir=output_dir, filename="transcript")
    export_end = datetime.now()
    
    # Print timing summary
    time_for_transcription = transcribe_end - transcribe_start
    time_for_diarization = diarize_end - diarize_start
    time_for_export = export_end - export_start
    time_total = export_end - total_start
    
    print(bcolors.OKBLUE + f"\n\nTime taken: {time_total.seconds // 60} minutes and {time_total.seconds % 60} seconds" + bcolors.ENDC)
    print(bcolors.OKBLUE + f" - Transcription time: {time_for_transcription.seconds // 60} minutes and {time_for_transcription.seconds % 60} seconds" + bcolors.ENDC)
    print(bcolors.OKBLUE + f" - Diarization time: {time_for_diarization.seconds // 60} minutes and {time_for_diarization.seconds % 60} seconds" + bcolors.ENDC)
    print(bcolors.OKBLUE + f" - Export time: {time_for_export.seconds // 60} minutes and {time_for_export.seconds % 60} seconds" + bcolors.ENDC)