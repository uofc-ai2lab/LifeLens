from whisper_trt import load_trt_model
import os
import dotenv
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
    dotenv.load_dotenv()
    
    device = os.getenv("DEVICE", "cuda")
    audio_from_env = os.getenv("AUDIO_FILE_PATH")
    audio_file_list = audio_from_env.split("/") if audio_from_env else []
    audio_file = os.path.join(*audio_file_list)
    
    model_size = os.getenv("MODEL_SIZE_TRT", "base.en")  # WhisperTRT models: tiny.en, base.en, small.en, medium.en
    model_cache_path = os.getenv("WHISPER_TRT_CACHE", None)  # Optional custom cache path
    output_dir = os.getenv("OUTPUT_DIR", "./output")
    
    use_offline_models = int(os.getenv("USE_OFFLINE_MODELS", "0"))
    hugging_face_token = os.getenv("HUGGING_FACE_TOKEN", "")
    
    print(bcolors.HEADER + 
        f"Using WhisperTRT with device {device}\n" +
        f"Model: {model_size}\n\n" + bcolors.ENDC
    )
    
    print(bcolors.OKGREEN + f"Transcribing {audio_file}...\n" + bcolors.ENDC)
    
    return device, audio_file, model_size, model_cache_path, output_dir, use_offline_models, hugging_face_token

async def transcribe_audio(audio_file: str, model_size: str, model_cache_path: str = None):
    """Transcribe audio using WhisperTRT"""
    print(bcolors.OKGREEN + f"Loading {model_size} model...\n" + bcolors.ENDC)
    print(bcolors.WARNING + "Note: First run will build TensorRT engine (takes 2-5 minutes)\n" + bcolors.ENDC)
    
    # Load model (first time builds TensorRT engine and caches it)
    if model_cache_path:
        model = load_trt_model(model_size, path=model_cache_path)
    else:
        model = load_trt_model(model_size)  # Uses default cache: ~/.cache/whisper_trt/
    
    print(bcolors.OKGREEN + "Transcribing audio...\n" + bcolors.ENDC)
    transcribe_start = datetime.now()
    
    # Transcribe (returns dict with 'text' and 'segments')
    result = model.transcribe(audio_file)
    
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
    
    # WhisperTRT result has 'segments' key with list of segment dicts
    # Each segment has: start, end, text
    if 'segments' in result and result['segments']:
        segments = result['segments']
    else:
        # If no segments, create one from full text
        segments = [{'start': 0.0, 'end': 0.0, 'text': result.get('text', '')}]
    
    # Assign speakers to transcription segments
    for seg in segments:
        seg_start = seg.get("start", 0.0)
        seg_end = seg.get("end", 0.0)
        seg_mid = (seg_start + seg_end) / 2
        seg["speaker"] = "UNKNOWN"
        
        for diar in diarize_segments:
            if diar["start"] <= seg_mid <= diar["end"]:
                seg["speaker"] = diar["speaker"]
                break
    
    # Update result with speaker-assigned segments
    result['segments'] = segments
    
    diarize_end = datetime.now()
    print(bcolors.OKGREEN + f"Diarization completed, took {diarize_end - diarize_start}.\n" + bcolors.ENDC)
    
    return result

def export_results(result: dict, output_dir: str = "output", filename: str = "transcript"):
    """Export results to CSV format with start/end timestamps, text, and speaker labels"""
    os.makedirs(output_dir, exist_ok=True)
    csv_path = f"{output_dir}/{filename}.csv"
    df_data = []
    
    segments = result.get('segments', [])
    
    for seg in segments:
        speaker = seg.get("speaker", "UNKNOWN")
        start_time = format_timestamp(seg.get("start", 0.0))
        end_time = format_timestamp(seg.get("end", 0.0))
        text = seg.get("text", "")
        
        df_data.append({
            "start": start_time,
            "end": end_time,
            "text": text.strip() if text else "",
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

async def run_whispertrt():
    """Main runner function for WhisperTRT transcription with diarization"""
    # Load environment variables
    device, audio_file, model_size, model_cache_path, output_dir, use_offline_models, hugging_face_token = load_env()
    
    # Track total time
    total_start = datetime.now()
    
    # Transcribe
    transcribe_start = datetime.now()
    result = await transcribe_audio(audio_file, model_size, model_cache_path)
    transcribe_end = datetime.now()
    
    # Diarization
    print(bcolors.OKGREEN + "\nDiarizing with pyannote...\n" + bcolors.ENDC)
    diarize_start = datetime.now()
    result = await assign_speakers(device, audio_file, result, use_offline_models, hugging_face_token)
    diarize_end = datetime.now()
    
    # Export
    print(bcolors.OKGREEN + "\nFinished diarization, exporting results...\n" + bcolors.ENDC)
    export_start = datetime.now()
    export_results(result, output_dir=output_dir, filename="transcript_trt")
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