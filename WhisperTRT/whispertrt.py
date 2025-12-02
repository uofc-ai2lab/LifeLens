from whisper_trt import load_trt_model
import os
import dotenv
from datetime import datetime
# from pyannote.audio import Pipeline
import pandas as pd
# import torch

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
    
    # ==================== VERIFICATION STEP 1: CHECK INPUT FILE ====================
    print("\n" + "="*70)
    print(bcolors.OKCYAN + "STEP 1: VERIFYING INPUT AUDIO FILE" + bcolors.ENDC)
    print("="*70)
    
    if not os.path.exists(audio_file):
        print(bcolors.FAIL + f"ERROR: Audio file not found: {audio_file}" + bcolors.ENDC)
        return None
    
    file_size = os.path.getsize(audio_file)
    print(bcolors.OKGREEN + f"Audio file found: {audio_file}" + bcolors.ENDC)
    print(bcolors.OKGREEN + f"File size: {file_size / (1024*1024):.2f} MB ({file_size:,} bytes)" + bcolors.ENDC)
    
    if file_size == 0:
        print(bcolors.FAIL + "ERROR: Audio file is empty!" + bcolors.ENDC)
        return None
    
    # Try to get audio file info
    try:
        import soundfile as sf
        info = sf.info(audio_file)
        print(bcolors.OKGREEN + f"Audio properties:" + bcolors.ENDC)
        print(bcolors.OKGREEN + f"  Duration: {info.duration:.2f} seconds ({info.duration/60:.2f} minutes)" + bcolors.ENDC)
        print(bcolors.OKGREEN + f"  Sample rate: {info.samplerate} Hz" + bcolors.ENDC)
        print(bcolors.OKGREEN + f"  Channels: {info.channels}" + bcolors.ENDC)
    except Exception as e:
        print(bcolors.WARNING + f"WARNING: Could not read audio properties: {e}" + bcolors.ENDC)
        print(bcolors.WARNING + f"  (This might be okay if the file format is supported by Whisper)" + bcolors.ENDC)
    
    # ==================== VERIFICATION STEP 2: LOAD MODEL ====================
    print("\n" + "="*70)
    print(bcolors.OKCYAN + f"STEP 2: LOADING {model_size.upper()} MODEL" + bcolors.ENDC)
    print("="*70)
    
    print(bcolors.OKGREEN + f"Loading {model_size} model..." + bcolors.ENDC)
    print(bcolors.WARNING + "Note: First run will build TensorRT engine (takes 2-5 minutes)\n" + bcolors.ENDC)
    
    try:
        # Load model (first time builds TensorRT engine and caches it)
        if model_cache_path:
            # setup cache file
            print(bcolors.OKBLUE + f"Using custom cache path: {model_cache_path}" + bcolors.ENDC)
            model_file_path = os.path.join(model_cache_path, f"{model_size}.pth")
            print(bcolors.OKBLUE + f"Model file path: {model_file_path}" + bcolors.ENDC)
            model = load_trt_model(model_size, path=model_file_path)
        else:
            print(bcolors.OKBLUE + f"Using default cache: ~/.cache/whisper_trt/" + bcolors.ENDC)
            model = load_trt_model(model_size)  # Uses default cache: ~/.cache/whisper_trt/
        
        print(bcolors.OKGREEN + f"Model loaded successfully" + bcolors.ENDC)
        print(bcolors.OKGREEN + f"  Model type: {type(model)}" + bcolors.ENDC)
    except Exception as e:
        print(bcolors.FAIL + f"ERROR loading model: {e}" + bcolors.ENDC)
        raise
    
    # ==================== VERIFICATION STEP 3: TRANSCRIBE ====================
    print("\n" + "="*70)
    print(bcolors.OKCYAN + "STEP 3: RUNNING TRANSCRIPTION" + bcolors.ENDC)
    print("="*70)
    
    print(bcolors.OKGREEN + "Starting transcription with WhisperTRT..." + bcolors.ENDC)
    transcribe_start = datetime.now()
    
    try:
        # Transcribe with WhisperTRT (fast, but no segments)
        result = model.transcribe(audio_file)
        
        transcribe_end = datetime.now()
        print(bcolors.OKGREEN + f"WhisperTRT transcription completed in {transcribe_end - transcribe_start}\n" + bcolors.ENDC)
        
        # Check if segments exist
        if 'segments' not in result or len(result.get('segments', [])) == 0:
            print(bcolors.WARNING + "WhisperTRT did not return segments. Loading original Whisper for timestamps..." + bcolors.ENDC)
            
            # Load original Whisper model to get segments
            import whisper
            segment_start = datetime.now()
            
            print(bcolors.OKBLUE + f"Loading original Whisper {model_size} model for segment extraction..." + bcolors.ENDC)
            whisper_model = whisper.load_model(model_size)
            
            print(bcolors.OKBLUE + "Extracting segments with timestamps..." + bcolors.ENDC)
            detailed_result = whisper_model.transcribe(audio_file)
            
            # Keep WhisperTRT's fast transcription text, but use Whisper's segments
            result['segments'] = detailed_result.get('segments', [])
            
            segment_end = datetime.now()
            print(bcolors.OKGREEN + f"Segment extraction completed in {segment_end - segment_start}\n" + bcolors.ENDC)
        
        # ==================== VERIFICATION STEP 4: CHECK OUTPUT ====================
        print("\n" + "="*70)
        print(bcolors.OKCYAN + "STEP 4: VERIFYING TRANSCRIPTION OUTPUT" + bcolors.ENDC)
        print("="*70)
        
        print(bcolors.OKBLUE + f"Result type: {type(result)}" + bcolors.ENDC)
        
        if isinstance(result, dict):
            print(bcolors.OKBLUE + f"Result keys: {list(result.keys())}" + bcolors.ENDC)
            
            # Check full text
            full_text = result.get('text', '')
            print(bcolors.OKGREEN + f"Full text length: {len(full_text)} characters" + bcolors.ENDC)
            
            if len(full_text) > 0:
                print(bcolors.OKGREEN + f"First 200 characters:" + bcolors.ENDC)
                print(bcolors.OKGREEN + f"  '{full_text[:200]}...'" + bcolors.ENDC)
            else:
                print(bcolors.FAIL + "WARNING: Full text is EMPTY!" + bcolors.ENDC)
            
            # Check segments
            segments = result.get('segments', [])
            print(bcolors.OKGREEN + f"Number of segments: {len(segments)}" + bcolors.ENDC)
            
            if len(segments) > 0:
                print(bcolors.OKGREEN + f"First segment example:" + bcolors.ENDC)
                first_seg = segments[0]
                print(bcolors.OKGREEN + f"  Keys: {list(first_seg.keys())}" + bcolors.ENDC)
                print(bcolors.OKGREEN + f"  Start: {first_seg.get('start', 'N/A')}" + bcolors.ENDC)
                print(bcolors.OKGREEN + f"  End: {first_seg.get('end', 'N/A')}" + bcolors.ENDC)
                print(bcolors.OKGREEN + f"  Text: '{first_seg.get('text', 'N/A')}'" + bcolors.ENDC)
                
                if len(segments) > 1:
                    print(bcolors.OKGREEN + f"Last segment example:" + bcolors.ENDC)
                    last_seg = segments[-1]
                    print(bcolors.OKGREEN + f"  Start: {last_seg.get('start', 'N/A')}" + bcolors.ENDC)
                    print(bcolors.OKGREEN + f"  End: {last_seg.get('end', 'N/A')}" + bcolors.ENDC)
                    print(bcolors.OKGREEN + f"  Text: '{last_seg.get('text', 'N/A')}'" + bcolors.ENDC)
            else:
                print(bcolors.FAIL + "WARNING: No segments found!" + bcolors.ENDC)
        else:
            print(bcolors.FAIL + f"ERROR: Result is not a dictionary! Got: {result}" + bcolors.ENDC)
        
        print("="*70 + "\n")
        
        return result
        
    except Exception as e:
        print(bcolors.FAIL + f"ERROR during transcription: {e}" + bcolors.ENDC)
        import traceback
        traceback.print_exc()
        raise

# async def assign_speakers(device: str, audio_file: str, result: dict, use_offline_models: int, hugging_face_token: str):
#     """Assign speakers using pyannote"""
#     print(bcolors.OKBLUE + "Running diarization...\n" + bcolors.ENDC)
#     diarize_start = datetime.now()
    
#     if use_offline_models:
#         print(bcolors.OKBLUE + "Using offline pyannote models for diarization.\n" + bcolors.ENDC)
#         pyannote_cache_dir = os.getenv("PYANNOTE_CACHE_DIR", "./pyannote_models")
#         os.environ['HF_HUB_OFFLINE'] = '1'
        
#         diarize_model = Pipeline.from_pretrained(
#             "pyannote/speaker-diarization-3.1",
#             cache_dir=pyannote_cache_dir
#         )
#     else:
#         print(bcolors.OKBLUE + "Using online pyannote models for diarization.\n" + bcolors.ENDC)
#         if not hugging_face_token:
#             raise ValueError("HUGGING_FACE_TOKEN environment variable not set.")
        
#         diarize_model = Pipeline.from_pretrained(
#             "pyannote/speaker-diarization-3.1",
#             use_auth_token=hugging_face_token
#         )
    
#     # Move to GPU if available
#     if device == "cuda":
#         diarize_model.to(torch.device("cuda"))
    
#     # Run diarization
#     diarization = diarize_model(audio_file)
    
#     # Convert pyannote output to segments
#     diarize_segments = []
#     for turn, _, speaker in diarization.itertracks(yield_label=True):
#         diarize_segments.append({
#             'start': turn.start,
#             'end': turn.end,
#             'speaker': speaker
#         })
    
#     # WhisperTRT result has 'segments' key with list of segment dicts
#     # Each segment has: start, end, text
#     if 'segments' in result and result['segments']:
#         segments = result['segments']
#     else:
#         # If no segments, create one from full text
#         segments = [{'start': 0.0, 'end': 0.0, 'text': result.get('text', '')}]
    
#     # Assign speakers to transcription segments
#     for seg in segments:
#         seg_start = seg.get("start", 0.0)
#         seg_end = seg.get("end", 0.0)
#         seg_mid = (seg_start + seg_end) / 2
#         seg["speaker"] = "UNKNOWN"
        
#         for diar in diarize_segments:
#             if diar["start"] <= seg_mid <= diar["end"]:
#                 seg["speaker"] = diar["speaker"]
#                 break
    
#     # Update result with speaker-assigned segments
#     result['segments'] = segments
    
#     diarize_end = datetime.now()
#     print(bcolors.OKGREEN + f"Diarization completed, took {diarize_end - diarize_start}.\n" + bcolors.ENDC)
    
#     return result

def export_results(result: dict, output_dir: str = "output", filename: str = "transcript"):
    """Export results to CSV format with start/end timestamps, text, and speaker labels"""
    
    # ==================== VERIFICATION STEP 5: CHECK EXPORT ====================
    print("\n" + "="*70)
    print(bcolors.OKCYAN + "STEP 5: EXPORTING RESULTS" + bcolors.ENDC)
    print("="*70)
    
    # Check if result is valid
    if result is None:
        print(bcolors.FAIL + "ERROR: Result is None, cannot export!" + bcolors.ENDC)
        return
    
    if not isinstance(result, dict):
        print(bcolors.FAIL + f"ERROR: Result is not a dict: {type(result)}" + bcolors.ENDC)
        return
    
    # Create output directory
    print(bcolors.OKBLUE + f"Creating output directory: {output_dir}" + bcolors.ENDC)
    os.makedirs(output_dir, exist_ok=True)
    
    if not os.path.exists(output_dir):
        print(bcolors.FAIL + f"ERROR: Could not create output directory!" + bcolors.ENDC)
        return
    
    print(bcolors.OKGREEN + f"Output directory ready: {os.path.abspath(output_dir)}" + bcolors.ENDC)
    
    csv_path = f"{output_dir}/{filename}.csv"
    print(bcolors.OKBLUE + f"Output file will be: {os.path.abspath(csv_path)}" + bcolors.ENDC)
    
    df_data = []
    
    segments = result.get('segments', [])
    print(bcolors.OKBLUE + f"Processing {len(segments)} segments..." + bcolors.ENDC)
    
    if len(segments) == 0:
        print(bcolors.WARNING + "WARNING: No segments to export!" + bcolors.ENDC)
        print(bcolors.WARNING + "  Creating single row with full text..." + bcolors.ENDC)
        # Create a single entry with the full text
        full_text = result.get('text', '')
        df_data.append({
            "start": "00:00:00.000",
            "end": "00:00:00.000",
            "text": full_text.strip() if full_text else "",
            "speaker": "UNKNOWN",
        })
    else:
        for i, seg in enumerate(segments):
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
            
            # Show first few segments
            if i < 3:
                print(bcolors.OKGREEN + f"  Segment {i+1}: [{start_time} -> {end_time}] '{text[:50]}...'" + bcolors.ENDC)
    
    print(bcolors.OKBLUE + f"Writing {len(df_data)} rows to CSV..." + bcolors.ENDC)
    
    try:
        df = pd.DataFrame(df_data)
        df.to_csv(csv_path, index=False)
        
        # Verify file was written
        if os.path.exists(csv_path):
            file_size = os.path.getsize(csv_path)
            print(bcolors.OKGREEN + f"File saved successfully!" + bcolors.ENDC)
            print(bcolors.OKGREEN + f"  Path: {os.path.abspath(csv_path)}" + bcolors.ENDC)
            print(bcolors.OKGREEN + f"  Size: {file_size} bytes" + bcolors.ENDC)
            
            # Read back first few lines to verify
            print(bcolors.OKBLUE + f"Reading back file to verify..." + bcolors.ENDC)
            with open(csv_path, 'r') as f:
                first_lines = [f.readline() for _ in range(min(5, len(df_data) + 1))]
                for i, line in enumerate(first_lines):
                    print(bcolors.OKGREEN + f"  Line {i+1}: {line.strip()[:100]}..." + bcolors.ENDC)
        else:
            print(bcolors.FAIL + f"ERROR: File was not created!" + bcolors.ENDC)
            
    except Exception as e:
        print(bcolors.FAIL + f"ERROR saving file: {e}" + bcolors.ENDC)
        import traceback
        traceback.print_exc()
        raise
    
    print("="*70 + "\n")
    
    print(bcolors.OKCYAN + f"\nResults exported to '{output_dir}/' directory:" + bcolors.ENDC)
    print(bcolors.OKGREEN + f"   {filename}.csv (timestamps + text)" + bcolors.ENDC)

def format_timestamp(seconds: float) -> str:
    """Convert seconds to timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

async def run_whispertrt():
    """Main runner function for WhisperTRT transcription with diarization"""
    
    print("\n" + "="*70)
    print(bcolors.HEADER + bcolors.BOLD + "WHISPERTRT TRANSCRIPTION PIPELINE" + bcolors.ENDC)
    print("="*70 + "\n")
    
    # Load environment variables
    device, audio_file, model_size, model_cache_path, output_dir, use_offline_models, hugging_face_token = load_env()
    
    # Track total time
    total_start = datetime.now()
    
    # Transcribe
    transcribe_start = datetime.now()
    result = await transcribe_audio(audio_file, model_size, model_cache_path)
    transcribe_end = datetime.now()
    
    # Check if transcription failed
    if result is None:
        print(bcolors.FAIL + "\nTRANSCRIPTION FAILED - STOPPING PIPELINE" + bcolors.ENDC)
        return
    
    # Diarization
    # print(bcolors.OKGREEN + "\nDiarizing with pyannote...\n" + bcolors.ENDC)
    # diarize_start = datetime.now()
    # result = await assign_speakers(device, audio_file, result, use_offline_models, hugging_face_token)
    # diarize_end = datetime.now()
    
    # Export
    print(bcolors.OKGREEN + "\nFinished transcription, exporting results...\n" + bcolors.ENDC)
    export_start = datetime.now()
    export_results(result, output_dir=output_dir, filename="transcript_trt")
    export_end = datetime.now()
    
    # Print timing summary
    time_for_transcription = transcribe_end - transcribe_start
    # time_for_diarization = diarize_end - diarize_start
    time_for_export = export_end - export_start
    time_total = export_end - total_start
    
    print("\n" + "="*70)
    print(bcolors.HEADER + bcolors.BOLD + "TIMING SUMMARY" + bcolors.ENDC)
    print("="*70)
    print(bcolors.OKBLUE + f"Total time: {time_total.seconds // 60} minutes and {time_total.seconds % 60} seconds" + bcolors.ENDC)
    print(bcolors.OKBLUE + f"  Transcription: {time_for_transcription.seconds // 60} minutes and {time_for_transcription.seconds % 60} seconds" + bcolors.ENDC)
    # print(bcolors.OKBLUE + f" - Diarization: {time_for_diarization.seconds // 60} minutes and {time_for_diarization.seconds % 60} seconds" + bcolors.ENDC)
    print(bcolors.OKBLUE + f"  Export: {time_for_export.seconds // 60} minutes and {time_for_export.seconds % 60} seconds" + bcolors.ENDC)
    print("="*70 + "\n")
    
    print(bcolors.OKGREEN + bcolors.BOLD + "PIPELINE COMPLETED SUCCESSFULLY!" + bcolors.ENDC + "\n")