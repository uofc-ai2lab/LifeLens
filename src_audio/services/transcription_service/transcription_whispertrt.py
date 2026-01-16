import os
from pathlib import Path
from datetime import datetime
import pandas as pd
import soundfile as sf
from src_audio.domain.constants import bcolors
from src_audio.utils.export_to_csv import export_to_csv
from config.audio_settings import AUDIO_FILES_LIST, IS_JETSON, MODEL_SIZE, MODEL_CACHE_PATH, TRANSCRIPT_DIR

def print_formatting(type: str, text: str):
    print("\n" + "="*70)
    if type=="title":
        print(bcolors.HEADER + bcolors.BOLD + f"{text}" + bcolors.ENDC)
    elif type=="heading":
        print(bcolors.OKCYAN + f"{text}" + bcolors.ENDC)
    print("="*70 + "\n") 
    
def load_whisper_model(model_size: str, model_cache_path: str = None):
    """Transcribe audio using WhisperTRT or fallback to original Whisper"""
    print_formatting("heading",f"STEP 1: LOADING {MODEL_SIZE.upper()} MODEL")
    model = None
    
    # Determine if we should use WhisperTRT or original Whisper
    use_whispertrt = IS_JETSON and MODEL_SIZE in ["tiny.en", "base.en"]
    
    if use_whispertrt:
        from whisper_trt import load_trt_model
        print(bcolors.OKGREEN + f"Using WhisperTRT for {model_size} (TensorRT accelerated)" + bcolors.ENDC)
        print(bcolors.WARNING + "Note: First run will build TensorRT engine (takes 2-5 minutes)\n" + bcolors.ENDC)
        
        try:
            if model_cache_path:
                print(bcolors.OKBLUE + f"Using custom cache path: {model_cache_path}" + bcolors.ENDC)
                model_file_path = os.path.join(model_cache_path, f"{model_size}.pth")
                print(bcolors.OKBLUE + f"Model file path: {model_file_path}" + bcolors.ENDC)
                model = load_trt_model(model_size, path=model_file_path)
            else:
                print(bcolors.OKBLUE + f"Using default cache: ~/.cache/whisper_trt/" + bcolors.ENDC)
                model = load_trt_model(model_size)
            
            print(bcolors.OKGREEN + f"Model loaded successfully" + bcolors.ENDC)
            print(bcolors.OKGREEN + f"  Model type: {type(model)}" + bcolors.ENDC)
            
        except Exception as e:
            print(bcolors.FAIL + f"ERROR loading WhisperTRT model: {e}" + bcolors.ENDC)
            print(bcolors.WARNING + f"Falling back to original Whisper..." + bcolors.ENDC)
            use_whispertrt = False
    
    if not use_whispertrt:
        print(bcolors.OKGREEN + f"Using original Whisper for {model_size}" + bcolors.ENDC)
        print(bcolors.WARNING + f"Note: Slower than TensorRT but more memory efficient\n" + bcolors.ENDC)
        
        try:
            import whisper
            print(bcolors.OKBLUE + f"Loading Whisper {model_size} model..." + bcolors.ENDC)
            model = whisper.load_model(model_size)
            
            print(bcolors.OKGREEN + f"Model loaded successfully" + bcolors.ENDC)
            print(bcolors.OKGREEN + f"  Model type: {type(model)}" + bcolors.ENDC)
            
        except Exception as e:
            print(bcolors.FAIL + f"ERROR loading model: {e}" + bcolors.ENDC)
            raise
 
    return model  
 
def verify_audio_file_exists(audio_file: str) -> bool:
    print_formatting("heading",f"STEP 2: VERIFYING INPUT AUDIO FILE ({Path(audio_file).name})")
    
    if not os.path.exists(audio_file):
        print(bcolors.FAIL + f"ERROR: Audio file not found: {audio_file}" + bcolors.ENDC)
        return False
    
    file_size = os.path.getsize(audio_file)
    print(bcolors.OKGREEN + f"Audio file found: {audio_file}" + bcolors.ENDC)
    print(bcolors.OKGREEN + f"File size: {file_size / (1024*1024):.2f} MB ({file_size:,} bytes)" + bcolors.ENDC)
    
    if file_size == 0:
        print(bcolors.FAIL + "ERROR: Audio file is empty!" + bcolors.ENDC)
        return False
    
    # Try to get audio file info
    try:
        info = sf.info(audio_file)
        print(bcolors.OKGREEN + f"Audio properties:" + bcolors.ENDC)
        print(bcolors.OKGREEN + f"  Duration: {info.duration:.2f} seconds ({info.duration/60:.2f} minutes)" + bcolors.ENDC)
        print(bcolors.OKGREEN + f"  Sample rate: {info.samplerate} Hz" + bcolors.ENDC)
        print(bcolors.OKGREEN + f"  Channels: {info.channels}" + bcolors.ENDC)
    except Exception as e:
        print(bcolors.WARNING + f"WARNING: Could not read audio properties: {e}" + bcolors.ENDC)
        print(bcolors.WARNING + f"  (This might be okay if the file format is supported by Whisper)" + bcolors.ENDC)
    
    return True

def verify_transcription_output(result: dict):
    print_formatting("heading","STEP 4: VERIFYING TRANSCRIPTION OUTPUT")

    print(bcolors.OKBLUE + f"Result type: {type(result)}" + bcolors.ENDC)
    
    if not isinstance(result, dict):
        print(bcolors.FAIL + f"ERROR: Result is not a dictionary! Got: {result}" + bcolors.ENDC)
        
    # Check full text
    full_text = result.get('text', '')
    print(bcolors.OKGREEN + f"Full text length: {len(full_text)} characters" + bcolors.ENDC)
    if not full_text.strip:
        print(bcolors.FAIL + "WARNING: Transcription text is EMPTY!" + bcolors.ENDC)
    
    # Check segments
    segments = result.get('segments', [])
    print(bcolors.OKGREEN + f"Number of segments: {len(segments)}" + bcolors.ENDC)
    if segments:
        print(bcolors.OKGREEN + f"First segment: {segments[0].get('text', '')[:50]}..." + bcolors.ENDC)
        return segments
    else:
        print(bcolors.WARNING + "WARNING: No segments found" + bcolors.ENDC)
        return full_text

def normalize_whisper_segments(segments):
    """
    Convert Whisper segment keys to pipeline-standard keys.
    """
    normalized = []
    for seg in segments:
        normalized.append({
            "start_time": seg["start"],
            "end_time": seg["end"],
            "text": seg.get("text", ""),
            "speaker": seg.get("speaker", "UNKNOWN")
        })
    return normalized
           
async def transcribe_audio(audio_file: str, model):
    print_formatting("heading","STEP 3: RUNNING TRANSCRIPTION")
    transcribe_start = datetime.now()
    try:
        result = model.transcribe(str(audio_file))
        transcribe_end = datetime.now()
        print(bcolors.OKGREEN + f"Transcription completed in {transcribe_end - transcribe_start}\n" + bcolors.ENDC)
        return result
    except Exception as e:
        print(bcolors.FAIL + f"ERROR during transcription: {e}" + bcolors.ENDC)
        import traceback
        traceback.print_exc()
        raise

async def run_transcription():
    print_formatting("title", "TRANSCRIPTION PIPELINE")
    
    """Main runner function for WhisperTRT (or Whisper) transcription """
    # ==================== VERIFICATION STEP 1: LOAD MODEL ====================
    model = load_whisper_model(MODEL_SIZE, MODEL_CACHE_PATH)
    
    for audio_file in AUDIO_FILES_LIST:
        print(bcolors.OKGREEN + f"Transcribing {Path(audio_file).name}...\n" + bcolors.ENDC)
        
        # ==================== VERIFICATION STEP 2: CHECK INPUT FILE ====================
        if verify_audio_file_exists(audio_file) is False:
            print(bcolors.FAIL + f"{audio_file} does not exist. Stopping transcription, moving to next audio file." + bcolors.ENDC)
            continue 
    
        # Track total time
        total_start = datetime.now()

        # ==================== VERIFICATION STEP 3: TRANSCRIBE ====================
        transcribe_start = datetime.now()
        result = await transcribe_audio(audio_file, model)
       
        # ==================== VERIFICATION STEP 4: CHECK OUTPUT ====================
        verified_result = verify_transcription_output(result)
        transcribe_end = datetime.now()

        # Check if transcription failed
        if verified_result is None:
            print(bcolors.FAIL + "\nTRANSCRIPTION FAILED - STOPPING PIPELINE" + bcolors.ENDC)
            return

        normalized_result = normalize_whisper_segments(verified_result)

        # ==================== VERIFICATION STEP 5: CHECK EXPORT ====================
        export_start = datetime.now()
        columns=["start_time", "end_time", "text", "speaker"]
        print_formatting("heading","STEP 5: EXPORTING RESULTS")
        
        export_to_csv(
            data=normalized_result,
            output_path=TRANSCRIPT_DIR,
            input_filename=Path(audio_file).name,
            service="transcript",
            columns=columns,
        )
        export_end = datetime.now()

        # Print timing summary
        time_for_transcription = transcribe_end - transcribe_start
        time_for_export = export_end - export_start
        time_total = export_end - total_start

        print_formatting("title", "TIMING SUMMARY")
        print(bcolors.OKBLUE + f"Total time: {time_total.seconds // 60} minutes and {time_total.seconds % 60} seconds" + bcolors.ENDC)
        print(bcolors.OKBLUE + f"  Transcription: {time_for_transcription.seconds // 60} minutes and {time_for_transcription.seconds % 60} seconds" + bcolors.ENDC)
        print(bcolors.OKBLUE + f"  Export: {time_for_export.seconds // 60} minutes and {time_for_export.seconds % 60} seconds" + bcolors.ENDC)
        print(bcolors.OKGREEN + bcolors.BOLD + "\nPIPELINE COMPLETED SUCCESSFULLY!" + bcolors.ENDC + "\n")
