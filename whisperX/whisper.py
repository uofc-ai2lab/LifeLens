import whisperx
import os
import dotenv
import pandas as pd
from datetime import datetime
import gc
import torch
from pyannote.audio import Pipeline

# class for printing in colour to terminal
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

dotenv.load_dotenv()  # take environment variables

device = os.getenv("DEVICE", "cpu")
audio_from_env = os.getenv("AUDIO_FILE_PATH")
audio_file_list = audio_from_env.split("/") if audio_from_env else []
print(f"Audio file path from env: {audio_file_list} from {audio_from_env}")
audio_file = os.path.join(*audio_file_list)
batch_size = 4 # reduce if low on GPU mem
compute_type = "int8" if device == "cpu" else "float16"

print(bcolors.HEADER +
    f"Using device {device} with compute type {compute_type} for whisperX.\n\n" + bcolors.ENDC
)

print(bcolors.OKGREEN + f"Transcribing {audio_file}...\n" + bcolors.ENDC)

# track time taken in total for loading transcription, alignment, diarization and exporting and total time
transcribe_start_time = datetime.now()
# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)
# audio is a np.ndarray containing the audio waveform in float32 dtype.
audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)

# delete model if low on GPU resources


gc.collect()
torch.cuda.empty_cache()
del model

transcribe_end_time = datetime.now()
print(bcolors.OKGREEN + f"Transcription completed, took {transcribe_end_time - transcribe_start_time}.\n" + bcolors.ENDC)
print(bcolors.OKGREEN + "\nAligning with whisperX...\n" + bcolors.ENDC)
# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# delete model if low on GPU resources
gc.collect()
torch.cuda.empty_cache()
del model_a

align_end_time = datetime.now()
print(bcolors.OKGREEN + f"Alignment completed, took {align_end_time - transcribe_end_time}.\n" + bcolors.ENDC)
print(bcolors.OKGREEN + "\nDiarizing with whisperX...\n" + bcolors.ENDC)
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")


# 3. Assign speaker labels
USE_OFFLINE_MODELS = int(os.getenv("USE_OFFLINE_MODELS", "0"))

if USE_OFFLINE_MODELS:
    print(bcolors.OKBLUE + "Using offline pyannote models for diarization.\n" + bcolors.ENDC)
    pyannote_cache_dir = os.getenv("PYANNOTE_CACHE_DIR", "./pyannote_models")
    
    # Enable offline mode
    os.environ['HF_HUB_OFFLINE'] = '1'
    
    # Load pipeline from local cache
    diarize_model = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        cache_dir=pyannote_cache_dir
    )
else:
    print(bcolors.OKBLUE + "Using online pyannote models for diarization.\n" + bcolors.ENDC)
    diarize_model = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=HUGGING_FACE_TOKEN
    )

# Move to appropriate device
if device == "cuda":
    diarize_model.to(torch.device("cuda"))

# Run diarization (add min/max speakers if known)
# For known speaker counts: diarization = diarize_model(audio_file, num_speakers=2)
# For range: diarization = diarize_model(audio_file, min_speakers=2, max_speakers=5)
# Note: Pass the file path, not the loaded audio array
diarization = diarize_model(audio_file)

# Convert pyannote output to WhisperX-compatible format
diarize_segments = []
for turn, _, speaker in diarization.itertracks(yield_label=True):
    diarize_segments.append({
        'start': turn.start,
        'end': turn.end,
        'speaker': speaker
    })

# Segments are now assigned with speaker labels
# whisperx.assign_word_speakers expects (diarization, aligned_transcript) where diarization is a DataFrame
diarize_df = pd.DataFrame(diarize_segments)  # columns: start, end, speaker
# Some versions of whisperx/pyannote expect a 'label' column; ensure both are present
if 'label' not in diarize_df.columns:
    diarize_df['label'] = diarize_df['speaker']

# Pass the full aligned transcription result object (not just its "segments" list)
result = whisperx.assign_word_speakers(diarize_df, result)

diarize_end_time = datetime.now()
print(bcolors.OKGREEN + "\nFinished diarization, exporting results...\n" + bcolors.ENDC)

# 4. Export results
def export_results(result, output_dir="output", filename="transcript"):
    """Export results to csv format with start/end timestamps, text, and speaker labels"""
    csv_path = f"{output_dir}/{filename}.csv"
    df_data = []

    # Export start/end as seconds relative to start of the audio (no absolute anchoring).
    # Use 3 decimal places for milliseconds precision.
    for seg in result["segments"]:
        speaker = seg.get("speaker", "UNKNOWN")  # <- avoids KeyError
        start_time = format_timestamp_vtt(seg["start"])
        end_time = format_timestamp_vtt(seg["end"])
        df_data.append(
            {
                "start": start_time,
                "end": end_time,
                "text": seg["text"].strip(),
                "speaker": speaker,
            }
        )
    pd.DataFrame(df_data).to_csv(csv_path, index=False)

    print(bcolors.OKCYAN + f"\nResults exported to '{output_dir}/' directory:" + bcolors.ENDC)
    print(bcolors.OKGREEN + f"   ✓ {filename}.csv (timestamps + text)" + bcolors.ENDC)

def format_timestamp_vtt(seconds):
    """Convert seconds to VTT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"

output = os.getenv("OUTPUT_DIR", "output")
export_results(result, output_dir=output, filename="transcript")
export_end_time = datetime.now()

time_for_transcription = transcribe_end_time - transcribe_start_time
time_for_alignment = align_end_time - transcribe_end_time
time_for_diarization = diarize_end_time - align_end_time
time_for_export = export_end_time - diarize_end_time
time_total = export_end_time - transcribe_start_time

# print time taken mm:ss format
print(bcolors.OKBLUE + f"\n\nTime taken: {time_total.seconds // 60} minutes and {time_total.seconds % 60} seconds" + bcolors.ENDC)
print(bcolors.OKBLUE + f" - Transcription time: {time_for_transcription.seconds // 60} minutes and {time_for_transcription.seconds % 60} seconds" + bcolors.ENDC)
print(bcolors.OKBLUE + f" - Alignment time: {time_for_alignment.seconds // 60} minutes and {time_for_alignment.seconds % 60} seconds" + bcolors.ENDC)
print(bcolors.OKBLUE + f" - Diarization time: {time_for_diarization.seconds // 60} minutes and {time_for_diarization.seconds % 60} seconds" + bcolors.ENDC)
print(bcolors.OKBLUE + f" - Export time: {time_for_export.seconds // 60} minutes and {time_for_export.seconds % 60} seconds" + bcolors.ENDC)
