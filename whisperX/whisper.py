import whisperx
import os
import dotenv
import json
import pandas as pd
from datetime import datetime

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

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
# print(result["segments"]) # before alignment

# delete model if low on GPU resources
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
del model

transcribe_end_time = datetime.now()
print(bcolors.OKGREEN + f"Transcription completed, took {transcribe_end_time - transcribe_start_time}.\n" + bcolors.ENDC)
print(bcolors.OKGREEN + "\nAligning with whisperX...\n" + bcolors.ENDC)
# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# print(result["segments"]) # after alignment

# delete model if low on GPU resources
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
del model_a

align_end_time = datetime.now()
print(bcolors.OKGREEN + f"Alignment completed, took {align_end_time - transcribe_end_time}.\n" + bcolors.ENDC)
print(bcolors.OKGREEN + "\nDiarizing with whisperX...\n" + bcolors.ENDC)
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
# 3. Assign speaker labels
# diarization can be slow since we have to use the CPU, and the model is online on huggingface
# local model can be cached first, requires more disk space
# local model needs to be downloaded from https://huggingface.co/pyannote/speaker-diarization
diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HUGGING_FACE_TOKEN, device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)

result = whisperx.assign_word_speakers(diarize_segments, result)
# print(diarize_segments)
# print(result["segments"]) # segments are now assigned speaker IDs

diarize_end_time = datetime.now()
print(bcolors.OKGREEN + "\nFinished diarization, exporting results...\n" + bcolors.ENDC)
# 4. Export results

def export_results(result, output_dir="output", filename="transcript"):
    """Export results in multiple formats"""
    os.makedirs(output_dir, exist_ok=True)

    json_path = f"{output_dir}/{filename}.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)

    srt_path = f"{output_dir}/{filename}.srt"
    with open(srt_path, "w", encoding="utf-8") as f:
        for i, seg in enumerate(result["segments"], 1):
            start = format_timestamp(seg["start"])
            end = format_timestamp(seg["end"])
            f.write(f"{i}\n{start} --> {end}\n{seg['text'].strip()}\n\n")

    vtt_path = f"{output_dir}/{filename}.vtt"
    with open(vtt_path, "w", encoding="utf-8") as f:
        f.write("WEBVTT\n\n")
        for i, seg in enumerate(result["segments"], 1):
            start = format_timestamp_vtt(seg["start"])
            end = format_timestamp_vtt(seg["end"])
            f.write(f"{start} --> {end}\n{seg['text'].strip()}\n\n")

    txt_path = f"{output_dir}/{filename}.txt"
    with open(txt_path, "w", encoding="utf-8") as f:
        for seg in result["segments"]:
            f.write(f"{seg['text'].strip()}\n")

    csv_path = f"{output_dir}/{filename}.csv"
    df_data = []

    # get length of audio
    AUDIO_LENGTH = result["segments"][-1]["end"]
    BASE_TIMESTAMP: float = datetime.now().timestamp() - AUDIO_LENGTH # set to current time - length of audio
    for seg in result["segments"]:
        # convert start and end to realtime format
        # start is base + seg start
        start_time = datetime.fromtimestamp((BASE_TIMESTAMP + seg["start"]))
        # end is base + seg end
        end_time = datetime.fromtimestamp((BASE_TIMESTAMP + seg["end"]))

        df_data.append({
            "start": start_time,
            "end": end_time,
            "text": seg["text"].strip()
        })
    pd.DataFrame(df_data).to_csv(csv_path, index=False)

    print(bcolors.OKCYAN + f"\nResults exported to '{output_dir}/' directory:" + bcolors.ENDC)
    print(bcolors.OKGREEN + f"   ✓ {filename}.json (full structured data)" + bcolors.ENDC)
    print(bcolors.OKGREEN + f"   ✓ {filename}.srt (subtitles)" + bcolors.ENDC)
    print(bcolors.OKGREEN + f"   ✓ {filename}.vtt (web video subtitles)" + bcolors.ENDC)
    print(bcolors.OKGREEN + f"   ✓ {filename}.txt (plain text)" + bcolors.ENDC)
    print(bcolors.OKGREEN + f"   ✓ {filename}.csv (timestamps + text)" + bcolors.ENDC)

def format_timestamp(seconds):
    """Convert seconds to SRT timestamp format"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    millis = int((seconds % 1) * 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

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
