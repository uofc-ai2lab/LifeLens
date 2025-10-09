import whisperx
import os
import dotenv
import json
import pandas as pd
from datetime import datetime

device = os.getenv("DEVICE", "cpu")
audio_from_env = os.getenv("AUDIO_FILE_PATH")
audio_file_list = audio_from_env.split("/") if audio_from_env else []
audio_file = os.path.join(*audio_file_list)
batch_size = 4 # reduce if low on GPU mem
compute_type = "int8" if device == "cpu" else "float16"

# 1. Transcribe with original whisper (batched)
model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# save model to local path (optional)
# model_dir = "/path/"
# model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

audio = whisperx.load_audio(audio_file)
result = model.transcribe(audio, batch_size=batch_size)
print(result["segments"]) # before alignment

# delete model if low on GPU resources
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
del model

# 2. Align whisper output
model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

print(result["segments"]) # after alignment

# delete model if low on GPU resources
import gc
import torch

gc.collect()
torch.cuda.empty_cache()
del model_a

dotenv.load_dotenv()
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
# 3. Assign speaker labels
diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HUGGING_FACE_TOKEN, device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
diarize_model(audio)

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs


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

    print(f"\n💾 Results exported to '{output_dir}/' directory:")
    print(f"   ✓ {filename}.json (full structured data)")
    print(f"   ✓ {filename}.srt (subtitles)")
    print(f"   ✓ {filename}.vtt (web video subtitles)")
    print(f"   ✓ {filename}.txt (plain text)")
    print(f"   ✓ {filename}.csv (timestamps + text)")

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
