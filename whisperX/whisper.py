import whisperx
import os
import dotenv
import json
import pandas as pd
from datetime import datetime
from rapidfuzz import process, fuzz
import jellyfish, re, pandas as pd
import spacy

from sentence_transformers import SentenceTransformer, util
model = SentenceTransformer("all-MiniLM-L6-v2") 

nlp = spacy.load("en_core_sci_sm")

dotenv.load_dotenv()  # take environment variables

device = os.getenv("DEVICE", "cpu")
audio_from_env = os.getenv("AUDIO_FILE_PATH")
audio_file_list = audio_from_env.split("/") if audio_from_env else []
print(f"Audio file path from env: {audio_file_list} from {audio_from_env}")
audio_file = os.path.join(*audio_file_list)
batch_size = 4 # reduce if low on GPU mem
compute_type = "int8" if device == "cpu" else "float16"

print(
    f"Using device {device} with compute type {compute_type} for whisperX.\n\nTranscribing {audio_file}...\n"
)

# track time taken in total for loading transcription, alignment, diarization and exporting and total time
transcribe_start_time = datetime.now()
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

transcribe_end_time = datetime.now()
print("\nAligning with whisperX...\n")
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

align_end_time = datetime.now()
print("\nDiarizing with whisperX...\n")
HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
# 3. Assign speaker labels
# diarization can be slow since we have to use the CPU, and the model is online on huggingface
# local model can be cached first, requires more disk space
# local model needs to be downloaded from https://huggingface.co/pyannote/speaker-diarization
diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HUGGING_FACE_TOKEN, device=device)

# add min/max number of speakers if known
diarize_segments = diarize_model(audio)
diarize_model(audio)

result = whisperx.assign_word_speakers(diarize_segments, result)

print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs

diarize_end_time = datetime.now()

print("\nFinished diarization, exporting results...\n")


# 4. Export results
# These are random words for testing. we won't do a key search going fwd anyways so no need for anything bigger 
KEYWORDS = ["shot", "compression", "blood"]


def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def phonetic_matches(phrase, words):
    pm = jellyfish.metaphone(phrase)
    return [w for w in words if jellyfish.metaphone(w) == pm]

def export_results(result, output_dir="output", filename="filtered", fuzzy_thresh=80):
    os.makedirs(output_dir, exist_ok=True)
    df_data = []

    for seg in result["segments"]:  # make sure it's .segments
        text = normalize(seg["text"])
        tokens = text.split()
        matched = False  # track if anything matched
        reason = ""

        # 1. keyword / fuzzy / phonetic
        for n in range(1, min(3, len(tokens)) + 1):
            for i in range(len(tokens) - n + 1):
                ng = " ".join(tokens[i:i+n])

                if ng in KEYWORDS:
                    matched = True
                    reason = f"exact match: {ng}"
                match = process.extractOne(ng, KEYWORDS, scorer=fuzz.token_sort_ratio)
                if match and match[1] >= fuzzy_thresh:
                    matched = True
                    reason = f"fuzzy match: {match[0]}"
                ph = phonetic_matches(ng, KEYWORDS)
                if ph:
                    matched = True
                    reason = f"phonetic match: {ph[0]}"

        # NLP entities --> need to replace with llm its not doing much smh
        doc = nlp(seg["text"])
        for ent in doc.ents:
            if ent.label_ in ["DISEASE", "CHEMICAL", "PROCEDURE"]:
                matched = True
                reason = f"nlp entity: {ent.text}"
                break

        # Only save if matched
        if matched:
            df_data.append({
                "start": seg["start"],
                "end": seg["end"],
                "phrase": seg["text"].strip(),
                "reason": reason
            })

    if not df_data:
        print("No important segments found.")
        return

    pd.DataFrame(df_data).to_csv(f"{output_dir}/{filename}.csv", index=False)
    print(f" Exported filtered matches to {output_dir}/{filename}.csv")


output = os.getenv("OUTPUT_DIR", "output")
export_results(result, output_dir=output, filename="transcript")
export_end_time = datetime.now()

time_for_transcription = transcribe_end_time - transcribe_start_time
time_for_alignment = align_end_time - transcribe_end_time
time_for_diarization = diarize_end_time - align_end_time
time_for_export = export_end_time - diarize_end_time
time_total = export_end_time - transcribe_start_time

# print time taken mm:ss format
print(f"Time taken: {time_total.seconds // 60} minutes and {time_total.seconds % 60} seconds")
print(f" - Transcription time: {time_for_transcription.seconds // 60} minutes and {time_for_transcription.seconds % 60} seconds")
print(f" - Alignment time: {time_for_alignment.seconds // 60} minutes and {time_for_alignment.seconds % 60} seconds")
print(f" - Diarization time: {time_for_diarization.seconds // 60} minutes and {time_for_diarization.seconds % 60} seconds")
print(f" - Export time: {time_for_export.seconds // 60} minutes and {time_for_export.seconds % 60} seconds")
