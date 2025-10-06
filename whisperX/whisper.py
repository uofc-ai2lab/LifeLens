import whisperx
import os
import dotenv

device = "cpu"  # or "cuda" if NVIDIA GPU
audio_file = os.path.join("test_data", "ShortParamedicClip.wav")
batch_size = 4 # reduce if low on GPU mem
compute_type = "int8"  # change to "float16" for GPU

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
diarize_model(audio, min_speakers=min_speakers, max_speakers=max_speakers)

result = whisperx.assign_word_speakers(diarize_segments, result)
print(diarize_segments)
print(result["segments"]) # segments are now assigned speaker IDs
