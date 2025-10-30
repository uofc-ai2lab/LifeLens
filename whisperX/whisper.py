import whisperx
import os
import dotenv
import json
import pandas as pd
from datetime import datetime
from rapidfuzz import process, fuzz
import jellyfish, re, pandas as pd
import spacy
from ollama import chat, ChatResponse

import subprocess

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

# print(
#     f"Using device {device} with compute type {compute_type} for whisperX.\n\nTranscribing {audio_file}...\n"
# )

# # track time taken in total for loading transcription, alignment, diarization and exporting and total time
# transcribe_start_time = datetime.now()
# # 1. Transcribe with original whisper (batched)
# model = whisperx.load_model("large-v2", device, compute_type=compute_type)

# # save model to local path (optional)
# # model_dir = "/path/"
# # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

# audio = whisperx.load_audio(audio_file)
# result = model.transcribe(audio, batch_size=batch_size)
# print(result["segments"]) # before alignment

# # delete model if low on GPU resources
# import gc
# import torch

# gc.collect()
# torch.cuda.empty_cache()
# del model

# transcribe_end_time = datetime.now()
# print("\nAligning with whisperX...\n")
# # 2. Align whisper output
# model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
# result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

# print(result["segments"]) # after alignment

# # delete model if low on GPU resources
# import gc
# import torch

# gc.collect()
# torch.cuda.empty_cache()
# del model_a

# align_end_time = datetime.now()
# print("\nDiarizing with whisperX...\n")
# HUGGING_FACE_TOKEN = os.getenv("HUGGING_FACE_TOKEN")
# # 3. Assign speaker labels
# # diarization can be slow since we have to use the CPU, and the model is online on huggingface
# # local model can be cached first, requires more disk space
# # local model needs to be downloaded from https://huggingface.co/pyannote/speaker-diarization
# diarize_model = whisperx.diarize.DiarizationPipeline(use_auth_token=HUGGING_FACE_TOKEN, device=device)

# # add min/max number of speakers if known
# diarize_segments = diarize_model(audio)
# diarize_model(audio)

# result = whisperx.assign_word_speakers(diarize_segments, result)

# print(diarize_segments)
# print(result["segments"]) # segments are now assigned speaker IDs


result = [{'start': 26.39, 'end': 49.185, 'text': " What's that?", 'words': [{'word': "What's", 'start': 26.39, 'end': 48.363, 'score': 0.557}, {'word': 'that?', 'start': 48.383, 'end': 49.185, 'score': 0.651}]}, {'start': 49.285, 'end': 54.152, 'text': 'A little bit.', 'words': [{'word': 'A', 'start': 49.285, 'end': 50.146, 'score': 0.998}, {'word': 'little', 'start': 50.226, 'end': 53.471, 'score': 0.526}, {'word': 'bit.', 'start': 53.791, 'end': 54.152, 'score': 0.68}]}, {'start': 75.31, 'end': 81.618, 'text': ' Ready, on three.', 'words': [{'word': 'Ready,', 'start': 75.31, 'end': 80.857, 'score': 0.63}, {'word': 'on', 'start': 80.877, 'end': 80.917, 'score': 0.0}, {'word': 'three.', 'start': 80.937, 'end': 81.618, 'score': 0.472}]}, {'start': 81.638, 'end': 91.591, 'text': 'Ready, one, two, three.', 'words': [{'word': 'Ready,', 'start': 81.638, 'end': 82.84, 'score': 0.499}, {'word': 'one,', 'start': 83.3, 'end': 84.462, 'score': 0.749}, {'word': 'two,', 'start': 84.482, 'end': 89.588, 'score': 0.619}, {'word': 'three.', 'start': 89.608, 'end': 91.591, 'score': 0.437}]}, {'start': 91.711, 'end': 95.376, 'text': 'Go ahead, man.', 'words': [{'word': 'Go', 'start': 91.711, 'end': 91.751, 'score': 0.035}, {'word': 'ahead,', 'start': 91.771, 'end': 92.852, 'score': 0.645}, {'word': 'man.', 'start': 93.013, 'end': 95.376, 'score': 0.682}]}, {'start': 95.396, 'end': 98.64, 'text': 'No gag.', 'words': [{'word': 'No', 'start': 95.396, 'end': 98.44, 'score': 0.851}, {'word': 'gag.', 'start': 98.46, 'end': 98.64, 'score': 0.602}]}, {'start': 99.241, 'end': 101.183, 'text': 'Ready, one, two, three.', 'words': [{'word': 'Ready,', 'start': 99.241, 'end': 100.542, 'score': 0.31}, {'word': 'one,', 'start': 100.562, 'end': 100.923, 'score': 0.514}, {'word': 'two,', 'start': 100.943, 'end': 101.043, 'score': 0.297}, {'word': 'three.', 'start': 101.063, 'end': 101.183, 'score': 0.0}]}, {'start': 108.453, 'end': 119.618, 'text': ' You got pre-charge and all that?', 'words': [{'word': 'You', 'start': 108.453, 'end': 112.161, 'score': 0.495}, {'word': 'got', 'start': 112.201, 'end': 113.645, 'score': 0.902}, {'word': 'pre-charge', 'start': 113.965, 'end': 117.393, 'score': 0.675}, {'word': 'and', 'start': 117.433, 'end': 117.614, 'score': 0.76}, {'word': 'all', 'start': 117.654, 'end': 118.415, 'score': 0.678}, {'word': 'that?', 'start': 118.476, 'end': 119.618, 'score': 0.468}]}, {'start': 120.179, 'end': 125.331, 'text': 'Pre-charge', 'words': [{'word': 'Pre-charge', 'start': 120.179, 'end': 125.331, 'score': 0.238}]}, {'start': 132.685, 'end': 140.913, 'text': ' Are we bagging and all that?', 'words': [{'word': 'Are', 'start': 132.685, 'end': 133.846, 'score': 0.814}, {'word': 'we', 'start': 137.51, 'end': 137.73, 'score': 0.702}, {'word': 'bagging', 'start': 137.75, 'end': 138.771, 'score': 0.359}, {'word': 'and', 'start': 138.791, 'end': 139.732, 'score': 0.292}, {'word': 'all', 'start': 139.752, 'end': 139.812, 'score': 0.232}, {'word': 'that?', 'start': 139.972, 'end': 140.913, 'score': 0.74}]}, {'start': 140.933, 'end': 144.277, 'text': 'You get an eye gel?', 'words': [{'word': 'You', 'start': 140.933, 'end': 140.993, 'score': 0.193}, {'word': 'get', 'start': 141.213, 'end': 141.854, 'score': 0.616}, {'word': 'an', 'start': 141.894, 'end': 143.796, 'score': 0.478}, {'word': 'eye', 'start': 143.816, 'end': 143.916, 'score': 0.332}, {'word': 'gel?', 'start': 143.956, 'end': 144.277, 'score': 0.487}]}, {'start': 144.977, 'end': 148.941, 'text': "Um, just for now, I don't want to get an eye gel.", 'words': [{'word': 'Um,', 'start': 144.977, 'end': 145.037, 'score': 0.001}, {'word': 'just', 'start': 145.057, 'end': 145.137, 'score': 0.001}, {'word': 'for', 'start': 145.157, 'end': 145.217, 'score': 0.172}, {'word': 'now,', 'start': 145.258, 'end': 146.679, 'score': 0.684}, {'word': 'I', 'start': 146.699, 'end': 146.719, 'score': 0.114}, {'word': "don't", 'start': 146.739, 'end': 146.839, 'score': 0.018}, {'word': 'want', 'start': 146.859, 'end': 146.939, 'score': 0.039}, {'word': 'to', 'start': 146.959, 'end': 146.999, 'score': 0.008}, {'word': 'get', 'start': 147.019, 'end': 147.4, 'score': 0.684}, {'word': 'an', 'start': 147.44, 'end': 147.78, 'score': 0.402}, {'word': 'eye', 'start': 147.8, 'end': 147.86, 'score': 0.044}, {'word': 'gel.', 'start': 147.88, 'end': 148.941, 'score': 0.759}]}, {'start': 149.001, 'end': 149.502, 'text': 'You do have one?', 'words': [{'word': 'You', 'start': 149.001, 'end': 149.061, 'score': 0.0}, {'word': 'do', 'start': 149.081, 'end': 149.121, 'score': 0.001}, {'word': 'have', 'start': 149.141, 'end': 149.342, 'score': 0.51}, {'word': 'one?', 'start': 149.402, 'end': 149.502, 'score': 0.232}]}, {'start': 149.522, 'end': 151.744, 'text': 'Oh, okay, cool.', 'words': [{'word': 'Oh,', 'start': 149.522, 'end': 151.223, 'score': 0.462}, {'word': 'okay,', 'start': 151.243, 'end': 151.624, 'score': 0.516}, {'word': 'cool.', 'start': 151.644, 'end': 151.744, 'score': 0.032}]}, {'start': 151.764, 'end': 157.51, 'text': 'Alright, 30 seconds for a pre-charge.', 'words': [{'word': 'Alright,', 'start': 151.764, 'end': 152.285, 'score': 0.172}, {'word': '30', 'start': 152.305, 'end': 153.386, 'score': 0.836}, {'word': 'seconds', 'start': 153.406, 'end': 153.546, 'score': 0.016}, {'word': 'for', 'start': 153.566, 'end': 153.626, 'score': 0.004}, {'word': 'a', 'start': 153.646, 'end': 153.726, 'score': 0.555}, {'word': 'pre-charge.', 'start': 154.447, 'end': 157.51, 'score': 0.143}]}, {'start': 157.53, 'end': 158.811, 'text': 'Alright, get a pre-charge.', 'words': [{'word': 'Alright,', 'start': 157.53, 'end': 157.73, 'score': 0.325}, {'word': 'get', 'start': 157.75, 'end': 157.81, 'score': 0.315}, {'word': 'a', 'start': 157.85, 'end': 157.89, 'score': 0.551}, {'word': 'pre-charge.', 'start': 157.93, 'end': 158.811, 'score': 0.185}]}, {'start': 159.567, 'end': 164.672, 'text': " I got it, I'll do it again.", 'words': [{'word': 'I', 'start': 159.567, 'end': 160.908, 'score': 0.834}, {'word': 'got', 'start': 160.928, 'end': 162.85, 'score': 0.549}, {'word': 'it,', 'start': 162.87, 'end': 163.611, 'score': 0.649}, {'word': "I'll", 'start': 163.631, 'end': 163.912, 'score': 0.549}, {'word': 'do', 'start': 164.152, 'end': 164.232, 'score': 0.594}, {'word': 'it', 'start': 164.312, 'end': 164.472, 'score': 0.316}, {'word': 'again.', 'start': 164.492, 'end': 164.672, 'score': 0.158}]}, {'start': 164.692, 'end': 165.834, 'text': "I'll do it.", 'words': [{'word': "I'll", 'start': 164.692, 'end': 164.792, 'score': 0.369}, {'word': 'do', 'start': 164.833, 'end': 165.393, 'score': 0.536}, {'word': 'it.', 'start': 165.413, 'end': 165.834, 'score': 0.311}]}, {'start': 165.854, 'end': 172.581, 'text': 'Oh no, keep going, keep going.', 'words': [{'word': 'Oh', 'start': 165.854, 'end': 165.894, 'score': 0.0}, {'word': 'no,', 'start': 165.914, 'end': 168.677, 'score': 0.674}, {'word': 'keep', 'start': 168.697, 'end': 168.777, 'score': 0.01}, {'word': 'going,', 'start': 168.797, 'end': 168.957, 'score': 0.164}, {'word': 'keep', 'start': 168.977, 'end': 171.72, 'score': 0.539}, {'word': 'going.', 'start': 171.74, 'end': 172.581, 'score': 0.256}]}, {'start': 172.601, 'end': 178.487, 'text': "Alright, we're gonna check in five seconds, y'all.", 'words': [{'word': 'Alright,', 'start': 172.601, 'end': 173.502, 'score': 0.189}, {'word': "we're", 'start': 174.002, 'end': 177.606, 'score': 0.666}, {'word': 'gonna', 'start': 177.626, 'end': 177.726, 'score': 0.018}, {'word': 'check', 'start': 177.746, 'end': 177.846, 'score': 0.001}, {'word': 'in', 'start': 177.866, 'end': 177.906, 'score': 0.026}, {'word': 'five', 'start': 177.926, 'end': 178.006, 'score': 0.002}, {'word': 'seconds,', 'start': 178.026, 'end': 178.287, 'score': 0.156}, {'word': "y'all.", 'start': 178.307, 'end': 178.487, 'score': 0.176}]}, {'start': 178.507, 'end': 180.709, 'text': 'Three, two, one.', 'words': [{'word': 'Three,', 'start': 178.507, 'end': 178.827, 'score': 0.57}, {'word': 'two,', 'start': 179.608, 'end': 180.329, 'score': 0.316}, {'word': 'one.', 'start': 180.349, 'end': 180.709, 'score': 0.349}]}, {'start': 180.729, 'end': 181.29, 'text': 'Alright, full check.', 'words': [{'word': 'Alright,', 'start': 180.729, 'end': 180.889, 'score': 0.044}, {'word': 'full', 'start': 180.909, 'end': 180.989, 'score': 0.18}, {'word': 'check.', 'start': 181.13, 'end': 181.29, 'score': 0.411}]}, {'start': 181.37, 'end': 182.952, 'text': 'I got it.', 'words': [{'word': 'I', 'start': 181.37, 'end': 181.59, 'score': 0.949}, {'word': 'got', 'start': 182.591, 'end': 182.751, 'score': 0.419}, {'word': 'it.', 'start': 182.791, 'end': 182.952, 'score': 0.398}]}, {'start': 182.972, 'end': 185.634, 'text': 'Yeah, okay.', 'words': [{'word': 'Yeah,', 'start': 182.972, 'end': 185.334, 'score': 0.534}, {'word': 'okay.', 'start': 185.534, 'end': 185.634, 'score': 0.142}]}, {'start': 185.654, 'end': 187.576, 'text': 'You look for it.', 'words': [{'word': 'You', 'start': 185.654, 'end': 185.714, 'score': 0.0}, {'word': 'look', 'start': 185.734, 'end': 185.815, 'score': 0.035}, {'word': 'for', 'start': 185.835, 'end': 186.015, 'score': 0.527}, {'word': 'it.', 'start': 186.055, 'end': 187.576, 'score': 0.663}]}, {'start': 187.596, 'end': 187.957, 'text': 'Yep, yep, shot it.', 'words': [{'word': 'Yep,', 'start': 187.596, 'end': 187.677, 'score': 0.002}, {'word': 'yep,', 'start': 187.697, 'end': 187.777, 'score': 0.006}, {'word': 'shot', 'start': 187.797, 'end': 187.877, 'score': 0.001}, {'word': 'it.', 'start': 187.897, 'end': 187.957, 'score': 0.0}]}, {'start': 187.977, 'end': 188.257, 'text': 'Alright, shot.', 'words': [{'word': 'Alright,', 'start': 187.977, 'end': 188.137, 'score': 0.0}, {'word': 'shot.', 'start': 188.157, 'end': 188.257, 'score': 0.0}]}, {'start': 188.277, 'end': 188.778, 'text': 'You got that ready to go?', 'words': [{'word': 'You', 'start': 188.277, 'end': 188.337, 'score': 0.0}, {'word': 'got', 'start': 188.357, 'end': 188.417, 'score': 0.0}, {'word': 'that', 'start': 188.437, 'end': 188.517, 'score': 0.0}, {'word': 'ready', 'start': 188.537, 'end': 188.638, 'score': 0.0}, {'word': 'to', 'start': 188.658, 'end': 188.698, 'score': 0.02}, {'word': 'go?', 'start': 188.718, 'end': 188.778, 'score': 0.003}]}, {'start': 188.798, 'end': 189.118, 'text': 'Go a little bit.', 'words': [{'word': 'Go', 'start': 188.798, 'end': 188.838, 'score': 0.0}, {'word': 'a', 'start': 188.858, 'end': 188.878, 'score': 0.0}, {'word': 'little', 'start': 188.898, 'end': 189.018, 'score': 0.0}, {'word': 'bit.', 'start': 189.038, 'end': 189.118, 'score': 0.0}]}, {'start': 197.367, 'end': 203.794, 'text': ' Hi Joseph.', 'words': [{'word': 'Hi', 'start': 197.367, 'end': 197.868, 'score': 0.674}, {'word': 'Joseph.', 'start': 197.908, 'end': 203.794, 'score': 0.565}]}, {'start': 203.814, 'end': 206.957, 'text': 'Hi Joseph.', 'words': [{'word': 'Hi', 'start': 203.814, 'end': 203.994, 'score': 0.838}, {'word': 'Joseph.', 'start': 206.097, 'end': 206.957, 'score': 0.447}]}, {'start': 208.159, 'end': 210.922, 'text': 'Thank you.', 'words': [{'word': 'Thank', 'start': 208.159, 'end': 210.021, 'score': 0.367}, {'word': 'you.', 'start': 210.041, 'end': 210.922, 'score': 0.498}]}, {'start': 210.942, 'end': 212.023, 'text': 'Interlake.', 'words': [{'word': 'Interlake.', 'start': 210.942, 'end': 212.023, 'score': 0.423}]}, {'start': 212.063, 'end': 216.167, 'text': 'Interlake.', 'words': [{'word': 'Interlake.', 'start': 212.063, 'end': 216.167, 'score': 0.563}]}, {'start': 221.834, 'end': 225.137, 'text': 'Interlake.', 'words': [{'word': 'Interlake.', 'start': 221.834, 'end': 225.137, 'score': 0.439}]}, {'start': 225.177, 'end': 225.838, 'text': 'Hi Joseph.', 'words': [{'word': 'Hi', 'start': 225.177, 'end': 225.317, 'score': 0.678}, {'word': 'Joseph.', 'start': 225.337, 'end': 225.838, 'score': 0.26}]}, {'start': 233.21, 'end': 262.018, 'text': ' 30 more seconds of compression until next pulse check.', 'words': [{'word': '30', 'start': 233.21, 'end': 233.911, 'score': 0.684}, {'word': 'more', 'start': 233.931, 'end': 243.5, 'score': 0.82}, {'word': 'seconds', 'start': 243.64, 'end': 254.21, 'score': 0.73}, {'word': 'of', 'start': 254.25, 'end': 254.311, 'score': 0.523}, {'word': 'compression', 'start': 254.511, 'end': 259.175, 'score': 0.548}, {'word': 'until', 'start': 259.195, 'end': 259.816, 'score': 0.737}, {'word': 'next', 'start': 260.016, 'end': 260.637, 'score': 0.512}, {'word': 'pulse', 'start': 260.657, 'end': 261.878, 'score': 0.315}, {'word': 'check.', 'start': 261.898, 'end': 262.018, 'score': 0.0}]}, {'start': 263.753, 'end': 276.192, 'text': ' Click the link.', 'words': [{'word': 'Click', 'start': 263.753, 'end': 273.849, 'score': 0.717}, {'word': 'the', 'start': 275.772, 'end': 275.832, 'score': 0.01}, {'word': 'link.', 'start': 275.852, 'end': 276.192, 'score': 0.447}]}, {'start': 276.212, 'end': 278.716, 'text': 'Click the link.', 'words': [{'word': 'Click', 'start': 276.212, 'end': 276.513, 'score': 0.425}, {'word': 'the', 'start': 276.813, 'end': 276.873, 'score': 0.049}, {'word': 'link.', 'start': 276.893, 'end': 278.716, 'score': 0.593}]}, {'start': 282.021, 'end': 282.522, 'text': 'Click the link.', 'words': [{'word': 'Click', 'start': 282.021, 'end': 282.121, 'score': 0.0}, {'word': 'the', 'start': 282.141, 'end': 282.202, 'score': 0.003}, {'word': 'link.', 'start': 282.222, 'end': 282.522, 'score': 0.169}]}, {'start': 286.113, 'end': 294.069, 'text': ' 5, 4, 3, 2, 1, pulse check.', 'words': [{'word': '5,', 'start': 286.113, 'end': 286.294, 'score': 0.569}, {'word': '4,', 'start': 286.314, 'end': 286.716, 'score': 0.447}, {'word': '3,', 'start': 286.736, 'end': 287.62, 'score': 0.626}, {'word': '2,', 'start': 287.64, 'end': 287.74, 'score': 0.921}, {'word': '1,', 'start': 287.76, 'end': 288.062, 'score': 0.73}, {'word': 'pulse', 'start': 288.082, 'end': 289.99, 'score': 0.696}, {'word': 'check.', 'start': 290.252, 'end': 294.069, 'score': 0.438}]}, {'start': 294.089, 'end': 297.825, 'text': 'Yeah, I got VTIB.', 'words': [{'word': 'Yeah,', 'start': 294.089, 'end': 295.636, 'score': 0.774}, {'word': 'I', 'start': 295.656, 'end': 295.676, 'score': 0.353}, {'word': 'got', 'start': 295.736, 'end': 295.897, 'score': 0.769}, {'word': 'VTIB.', 'start': 296.178, 'end': 297.825, 'score': 0.738}]}, {'start': 298.85, 'end': 299.312, 'text': 'Clear.', 'words': [{'word': 'Clear.', 'start': 298.85, 'end': 299.312, 'score': 0.592}]}]

diarize_end_time = datetime.now()

print("\nFinished diarization, exporting results...\n")


# 4. Export results
# These are random words for testing. we won't do a key search going fwd anyways so no need for anything bigger 
KEYWORDS = ["shot", "compression", "blood"]

def is_important_llm(prompt):
    
    response: ChatResponse = chat(model='gemma3:4b', messages=[
    {
        'role': 'user',
        'content': prompt,
    },
    ])
    print(response['message']['content'])
    # # or access fields directly from the response object
    # print(response.message.content)
    # print("yes" in response.message.content.strip().lower())

    return response.message.content.strip().lower()



def normalize(text):
    text = text.lower()
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text

def phonetic_matches(phrase, words):
    pm = jellyfish.metaphone(phrase)
    return [w for w in words if jellyfish.metaphone(w) == pm]

texts = """So we are going to facilitate a trauma scenario now.
So you can see we have a mannequin on the floor with blankets.
The blankets are supposed to be snow.
It's not really snow, but we're hoping it's just enough to get our participants in that mindset to go with it.
Okay, so these are the vital signs.
And really the point of this scenario is for our responders to respond to the scene, perform an assessment.
And what they're going to find is that the patient has a pneumo.
I'll show you that again.
Clicking on the lungs, and then clicking on pneumothorax, I can then choose which side.
So I'm going to give them a right-sided pneumo, and then activate.
There might be some other injuries that are hard to simulate using the mannequin.
So as the facilitator, I will be verbalizing what those are if the participants actually assess them or look for those things.
OK.
Chart 53.
27-year-old male, ATV rollover, no helmet, County Road 1643.
Time now, 1430.
Roger that, we're on it.
Hello!
Alright, DSI, scene safe.
Looks like we have a 27-year-old male.
Unresponsive.
Sir, can you hear me?
My name is Hans with the ambulance service.
My partner, Ted.
Can you hear me, sir?
Can you hear me?
Unresponsive.
Let's get to the vitals here.
I'm going to start checking for any injuries.
My pen light, shiny eyes.
Pupils react to light?
Pupils are reactive to light.
Checking for any decapital loss in the head, eye injuries, mouth, nose.
Negative.
JVD, tracheal deviation.
Negative.
Any clavicle injuries?
Negative.
I'm going to also take the lung sounds here.
Okay, no lung sounds on the right.
That's correct.
I'm going to confirm that.
I only have the chest wires on the left.
That's correct, yep.
Okay, so we have a possible tension pneumo.
I'm going to stop right there and we're going to actually, I'm going to grab you the needle out of the bag there.
We're going to needle the chest here.
I'm going to do a second intercostal space, mid-tellicular, just above the rib.
I'm actually going to need a little chest because that's expensive.
Okay, need a little chest.
Do I hear any type of... You do.
Okay, so I'm going to reconfer and I do have chest rise on the right side now.
You do.
Abdominal area.
Abdomen's soft.
Okay.
Pelvis.
Some instability.
Okay, so notice I'm going to stop there.
I'm going to put a pelvic binder on my partner and my arm.
Okay, see you partner.
The woman first when we were levitating.
A little higher.
Perfect, okay.
Okay, hold down.
Okay, pelvic stabilized.
Now down on the left leg here.
Checking for any injuries.
Negative.
Pulses?
Pulses are intact.
Okay.
Circuitry with your toes?
No response.
No response.
Okay, down on the other leg here.
Negative.
Okay.
Start blood pressure.
Don't prep pulse.
Pulse is intact.
Okay.
Left arm, same pulse.
Pulse is intact.
No response.
Right arm pulses.
Pulse is intact.
Blood pressure is 90 over 40.
Heart rate is 125.
Alright, the syringes we have are the right pneumo which has been something.
It's been fixed.
We'll keep an eye on the right side in case it reinflates.
Pelvis is stabilized.
Let's go ahead and get him on a longboard and we'll get out of here."""

def export_results(texts, result, output_dir="output", filename="filtered", fuzzy_thresh=80):
    os.makedirs(output_dir, exist_ok=True)
    df_data = []

    med_prompt = f"""which segments in the follwoing transcript talk about a medication, its dosage, start time, or calculated end time based on the start time? 
    return ONLY a list of dictionaries like:
    [{{'start': <float>, 'end': <float>, 'phrase': '<text>', 'reason': 'mentions medication or dosage'}}]
    no explanations. transcript: {texts}"""

    llm_medication_res = is_important_llm(med_prompt)
    df_data.append(llm_medication_res)

    proc_prompt = f"""which segments in the following transcript describe a medical intervention or procedure (e.g., IV insertion, tube insertion, flow rate, compression)?
    return ONLY a list of dictionaries like:
    [{{'start': <float>, 'end': <float>, 'phrase': '<text>', 'reason': 'mentions procedure or intervention'}}]
    no explanations. transcript: {texts}"""

    llm_proc_res = is_important_llm(proc_prompt)
    df_data.append(llm_proc_res)


    for seg in result:  # make sure it's .segments
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
            if ent.label_ in ["ORG", "PRODUCT", "NORP", "FAC", "EVENT"] or any(
                kw in ent.text.lower() for kw in ["med", "drug", "tube", "injection", "flow", "iv", "dose", "compression"]
            ):
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
export_results(texts, result, output_dir=output, filename="transcript")
export_end_time = datetime.now()

# time_for_transcription = transcribe_end_time - transcribe_start_time
# time_for_alignment = align_end_time - transcribe_end_time
# time_for_diarization = diarize_end_time - align_end_time
# time_for_export = export_end_time - diarize_end_time
# time_total = export_end_time - transcribe_start_time

# # print time taken mm:ss format
# print(f"Time taken: {time_total.seconds // 60} minutes and {time_total.seconds % 60} seconds")
# print(f" - Transcription time: {time_for_transcription.seconds // 60} minutes and {time_for_transcription.seconds % 60} seconds")
# print(f" - Alignment time: {time_for_alignment.seconds // 60} minutes and {time_for_alignment.seconds % 60} seconds")
# print(f" - Diarization time: {time_for_diarization.seconds // 60} minutes and {time_for_diarization.seconds % 60} seconds")
# print(f" - Export time: {time_for_export.seconds // 60} minutes and {time_for_export.seconds % 60} seconds")
