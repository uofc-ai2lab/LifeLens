from nemo.collections.asr.models import SortformerEncLabelModel
import torch
import torchaudio

# --- 1. Load model ---
# Use CPU or MPS (Mac Metal GPU)
device = "mps" if torch.backends.mps.is_available() else "cpu"

# load model from Hugging Face model card directly (need a Hugging Face token)
diar_model = SortformerEncLabelModel.from_pretrained(
    "nvidia/diar_streaming_sortformer_4spk-v1",
    map_location=device
)

# switch to inference mode
diar_model.eval()
diar_model.to(device)


# load audio (source: https://www.youtube.com/shorts/edmW6PyD8UQ)
audio_path = "/multispeaker_audio1.wav"
audio, sr = torchaudio.load(audio_path)

# convert to mono if stereo
if audio.shape[0] > 1:
    audio = torch.mean(audio, dim=0, keepdim=True) 

# Resample to 16kHz if needed
if sr != 16000:
    resampler = torchaudio.transforms.Resample(sr, 16000)
    audio = resampler(audio)

# add batch dim -> [1, num_samples]
audio = audio.unsqueeze(0).to(device)

# run inference
with torch.no_grad():
    logits = diar_model.forward(input_signal=audio)

# logits shape: [batch, frames, speakers], you can threshold >0 for speech
speech_detected = (logits.sum(dim=-1) > 0).any().item()
print("Speech detected" if speech_detected else "No speech")