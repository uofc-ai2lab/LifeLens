from nemo.collections.asr.models import SortformerEncLabelModel
import torch, torchaudio
import json, os, tempfile
import matplotlib.pyplot as plt

def preprocess_audio(audio_path):
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    audio, sr = torchaudio.load(audio_path)

    # Convert to mono
    if audio.shape[0] > 1:
        audio = torch.mean(audio, dim=0, keepdim=True)

    # Resample if needed
    if sr != 16000:
        resampler = torchaudio.transforms.Resample(sr, 16000)
        audio = resampler(audio)

    # Save to temp file
    tmp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    torchaudio.save(tmp_file.name, audio, 16000)
    return tmp_file.name

def post_process_output(json_file):
    # Load the JSON file
    with open(json_file, "r") as f:
        data = json.load(f)

    # Flatten the data if needed (your example is nested)
    segments = data[0]  # assuming single list inside list

    # Parse segments into structured form
    parsed = []
    for seg in segments:
        start, end, speaker = seg.split()
        parsed.append({
            "speaker": speaker,
            "start": float(start),
            "end": float(end)
        })

    # Get unique speakers
    speakers = sorted(set(seg["speaker"] for seg in parsed))

    # Assign y-axis positions to speakers
    y_pos = {spk: i for i, spk in enumerate(speakers)}

    # Create plot
    plt.figure(figsize=(12, len(speakers) * 1.5))
    for seg in parsed:
        spk = seg["speaker"]
        start = seg["start"]
        end = seg["end"]
        plt.hlines(y=y_pos[spk], xmin=start, xmax=end, colors="tab:blue", linewidth=6)

    plt.yticks(list(y_pos.values()), list(y_pos.keys()))
    plt.xlabel("Time (s)")
    plt.ylabel("Speakers")
    plt.title("Speaker Diarization Timeline")
    plt.grid(True, axis="x", linestyle="--", alpha=0.5)
    plt.tight_layout()
    plt.show()

def main():
    # Path to your audio file
    audio_path = "multispeaker_audio1.wav" # source: https://www.youtube.com/shorts/edmW6PyD8UQ
    # audio = load_and_prepare_audio(audio_path)
    
    tmp_audio_path = preprocess_audio(audio_path)

    # Check if CUDA is available
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    device="cpu"
    print(f"Using device: {device}")

    # Load pretrained diarization model from Hugging Face
    print("Loading model... this may take a while the first time.")
    diar_model = SortformerEncLabelModel.from_pretrained(
        "nvidia/diar_sortformer_4spk-v1",
        map_location=device
    )

    # to perform speaker diarization and get a list of speaker-marked speech segments in the format 
    # 'begin_seconds, end_seconds, speaker_index', simply use:
    # Run diarization
    print(f"Running diarization on: {audio_path}")
    predicted_segments = diar_model.diarize(
        audio=tmp_audio_path, 
        batch_size=1 # keep this 1 for best accuracy
    )


    # Save segments to JSON
    output_file = "predicted_segments.json"
    with open(output_file, "w") as f:
        json.dump(predicted_segments, f, indent=4)

    print(f"Predicted segments saved to {output_file}")
    post_process_output(output_file)

    # # Print results
    # print("\nPredicted segments:")
    # for seg in predicted_segments:
    #     start, end, spk = seg
    #     print(f"Start: {start:.2f}s | End: {end:.2f}s | Speaker: {spk}")

    # Clean up temporary file
    os.unlink(tmp_audio_path)


if __name__ == "__main__":
    main()