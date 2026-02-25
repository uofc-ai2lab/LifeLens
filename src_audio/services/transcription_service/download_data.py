import os
import json
from io import BytesIO
import soundfile as sf
import torch
import torchaudio
from datasets import load_dataset, Audio
from tqdm import tqdm

# Force Hugging Face to use soundfile instead of torchcodec
os.environ["DATASETS_AUDIO_BACKEND"] = "soundfile"

def prepare_nemo_data(split_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    # Map the dataset split name to a safe filename
    safe_name = split_name.replace(".", "_")
    manifest_path = f"multimed_{safe_name}_manifest.json"

    print(f"--- Loading MultiMed-ST {split_name} split ---")
    
    # load_dataset returns a generator. 'decode=False' keeps audio as raw bytes.
    ds = load_dataset("leduckhai/MultiMed-ST", "English", split=split_name)
    ds = ds.cast_column("audio", Audio(decode=False))

    with open(manifest_path, 'w', encoding='utf-8') as f:
        for i, item in enumerate(tqdm(ds, desc=f"Processing {split_name}")):
            try:
                # 1. Get raw bytes and text
                audio_bytes = item['audio']['bytes']
                # MultiMed-ST usually uses 'transcription' but we check 'text' as fallback
                text = item.get('text') or item.get('text') or ""

                # 2. Decode bytes to numpy using soundfile (No torchcodec needed)
                audio_array, sr = sf.read(BytesIO(audio_bytes))

                # 3. Convert to Torch Tensor for resampling
                waveform = torch.from_numpy(audio_array).float()
                
                # Ensure shape is [channels, time]
                if waveform.dim() == 1:
                    waveform = waveform.unsqueeze(0)
                else:
                    waveform = waveform.transpose(0, 1)

                # 4. Resample to 16kHz (Standard for NeMo/ASR)
                target_sr = 16000
                if sr != target_sr:
                    waveform = torchaudio.functional.resample(waveform, sr, target_sr)
                    sr = target_sr

                # 5. Save to .wav using soundfile (Avoids torchaudio.save backend errors)
                audio_filename = f"{safe_name}_{i}.wav"
                audio_path = os.path.join(output_dir, audio_filename)
                
                # sf.write expects [time, channels]
                sf.write(audio_path, waveform.t().numpy(), sr)

                # 6. Create NeMo JSON entry
                duration = waveform.shape[1] / sr
                entry = {
                    "audio_filepath": os.path.abspath(audio_path),
                    "duration": float(duration),
                    "text": text
                }
                f.write(json.dumps(entry) + '\n')

            except Exception as e:
                print(f"Skipping sample {i} due to error: {e}")
                continue

    print(f"✅ Success! Manifest saved to: {manifest_path}")
    return manifest_path

if __name__ == "__main__":
    # Prepare both splits
    # Note: 'train' and 'test' are the standard split names
    test_manifest = prepare_nemo_data("corrected.test", "data/test")
    try:
        train_manifest = prepare_nemo_data("train", "data/train")
        print(f"Train manifest: {train_manifest}")
    except Exception as e:
        print(f"Train failed at: {e}")
        raise