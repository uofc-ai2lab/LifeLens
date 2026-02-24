import os
import json
import torch
import soundfile as sf
from datasets import load_dataset
from tqdm import tqdm

def prepare_nemo_data(split_name, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    manifest_path = f"multimed_{split_name}_manifest.json"
    
    print(f"Loading MultiMed-ST {split_name} split...")
    ds = load_dataset("leduckhai/MultiMed-ST", "English", split=split_name)

    with open(manifest_path, 'w') as f:
        for i, item in enumerate(tqdm(ds)):
            audio = item['audio']['array']
            sr = item['audio']['sampling_rate']
            text = item['transcription']
            
            # Save audio to local disk for NeMo
            audio_filename = f"{split_name}_{i}.wav"
            audio_path = os.path.join(output_dir, audio_filename)
            sf.write(audio_path, audio, sr)
            
            # Create NeMo entry
            entry = {
                "audio_filepath": os.path.abspath(audio_path),
                "duration": len(audio) / sr,
                "text": text
            }
            f.write(json.dumps(entry) + '\n')
            
    return manifest_path

# Prepare both splits
train_manifest = prepare_nemo_data("train", "data/train")
test_manifest = prepare_nemo_data("test", "data/test")