# This file takes medical text transcripts, converts them to audio using TTS,
# then adds noise and reverberation to simulate real-world medical audio environments.
# Output: Noisy medical audio files

import os
import random
import numpy as np
from pathlib import Path
from typing import List, Optional

# TTS
try:
    import pyttsx3
except ImportError:
    raise ImportError("Install TTS engine: pip install pyttsx3")

# Audio processing
try:
    import librosa
    from scipy.io import wavfile
    from audiomentations import Compose, AddBackgroundNoise
except ImportError:
    raise ImportError(
        "Install audio deps: pip install librosa scipy audiomentations numpy"
    )

def text_to_speech(text: str, output_path: str) -> None:
    """Converts text to speech using pyttsx3 and saves as WAV."""
    engine = pyttsx3.init()
    engine.setProperty('rate', 160)      # Words per minute
    engine.setProperty('volume', 1.0)

    # Optional: vary voice if multiple available
    voices = engine.getProperty('voices')
    if voices:
        voice = random.choice(voices)
        engine.setProperty('voice', voice.id)

    engine.save_to_file(text, output_path)
    engine.runAndWait()


def add_noise_and_reverb(
    audio_path: str,
    noise_dir: str = "noise_library",
    noise_level: float = 0.2,
    reverb_level: float = 0.1,
    output_path: Optional[str] = None
) -> None:
    """
    Adds realistic background noise (helicopter, sirens, etc.) and optional reverb.
    """
    if output_path is None:
        output_path = audio_path

    # Load audio
    audio, sr = librosa.load(audio_path, sr=16000)
    if audio.dtype != np.float32:
        audio = audio.astype(np.float32)

    # Discover noise files
    noise_root = Path(noise_dir)
    if not noise_root.exists():
        raise FileNotFoundError(f"Noise library not found at {noise_root}. Run generate_noise_library.py first.")

    noise_files = []
    for subdir in ["helicopter", "sirens", "traffic", "crowd", "equipment"]:
        p = noise_root / subdir
        if p.exists():
            noise_files.extend(list(p.glob("*.wav")))

    if not noise_files:
        raise FileNotFoundError(f"No .wav noise files found in {noise_root}")

    # Prioritize helicopter (60% chance)
    helicopter_files = list((noise_root / "helicopter").glob("*.wav")) if (noise_root / "helicopter").exists() else []
    if helicopter_files and random.random() < 0.6:
        chosen_noise = random.choice(helicopter_files)
    else:
        chosen_noise = random.choice(noise_files)

    # Map noise_level (0–1) to SNR (20 → 0 dB)
    snr_db = max(0.0, 20.0 - (noise_level * 20.0))

    augment = Compose([
        AddBackgroundNoise(
            sounds_path=str(chosen_noise.parent),
            min_snr_db=snr_db,
            max_snr_db=snr_db,
            noise_rms="relative",
            p=1.0
        )
    ])

    augmented = augment(samples=audio, sample_rate=sr)

    # Simple reverb tail
    if reverb_level > 0:
        tail_len = int(0.4 * sr)
        decay = np.exp(-np.linspace(0, 4 * reverb_level, tail_len))
        tail = np.zeros_like(augmented)
        for i in range(len(augmented) - tail_len):
            tail[i:i+tail_len] += augmented[i] * decay
        augmented = np.clip(augmented + tail, -1.0, 1.0)

    # Save
    wavfile.write(output_path, sr, augmented.astype(np.float32))


def create_synthetic_medical_audio_dataset(
    texts: List[str],
    output_dir: str = "synthetic_medical_audio",
    noise_dir: str = "noise_library",
    noise_level: float = 0.25,
    reverb_level: float = 0.1
) -> None:
    """Generates a full synthetic EMS audio dataset with noise and reverb."""
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    for i, text in enumerate(texts):
        clean_file = out_path / f"medical_audio_{i:04d}_clean.wav"
        final_file = out_path / f"medical_audio_{i:04d}.wav"

        # TTS
        text_to_speech(text, str(clean_file))

        # Add noise + reverb → final output
        add_noise_and_reverb(
            str(clean_file),
            noise_dir=noise_dir,
            noise_level=noise_level,
            reverb_level=reverb_level,
            output_path=str(final_file)
        )

        # Clean up intermediate file
        clean_file.unlink()

    print(f"✅ Synthetic EMS audio dataset saved to: {out_path.absolute()}")


if __name__ == "__main__":
    medical_texts = import_medical_texts()
    create_synthetic_medical_audio_dataset(
        texts=medical_texts,
        output_dir="synthetic_medical_audio",
        noise_dir="noise_library",
        noise_level=0.25,
        reverb_level=0.1
    )