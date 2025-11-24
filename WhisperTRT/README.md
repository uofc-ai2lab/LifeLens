# Installation

Note: Only used on the Jetson Nano

# Jetson containers setup
Initial setup is found here: https://github.com/dusty-nv/jetson-containers/blob/master/docs/setup.md 

```bash
cd ~
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers
bash install.sh
source ~/.bashrc
```

# create .env
Create a `.env` file in your project directory:
```properties
#### faster-whisper Configuration ####

# Hugging Face Token (get from: https://huggingface.co/settings/tokens)
# Required for speaker diarization
HUGGING_FACE_TOKEN=hf_your_token_here

# Device: cuda (GPU) or cpu
DEVICE=cuda

# Model: tiny, base, small, medium, large-v2, large-v3
# Recommended: base (good balance of speed/accuracy)
MODEL_SIZE=base

# Compute Type: float16 (better quality) or int8 (faster)
COMPUTE_TYPE=float16

# Audio file path (relative to .env file, use forward slashes)
AUDIO_FILE_PATH=./audio_data/your_audio.mp3

# Output directory
OUTPUT_DIR=./output

# Pyannote cache directory (for offline diarization models)
PYANNOTE_CACHE_DIR=./pyannote_models

# Use offline models: 0 (download from internet) or 1 (use cached)
USE_OFFLINE_MODELS=0
```

### Environment Variables Explained

| Variable | Options | Description |
|----------|---------|-------------|
| `HUGGING_FACE_TOKEN` | Your token | Required for diarization. Get from [huggingface.co](https://huggingface.co/settings/tokens) |
| `DEVICE` | `cuda`, `cpu` | Use `cuda` for GPU (recommended) |
| `MODEL_SIZE` | `tiny`, `base`, `small`, `medium`, `large-v2`, `large-v3` | Larger = better accuracy but slower |
| `COMPUTE_TYPE` | `float16`, `int8` | `float16` = better quality, `int8` = faster |
| `AUDIO_FILE_PATH` | File path | Path to your audio file (mp3, wav, flac, etc.) |
| `OUTPUT_DIR` | Directory path | Where to save transcription results |
| `PYANNOTE_CACHE_DIR` | Directory path | Where pyannote models are cached |
| `USE_OFFLINE_MODELS` | `0`, `1` | `0` = download models, `1` = use cached |


# WhisperTRT container running

To run on the Jetson Nano:
```bash
cd lifelens/
jetson-containers run -v $(pwd):/app -w /app $(autotag whisper_trt) python3 main.py whispertrt 
```

To learn more about running on jetson containers check out the docs here: https://github.com/dusty-nv/jetson-containers/blob/master/docs/run.md
