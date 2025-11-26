# faster-whisper Setup for Jetson

Quick setup guide for running faster-whisper on NVIDIA Jetson devices using Docker.

> **Note:** This setup is for Jetson devices only. For Windows/Mac development, use: `pip install faster-whisper`

---

## Installation

### 1. Install jetson-containers
```bash
cd ~
git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers
bash install.sh
source ~/.bashrc
```

### 2. Pull faster-whisper Container
```bash
# This downloads/builds the faster-whisper container (first time takes 10-30 min)
jetson-containers run $(autotag faster-whisper)

# Test it works
python3 -c "from faster_whisper import WhisperModel; print('✓ faster-whisper ready')"
exit
```

---

## Configuration

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

---

## Usage

### Project Structure
```
your-project/
├── transcribe_faster.py    # Transcription script
├── main.py                  # Entry point
├── .env                     # Configuration
├── audio_data/              # Input audio
└── output/                  # Results
```

### Run Transcription
```bash
cd ~/your-project

# Method 1: Using helper script
jetson-containers run \
  -v $(pwd):/app \
  --workdir /app \
  $(autotag faster-whisper) \
  python3 main.py
```

* Method 2

Or create `run_docker.sh`:
```bash
#!/bin/bash
jetson-containers run \
  -v $(pwd):/app \
  --workdir /app \
  $(autotag faster-whisper) \
  python3 main.py
```
```bash
chmod +x run_docker.sh
./run_docker.sh
```

## Performance (Jetson Orin Nano)

**20-second audio:**
- `tiny`: 1-2s
- `base`: 2-3s (recommended)
- `small`: 4-5s
- `medium`: 8-10s

**10-minute audio:**
- `base`: 5-10 minutes
- Add 2-5 minutes for diarization

---

## Troubleshooting

**"autotag: command not found"**
```bash
source ~/.bashrc
# or
export PATH="$PATH:~/jetson-containers"
```

**"CUDA out of memory"**
- Use smaller model: `MODEL_SIZE=tiny` or `MODEL_SIZE=base`
- Use int8: `COMPUTE_TYPE=int8`

**Diarization fails**
- Check Hugging Face token is valid
- Accept model terms at [pyannote/speaker-diarization-3.1](https://huggingface.co/pyannote/speaker-diarization-3.1)

---

## Quick Start
```bash
# 1. Install jetson-containers
cd ~ && git clone https://github.com/dusty-nv/jetson-containers
cd jetson-containers && bash install.sh && source ~/.bashrc

# 2. Create project with .env (see above)
cd ~/your-project
nano .env

# 3. Add audio file
cp /path/to/audio.mp3 audio_data/

# 4. Run
jetson-containers run -v $(pwd):/app --workdir /app \
  $(autotag faster-whisper) python3 main.py
```

Results in `output/transcript.csv` with timestamps, text, and speaker labels.