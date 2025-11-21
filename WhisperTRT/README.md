# WhisperTRT Installation and Usage Guide

## Important: Platform Considerations

**WhisperTRT is ONLY for NVIDIA Jetson devices.** It is specifically optimized for Jetson's Tegra GPU and TensorRT integration.

### Should You Use WhisperTRT?

- ✅ **YES - If you have a Jetson device** (Orin Nano, Orin NX, AGX Orin, Xavier, etc.)
  - ~3x faster than standard Whisper
  - ~40% less memory usage
  - Optimized for Jetson hardware

- ❌ **NO - If you're on Windows, macOS, or regular Linux PC**
  - WhisperTRT is not optimized for desktop GPUs
  - You'll get better performance with `faster-whisper` or standard Whisper
  - Installation is complex and offers no performance benefit

### For Non-Jetson Users

If you're developing on Windows/Mac, use **faster-whisper** instead:
```bash
# On Windows/Mac - Use faster-whisper
pip install faster-whisper

# Usage
from faster_whisper import WhisperModel
model = WhisperModel("base", device="cpu", compute_type="int8")
segments, info = model.transcribe("audio.wav")
```

Then deploy to Jetson for WhisperTRT when you're ready to run on the device.

---

## Installation on Jetson Devices

### Prerequisites

Before starting, ensure you have:
- ✅ NVIDIA Jetson device (Orin Nano, Orin NX, AGX Orin, Xavier NX, etc.)
- ✅ JetPack 5.0+ or 6.0+ installed
- ✅ Python 3.8 or higher
- ✅ At least 10GB free disk space
- ✅ Internet connection for downloading models

### Step 1: Verify Your Jetson Setup
```bash
# Check you're on a Jetson (should show aarch64)
uname -m

# Check JetPack version
dpkg -l | grep nvidia-jetpack

# Check CUDA is available
nvcc --version

# Check GPU is detected
nvidia-smi
```

**Expected output for `nvidia-smi`:**
```
GPU 0: Orin (nvgpu)
```

If any of these fail, install/reinstall JetPack first.

### Step 2: Install System Dependencies
```bash
# Update package list
sudo apt-get update

# Install required system packages
sudo apt-get install -y \
    ffmpeg \
    libsndfile1 \
    build-essential \
    git \
    python3-pip
```

### Step 3: Install Python Dependencies

**Important:** Install dependencies in this exact order to avoid errors.
```bash
# 1. Core dependencies
pip3 install numpy psutil

# 2. Verify PyTorch is installed (comes with JetPack)
python3 -c "import torch; print(f'PyTorch {torch.__version__} installed')"

# If PyTorch is not installed:
# pip3 install torch torchvision torchaudio

# 3. Install Whisper and audio processing
pip3 install openai-whisper tiktoken librosa soundfile ffmpeg-python torchaudio

# 4. Verify all dependencies
python3 << 'EOF'
import torch
import numpy
import whisper
import librosa
print("✓ All dependencies installed successfully")
EOF
```

### Step 4: Clone WhisperTRT Repository

**Important:** Clone WhisperTRT **outside** your project directory.
```bash
# Navigate to home directory
cd ~

# Clone WhisperTRT
git clone https://github.com/NVIDIA-AI-IOT/whisper_trt

# Navigate into the directory
cd whisper_trt
```

### Step 5: Install WhisperTRT
```bash
# Install in editable mode
pip3 install -e .
```

If you encounter errors about missing modules during installation, install them first:
```bash
pip3 install openai-whisper psutil
pip3 install -e .
```

### Step 6: Verify Installation
```bash
# Test import
python3 -c "from whisper_trt import load_trt_model; print('✓ WhisperTRT installed successfully!')"
```

If this works, you're ready to go!

---

## Using WhisperTRT with Your Project

### Project Structure

Your project should look like this:
```
~/ (Jetson home directory)
├── whisper_trt/                    # WhisperTRT installation (cloned above)
└── your-project/
    ├── whisperTRT.py               # Your transcription script
    ├── main.py                     # Main entry point
    ├── .env                        # Configuration
    ├── audio_data/                 # Input audio files
    │   └── your_audio.mp3
    ├── output/                     # Transcription results
    └── pyannote_models/            # Diarization models (if using offline)
```

### Step 1: Set Up Your Project
```bash
# Navigate to your project directory
cd ~/your-project

# Create necessary directories
mkdir -p audio_data output pyannote_models
```

### Step 2: Configure Environment Variables

Create a `.env` file in your project root:
```bash
nano .env
```

Add the following configuration:
```properties
#### WhisperTRT Configuration ####

# Device (always use cuda on Jetson)
DEVICE=cuda

# WhisperTRT Model Selection
# Options: tiny.en, base.en, small.en, medium.en
MODEL_SIZE_TRT=base.en

# Optional: Custom cache path for TensorRT engines
# Leave empty to use default: ~/.cache/whisper_trt/
WHISPER_TRT_CACHE=

# Audio file path (relative to .env file)
AUDIO_FILE_PATH=./audio_data/your_audio.mp3

# Output directory
OUTPUT_DIR=./output

# Hugging Face token (required for diarization)
HUGGING_FACE_TOKEN=your_token_here

# Pyannote diarization configuration
PYANNOTE_CACHE_DIR=./pyannote_models
# 0 = use online model, 1 = use offline model
USE_OFFLINE_MODELS=1
```

Save and exit (`Ctrl+X`, then `Y`, then `Enter`).

### Step 3: Copy Your Audio File
```bash
# Copy your audio file to the audio_data directory
cp /path/to/your/audio.mp3 ~/your-project/audio_data/
```

### Step 4: Run WhisperTRT Transcription

Using the `whisperTRT.py` file provided:
```bash
cd ~/your-project

# Run transcription
python3 main.py
```

Or call the function directly:
```python
# main.py
import asyncio
from whisperTRT import run_whisper_trt

async def main():
    print("Starting WhisperTRT transcription...\n")
    await run_whisper_trt()
    print("\nTranscription complete!")

if __name__ == "__main__":
    asyncio.run(main())
```

### Step 5: First Run - Engine Building

**Important:** The first time you run WhisperTRT with a model, it will build the TensorRT engine:
```
Loading base.en model...
Note: First run will build TensorRT engine (takes 2-5 minutes)
Building TensorRT engine... (this may take a while)
```

This is **normal** and only happens once. The engine is cached at `~/.cache/whisper_trt/`.

**Subsequent runs will be fast** (~1 second for transcription).

### Step 6: View Results

After transcription completes, your results will be in the output directory:
```bash
# View the CSV output
cat output/transcript_trt.csv
```

The CSV contains:
- `start` - Timestamp when speech starts
- `end` - Timestamp when speech ends
- `text` - Transcribed text
- `speaker` - Speaker label (if diarization is enabled)

---

## Performance Expectations

### First Run (Engine Build)
- **tiny.en**: 1-2 minutes
- **base.en**: 2-5 minutes
- **small.en**: 5-10 minutes
- **medium.en**: 10-20 minutes

This happens **once per model**, then the engine is cached.

### Subsequent Runs (with cached engine)

For a 20-second audio clip on Jetson Orin Nano:

| Model | Time | Memory |
|-------|------|--------|
| tiny.en | 0.64s | 488 MB |
| base.en | 0.86s | 439 MB |
| small.en | ~1.5s | ~600 MB |

**For longer audio** (10-minute clip):
- **base.en**: 3-5 minutes
- **small.en**: 5-8 minutes

---

## Using whisperTRT.py Script

The `whisperTRT.py` script provides a complete pipeline:

1. **Load configuration** from `.env`
2. **Transcribe audio** using WhisperTRT
3. **Diarize speakers** using pyannote.audio
4. **Export results** to CSV with timestamps and speaker labels

### Features

- ✅ Async/await support for integration
- ✅ Colored terminal output for progress tracking
- ✅ Automatic speaker diarization
- ✅ CSV export with timestamps
- ✅ Detailed timing information

### Function Usage
```python
import asyncio
from whisperTRT import run_whisper_trt

# Run the complete pipeline
async def transcribe():
    await run_whisper_trt()

asyncio.run(transcribe())
```

### Output Format

The script outputs timing information:
```
Time taken: 5 minutes and 23 seconds
 - Transcription time: 3 minutes and 12 seconds
 - Diarization time: 2 minutes and 8 seconds
 - Export time: 0 minutes and 3 seconds
```

And creates a CSV file with this structure:

| start | end | text | speaker |
|-------|-----|------|---------|
| 00:00:00.000 | 00:00:03.450 | Hello, how are you? | SPEAKER_00 |
| 00:00:03.500 | 00:00:06.120 | I'm doing well, thanks! | SPEAKER_01 |

---

## Troubleshooting

### "ModuleNotFoundError: No module named 'whisper_trt'"

**Solution:**
```bash
# Verify WhisperTRT is installed
cd ~/whisper_trt
pip3 install -e .
```

### "CUDA not available" or GPU errors

**Solution:**
```bash
# Check CUDA
nvcc --version
nvidia-smi

# Reinstall JetPack if needed
sudo apt install nvidia-jetpack
```

### TensorRT engine build fails

**Solution:**
- Check disk space: `df -h` (need ~2-3GB free)
- Try a smaller model first: `tiny.en`
- Check CUDA memory: `nvidia-smi` during build
- Restart Jetson if low on memory

### "RuntimeError: Couldn't load custom C++ ops"

**Solution:**
```bash
pip3 uninstall torch torchvision torchaudio
pip3 install torch torchvision torchaudio
cd ~/whisper_trt
pip3 install -e .
```

### Transcription is very slow

**Check these:**
1. Is this the first run? (Engine building is slow, only happens once)
2. Is CUDA being used? Check `.env` has `DEVICE=cuda`
3. Is GPU active? Run `nvidia-smi` during transcription
4. Try a smaller model: `tiny.en` or `base.en`

### Diarization fails

**Solution:**
```bash
# Install pyannote
pip3 install pyannote.audio torch-audiomentations

# Or disable diarization by commenting out in whisperTRT.py
```

---

## Model Selection Guide

Choose based on your needs:

### tiny.en
- ⚡ **Fastest** (~0.6s for 20s audio)
- 📉 Lowest accuracy
- 💾 Least memory (488 MB)
- ✅ Good for: Quick testing, real-time applications

### base.en (Recommended)
- ⚡ Fast (~0.9s for 20s audio)
- 📊 Good accuracy
- 💾 Low memory (439 MB)
- ✅ Good for: Most use cases, production

### small.en
- ⚡ Medium speed (~1.5s for 20s audio)
- 📈 Better accuracy
- 💾 Moderate memory (~600 MB)
- ✅ Good for: When accuracy is important

### medium.en
- 🐌 Slower (~3s for 20s audio)
- 📈 Best accuracy
- 💾 Higher memory (~1GB)
- ✅ Good for: Maximum accuracy needed

---

## Cleaning Up

### Clear TensorRT Engine Cache

If you want to rebuild engines or free up space:
```bash
# Remove cached engines
rm -rf ~/.cache/whisper_trt/

# Next run will rebuild engines (takes 2-5 minutes)
```

### Uninstall WhisperTRT
```bash
cd ~/whisper_trt
pip3 uninstall whisper-trt

# Optionally remove the directory
cd ~
rm -rf whisper_trt/
```

---

## Comparison: WhisperTRT vs faster-whisper

| Feature | WhisperTRT | faster-whisper |
|---------|------------|----------------|
| Platform | Jetson only | Any platform |
| Speed on Jetson | ~3x faster | Baseline |
| Memory on Jetson | ~40% less | Baseline |
| First run | 2-5 min (engine build) | Instant |
| Custom models | ❌ Not supported | ✅ Supported |
| Installation | Native only | Native or Docker |

**Recommendation:** Use WhisperTRT for production on Jetson, use faster-whisper for development on Windows/Mac.

---

## Additional Resources

- [WhisperTRT GitHub](https://github.com/NVIDIA-AI-IOT/whisper_trt)
- [NVIDIA Jetson AI Lab](https://www.jetson-ai-lab.com)
- [OpenAI Whisper](https://github.com/openai/whisper)
- [pyannote.audio Documentation](https://github.com/pyannote/pyannote-audio)

---

## Getting Help

If you encounter issues:

1. Check the [Troubleshooting](#troubleshooting) section above
2. Review [WhisperTRT Issues on GitHub](https://github.com/NVIDIA-AI-IOT/whisper_trt/issues)
3. Visit [NVIDIA Jetson Forums](https://forums.developer.nvidia.com/c/agx-autonomous-machines/jetson-embedded-systems/)

---

## Summary: Quick Start
```bash
# 1. On your Jetson, install dependencies
sudo apt-get update && sudo apt-get install -y ffmpeg libsndfile1 build-essential git
pip3 install numpy psutil openai-whisper tiktoken librosa soundfile ffmpeg-python torchaudio

# 2. Clone and install WhisperTRT
cd ~ && git clone https://github.com/NVIDIA-AI-IOT/whisper_trt
cd whisper_trt && pip3 install -e .

# 3. Set up your project
cd ~/your-project
# Create .env with configuration
# Copy audio files to audio_data/

# 4. Run transcription
python3 main.py

# First run: 2-5 minutes (builds engine)
# Subsequent runs: Fast! (~1 second per 20s audio)
```

That's it! You're ready to use WhisperTRT on your Jetson device.