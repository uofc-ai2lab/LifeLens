#!/bin/bash

# Run faster-whisper container with mounted directories
jetson-containers run \
  -v $(pwd)/audio_data:/audio_data \
  -v $(pwd)/output:/output \
  -v $(pwd)/models:/models \
  -v $(pwd)/transcribe_faster.py:/app/transcribe_faster.py \
  --workdir /app \
  $(autotag faster-whisper) \
  python3 transcribe_faster.py