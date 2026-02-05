# Migration Guide: From Subprocess to GStreamer

## Overview

This guide helps developers understand the changes from the previous subprocess-based architecture to the new GStreamer pipeline architecture.

## What Changed

### Video Pipeline

#### Before (OpenCV Direct)
```python
# src_video/main.py - OLD
from src_video.services.camera_capture_service.capture_img import initialize_camera

def main():
    video_capture = initialize_camera()  # Returns cv2.VideoCapture
    
    while True:
        ok, frame = video_capture.read()  # Direct OpenCV call
        # ...
```

#### After (GStreamer Wrapper)
```python
# src_video/main.py - NEW
from src_video.services.camera_capture_service.gstreamer_video_pipeline import GStreamerVideoPipeline

def main(video_pipeline: Optional[GStreamerVideoPipeline] = None):
    # Use passed pipeline or create new one
    if video_pipeline is None:
        video_pipeline = initialize_camera()
        owns_pipeline = True
    
    while True:
        ok, frame = video_pipeline.read_frame()  # Wrapper method
        # ...
```

### Audio Pipeline

#### Before (Subprocess)
```python
# src_audio/services/recording_audio_service/record_functions.py - OLD
import subprocess

def start_recording():
    process = subprocess.Popen([
        "arecord",
        "-D", ARECORD_DEVICE,
        "-f", "S16_LE",
        "-r", "16000",
        "-c", "6",
        output_file
    ], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return process, output_file

async def run_recording_service():
    process, output_file = start_recording()
    # ...
    process.send_signal(signal.SIGINT)
    process.wait()
```

#### After (GStreamer)
```python
# src_audio/services/recording_audio_service/record_functions.py - NEW
from src_audio.services.recording_audio_service.gstreamer_audio_pipeline import GStreamerAudioPipeline

def start_recording():
    pipeline = GStreamerAudioPipeline(output_file)
    if not pipeline.start():
        raise RuntimeError("Failed to start GStreamer pipeline")
    return pipeline, output_file

async def run_recording_service():
    pipeline, output_file = start_recording()
    # ...
    pipeline.stop()
    pipeline.cleanup()
```

### Main Orchestrator

#### Before (Direct Initialization)
```python
# main.py - OLD
from src_video.main import main as video_main, initialize_camera
from src_audio.main import main as audio_main

def run_video_pipeline(video_ready, video_failed):
    video_capture = None
    try:
        video_capture = initialize_camera(flip_method=0)
        video_ready.set()
        asyncio.run(video_main(video_capture))
    finally:
        if video_capture is not None:
            video_capture.release()
```

#### After (Orchestrated Initialization)
```python
# main.py - NEW
from src_video.services.camera_capture_service.gstreamer_video_pipeline import GStreamerVideoPipeline
from src_video.main import main as video_main
from src_audio.main import main as audio_main

def run_video_pipeline(video_ready, video_failed):
    video_pipeline = None
    try:
        video_pipeline = GStreamerVideoPipeline(flip_method=0)
        if not video_pipeline.start():
            video_failed.set()
            return
        
        video_ready.set()
        asyncio.run(video_main(video_pipeline))
    finally:
        if video_pipeline:
            video_pipeline.cleanup()
```

## API Comparison

### Video Capture

| Operation | Before | After |
|-----------|--------|-------|
| Initialize | `cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)` | `GStreamerVideoPipeline(flip_method=0)` |
| Start | (implicit in init) | `pipeline.start()` |
| Read Frame | `video_capture.read()` | `pipeline.read_frame()` |
| Stop | `video_capture.release()` | `pipeline.stop()` |
| Cleanup | (implicit in release) | `pipeline.cleanup()` |

### Audio Recording

| Operation | Before | After |
|-----------|--------|-------|
| Initialize | `subprocess.Popen(["arecord", ...])` | `GStreamerAudioPipeline(output_file)` |
| Start | (subprocess automatic) | `pipeline.start()` |
| Stop | `process.send_signal(signal.SIGINT)` | `pipeline.stop()` |
| Wait | `process.wait()` | (handled by pipeline) |
| Cleanup | (manual) | `pipeline.cleanup()` |

## Class Reference

### GStreamerVideoPipeline

```python
class GStreamerVideoPipeline:
    """
    Manages a GStreamer video capture pipeline.
    
    Replaces: cv2.VideoCapture(gstreamer_pipeline, cv2.CAP_GSTREAMER)
    
    Usage:
        pipeline = GStreamerVideoPipeline(flip_method=0)
        if pipeline.start():
            ok, frame = pipeline.read_frame()
            pipeline.stop()
        pipeline.cleanup()
    """
    
    def __init__(self, flip_method: int = 0)
    def start(self) -> bool:  # Was: implicit in OpenCV init
    def read_frame(self) -> (bool, numpy.ndarray):  # Was: video_capture.read()
    def stop(self) -> bool:  # Was: video_capture.release()
    def cleanup(self):  # Was: implicit
    
    @property
    def is_initialized(self) -> bool:  # New property
```

### GStreamerAudioPipeline

```python
class GStreamerAudioPipeline:
    """
    Manages a GStreamer audio recording pipeline.
    
    Replaces: subprocess.Popen(["arecord", ...])
    
    Usage:
        pipeline = GStreamerAudioPipeline(output_file)
        if pipeline.start():
            time.sleep(duration)
            pipeline.stop()
        pipeline.cleanup()
    """
    
    def __init__(self, output_file: str)
    def start(self) -> bool:  # Was: subprocess automatic
    def stop(self) -> bool:  # Was: process.send_signal(SIGINT)
    def is_recording_active(self) -> bool:  # Was: process.poll() == None
    def cleanup(self):  # New method
```

## Configuration Changes

### Video Configuration

**File**: `src_video/domain/constants.py`

**Before**: 
```python
# Just constants, no functions
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080
# ...
```

**After**:
```python
# Constants + helper function
CAPTURE_WIDTH = 1920
CAPTURE_HEIGHT = 1080

def get_gstreamer_video_pipeline(...) -> str:
    """Returns GStreamer pipeline string"""
```

### Audio Configuration

**File**: `src_audio/domain/constants.py`

**Before**:
```python
ARECORD_DEVICE = "hw:CARD=ArrayUAC10,DEV=0"
# Hardcoded pipeline commented out
pipeline = f"alsasrc device=... ! ... ! filesink location={output_file}"
```

**After**:
```python
ARECORD_DEVICE = "hw:CARD=ArrayUAC10,DEV=0"
AUDIO_SAMPLE_RATE = 16000
AUDIO_CHANNELS = 6
AUDIO_FORMAT = "S16LE"

def get_gstreamer_audio_pipeline(output_file: str) -> str:
    """Returns GStreamer pipeline string"""
```

## Breaking Changes

### 1. Video main() Function Signature

**Before**:
```python
def main() -> int:
    video_capture = initialize_camera()
    asyncio.run(video_main(video_capture))
```

**After**:
```python
def main(video_pipeline: Optional[GStreamerVideoPipeline] = None) -> int:
    if video_pipeline is None:
        video_pipeline = initialize_camera()
        owns_pipeline = True
    else:
        owns_pipeline = False
    # ...
```

**Migration**: Update calls to video_main() to pass pipeline object, OR use dev mode for standalone operation.

### 2. Audio Recording Function

**Before**:
```python
def start_recording() -> (subprocess.Popen, str):
    # Returns process object
```

**After**:
```python
def start_recording() -> (GStreamerAudioPipeline, str):
    # Returns pipeline object
```

**Migration**: Update code expecting subprocess.Popen to use GStreamerAudioPipeline API.

### 3. Import Changes

**Before**:
```python
import subprocess
import signal
from src_video.main import initialize_camera
```

**After**:
```python
from src_audio.services.recording_audio_service.gstreamer_audio_pipeline import GStreamerAudioPipeline
from src_video.services.camera_capture_service.gstreamer_video_pipeline import GStreamerVideoPipeline
```

## Behavioral Changes

### Error Handling

**Before**: subprocess errors returned via returncode
```python
if process.poll() is not None:  # Process exited
    error = process.returncode
```

**After**: Pipeline methods return bool
```python
if not pipeline.start():  # Returns False on error
    handle_error()
```

### State Management

**Before**: No explicit state (implicit in subprocess)
**After**: Explicit GStreamer states (NULL → PLAYING → NULL)

```python
# Video
pipeline.is_initialized  # Check if ready

# Audio  
pipeline.is_recording_active()  # Check if recording
```

### Resource Cleanup

**Before**: Implicit or manual
```python
video_capture.release()  # Single method
process.wait()  # Manual wait
```

**After**: Explicit cleanup
```python
pipeline.stop()      # Stop processing
pipeline.cleanup()    # Release resources
```

## Migration Checklist

- [ ] Update imports in all modules using video/audio pipelines
- [ ] Change `initialize_camera()` calls to `GStreamerVideoPipeline()`
- [ ] Update `video_capture.read()` to `pipeline.read_frame()`
- [ ] Replace subprocess Popen with `GStreamerAudioPipeline()`
- [ ] Change signal.SIGINT calls to `pipeline.stop()`
- [ ] Add `pipeline.cleanup()` calls
- [ ] Update error handling (subprocess errors → bool returns)
- [ ] Test video pipeline with camera connected
- [ ] Test audio pipeline with USB device connected
- [ ] Verify both pipelines work concurrently
- [ ] Check output files are created correctly
- [ ] Validate downstream services receive same input format

## Testing Migration

### Quick Test Script

```python
#!/usr/bin/env python3

import time
from src_video.services.camera_capture_service.gstreamer_video_pipeline import GStreamerVideoPipeline
from src_audio.services.recording_audio_service.gstreamer_audio_pipeline import GStreamerAudioPipeline

print("Testing Video Pipeline...")
vpipe = GStreamerVideoPipeline(flip_method=0)
if vpipe.start():
    for i in range(10):
        ok, frame = vpipe.read_frame()
        if ok:
            print(f"  Frame {i+1}: {frame.shape}")
    vpipe.stop()
    vpipe.cleanup()
    print("✓ Video pipeline OK")
else:
    print("✗ Video pipeline FAILED")

print("\nTesting Audio Pipeline...")
apipe = GStreamerAudioPipeline("/tmp/test.wav")
if apipe.start():
    for i in range(5):
        time.sleep(1)
        if apipe.is_recording_active():
            print(f"  Recording {i+1}/5 seconds...")
    apipe.stop()
    apipe.cleanup()
    print("✓ Audio pipeline OK")
else:
    print("✗ Audio pipeline FAILED")

print("\nAll tests completed!")
```

## Common Pitfalls

### 1. Forgetting cleanup()
```python
# WRONG
pipeline = GStreamerVideoPipeline()
pipeline.start()
# ... forgot pipeline.cleanup()
# Resources leak, next run fails

# RIGHT
try:
    pipeline = GStreamerVideoPipeline()
    if pipeline.start():
        # use pipeline
finally:
    pipeline.cleanup()
```

### 2. Not checking start() return value
```python
# WRONG
pipeline = GStreamerAudioPipeline(output_file)
pipeline.start()  # Returns False if error
time.sleep(10)  # Never records anything

# RIGHT
if not pipeline.start():
    print("ERROR: Failed to start")
    return
```

### 3. Wrong method calls
```python
# WRONG (old API)
ok, frame = video_pipeline.read()
process.wait()

# RIGHT (new API)
ok, frame = video_pipeline.read_frame()
pipeline.stop()
```

### 4. Incorrect import paths
```python
# WRONG
from src_video.main import GStreamerVideoPipeline

# RIGHT
from src_video.services.camera_capture_service.gstreamer_video_pipeline import GStreamerVideoPipeline
```

## Rollback Plan

If you need to revert to subprocess-based audio:

1. Keep old record_functions.py backup
2. Restore from git: `git show HEAD~1:src_audio/services/recording_audio_service/record_functions.py`
3. Remove GStreamerAudioPipeline import
4. Update main.py to not pass pipeline objects

## Support Resources

- See [GSTREAMER_PIPELINES.md](GSTREAMER_PIPELINES.md) for comprehensive architecture
- See [GSTREAMER_QUICK_REFERENCE.md](GSTREAMER_QUICK_REFERENCE.md) for API reference
- See [GSTREAMER_ARCHITECTURE.md](GSTREAMER_ARCHITECTURE.md) for visual diagrams

---

**Migration Date**: February 4, 2026  
**From**: Subprocess/Direct OpenCV  
**To**: GStreamer Pipelines with Threading  
**Status**: Production Ready
