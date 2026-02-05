# GStreamer Pipelines Quick Reference

## Quick Start

```bash
# Run both pipelines (video + audio)
python main.py

# Run video only (dev mode)
python -m src_video.main --dev

# Run specific audio service
python -m src_audio.main record         # Recording
python -m src_audio.main transcribe     # Transcription
python -m src_audio.main meds           # Medication extraction
python -m src_audio.main inter          # Intervention extraction
```

## Pipeline Classes

### Video Pipeline

```python
from src_video.services.camera_capture_service.gstreamer_video_pipeline import GStreamerVideoPipeline

# Create pipeline
pipeline = GStreamerVideoPipeline(flip_method=0)

# Start capturing
if pipeline.start():
    # Read frames
    ok, frame = pipeline.read_frame()
    
    if ok:
        # Process frame
        process(frame)

# Stop and cleanup
pipeline.stop()
pipeline.cleanup()
```

### Audio Pipeline

```python
from src_audio.services.recording_audio_service.gstreamer_audio_pipeline import GStreamerAudioPipeline

# Create pipeline
pipeline = GStreamerAudioPipeline(output_file="/path/to/output.wav")

# Start recording
if pipeline.start():
    # Let it record
    time.sleep(10)
    
    # Check status
    if pipeline.is_recording_active():
        print("Recording...")

# Stop and cleanup
pipeline.stop()
pipeline.cleanup()
```

## Configuration

### Video Settings

**File**: `src_video/domain/constants.py`

```python
CAPTURE_WIDTH = 1920          # Camera capture width
CAPTURE_HEIGHT = 1080         # Camera capture height
DISPLAY_WIDTH = 960           # Display window width
DISPLAY_HEIGHT = 540          # Display window height
FRAME_RATE = 30               # FPS target
FLIP_METHOD = 0               # Image flip: 0=none, 2=180°
```

### Audio Settings

**File**: `src_audio/domain/constants.py`

```python
ARECORD_DEVICE = "hw:CARD=ArrayUAC10,DEV=0"  # ALSA device ID
AUDIO_SAMPLE_RATE = 16000                     # Hz
AUDIO_CHANNELS = 6                            # Channels
AUDIO_FORMAT = "S16LE"                        # Format (Signed 16-bit LE)
```

## Common Tasks

### Get Current Video Settings

```python
from src_video.domain.constants import CAPTURE_WIDTH, CAPTURE_HEIGHT, FRAME_RATE

print(f"Video: {CAPTURE_WIDTH}x{CAPTURE_HEIGHT} @ {FRAME_RATE}fps")
```

### Get Current Audio Settings

```python
from src_audio.domain.constants import AUDIO_SAMPLE_RATE, AUDIO_CHANNELS

print(f"Audio: {AUDIO_CHANNELS}-channel @ {AUDIO_SAMPLE_RATE}Hz")
```

### Check Camera Status

```bash
# List available cameras
v4l2-ctl --list-devices

# Check NVIDIA Jetson camera
ls -la /dev/video*

# Test with GStreamer
gst-launch-1.0 nvarguscamerasrc ! fakesink
```

### Check Audio Devices

```bash
# List ALSA devices
arecord -l

# List PulseAudio devices
pactl list sources

# Test USB audio device
arecord -D "hw:CARD=ArrayUAC10,DEV=0" -f S16_LE -r 16000 -c 6 test.wav
```

### Enable GStreamer Debug

```bash
# Level 0-5 (0=none, 5=debug info)
export GST_DEBUG=3
python main.py

# Debug specific element
export GST_DEBUG=nvarguscamerasrc:5
python main.py

# Save to file
export GST_DEBUG_FILE=gstreamer.log
python main.py 2>&1 | tee execution.log
```

## Troubleshooting

### Video Not Starting

```
[video][camera] ERROR: Unable to open camera
```

**Solutions**:
1. Restart Argus daemon: `sudo systemctl restart nvargus-daemon`
2. Check camera connection (ribbon cable)
3. Verify no other process using camera: `lsof /dev/video*`

### Audio Not Recording

```
[GStreamer Audio] ERROR: Failed to create pipeline
```

**Solutions**:
1. Check USB device: `lsusb | grep -i array`
2. Check ALSA device: `arecord -l`
3. Verify device permissions: `id -a`
4. Update device ID in constants.py if needed

### GStreamer Errors

```
ERROR: from element /GstPipeline:pipeline0/...
```

**Solutions**:
1. Enable debug: `export GST_DEBUG=4`
2. Check plugin availability: `gst-inspect-1.0 nvarguscamerasrc`
3. Verify GStreamer installation: `gst-launch-1.0 --version`

## Performance Tips

### Video Performance

```python
# Reduce resolution for faster processing
DISPLAY_WIDTH = 640
DISPLAY_HEIGHT = 480

# Reduce FPS if not needed
FRAME_RATE = 15
```

### Audio Performance

```python
# Reduce channels if not needed
AUDIO_CHANNELS = 2  # Instead of 6

# Lower sample rate for faster processing (with quality trade-off)
AUDIO_SAMPLE_RATE = 8000  # Instead of 16000
```

### Monitor Resources

```bash
# Watch GPU/CPU usage
tegrastats --interval 500

# Watch memory
watch -n 1 'free -h'

# Watch disk I/O
iotop -b -n 1
```

## File Locations

### Output Directories

```
Video Outputs:
  /home/capstone/LifeLens/data/video/output_files/

Audio Outputs:
  /home/capstone/LifeLens/data/audio/audio_files/
  /home/capstone/LifeLens/data/audio/processed_audio/

Recordings:
  /home/capstone/recordings/
```

### Configuration Files

```
Video Config:     config/video_settings.py
Audio Config:     config/audio_settings.py
Video Constants:  src_video/domain/constants.py
Audio Constants:  src_audio/domain/constants.py
```

### Pipeline Code

```
Main Orchestrator:           main.py
Video Pipeline Class:        src_video/services/camera_capture_service/gstreamer_video_pipeline.py
Audio Pipeline Class:        src_audio/services/recording_audio_service/gstreamer_audio_pipeline.py
Video Processing:            src_video/main.py
Audio Processing:            src_audio/main.py
```

## API Reference

### GStreamerVideoPipeline

```python
class GStreamerVideoPipeline:
    def __init__(self, flip_method: int = 0)
    def start(self) -> bool:           # Start pipeline
    def read_frame(self) -> (bool, ndarray):  # Read single frame
    def stop(self) -> bool:            # Stop pipeline
    def cleanup(self):                 # Cleanup resources
    @property
    def is_initialized(self) -> bool:  # Check if ready
```

### GStreamerAudioPipeline

```python
class GStreamerAudioPipeline:
    def __init__(self, output_file: str)
    def start(self) -> bool:           # Start recording
    def stop(self) -> bool:            # Stop recording
    def is_recording_active(self) -> bool:  # Check if recording
    def cleanup(self):                 # Cleanup resources
```

### Helper Functions

```python
# Get video pipeline string
def get_gstreamer_video_pipeline(
    sensor_id: int = 0,
    capture_width: int = 1920,
    capture_height: int = 1080,
    display_width: int = 960,
    display_height: int = 540,
    framerate: int = 30,
    flip_method: int = 0
) -> str

# Get audio pipeline string
def get_gstreamer_audio_pipeline(output_file: str) -> str
```

## Integration Examples

### Custom Video Processing Loop

```python
from src_video.services.camera_capture_service.gstreamer_video_pipeline import GStreamerVideoPipeline
import cv2

pipeline = GStreamerVideoPipeline(flip_method=0)

if pipeline.start():
    frame_count = 0
    while frame_count < 1000:
        ok, frame = pipeline.read_frame()
        
        if not ok:
            break
        
        # Your processing here
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        frame_count += 1
    
    pipeline.cleanup()
```

### Custom Audio Recording

```python
from src_audio.services.recording_audio_service.gstreamer_audio_pipeline import GStreamerAudioPipeline
import time

pipeline = GStreamerAudioPipeline("/tmp/recording.wav")

if pipeline.start():
    # Record for 10 seconds
    for i in range(10):
        time.sleep(1)
        status = "recording..." if pipeline.is_recording_active() else "ERROR"
        print(f"Second {i+1}: {status}")
    
    pipeline.stop()
    pipeline.cleanup()
```

## Environment Variables

```bash
# GStreamer debug level (0-5)
export GST_DEBUG=3

# GStreamer debug output file
export GST_DEBUG_FILE=gstreamer.log

# Video camera debug
export VIDEO_CAMERA_DEBUG=1

# Python asyncio debug
export PYTHONASYNCDEBUG=1
```

## Status Codes

### Video Pipeline
- `0` - Not initialized
- `1` - Initializing
- `2` - Running
- `-1` - Error

### Audio Pipeline
- `False` - Not recording
- `True` - Recording active

## Maintenance

### Regular Checks

```bash
# Monthly: Restart Argus daemon
sudo systemctl restart nvargus-daemon

# Monthly: Check GStreamer version
gst-launch-1.0 --version

# Quarterly: Update system packages
sudo apt update && sudo apt upgrade -y
```

### Log Inspection

```bash
# Check system logs for GStreamer issues
journalctl -u nvargus-daemon -n 50

# Check Python logs
grep "ERROR\|FAIL" execution.log | tail -20
```

---

**Last Updated**: February 4, 2026  
For detailed documentation, see [GSTREAMER_PIPELINES.md](GSTREAMER_PIPELINES.md) and [GSTREAMER_ARCHITECTURE.md](GSTREAMER_ARCHITECTURE.md)
