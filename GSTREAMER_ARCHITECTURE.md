# GStreamer Pipeline Architecture Diagrams

## 1. Overall System Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Main Python Process                          │
│                                                                   │
│  ┌──────────────────────────────────────────────────────────┐   │
│  │  main.py - Orchestrator                                 │   │
│  │  - Manages startup synchronization                       │   │
│  │  - Coordinates both pipeline threads                     │   │
│  │  - Handles graceful shutdown                             │   │
│  └───────────────┬──────────────────────────┬───────────────┘   │
│                  │                          │                   │
│    ┌─────────────▼─────┐      ┌─────────────▼──────────┐        │
│    │ VIDEO THREAD      │      │ AUDIO THREAD           │        │
│    │ (asyncio loop)    │      │ (asyncio loop)         │        │
│    │                   │      │                        │        │
│    │ ┌───────────────┐ │      │ ┌──────────────────┐  │        │
│    │ │ GStreamer     │ │      │ │ GStreamer        │  │        │
│    │ │ Video         │ │      │ │ Audio Pipeline   │  │        │
│    │ │ Pipeline      │ │      │ │                  │  │        │
│    │ └───────────────┘ │      │ └──────────────────┘  │        │
│    │                   │      │                        │        │
│    │ - Camera frame    │      │ - Waveform output     │        │
│    │   capture         │      │ - Metadata output     │        │
│    │ - Detection       │      │ - Transcription       │        │
│    │ - Classification  │      │ - Med extraction      │        │
│    │ - Deidentify      │      │ - Anonymization       │        │
│    └───────────────────┘      └────────────────────────┘        │
└─────────────────────────────────────────────────────────────────┘

Synchronization:
  1. Video thread starts, initializes GStreamer pipeline
  2. Camera warmup sequence completes
  3. video_ready event set → Audio thread starts
  OR
  4. video_failed event set → Audio thread never starts
```

## 2. Video Pipeline Details

```
CSI Camera (NVIDIA Jetson)
    │
    ▼
┌─────────────────────────────────────────────────────────────┐
│         GStreamer Video Pipeline                             │
│                                                               │
│  nvarguscamerasrc                                             │
│  (Hardware accelerated camera source)                         │
│         │ format: NVMM (GPU memory)                           │
│         │ resolution: 1920x1080                               │
│         │ framerate: 30 fps                                   │
│         ▼                                                      │
│  nvvidconv                                                    │
│  (NVIDIA video converter)                                     │
│         │ flip-method: 0 (none) / 2 (rotate-180)             │
│         │ output: BGRx                                        │
│         ▼                                                      │
│  videoconvert + BGR format                                    │
│  (Convert to OpenCV-compatible format)                        │
│         │                                                      │
│         ▼                                                      │
│  appsink                                                      │
│  (Feed frames to Python application)                          │
│                                                               │
└─────────────────────────────────────────────────────────────┘
         │
         ▼
   Frame to Python
   (cv2.VideoCapture)
         │
         ├─► Marker detection (detect_apriltags)
         │
         ├─► Frame saving (capture_frame_from_pipeline)
         │
         ├─► Detection (YOLO)
         │
         ├─► Classification (SwinTransformer)
         │
         └─► Deidentification (face/body blur)

Classes & Functions:
  - GStreamerVideoPipeline (class)
    ├── get_gstreamer_video_pipeline() → str
    ├── start() → bool
    ├── read_frame() → (bool, numpy.ndarray)
    ├── stop() → bool
    └── cleanup()
```

## 3. Audio Pipeline Details

```
USB Audio Device (6-channel ArrayUAC10)
    │
    ▼
┌────────────────────────────────────────────────────────────┐
│       GStreamer Audio Pipeline                              │
│                                                              │
│  alsasrc                                                     │
│  (ALSA audio input source)                                   │
│         │ device: hw:CARD=ArrayUAC10,DEV=0                   │
│         │ channels: 6                                        │
│         │ rate: 16000 Hz                                     │
│         ▼                                                     │
│  audioconvert                                                │
│  (Convert audio format)                                      │
│         │                                                     │
│         ▼                                                     │
│  audioresample                                               │
│  (Resample to target rate)                                   │
│         │ output: S16LE (Signed 16-bit LE)                   │
│         │ rate: 16000 Hz                                     │
│         ▼                                                     │
│  wavenc                                                      │
│  (Encode to WAV format)                                      │
│         │                                                     │
│         ▼                                                     │
│  filesink                                                    │
│  (Write to file)                                             │
│                                                              │
└────────────────────────────────────────────────────────────┘
         │
         ▼
   WAV File Output
   /home/capstone/recordings/recording_YYYYMMDD_HHMMSS.wav
         │
         ├─► Copy to /home/capstone/data/audio/audio_files/
         │
         ├─► Transcription (WhisperTRT)
         │
         ├─► Medication extraction
         │
         ├─► Intervention extraction
         │
         ├─► Anonymization
         │
         └─► Output artifacts
             ├── Transcripts
             ├── Medications CSV
             ├── Interventions CSV
             └── Metadata JSON

Classes & Functions:
  - GStreamerAudioPipeline (class)
    ├── get_gstreamer_audio_pipeline() → str
    ├── start() → bool
    ├── stop() → bool
    ├── is_recording_active() → bool
    └── cleanup()
  
  Recording Service:
    ├── start_recording() → (GStreamerAudioPipeline, str)
    ├── wait_for_recording(output_file: str)
    ├── copy_to_target(recording_path: str)
    └── run_recording_service() → async
```

## 4. Thread Execution Timeline

```
Time  Video Thread                Audio Thread         Status
────────────────────────────────────────────────────────────────
  T0  main() starts
      Video thread created
                                                       INIT
  
  T1  Video thread starts
      GStreamer init
      Camera open                                      VIDEO_INIT
  
  T2  Camera warmup
      (20 frames)                                      WARMING_UP
  
  T3  Warmup complete
      video_ready.set()
                                                       VIDEO_READY
  
  T4                            Audio thread starts
                                GStreamer init
                                Pipeline start       AUDIO_INIT
  
  T5  Reading frames            Recording audio      RUNNING
      Processing                Processing
      
  Tn  (running concurrently)    (running concurrently) RUNNING
      No blocking               No blocking
  
  Tm  User interrupts or        
      processing completes      
      
  Tm+1 Cleanup video
       Pipeline stop
       Resources release        Still running         CLEANUP_VIDEO
  
  Tm+2                          Audio completes
                                Cleanup audio
                                Resources release    CLEANUP_AUDIO
  
  Tm+3                          Threads joined
                                Process exit         COMPLETE
```

## 5. GStreamer Plugin Chain Visualization

### Video Chain
```
Input Source          Format Processing        Output Sink
────────────         ──────────────────        ───────────
nvarguscamerasrc  ── nvvidconv ── videoconvert ── appsink
   ↓                    ↓               ↓              ↓
[NVMM Memory]      [GPU Convert]   [BGR Format]   [Python App]
[1920x1080]        [flip-method]   [24-bit RGB]   [cv2.VideoCapture]
[30 fps]           [scale]         [OpenCV Ready] [Frame Access]
```

### Audio Chain  
```
Input Source      Format Processing        Output Sink
────────────      ──────────────────       ───────────
alsasrc  ──── audioconvert ── audioresample ── wavenc ── filesink
  ↓              ↓                ↓              ↓        ↓
[ALSA In]    [Format Conv]   [Rate Conv]   [WAV Enc]   [File Out]
[6 channels] [Any Format]    [16000 Hz]    [PCM WAV]   [.wav File]
[Raw Audio]  [→ Raw]         [S16LE]       [Valid]     [Stored]
```

## 6. Memory & Resource Management

```
┌─────────────────────────────────────────────────────────┐
│         GStreamer Pipeline Resource Lifecycle            │
├─────────────────────────────────────────────────────────┤
│                                                           │
│  Initialization Phase:                                   │
│  ┌──────────────────────────────────────────────────┐   │
│  │ 1. Pipeline string created                       │   │
│  │ 2. Gst.parse_launch() → Pipeline object          │   │
│  │ 3. Bus created for event monitoring              │   │
│  │ 4. set_state(PLAYING) → Start processing         │   │
│  └──────────────────────────────────────────────────┘   │
│                                                           │
│  Running Phase:                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ GStreamer handles:                               │   │
│  │ - Buffer allocation & management                 │   │
│  │ - Element padding & negotiation                  │   │
│  │ - Data flow synchronization                      │   │
│  │ - Thread management per element                  │   │
│  │ - Hardware resource access (GPU, devices)        │   │
│  └──────────────────────────────────────────────────┘   │
│                                                           │
│  Cleanup Phase:                                          │
│  ┌──────────────────────────────────────────────────┐   │
│  │ 1. set_state(NULL) → Stop all elements           │   │
│  │ 2. Release device/file handles                   │   │
│  │ 3. Free allocated buffers                        │   │
│  │ 4. bus.remove_signal_watch() → Cleanup bus       │   │
│  │ 5. Unref pipeline → Destroy object               │   │
│  └──────────────────────────────────────────────────┘   │
│                                                           │
└─────────────────────────────────────────────────────────┘
```

## 7. Error Handling & Recovery

```
Video Pipeline Initialization
            │
            ▼
  ┌─────────────────────┐
  │ GStreamerVideoPipeline │
  │ .start()                │
  └──────────┬──────────┘
             │
      ┌──────┴──────┐
      ▼             ▼
   SUCCESS      FAILURE
      │             │
      │     ┌───────┴──────────────┐
      │     │  Error States:        │
      │     │  - Device not found   │
      │     │  - Permission denied  │
      │     │  - Warmup failed      │
      │     │  - Pipeline error     │
      │     └─────────┬─────────────┘
      │               │
      │               ▼
      │         video_failed.set()
      │               │
      │               ▼
      │         Audio never starts
      │               │
      │               ▼
      ▼               ▼
   video_ready   Return False
      │               │
      ▼               ▼
  Audio start   Error logged
  (concurrent)  (Exit/Retry)
```

## 8. Configuration Parameters

```
Video Pipeline Configuration:
┌────────────────────────────────┐
│ Sensor ID              0        │
│ Capture Width          1920     │
│ Capture Height         1080     │
│ Display Width          960      │
│ Display Height         540      │
│ Framerate              30 fps   │
│ Flip Method            0        │
│ Hardware Accel         Yes      │
│ Input Format           NVMM     │
│ Output Format          BGR      │
│ Warmup Frames          20       │
└────────────────────────────────┘

Audio Pipeline Configuration:
┌────────────────────────────────┐
│ Device                 ALSA ID  │
│ Channels               6        │
│ Sample Rate            16000 Hz │
│ Format                 S16LE    │
│ Input Source           alsasrc  │
│ Output Format          PCM WAV  │
│ Output Extension       .wav     │
│ Recording Location     ~/records │
└────────────────────────────────┘
```

---

**Architecture**: Dual GStreamer Pipelines with Python Threading  
**Last Updated**: February 4, 2026
