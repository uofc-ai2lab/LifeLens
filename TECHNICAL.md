# LifeLens Technical Overview

This document summarizes the audio and video services, how they run in parallel, and the GStreamer pipeline pieces.

## Entry Points

- Full system (camera + mic): `python main.py`
- Audio only (mic + processing): `python -m src_audio.main_audio`
- Video only (cam + processing): `python -m src_video.main_video`
- Audio dev (process existing chunks): `python -m src_audio.main_audio --dev`
- Video dev (process images, no camera): `python -m src_video.main_video --dev`

Note: The live camera pipeline is started by `main.py`. 

## Audio Pipeline

**Modules**
- `src_audio/main_audio.py`
- `src_audio/services/recording_audio_service/gstreamer_audio_pipeline.py`
- `src_audio/services/transcription_service/transcription_whispertrt.py`
- `src_audio/services/anonymization_service/transcript_anonymization.py`
- `src_audio/services/medication_extraction_service/medication_extraction.py`
- `src_audio/services/intervention_extraction_service/intervention_extraction.py`

**Flow**
1. `record_one_chunk()` writes WAV chunks to `data/audio/audio_chunks/`.
2. A worker thread processes chunks:
   - Transcription
   - Anonymization
   - Medication extraction
   - Intervention extraction
3. Processed artifacts are written under `data/audio/processed_audio/<chunk_stem>/`.

**GStreamer audio chain**
```
alsasrc -> audioconvert -> audioresample -> wavenc -> filesink
```

## Video Pipeline

**Modules**
- `src_video/main_video.py`
- `src_video/services/camera_capture_service/gstreamer_video_pipeline.py`
- `src_video/services/detection_service/detect_body_parts.py`
- `src_video/services/classification_service/infer_injuries_on_crops.py`
- `src_video/services/deidentification_service/deidentify.py`
- `src_video/services/detect_marker_service/detect_marker.py`

**Flow**
1. `GStreamerVideoPipeline.start()` initializes the CSI camera.
2. Frames are read via `read_frame()` and checked for markers.
3. Snapshots are saved to `data/video/saved_imgs/`.
4. The batch worker runs detection, classification, and de-identification.
5. Outputs land under `data/video/output_files/`.

**GStreamer video chain**
```
nvarguscamerasrc -> nvvidconv -> videoconvert -> appsink
```

## Parallelism

- `main.py` runs video and audio in separate threads.
- Video starts first and signals readiness.
- Audio starts only after video is ready.
- `scripts/run_jetson_startup_tasks.sh` runs before initialization to reset the camera and validate GStreamer plugins.

## Quick Troubleshooting

- Camera reset: `sudo systemctl restart nvargus-daemon`
- Verify GStreamer tools: `gst-launch-1.0 --version`
- Check camera device: `ls -la /dev/video*`
