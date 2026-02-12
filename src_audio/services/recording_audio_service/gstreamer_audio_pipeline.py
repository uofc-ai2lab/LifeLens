"""
GStreamer Audio Pipeline Module

Provides GStreamer-based audio recording using alsasrc (ALSA audio input).
Handles multi-channel USB audio device recording with proper pipeline management.
"""

import os, time
from pathlib import Path
from config.logger import Logger
from config.audio_settings import IS_JETSON
from src_audio.domain.constants import CHUNK_SECONDS
from src_audio.domain.constants import (
    ARECORD_DEVICE,
    AUDIO_FORMAT,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS
)

# Initialize GStreamer bindings
if IS_JETSON:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst, GLib

log = Logger("[audio][microphone]")

def record_one_chunk(output_dir: str | Path, stop_event) -> bool:
    """
    Records ONE audio chunk using GStreamer pipeline.
    Records for CHUNK_SECONDS or until stop_event is set.
    Saves the chunk as WAV file in output_dir.
    
    Args:
        output_dir: Directory to save the chunk file
        stop_event: threading.Event to signal early stop
    
    Returns:
        True if a chunk file was successfully written, False otherwise
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    output_path = output_dir / f"recording_{ts}.wav"

    log.info(f"Recording chunk -> {output_path.name}")

    # Create GStreamer audio pipeline for this chunk
    pipeline = GStreamerAudioPipeline(str(output_path))

    try:
        # Start recording
        if not pipeline.start():
            log.error(f"Failed to start recording pipeline")
            return False

        # Record for CHUNK_SECONDS or until stop_event is set
        start_time = time.time()
        while True:
            # Check if stop event was set (user pressed ENTER)
            if stop_event.is_set():
                log.info("Stop event received, stopping chunk recording")
                break
            
            # Check if chunk duration exceeded
            elapsed = time.time() - start_time
            if elapsed >= CHUNK_SECONDS:
                log.info(f"Chunk duration ({elapsed:.1f}s) reached, saving chunk")
                break
            
            time.sleep(0.1)

        pipeline.stop()
        pipeline.cleanup()

    except Exception as e:
        log.error(f"Error during chunk recording: {e}")
        try:
            pipeline.stop()
            pipeline.cleanup()
        except:
            pass
        return False

    ok = output_path.exists() and output_path.stat().st_size > 0
    if ok:
        file_size_mb = output_path.stat().st_size / (1024 * 1024)
        log.success(f"Chunk written -> {output_path.name} ({file_size_mb:.2f}MB)")
    else:
        log.warning("Chunk missing or empty; skipping")

    return ok

def get_gstreamer_audio_pipeline(output_file: str) -> str:
    """
    Returns a GStreamer pipeline for audio recording from multi-channel USB audio device.
    """
    return (
        f"alsasrc device={ARECORD_DEVICE} ! "
        f"audioconvert ! "
        f"audioresample ! "
        f"audio/x-raw,format={AUDIO_FORMAT},rate={AUDIO_SAMPLE_RATE},channels={AUDIO_CHANNELS} ! "
        f"wavenc ! "
        f"filesink location={output_file}"
    )
    
class GStreamerAudioPipeline:
    """
    Manages a GStreamer audio recording pipeline using ALSA source.
    
    Features:
    - Multi-channel audio capture from USB audio devices
    - Automatic WAV encoding
    """
    
    def __init__(self, output_file: str):
        """
        Initialize the GStreamer audio pipeline.
        
        Args:
            output_file: Path where the output WAV file will be saved
        """
        if not Gst.is_initialized():
            Gst.init(None)
        
        self.output_file = output_file
        self.pipeline = None
        self.bus = None
        self.is_recording = False
        self._ensure_output_dir()
    
    def _ensure_output_dir(self):
        """Ensure output directory exists."""
        output_dir = os.path.dirname(self.output_file)
        if output_dir and not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
    
    def start(self) -> bool:
        """Start the audio recording pipeline."""
        
        try:
            pipeline_string = get_gstreamer_audio_pipeline(self.output_file)
            
            log.debug(f"Starting pipeline: {pipeline_string}")
            
            self.pipeline = Gst.parse_launch(pipeline_string)
            if not self.pipeline:
                log.error("Failed to create pipeline")
                return False
            
            self.bus = self.pipeline.get_bus()
            self.bus.add_signal_watch()
            
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                log.error("Failed to set pipeline to PLAYING state")
                return False
            
            self.is_recording = True
            log.success(f"Recording started to {self.output_file}")
            return True
            
        except Exception as e:
            log.error(f"Error starting pipeline: {e}")
            return False
    
    def stop(self) -> bool:
        """Stop the audio recording pipeline."""
        try:
            if self.pipeline is None:
                return False
            
            ret = self.pipeline.set_state(Gst.State.NULL)
            if ret == Gst.StateChangeReturn.FAILURE:
                log.warning("Pipeline may not have stopped cleanly")
            
            self.is_recording = False
            log.success(f"Recording stopped. File saved to {self.output_file}")
            return True
            
        except Exception as e:
            log.error(f"Error stopping pipeline: {e}")
            return False
    
    def is_recording_active(self) -> bool:
        """Check if the pipeline is actively recording."""
        if self.pipeline is None:
            return False
        
        state = self.pipeline.get_state(0).state
        return state == Gst.State.PLAYING and self.is_recording
    
    def cleanup(self):
        """Cleanup pipeline resources."""
        try:
            if self.bus:
                self.bus.remove_signal_watch()
            if self.pipeline:
                self.pipeline.set_state(Gst.State.NULL)
                self.pipeline = None
        except Exception as e:
            log.warning(f"Error during cleanup: {e}")
