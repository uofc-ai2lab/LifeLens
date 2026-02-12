"""
GStreamer Audio Pipeline Module

Provides GStreamer-based audio recording using alsasrc (ALSA audio input).
Handles multi-channel USB audio device recording with proper pipeline management.
"""

import gi, os

# Initialize GStreamer bindings
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib
from config.logger import Logger


from src_audio.domain.constants import (
    ARECORD_DEVICE,
    AUDIO_FORMAT,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS
)

log = Logger("[audio][microphone]")

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
