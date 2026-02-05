"""
GStreamer Audio Pipeline Module

Provides GStreamer-based audio recording using alsasrc (ALSA audio input).
Handles multi-channel USB audio device recording with proper pipeline management.
"""

import gi
import os
import time
from typing import Optional

# Initialize GStreamer bindings
gi.require_version("Gst", "1.0")
from gi.repository import Gst, GLib

from src_audio.domain.constants import get_gstreamer_audio_pipeline


class GStreamerAudioPipeline:
    """
    Manages a GStreamer audio recording pipeline using ALSA source.
    
    Features:
    - Multi-channel audio capture from USB audio devices
    - Automatic WAV encoding
    - Proper pipeline state management
    - Thread-safe operation
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
        """
        Start the audio recording pipeline.
        
        Returns:
            bool: True if pipeline started successfully, False otherwise
        """
        try:
            pipeline_string = get_gstreamer_audio_pipeline(self.output_file)
            
            print(f"[GStreamer Audio] Starting pipeline: {pipeline_string}")
            
            self.pipeline = Gst.parse_launch(pipeline_string)
            if not self.pipeline:
                print("[GStreamer Audio] ERROR: Failed to create pipeline")
                return False
            
            self.bus = self.pipeline.get_bus()
            self.bus.add_signal_watch()
            
            ret = self.pipeline.set_state(Gst.State.PLAYING)
            if ret == Gst.StateChangeReturn.FAILURE:
                print("[GStreamer Audio] ERROR: Failed to set pipeline to PLAYING state")
                return False
            
            self.is_recording = True
            print(f"[GStreamer Audio] Recording started to {self.output_file}")
            return True
            
        except Exception as e:
            print(f"[GStreamer Audio] ERROR starting pipeline: {e}")
            return False
    
    def stop(self) -> bool:
        """
        Stop the audio recording pipeline.
        
        Returns:
            bool: True if pipeline stopped successfully, False otherwise
        """
        try:
            if self.pipeline is None:
                return False
            
            ret = self.pipeline.set_state(Gst.State.NULL)
            if ret == Gst.StateChangeReturn.FAILURE:
                print("[GStreamer Audio] WARNING: Pipeline may not have stopped cleanly")
            
            self.is_recording = False
            print(f"[GStreamer Audio] Recording stopped. File saved to {self.output_file}")
            return True
            
        except Exception as e:
            print(f"[GStreamer Audio] ERROR stopping pipeline: {e}")
            return False
    
    def is_recording_active(self) -> bool:
        """
        Check if the pipeline is actively recording.
        
        Returns:
            bool: True if recording is active
        """
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
            print(f"[GStreamer Audio] WARNING: Error during cleanup: {e}")
