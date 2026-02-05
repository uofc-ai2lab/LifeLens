"""
GStreamer Audio Pipeline for USB Audio Device Recording

Handles multi-channel audio capture from USB devices using GStreamer.
Provides WAV file output with automatic format conversion.
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib, GObject
import threading
import time
from typing import Optional
from pathlib import Path

from src_audio.domain.constants import (
    ARECORD_DEVICE,
    AUDIO_SAMPLE_RATE,
    AUDIO_CHANNELS,
    AUDIO_FORMAT,
)


def get_gstreamer_audio_pipeline(output_file: str) -> str:
    """
    Generate GStreamer pipeline string for audio recording.
    
    Args:
        output_file: Path to output WAV file
    
    Returns:
        GStreamer pipeline string
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
    GStreamer-based audio recording pipeline for USB audio devices.
    
    Features:
        - Multi-channel ALSA audio input
        - Automatic format/sample rate conversion
        - WAV file output
        - Recording status monitoring
        - Proper state management and cleanup
    """
    
    def __init__(self, output_file: str):
        """
        Initialize GStreamer audio pipeline.
        
        Args:
            output_file: Path where WAV file will be saved
        """
        Gst.init(None)
        
        self.output_file = output_file
        self.pipeline_string = get_gstreamer_audio_pipeline(output_file)
        self.pipeline = None
        self.bus = None
        self.is_recording = False
        self._lock = threading.Lock()
        self._message_loop = None
        self._loop_thread = None
        self.start_time = None
        
        print(f"[audio] Initialized audio pipeline: {output_file}")
    
    def start(self) -> bool:
        """
        Start the GStreamer audio pipeline.
        
        Returns:
            True if pipeline started successfully, False otherwise
        """
        try:
            with self._lock:
                if self.is_recording:
                    print("[audio] ERROR: Pipeline already recording")
                    return False
                
                print("[audio] Starting GStreamer audio pipeline")
                print(f"[audio] Output file: {self.output_file}")
                print(f"[audio] Device: {ARECORD_DEVICE}")
                print(f"[audio] Format: {AUDIO_CHANNELS}ch, {AUDIO_SAMPLE_RATE}Hz, {AUDIO_FORMAT}")
                
                # Create pipeline from string
                self.pipeline = Gst.parse_launch(self.pipeline_string)
                if self.pipeline is None:
                    print("[audio] ERROR: Failed to create pipeline from string")
                    return False
                
                # Set up bus for error handling
                self.bus = self.pipeline.get_bus()
                self.bus.add_signal_watch()
                self.bus.connect("message", self._on_bus_message)
                
                # Start event loop for bus messages
                self._message_loop = GLib.MainLoop()
                self._loop_thread = threading.Thread(
                    target=self._run_message_loop,
                    daemon=True,
                    name="AudioBusThread"
                )
                self._loop_thread.start()
                
                # Set pipeline to PLAYING state
                ret = self.pipeline.set_state(Gst.State.PLAYING)
                if ret == Gst.StateChangeReturn.FAILURE:
                    print("[audio] ERROR: Failed to set pipeline to PLAYING state")
                    self._cleanup_pipeline()
                    return False
                
                self.is_recording = True
                self.start_time = time.time()
                
                # Give pipeline a moment to stabilize
                time.sleep(0.5)
                
                # Verify it's actually playing
                state = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)[1]
                if state != Gst.State.PLAYING:
                    print(f"[audio] ERROR: Pipeline not in PLAYING state (state={state})")
                    self._cleanup_pipeline()
                    return False
                
                print("[audio] GStreamer audio pipeline started")
                return True
                
        except Exception as e:
            print(f"[audio] ERROR: Failed to start pipeline: {e}")
            self._cleanup_pipeline()
            return False
    
    def stop(self) -> bool:
        """
        Stop the GStreamer audio pipeline.
        
        Returns:
            True if stopped successfully
        """
        try:
            with self._lock:
                if not self.is_recording:
                    return True
                
                print("[audio] Stopping audio pipeline")
                
                # Send EOS (End Of Stream) event to flush data and properly close the file
                if self.pipeline is not None:
                    print("[audio] Sending EOS event to pipeline")
                    self.pipeline.send_event(Gst.Event.new_eos())
                    
                    # Wait for EOS to be processed (with timeout)
                    bus = self.pipeline.get_bus()
                    if bus:
                        msg = bus.timed_pop_filtered(
                            2 * Gst.SECOND,  # 2 second timeout
                            Gst.MessageType.EOS | Gst.MessageType.ERROR
                        )
                        if msg:
                            if msg.type == Gst.MessageType.ERROR:
                                err, debug = msg.parse_error()
                                print(f"[audio] Error during EOS: {err.message}")
                            elif msg.type == Gst.MessageType.EOS:
                                print("[audio] EOS received, file closed properly")
                    
                    # Now set pipeline to NULL state
                    self.pipeline.set_state(Gst.State.NULL)
                
                self.is_recording = False
                
                # Stop message loop
                if self._message_loop is not None:
                    self._message_loop.quit()
                
                # Calculate duration
                if self.start_time:
                    duration = time.time() - self.start_time
                    print(f"[audio] Recording stopped (duration: {duration:.1f}s)")
                
                self._cleanup_pipeline()
                return True
                
        except Exception as e:
            print(f"[audio] ERROR: Failed to stop pipeline: {e}")
            return False
    
    def is_recording_active(self) -> bool:
        """
        Check if pipeline is actively recording.
        
        Returns:
            True if recording, False otherwise
        """
        with self._lock:
            if self.pipeline is None:
                return False
            
            state = self.pipeline.get_state(Gst.CLOCK_TIME_NONE)[1]
            return state == Gst.State.PLAYING
    
    def cleanup(self):
        """Clean up all resources."""
        self.stop()
        self._cleanup_pipeline()
    
    def _run_message_loop(self):
        """Run the GLib event loop for bus messages in a separate thread."""
        try:
            self._message_loop.run()
        except Exception as e:
            print(f"[audio] ERROR in message loop: {e}")
    
    def _on_bus_message(self, bus, message):
        """Handle GStreamer bus messages."""
        msg_type = message.type
        
        if msg_type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"[audio] GStreamer ERROR: {err.message}")
            if debug:
                print(f"[audio] Debug info: {debug}")
            self.stop()
            
        elif msg_type == Gst.MessageType.EOS:
            print("[audio] End of stream reached")
            self.stop()
            
        elif msg_type == Gst.MessageType.WARNING:
            err, debug = message.parse_warning()
            print(f"[audio] GStreamer WARNING: {err.message}")
            
        return True
    
    def _cleanup_pipeline(self):
        """Internal cleanup of pipeline resources."""
        try:
            if self.bus is not None:
                self.bus.remove_signal_watch()
                self.bus = None
            
            if self.pipeline is not None:
                self.pipeline = None
                
        except Exception as e:
            print(f"[audio] ERROR during cleanup: {e}")
