"""
GStreamer Video Pipeline for NVIDIA Jetson Camera Capture

Handles hardware-accelerated CSI camera capture using GStreamer.
Provides frame-by-frame access through OpenCV integration.
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import cv2, os
import threading
import time
from typing import Tuple, Optional
import numpy as np

from src_video.domain.constants import (
    CAPTURE_WIDTH,
    CAPTURE_HEIGHT,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
    FRAME_RATE,
    FLIP_METHOD,
)


def get_gstreamer_video_pipeline(
    sensor_id: int = 0,
    capture_width: int = CAPTURE_WIDTH,
    capture_height: int = CAPTURE_HEIGHT,
    display_width: int = DISPLAY_WIDTH,
    display_height: int = DISPLAY_HEIGHT,
    framerate: int = FRAME_RATE,
    flip_method: int = FLIP_METHOD,
) -> str:
    """
    Generate GStreamer pipeline string for NVIDIA Jetson CSI camera.
    
    Args:
        sensor_id: Camera sensor ID (0 for first camera)
        capture_width: Capture resolution width
        capture_height: Capture resolution height
        display_width: Display resolution width
        display_height: Display resolution height
        framerate: Target framerate in FPS
        flip_method: Image flip method (0=none, 1=clockwise, 2=rotate-180, 3=counter-clockwise)
    
    Returns:
        GStreamer pipeline string
    """
    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! bufapi-version=1 ! "
        f"video/x-raw(memory:NVMM), "
        f"width={capture_width}, height={capture_height}, "
        f"format=BGRx, framerate={framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width={display_width}, height={display_height} ! "
        f"videoconvert ! "
        f"video/x-raw, format=BGR ! "
        f"appsink max-buffers=2 drop=true"
    )


class GStreamerVideoPipeline:
    """
    NVIDIA Jetson hardware-accelerated video pipeline using GStreamer.
    
    Features:
        - Hardware acceleration via nvarguscamerasrc and nvvidconv
        - Frame warmup sequence for stable operation
        - OpenCV integration via appsink
        - Proper state management and resource cleanup
        - Automatic retry mechanism on initialization failure
    """
    
    # Default retry configuration
    DEFAULT_MAX_RETRIES = 3
    DEFAULT_RETRY_DELAY = 1.0  # seconds
    
    def __init__(self, flip_method: int = 0, warmup_frames: int = 20, max_retries: int = DEFAULT_MAX_RETRIES, retry_delay: float = DEFAULT_RETRY_DELAY):
        """
        Initialize GStreamer video pipeline.
        
        Args:
            flip_method: Image flip method (0=none, 2=180°)
            warmup_frames: Number of frames to skip during warmup
            max_retries: Maximum number of initialization retry attempts (default: 3)
            retry_delay: Delay in seconds between retry attempts (default: 1.0)
        """
        if not Gst.is_initialized():
            Gst.init(None)
        
        self.flip_method = flip_method
        self.warmup_frames = warmup_frames
        self.max_retries = max_retries
        self.retry_delay = retry_delay
        self.pipeline_string = get_gstreamer_video_pipeline(flip_method=flip_method)
        self.pipeline = None
        self.caps_filter = None
        self.appsink = None
        self.video_capture = None
        self.is_initialized = False
        self._lock = threading.Lock()
        self._frames_captured = 0
        self._attempts = 0
        self._last_error = None
        
    def start(self) -> bool:
        """
        Start the GStreamer pipeline with automatic retry on failure.
        
        Attempts to initialize the pipeline up to max_retries times with
        retry_delay seconds between attempts.
        
        Returns:
            True if pipeline started successfully, False if all retries exhausted
        """
        for attempt in range(self.max_retries):
            self._attempts = attempt + 1
            try:
                print(f"[video][camera] Initialization attempt {self._attempts}/{self.max_retries}")
                print(f"[video][camera] Pipeline: {self.pipeline_string}")
                
                debug = os.getenv("VIDEO_CAMERA_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
                if debug:
                    print(f"[video][camera] Attempt {attempt + 1}/{self.max_retries}")
                    print(f"[video][camera] GStreamer pipeline: {self.pipeline_string}")
                    
                # Create GStreamer pipeline
                self.video_capture = cv2.VideoCapture(
                    self.pipeline_string,
                    cv2.CAP_GSTREAMER
                )
                
                if not self.video_capture.isOpened():
                    error_msg = f"Unable to open camera (GStreamer pipeline failed to open)"
                    self._last_error = error_msg
                    print(f"[video][camera] ERROR: {error_msg} (attempt {self._attempts}/{self.max_retries})")
                    
                    # Cleanup before retry
                    if self.video_capture is not None:
                        self.video_capture.release()
                        self.video_capture = None
                    
                    # Retry with delay if not last attempt
                    if attempt < self.max_retries - 1:
                        print(f"[video][camera] Retrying in {self.retry_delay}s...")
                        time.sleep(self.retry_delay)
                    continue
                
                # Set buffer size to 1 for low-latency frame capture
                self.video_capture.set(cv2.CAP_PROP_BUFFERSIZE, 1)
                
                print("[video][camera] GStreamer pipeline opened, starting warmup sequence")
                
                # Warmup sequence: skip initial frames for stable operation
                warmup_failed = False
                for i in range(self.warmup_frames):
                    ret, frame = self.video_capture.read()
                    if not ret:
                        error_msg = f"Failed to read frame during warmup at frame {i}"
                        self._last_error = error_msg
                        print(f"[video][camera] ERROR: {error_msg}")
                        self.video_capture.release()
                        self.video_capture = None
                        warmup_failed = True
                        break
                    
                    if debug or (i + 1) % 5 == 0:
                        print(f"[video][camera] Warmup frame {i+1}/{self.warmup_frames}")
                
                if warmup_failed:
                    # Retry warmup failed attempt
                    if attempt < self.max_retries - 1:
                        print(f"[video][camera] Retrying in {self.retry_delay}s...")
                        time.sleep(self.retry_delay)
                    continue
                
                self.is_initialized = True
                self._frames_captured = 0
                print(f"[video][camera] ✓ GStreamer pipeline ready (attempt {self._attempts}/{self.max_retries})")
                return True
                
            except Exception as e:
                error_msg = str(e)
                self._last_error = error_msg
                print(f"[video][camera] ERROR: Failed to initialize: {error_msg} (attempt {self._attempts}/{self.max_retries})")
                
                if self.video_capture is not None:
                    try:
                        self.video_capture.release()
                    except:
                        pass
                    self.video_capture = None
                
                if attempt < self.max_retries - 1:
                    print(f"[video][camera] Retrying in {self.retry_delay}s...")
                    time.sleep(self.retry_delay)
        
        print(f"[video][camera] ✗ Failed to initialize after {self.max_retries} attempts")
        if self._last_error:
            print(f"[video][camera] Last error: {self._last_error}")
        return False
    
    def get_attempts(self) -> int:
        """Get the number of initialization attempts made."""
        return self._attempts
    
    def get_last_error(self) -> Optional[str]:
        """Get the last error message from initialization."""
        return self._last_error
    
    def read_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Read a single frame from the pipeline.
        
        Returns:
            Tuple of (success: bool, frame: ndarray or None)
        """
        if not self.is_initialized or self.video_capture is None:
            return False, None
        
        try:
            with self._lock:
                ret, frame = self.video_capture.read()
                if ret:
                    self._frames_captured += 1
                return ret, frame
        except Exception as e:
            print(f"[video][camera] ERROR: Failed to read frame: {e}")
            return False, None
    
    def stop(self) -> bool:
        """
        Stop the GStreamer pipeline.
        
        Returns:
            True if stopped successfully
        """
        try:
            if self.video_capture is not None:
                self.video_capture.release()
                self.video_capture = None
                self.is_initialized = False
                print(f"[video][camera] Pipeline stopped (captured {self._frames_captured} frames)")
            return True
        except Exception as e:
            print(f"[video][camera] ERROR: Failed to stop: {e}")
            return False
    
    def cleanup(self):
        """Clean up all resources."""
        self.stop()
        if self.pipeline is not None:
            self.pipeline = None
