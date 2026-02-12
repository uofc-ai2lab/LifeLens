"""
GStreamer Video Pipeline Module

Provides GStreamer-based video capture from CSI camera using NVIDIA Jetson hardware acceleration.
Handles pipeline initialization, frame capture, and state management.
"""

import cv2, os
from config.logger import Logger
from src_video.domain.constants import (
    CAPTURE_WIDTH,
    CAPTURE_HEIGHT,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
    COLOR_TEXT,
    FRAME_RATE,
    FLIP_METHOD,
)
from config.audio_settings import IS_JETSON

log = Logger("[video][camera]")

# Initialize GStreamer bindings
if IS_JETSON:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst

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
    Returns a GStreamer pipeline string for NVIDIA Jetson CSI camera capture.
    
    Uses hardware acceleration (nvarguscamerasrc, nvvidconv) for efficient video capture.
    Includes buffer management and stability optimizations for Jetson platform.
    
    Args:
        sensor_id: Camera sensor ID (default: 0)
        capture_width: Camera capture width in pixels
        capture_height: Camera capture height in pixels
        display_width: Display width in pixels
        display_height: Display height in pixels
        framerate: Target frame rate in FPS
        flip_method: Image flip method (0=none, 1=clockwise, 2=rotate-180, 3=counter-clockwise)
    
    Returns:
        GStreamer pipeline string for video capture via appsink
    """
    return (
        "nvarguscamerasrc sensor-id=%d bufapi-version=1 ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! "
        "appsink max-buffers=2 drop=true"
        % (
            sensor_id,
            capture_width,
            capture_height,
            framerate,
            flip_method,
            display_width,
            display_height,
        )
    )

def capture_frame_from_pipeline(frame, image_save_dir: str) -> bool:
    """
    Saves a single frame to disk from the video pipeline.
    """
    if frame is None:
        log.error("No frame to save")
        return False
    timestamp = cv2.getTickCount()
    filename = os.path.join(image_save_dir, f"captured_img_{timestamp}.jpg")

    if not cv2.imwrite(filename, frame):
        log.error(f"Failed to save frame to {filename}")
        return False
    
    log.info(f"Frame saved to {filename}")
    return True

def draw_overlay(frame, fps: float, processing: bool):
    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        COLOR_TEXT,
        2,
    )

    if processing:
        cv2.putText(
            frame,
            "PROCESSING...",
            (10, 60),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (0, 165, 255),
            2,
        )

class GStreamerVideoPipeline:
    """
    Manages a GStreamer video capture pipeline using NVIDIA Jetson CSI camera.
    
    Features:
    - Hardware-accelerated video capture (nvarguscamerasrc)
    - NVIDIA video format conversion (nvvidconv)
    - OpenCV integration via appsink
    - Proper pipeline state management
    - Warmup sequence to ensure stable operation
    """
    
    def __init__(self, flip_method: int = 0):
        """
        Initialize the GStreamer video pipeline.
        
        Args:
            flip_method: Image flip method (0=none, 2=rotate-180, etc.)
        """
        if not Gst.is_initialized():
            Gst.init(None)
        
        self.flip_method = flip_method
        self.video_capture = None
        self.is_initialized = False
    
    def start(self) -> bool:
        """
        Start the video capture pipeline.
        
        Performs warmup sequence to ensure camera is ready.
        Includes retry logic for transient buffer issues.
        
        Returns:
            bool: True if pipeline started successfully, False otherwise
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                pipeline = get_gstreamer_video_pipeline(flip_method=self.flip_method)
                
                debug = os.getenv("VIDEO_CAMERA_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
                if debug:
                    log.debug(f"Attempt {attempt + 1}/{max_retries}")
                    log.debug(f"Pipeline: {pipeline}")

                self.video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if not self.video_capture.isOpened():
                    log.error(f"Pipeline failed to open (attempt {attempt + 1}/{max_retries})")
                    if attempt < max_retries - 1:
                        import time
                        time.sleep(1)
                    continue

                log.info(f"Pipeline created, warming up (attempt {attempt + 1}/{max_retries})...")
                
                # Extended warmup sequence with more patience
                warmup_attempts = 30
                for i in range(warmup_attempts):
                    ok, frame = self.video_capture.read()
                    if ok and frame is not None:
                        log.success(f"Warmup complete on frame {i+1} - camera ready")
                        self.is_initialized = True
                        return True
                    import time
                    time.sleep(0.1)

                log.warning(f"Warmup failed after {warmup_attempts} frames (attempt {attempt + 1}/{max_retries})")
                self.video_capture.release()
                
                if attempt < max_retries - 1:
                    import time
                    time.sleep(2)
                    
            except Exception as e:
                log.error(f"Exception during startup (attempt {attempt + 1}/{max_retries}): {e}")
                if attempt < max_retries - 1:
                    import time
                    time.sleep(1)

        # All retries exhausted
        raise RuntimeError(
            f"Camera failed to initialize after {max_retries} attempts. "
            "Common solutions:\n"
            "  1. Restart nvargus-daemon: sudo systemctl restart nvargus-daemon\n"
            "  2. Check camera connection and ribbon cable\n"
            "  3. Verify no other process is using the camera (lsof /dev/video0)\n"
            "  4. Check dmesg for NVIDIA Argus errors: dmesg | tail -20"
        )


    def read_frame(self) -> tuple:
        """
        Read a frame from the video pipeline.
        
        Returns:
            tuple: (success: bool, frame: numpy.ndarray or None)
        """
        if not self.is_initialized or self.video_capture is None:
            return False, None
        
        try:
            ok, frame = self.video_capture.read()
            return ok, frame
        except Exception as e:
            print(f"[video][camera] ERROR reading frame: {e}")
            return False, None
    
    def stop(self) -> bool:
        """
        Stop the video capture pipeline.
        
        Returns:
            bool: True if pipeline stopped successfully
        """
        try:
            if self.video_capture is not None:
                self.video_capture.release()
                self.is_initialized = False
                print("[video][camera] Pipeline stopped")
                return True
            return False
        except Exception as e:
            print(f"[video][camera] ERROR stopping pipeline: {e}")
            return False
    
    def cleanup(self):
        """Cleanup pipeline resources."""
        self.stop()
