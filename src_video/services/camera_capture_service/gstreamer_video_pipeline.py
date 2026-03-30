"""
GStreamer Video Pipeline Module

Provides GStreamer-based video capture from CSI camera using NVIDIA Jetson hardware acceleration.
Handles pipeline initialization, frame capture, and state management.
"""

import cv2, os, time
from config.logger import Logger
from src_video.domain.constants import (
    CAPTURE_WIDTH,
    CAPTURE_HEIGHT,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
    SENSOR_ID,
    CAMERA_BRIGHTNESS,
    COLOR_TEXT,
    FRAME_RATE,
    FLIP_METHOD,
)
from config.audio_settings import IS_JETSON

log = Logger("[video][camera]")

configs = [
    (CAPTURE_WIDTH, CAPTURE_HEIGHT, DISPLAY_WIDTH, DISPLAY_HEIGHT),
    (1280, 720, 960, 540),
    (1280, 720, 640, 360),
    (640, 480, 640, 360),
]

# Initialize GStreamer bindings
if IS_JETSON:
    import gi
    gi.require_version("Gst", "1.0")
    from gi.repository import Gst
else:
    Gst = None

def get_gstreamer_video_pipeline(
    sensor_id: int = SENSOR_ID,
    capture_width: int = CAPTURE_WIDTH,
    capture_height: int = CAPTURE_HEIGHT,
    display_width: int = DISPLAY_WIDTH,
    display_height: int = DISPLAY_HEIGHT,
    framerate: int = FRAME_RATE,
    flip_method: int = FLIP_METHOD,
    brightness: float = CAMERA_BRIGHTNESS,
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
        brightness: videobalance brightness in [-1.0, 1.0]; positive brightens image
    
    Returns:
        GStreamer pipeline string for video capture via appsink
    """
    brightness = max(-1.0, min(1.0, float(brightness)))

    return (
        f"nvarguscamerasrc sensor-id={sensor_id} ! "
        f"video/x-raw(memory:NVMM), width=(int){capture_width}, height=(int){capture_height}, framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        f"videoconvert ! videobalance brightness={brightness:.3f} ! "
        f"video/x-raw, format=(string)BGR ! "
        f"appsink drop=true max-buffers=1 sync=false"
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
    
    def __init__(self, flip_method: int = FLIP_METHOD):
        """
        Initialize the GStreamer video pipeline.
        
        Args:
            flip_method: Image flip method (0=none, 2=rotate-180, etc.)
        """
        try:
            if not Gst.is_initialized():
                Gst.init(None)
        except Exception as e:
                raise RuntimeError(f"Failed to initialize GStreamer (Gst.init): {e}") from e
        
        self.flip_method = flip_method
        self.video_capture = None
        self.is_initialized = False

    def _release_capture(self, settle_s: float = 0.25) -> None:
        if self.video_capture is not None:
            try:
                self.video_capture.release()
            except Exception:
                pass
            finally:
                self.video_capture = None
        self.is_initialized = False
        if settle_s > 0:
            time.sleep(settle_s)

    def _warmup_capture(self, seconds: float = 3.0, min_good: int = 5) -> bool:
        """Warm up the capture by requiring a streak of good frames within a time budget.

        This is more robust than "first frame must be ok" and tolerates
        transient Argus / nvarguscamerasrc hiccups.
        """
        if self.video_capture is None or not self.video_capture.isOpened():
            return False

        deadline = time.time() + seconds
        good = 0

        while time.time() < deadline:
            ok, frame = self.video_capture.read()

            if ok and frame is not None and getattr(frame, "size", 0) != 0:
                good += 1
                if good >= min_good:
                    return True
            else:
                good = 0 # reset streak if we see a bad / empty frame

            time.sleep(0.01)

        return False
    
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
                self._release_capture(settle_s=0)
                pipeline = get_gstreamer_video_pipeline(flip_method=self.flip_method)
                backoff_s = 2 ** attempt
                
                debug = os.getenv("VIDEO_CAMERA_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}
                if debug:
                    log.debug(f"Attempt {attempt + 1}/{max_retries}")
                    log.debug(f"Pipeline: {pipeline}")

                self.video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
                if not self.video_capture.isOpened():
                    log.error(f"Pipeline failed to open (attempt {attempt + 1}/{max_retries})")
                    self._release_capture(settle_s=0.5)
                    if attempt < max_retries - 1:
                        log.warning(f"Retrying camera open in {backoff_s}s")
                        time.sleep(backoff_s)
                    continue

                log.info(f"Pipeline created, warming up (attempt {attempt + 1}/{max_retries})...")

                # Time-based warmup with streak of valid frames to avoid false negatives
                if self._warmup_capture(seconds=4.0, min_good=5):
                    log.success("Warmup complete - camera ready")
                    self.is_initialized = True
                    return True

                log.warning(
                    f"Warmup failed within time budget (attempt {attempt + 1}/{max_retries}); "
                    "releasing and retrying pipeline if retries remain."
                )
                self._release_capture(settle_s=0.5)

                if attempt < max_retries - 1:
                    log.warning(f"Retrying camera warmup in {backoff_s}s")
                    time.sleep(backoff_s)
                    
            except Exception as e:
                log.error(f"Exception during startup (attempt {attempt + 1}/{max_retries}): {e}")
                self._release_capture(settle_s=0.5)
                if attempt < max_retries - 1:
                    log.warning(f"Retrying camera startup in {backoff_s}s")
                    time.sleep(backoff_s)

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
                self._release_capture(settle_s=0)
                print("[video][camera] Pipeline stopped")
                return True
            return False
        except Exception as e:
            print(f"[video][camera] ERROR stopping pipeline: {e}")
            return False
    
    def cleanup(self):
        """Cleanup pipeline resources."""
        self.stop()
