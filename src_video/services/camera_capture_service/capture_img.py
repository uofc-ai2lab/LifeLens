# MIT License
# Copyright (c) 2019-2022 JetsonHacks
import time 
import cv2
import os
from config.logger import Logger
from config.video_settings import IMAGE_SAVE_DIR
from src_video.domain.constants import (
    CAPTURE_WIDTH,
    CAPTURE_HEIGHT,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT,
    COLOR_TEXT,
    FRAME_RATE,
    FLIP_METHOD
    )

log = Logger("[video][camera]")

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""

def initialize_camera(flip_method: int = 0) -> cv2.VideoCapture:
    debug = os.getenv("VIDEO_CAMERA_DEBUG", "0").strip().lower() in {"1", "true", "yes", "on"}

    configs = [
        (CAPTURE_WIDTH, CAPTURE_HEIGHT, DISPLAY_WIDTH, DISPLAY_HEIGHT),
        (1280, 720, 960, 540),
        (1280, 720, 640, 360),
        (640, 360, 640, 360),
    ]

    last_err = ""
    for capture_w, capture_h, display_w, display_h in configs:
        pipeline = gstreamer_pipeline(
            capture_width=capture_w,
            capture_height=capture_h,
            display_width=display_w,
            display_height=display_h,
            flip_method=flip_method,
        )
        if debug:
            log.debug(f"gstreamer pipeline: {pipeline}")

        video_capture = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
        if not video_capture.isOpened():
            last_err = "pipeline did not open"
            video_capture.release()
            continue

        # Warmup
        for _ in range(20):
            ok, frame = video_capture.read()
            if ok and frame is not None:
                log.success("Camera started successfully")
                return video_capture
            last_err = "read() returned no frame"
            time.sleep(0.05)

        video_capture.release()

    raise RuntimeError(
        "Error: Camera opened but no frames received. "
        "If you see 'Failed to create CaptureSession', restart `nvargus-daemon`, "
        "ensure no other process is using the CSI camera, and verify the camera ribbon/port. "
        f"({last_err})"
    )

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
