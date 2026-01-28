# MIT License
# Copyright (c) 2019-2022 JetsonHacks

import cv2
import os
from config.video_settings import IMAGE_SAVE_DIR
from src_video.domain.constants import (
    # Camera settings
    CAPTURE_WIDTH,
    CAPTURE_HEIGHT,
    DISPLAY_WIDTH,
    DISPLAY_HEIGHT
    )

""" 
gstreamer_pipeline returns a GStreamer pipeline for capturing from the CSI camera
Flip the image by setting the flip_method (most common values: 0 and 2)
display_width and display_height determine the size of each camera pane in the window on the screen
Default 1920x1080 displayd in a 1/4 size window
"""


def gstreamer_pipeline(
    sensor_id=0,
    capture_width=CAPTURE_WIDTH,
    capture_height=CAPTURE_HEIGHT,
    display_width=DISPLAY_WIDTH,
    display_height=DISPLAY_HEIGHT,
    framerate=30,
    flip_method=0,
):
    return (
        "nvarguscamerasrc sensor-id=%d ! "
        "video/x-raw(memory:NVMM), width=(int)%d, height=(int)%d, framerate=(fraction)%d/1 ! "
        "nvvidconv flip-method=%d ! "
        "video/x-raw, width=(int)%d, height=(int)%d, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
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


def capture_images(video_capture):
    ret_val, frame = video_capture.read()

    if not ret_val or frame is None:
        print("Error: Capturing images -> frame is None.")
        return False

    timestamp = cv2.getTickCount()
    filename = os.path.join(IMAGE_SAVE_DIR, f"captured_img_{timestamp}.jpg")

    if not cv2.imwrite(filename, frame):
        print("Error: Failed to save image.")
        return False

    print(f"Image saved as {filename}")
    return True

