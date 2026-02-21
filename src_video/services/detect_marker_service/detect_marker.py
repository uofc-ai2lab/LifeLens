# """
# AprilTag Marker Detection Service
# Calls the existing camera capture service and adds AprilTag detection
# """

import cv2
import numpy as np
import sys
import os
from pupil_apriltags import Detector
from src_video.domain.entities import AprilTagDetection
from config.logger import Logger
from config.video_settings import (
    # tag settings
    TAG_FAMILY,
    TAG_SIZE,
    TARGET_TAG_IDS,
    # Performance
    NTHREADS,
    QUAD_DECIMATE,
)

# ==================== CONSTANTS ====================
from src_video.domain.constants import (
    # Visualization colors
    COLOR_OUTLINE,
    COLOR_CORNERS,
    COLOR_CENTER,
    COLOR_ID_TEXT,
    COLOR_DISTANCE_TEXT,
    CAMERA_PARAMS
)
MIN_DECISION_MARGIN = 20  # Adjust based on environment

log = Logger("[video][apriltag]")

# ==================== APRILTAG DETECTOR ====================

# Suppress verbose apriltag warnings by redirecting stderr temporarily
log.info(f"Initializing AprilTag detector for {TAG_FAMILY}...")

# Redirect stderr to suppress C++ library warnings
stderr_fd = sys.stderr.fileno()
old_stderr = os.dup(stderr_fd)
devnull = os.open(os.devnull, os.O_WRONLY)
os.dup2(devnull, stderr_fd)

try:
    at_detector = Detector(
        families=TAG_FAMILY,
        nthreads=NTHREADS,
        quad_decimate=QUAD_DECIMATE,
        quad_sigma=0.0,
        refine_edges=1,
        decode_sharpening=0.25,
        debug=0
    )
finally:
    # Restore stderr
    os.dup2(old_stderr, stderr_fd)
    os.close(old_stderr)
    os.close(devnull)

log.success("AprilTag detector initialized successfully")

# ==================== DETECTION FUNCTIONS ====================

def detect_apriltags(source, show_visualization=True, print_info=True):
    """
    Detect AprilTags in a frame
    
    Args:
        frame: BGR image from camera
        show_visualization: Whether to draw detection results on frame
    
    Returns:
        List of detected tags
    """

    # Accept either a cv2.VideoCapture-like object (with .read()) or a raw frame (np.ndarray).
    if hasattr(source, "read"):
        ret_val, frame = source.read()
        if not ret_val:
            log.error("Failed to grab frame for AprilTag detection")
            return []
    else:
        frame = source
        if frame is None:
            log.error("No frame provided for AprilTag detection")
            return []
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect tags - suppress stderr warnings from C++ library
    stderr_fd = sys.stderr.fileno()
    old_stderr = os.dup(stderr_fd)
    devnull = os.open(os.devnull, os.O_WRONLY)
    os.dup2(devnull, stderr_fd)
    
    try:
        tags = at_detector.detect(
            gray,
            estimate_tag_pose=True,
            camera_params=CAMERA_PARAMS,
            tag_size=TAG_SIZE
        )
    finally:
        os.dup2(old_stderr, stderr_fd)
        os.close(old_stderr)
        os.close(devnull)

    tags = [tag for tag in tags if tag.decision_margin > MIN_DECISION_MARGIN]
    
    if show_visualization:
        draw_detections(frame, tags)
    
    # Filter by target IDs if specified
    if TARGET_TAG_IDS is not None:
        tags = [tag for tag in tags if tag.tag_id in TARGET_TAG_IDS]

    detections = detect_tags(tags)

    # Print detection information
    if tags and print_info:
        print_tag_info(tags)
    
    return detections

def draw_detections(frame, tags):
    """Draw detection visualizations on frame"""
    for tag in tags:
        # Draw tag outline
        corners = tag.corners.astype(int)
        cv2.polylines(frame, [corners], True, COLOR_OUTLINE, 3)
        
        # Draw corner points
        for corner in corners:
            cv2.circle(frame, tuple(corner.astype(int)), 5, COLOR_CORNERS, -1)
        
        # Draw center point
        center = tuple(tag.center.astype(int))
        cv2.circle(frame, center, 8, COLOR_CENTER, -1)
        
        # Display tag ID
        cv2.putText(
            frame,
            f"ID: {tag.tag_id}",
            (center[0] - 30, center[1] - 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            COLOR_ID_TEXT,
            2
        )
        
        # Display distance if pose was estimated
        if tag.pose_t is not None:
            distance = np.linalg.norm(tag.pose_t) * 100  # Convert to cm
            cv2.putText(
                frame,
                f"{distance:.1f}cm",
                (center[0] - 30, center[1] + 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                COLOR_DISTANCE_TEXT,
                2
            )


def print_tag_info(tags):
    """Print detailed information about detected tags"""
    if not tags:
        return

    log.info("==================================================")
    for tag in tags:
        log.info(f"[DETECTED] Tag ID: {tag.tag_id}")
        log.info(f"  Center: ({tag.center[0]:.1f}, {tag.center[1]:.1f})")
        log.info("  Corners:")
        for i, corner in enumerate(tag.corners):
            log.info(f"    Corner {i}: ({corner[0]:.1f}, {corner[1]:.1f})")

        if tag.pose_t is not None:
            distance = np.linalg.norm(tag.pose_t) * 100
            log.info(f"  Distance: {distance:.1f} cm")
            log.info(f"  Translation (x, y, z): ({tag.pose_t[0][0]:.3f}, {tag.pose_t[1][0]:.3f}, {tag.pose_t[2][0]:.3f})")

        if tag.pose_R is not None:
            log.info("  Rotation matrix available: Yes")

        log.info(f"  Decision margin: {tag.decision_margin:.2f}")
        log.info(f"  Hamming distance: {tag.hamming}")
        log.info("--------------------------------------------------")



def detect_tags(tags):
    # take photos, pause detection, call next service
    detections: list[AprilTagDetection] = []
    for tag in tags:
        detections.append(
            AprilTagDetection(
                tag_id=tag.tag_id,
                center_x=tag.center[0],
                center_y=tag.center[1],
                corners=[(corner[0], corner[1]) for corner in tag.corners],
                distance=(
                    np.linalg.norm(tag.pose_t)
                    if tag.pose_t is not None
                    else -1
                ),
                decision_margin=tag.decision_margin,
            )
        )

    for dt in detections:
        dt.print_info()

    return detections