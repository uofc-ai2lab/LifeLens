"""
AprilTag Marker Detection Service
Calls the existing camera capture service and adds AprilTag detection
"""

import cv2
import numpy as np
from pupil_apriltags import Detector
import time
from src_video.domain.entities import AprilTagDetection
from config.video_settings import (
    # tag settings
    TAG_FAMILY,
    TAG_SIZE,
    TARGET_TAG_IDS,
    CAMERA_PARAMS,
    # Performance
    NTHREADS,
    QUAD_DECIMATE,
    # Image save directory
    IMAGE_SAVE_DIR,
)
from src_video.services.camera_capture_service.capture_img import (
    gstreamer_pipeline,
    capture_images,
    video_stream,
)

COLOR_OUTLINE = (0, 255, 0)
COLOR_CORNERS = (255, 0, 0)
COLOR_CENTER = (0, 0, 255)
COLOR_TEXT = (0, 255, 255)
COLOR_ID_TEXT = (0, 255, 0)
COLOR_DISTANCE_TEXT = (255, 0, 0)

MIN_DECISION_MARGIN = 20  # Adjust based on environment

# ==================== APRILTAG DETECTOR ====================

# Initialize AprilTag detector
print(f"Initializing AprilTag detector for {TAG_FAMILY}...")
at_detector = Detector(
    families=TAG_FAMILY,
    nthreads=NTHREADS,
    quad_decimate=QUAD_DECIMATE,
    quad_sigma=0.0,
    refine_edges=1,
    decode_sharpening=0.25,
    debug=0
)
print("AprilTag detector initialized successfully!")

# ==================== DETECTION FUNCTIONS ====================

def detect_apriltags(frame, show_visualization=True):
    """
    Detect AprilTags in a frame
    
    Args:
        frame: BGR image from camera
        show_visualization: Whether to draw detection results on frame
    
    Returns:
        List of detected tags
    """
    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect tags
    tags = at_detector.detect(
        gray,
        estimate_tag_pose=True,
        camera_params=CAMERA_PARAMS,
        tag_size=TAG_SIZE
    )

    tags = [tag for tag in tags if tag.decision_margin > MIN_DECISION_MARGIN]
    
    if show_visualization:
        draw_detections(frame, tags)
    
    return tags


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

    print("\n" + "=" * 50)
    for tag in tags:
        print(f"[DETECTED] Tag ID: {tag.tag_id}")
        print(f"  Center: ({tag.center[0]:.1f}, {tag.center[1]:.1f})")
        print(f"  Corners:")
        for i, corner in enumerate(tag.corners):
            print(f"    Corner {i}: ({corner[0]:.1f}, {corner[1]:.1f})")

        if tag.pose_t is not None:
            distance = np.linalg.norm(tag.pose_t) * 100
            print(f"  Distance: {distance:.1f} cm")
            print(f"  Translation (x, y, z): ({tag.pose_t[0][0]:.3f}, {tag.pose_t[1][0]:.3f}, {tag.pose_t[2][0]:.3f})")

        if tag.pose_R is not None:
            print(f"  Rotation matrix available: Yes")

        print(f"  Decision margin: {tag.decision_margin:.2f}")
        print(f"  Hamming distance: {tag.hamming}")
        print("-" * 50)

# ==================== MAIN SERVICE ====================


async def run_marker_detection() -> bool:
    """Main detection service"""
    window_title = "AprilTag Marker Detection Service"

    print("\n" + "=" * 60)
    print("APRILTAG MARKER DETECTION SERVICE")
    print("=" * 60)
    print(f"Tag Family: {TAG_FAMILY}")
    print(f"Tag Size: {TAG_SIZE * 100}cm")
    print(f"Target IDs: {'All tags' if TARGET_TAG_IDS is None else TARGET_TAG_IDS}")
    print(f"Save Directory: {IMAGE_SAVE_DIR}")
    print(f"Threads: {NTHREADS}, Quad Decimate: {QUAD_DECIMATE}")
    print("\nControls:")
    print("  'q' or ESC  - Quit")
    print("  'e'         - Save current frame (single)")
    print("  'r'         - Record 10 frames (2s intervals)")
    print("  'i'         - Toggle info printing")
    print("=" * 60 + "\n")

    # Initialize camera using GStreamer pipeline
    pipeline = gstreamer_pipeline()
    print(f"GStreamer Pipeline:\n{pipeline}\n")

    video_capture = video_stream()

    if not video_capture.isOpened():
        print("ERROR: Unable to open camera!")
        print("Please check:")
        print("  1. Camera is properly connected")
        print("  2. GStreamer is installed")
        print("  3. nvarguscamerasrc is available")
        return

    try:
        window_handle = cv2.namedWindow(window_title, cv2.WINDOW_AUTOSIZE)

        # State variables
        frame_count = 0
        start_time = time.time()
        fps = 0.0
        show_visualization = True
        print_info = True
        DETECTED_TAG = False

        print("Camera started successfully! Detecting markers...\n")

        while True:
            if DETECTED_TAG:
                # take photos, pause detection, call next service
                capture_images(count=10, interval=2)
                detect_tags: list[AprilTagDetection] = []
                for tag in tags:
                    detect_tags.append(
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

                for dt in detect_tags:
                    dt.print_info()
                # pause
                time.sleep(5)
                break

            ret_val, frame = video_capture.read()

            if not ret_val:
                print("ERROR: Failed to grab frame")
                break

            # Detect AprilTags
            tags = detect_apriltags(frame, show_visualization=show_visualization)

            # Filter by target IDs if specified
            if TARGET_TAG_IDS is not None:
                tags = [tag for tag in tags if tag.tag_id in TARGET_TAG_IDS]

            # Print detection information
            if tags and print_info:
                # if we are here then we have detected tag!
                DETECTED_TAG = True
                print_tag_info(tags)

            # Calculate FPS
            frame_count += 1
            elapsed = time.time() - start_time
            if elapsed > 2.0:
                fps = frame_count / elapsed
                if print_info:
                    print(f"\n[PERFORMANCE] FPS: {fps:.1f}")
                frame_count = 0
                start_time = time.time()

            # Draw FPS and tag count on frame
            cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
            cv2.putText(frame, f"Tags: {len(tags)}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)
            cv2.putText(frame, f"Viz: {'ON' if show_visualization else 'OFF'}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, COLOR_TEXT, 2)

            # Display frame
            if cv2.getWindowProperty(window_title, cv2.WND_PROP_AUTOSIZE) >= 0:
                cv2.imshow(window_title, frame)
            else:
                break

            # Handle keyboard input
            keyCode = cv2.waitKey(10) & 0xFF

            if keyCode == 27 or keyCode == ord('q'):  # ESC or 'q' to quit
                print("\nQuitting...")
                break

            elif keyCode == ord('s'):  # Toggle visualization
                show_visualization = not show_visualization
                status = "ON" if show_visualization else "OFF"
                print(f"\n[TOGGLE] Visualization: {status}")

            elif keyCode == ord('i'):  # Toggle info printing
                print_info = not print_info
                status = "ON" if print_info else "OFF"
                print(f"\n[TOGGLE] Info printing: {status}")

    except KeyboardInterrupt:
        print("\n\nInterrupted by user (Ctrl+C)")

    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()

    finally:
        video_capture.release()
        cv2.destroyAllWindows()
        print("\n" + "=" * 60)
        print("Camera stopped. Service terminated.")
        print("=" * 60)
        return DETECTED_TAG
