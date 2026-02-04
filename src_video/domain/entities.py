from __future__ import annotations

from dataclasses import dataclass

@dataclass(frozen=True)
class CropPrediction:
    crop_path: str
    image_id: str
    body_part: str
    injury_pred: str
    injury_prob: float


@dataclass
class AprilTagDetection:
    tag_id: int
    center_x: float
    center_y: float
    corners: list[tuple[float, float]]  # List of (x, y) tuples for each corner
    distance: float  # Estimated distance to the tag in meters
    decision_margin: float  # Decision margin for the detection

    def print_info(self):
        print(f"Tag ID: {self.tag_id}")
        print(f"Center: ({self.center_x}, {self.center_y})")
        print(f"Corners: {self.corners}")
        print(f"Distance: {self.distance} meters")
        print(f"Decision Margin: {self.decision_margin}")

#basic body parts that will be updated with real patient info
def create_body_parts():
    return {
        "head": {"injuries": {}},
        "face": {"injuries": {}},
        "neck": {"injuries": {}},
        "arm": {"injuries": {}},
        "hand": {"injuries": {}},
        "chest": {"injuries": {}},
        "abdomen": {"injuries": {}},
        "back": {"injuries": {}},
        "leg": {"injuries": {}},
        "foot": {"injuries": {}},
    }
