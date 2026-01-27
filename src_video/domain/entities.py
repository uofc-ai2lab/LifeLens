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
@dataclass
def create_body_parts():
    return {
        "head": {"image_id": None, "injury_pred": None, "accuracy": None},
        "face": {"image_id": None, "injury_pred": None, "accuracy": None},
        "neck": {"image_id": None, "injury_pred": None, "accuracy": None},
        "arm": {"image_id": None, "injury_pred": None, "accuracy": None},
        "hand": {"image_id": None, "injury_pred": None, "accuracy": None},
        "chest": {"image_id": None, "injury_pred": None, "accuracy": None},
        "abdomen": {"image_id": None, "injury_pred": None, "accuracy": None},
        "back": {"image_id": None, "injury_pred": None, "accuracy": None},
        "leg": {"image_id": None, "injury_pred": None, "accuracy": None},
        "foot": {"image_id": None, "injury_pred": None, "accuracy": None},
    }

