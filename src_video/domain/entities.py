from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass(frozen=True)
class CropPrediction:
    crop_path: str
    image_id: str
    body_part: str
    injury_pred: str
    injury_prob: float


@dataclass
class BodyPartDetection:
    """Represents a detected body part in an image."""
    body_part: str
    crop_path: str
    bbox: tuple[int, int, int, int]  # (x1, y1, x2, y2)
    confidence: float
    image_id: str


@dataclass
class PersonTracking:
    """Tracks a person's body parts across multiple detections."""
    tracking_id: str
    reference_detections: Dict[str, BodyPartDetection] = field(default_factory=dict)
    detection_count: int = 0
    last_detection_time: Optional[float] = None
    
    def add_detection(self, body_part: str, detection: BodyPartDetection) -> None:
        """Add a reference detection for a body part."""
        self.reference_detections[body_part] = detection
    
    def has_body_part(self, body_part: str) -> bool:
        """Check if this person has a reference detection for a body part."""
        return body_part in self.reference_detections
    
    def get_detection(self, body_part: str) -> Optional[BodyPartDetection]:
        """Get the reference detection for a body part."""
        return self.reference_detections.get(body_part)


@dataclass
class MatchResult:
    """Result of comparing a detected body part against reference detections."""
    body_part: str
    is_match: bool
    confidence: float  # 0.0 to 1.0, higher = more similar to reference
    reference_tracking_id: Optional[str] = None
    spatial_distance: Optional[float] = None  # Pixel distance from reference location


@dataclass(frozen=True)
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

