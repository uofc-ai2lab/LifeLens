from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

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


@dataclass(frozen=True)
class BodyPartDetection:
    """A single body-part detection crop used by the ReID pipeline."""

    body_part: str
    crop_path: str
    bbox: Tuple[int, int, int, int]
    confidence: float
    image_id: str


@dataclass
class PersonTracking:
    """Tracks a person's reference detections for ReID matching."""

    tracking_id: str
    reference_detections: Dict[str, BodyPartDetection] = field(default_factory=dict)
    detection_count: int = 0
    last_detection_time: Optional[float] = None

    def add_detection(self, body_part: str, detection: BodyPartDetection) -> None:
        self.reference_detections[body_part] = detection

    def has_body_part(self, body_part: str) -> bool:
        return body_part in self.reference_detections

    def get_detection(self, body_part: str) -> Optional[BodyPartDetection]:
        return self.reference_detections.get(body_part)


@dataclass(frozen=True)
class MatchResult:
    """Result of attempting to re-identify a single body-part crop."""

    body_part: str
    is_match: bool
    confidence: float
    reference_tracking_id: Optional[str] = None
    spatial_distance: Optional[float] = None

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
