from __future__ import annotations

from dataclasses import dataclass

from src_video.domain.constants import DETECTION_PART_DEFAULT, SIDEABLE_PARTS

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
    parts: dict[str, dict] = {}

    # Base detection parts (non-sided)
    for part in DETECTION_PART_DEFAULT:
        if part in SIDEABLE_PARTS:
            continue
        parts[part] = {"injuries": {}}

    # Side-disambiguated limb parts (camera/image heuristic)
    for part in sorted(SIDEABLE_PARTS):
        parts[f"{part}1"] = {"injuries": {}}
        parts[f"{part}2"] = {"injuries": {}}

    # Additional placeholders used by downstream consumers
    parts.setdefault("chest", {"injuries": {}})
    parts.setdefault("abdomen", {"injuries": {}})
    parts.setdefault("back", {"injuries": {}})

    # Backwards-compat keys (older runs may still emit unsuffixed parts)
    for part in sorted(SIDEABLE_PARTS):
        parts.setdefault(part, {"injuries": {}})

    return parts
