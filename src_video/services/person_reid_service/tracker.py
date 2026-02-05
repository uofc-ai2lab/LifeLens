"""Person Re-Identification (ReID) Service

Main service that manages person identities and performs re-identification
by matching detected body parts across multiple image captures.
"""

import json
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime

from src_video.domain.entities import (
    BodyPartDetection,
    PersonTracking,
    MatchResult,
)
from src_video.services.person_reid_service.matcher import ReIDMatcher


class PersonReIDEngine:
    """
    Person Re-Identification Engine
    
    Maintains a gallery of known people and performs re-identification
    by matching detected body parts against previously registered identities.
    """
    
    def __init__(
        self,
        storage_dir: Optional[str] = None,
        matcher: Optional[ReIDMatcher] = None,
    ):
        """
        Initialize the ReID engine.
        
        Args:
            storage_dir: Directory to store ReID gallery/metadata. If None, uses in-memory only.
            matcher: Custom ReIDMatcher instance. Uses default if None.
        """
        self.gallery: Dict[str, PersonTracking] = {}  # Person ID → registered identity
        self.storage_dir = Path(storage_dir) if storage_dir else None
        self.matcher = matcher or ReIDMatcher(
            histogram_weight=0.4,
            structural_weight=0.4,
            spatial_weight=0.2,
            reid_threshold=0.65,
        )
        
        if self.storage_dir:
            self.storage_dir.mkdir(parents=True, exist_ok=True)
            self._load_reid_gallery()
    
    def _load_reid_gallery(self) -> None:
        """Load persisted ReID gallery from storage."""
        if not self.storage_dir:
            return
        
        gallery_file = self.storage_dir / "reid_gallery.json"
        if not gallery_file.exists():
            return
        
        try:
            with open(gallery_file, 'r') as f:
                gallery_data = json.load(f)
            print(f"[reid] Loaded gallery from {gallery_file}")
        except Exception as e:
            print(f"[reid] Error loading gallery: {e}")
    
    def _save_reid_gallery(self) -> None:
        """Persist current ReID gallery to storage."""
        if not self.storage_dir:
            return
        
        gallery_file = self.storage_dir / "reid_gallery.json"
        try:
            # Create a JSON-serializable version
            gallery_data = {
                "people": {},
                "timestamp": datetime.now().isoformat(),
            }
            
            for person_id, person in self.gallery.items():
                gallery_data["people"][person_id] = {
                    "person_id": person.tracking_id,
                    "detection_count": person.detection_count,
                    "last_detection_time": person.last_detection_time,
                    "body_parts": {
                        bp: {
                            "body_part": det.body_part,
                            "image_id": det.image_id,
                            "bbox": det.bbox,
                        }
                        for bp, det in person.reference_detections.items()
                    }
                }
            
            with open(gallery_file, 'w') as f:
                json.dump(gallery_data, f, indent=2)
        except Exception as e:
            print(f"[reid] Error saving gallery: {e}")
    
    def register_person(self) -> str:
        """
        Register a new person in the ReID gallery.
        
        Returns:
            Unique person ID
        """
        person_id = f"person_{uuid.uuid4().hex[:8]}"
        self.gallery[person_id] = PersonTracking(tracking_id=person_id)
        print(f"[reid] Registered person: {person_id}")
        return person_id
    
    def add_reference_detection(
        self,
        person_id: str,
        body_part: str,
        crop_path: str,
        bbox: Tuple[int, int, int, int],
        confidence: float,
        image_id: str,
    ) -> bool:
        """
        Add a reference detection for a person in the gallery.
        
        Reference detections are used to re-identify the same person in future frames.
        
        Args:
            person_id: ID of the person in gallery
            body_part: Name of the body part
            crop_path: Path to the crop image
            bbox: Bounding box (x1, y1, x2, y2)
            confidence: Detection confidence
            image_id: Image ID where detection was found
        
        Returns:
            True if successfully added
        """
        if person_id not in self.gallery:
            print(f"[reid] Person ID {person_id} not found in gallery")
            return False
        
        person = self.gallery[person_id]
        detection = BodyPartDetection(
            body_part=body_part,
            crop_path=crop_path,
            bbox=bbox,
            confidence=confidence,
            image_id=image_id,
        )
        
        person.add_detection(body_part, detection)
        person.detection_count += 1
        person.last_detection_time = time.time()
        
        print(f"[reid] Added {body_part} reference to {person_id}")
        return True
    
    def reid_detections(
        self,
        detections: List[Tuple[str, str, Tuple[int, int, int, int], float, str]],
    ) -> List[MatchResult]:
        """
        Perform re-identification on a set of detected body parts.
        
        Matches each detection against all people in the gallery.
        
        Args:
            detections: List of (body_part, crop_path, bbox, confidence, image_id)
        
        Returns:
            List of MatchResult objects
        """
        results = []
        
        for body_part, crop_path, bbox, confidence, image_id in detections:
            reid_result = self._reid_single_detection(
                body_part, crop_path, bbox, image_id
            )
            results.append(reid_result)
        
        return results
    
    def _reid_single_detection(
        self,
        body_part: str,
        crop_path: str,
        bbox: Tuple[int, int, int, int],
        image_id: str,
    ) -> MatchResult:
        """
        Re-identify a single detected body part against gallery.
        
        Args:
            body_part: Name of the body part
            crop_path: Path to the crop image
            bbox: Bounding box
            image_id: Image ID
        
        Returns:
            MatchResult with re-id decision
        """
        best_reid = False
        best_confidence = 0.0
        best_person_id = None
        
        # Try to re-identify against each person in gallery
        for person_id, person in self.gallery.items():
            if not person.has_body_part(body_part):
                continue
            
            reference_detection = person.get_detection(body_part)
            if reference_detection is None:
                continue
            
            # Use matcher to compare
            is_same_person, confidence = self.matcher.match(
                crop_path,
                bbox,
                reference_detection.crop_path,
                reference_detection.bbox,
            )
            
            if confidence > best_confidence:
                best_confidence = confidence
                best_reid = is_same_person
                best_person_id = person_id
        
        spatial_dist = None
        if best_person_id and best_reid:
            ref_det = self.gallery[best_person_id].get_detection(body_part)
            if ref_det:
                spatial_dist = (
                    (bbox[0] - ref_det.bbox[0]) ** 2 +
                    (bbox[1] - ref_det.bbox[1]) ** 2
                ) ** 0.5
        
        return MatchResult(
            body_part=body_part,
            is_match=best_reid,
            confidence=best_confidence,
            reference_tracking_id=(best_person_id if best_reid else None),
            spatial_distance=spatial_dist,
        )
    
    def get_match_summary(
        self,
        match_results: List[MatchResult],
    ) -> Dict[str, any]:
        """
        Generate a summary of re-identification results.
        
        Args:
            match_results: List of MatchResult objects
        
        Returns:
            Dictionary with summary statistics
        """
        total = len(match_results)
        identified = sum(1 for r in match_results if r.is_match)
        new = total - identified
        
        identified_by_part = {}
        for result in match_results:
            if result.is_match:
                identified_by_part[result.body_part] = result.reference_tracking_id
        
        return {
            "total_detections": total,
            "identified_count": identified,
            "new_count": new,
            "reid_rate": identified / total if total > 0 else 0.0,
            "identified_by_part": identified_by_part,
            "confidence_scores": {
                r.body_part: {
                    "confidence": r.confidence,
                    "is_identified": r.is_match,
                }
                for r in match_results
            }
        }
    
    def get_person_info(self, person_id: str) -> Optional[Dict]:
        """
        Get detailed information about a person in the gallery.
        
        Args:
            person_id: ID of the person
        
        Returns:
            Dictionary with person's ReID data
        """
        if person_id not in self.gallery:
            return None
        
        person = self.gallery[person_id]
        return {
            "person_id": person.tracking_id,
            "detection_count": person.detection_count,
            "last_detection_time": person.last_detection_time,
            "body_parts": list(person.reference_detections.keys()),
            "reference_crops": {
                bp: det.crop_path
                for bp, det in person.reference_detections.items()
            }
        }
    
    def get_gallery(self) -> Dict[str, Dict]:
        """Get information about all people in the gallery."""
        return {
            pid: self.get_person_info(pid)
            for pid in self.gallery.keys()
        }
    
    def export_reid_results(
        self,
        match_results: List[MatchResult],
        output_path: str,
    ) -> bool:
        """
        Export re-identification results to a JSON file.
        
        Args:
            match_results: List of MatchResult objects
            output_path: Path to save results
        
        Returns:
            True if successful
        """
        try:
            output_file = Path(output_path)
            output_file.parent.mkdir(parents=True, exist_ok=True)
            
            results_data = {
                "timestamp": datetime.now().isoformat(),
                "summary": self.get_match_summary(match_results),
                "detections": [
                    {
                        "body_part": r.body_part,
                        "is_identified": r.is_match,
                        "confidence": r.confidence,
                        "person_id": r.reference_tracking_id,
                        "spatial_distance": r.spatial_distance,
                    }
                    for r in match_results
                ]
            }
            
            with open(output_file, 'w') as f:
                json.dump(results_data, f, indent=2)
            
            print(f"[reid] Exported results to {output_file}")
            return True
        except Exception as e:
            print(f"[reid] Error exporting results: {e}")
            return False