"""Integration module for person ReID with detection pipeline.

This module demonstrates how to integrate the PersonReIDEngine service
with the existing body part detection pipeline.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from src_video.services.person_reid_service.tracker import PersonReIDEngine
from src_video.domain.entities import MatchResult


def extract_detections_from_crops(
    crops_root: Path,
    image_id: str,
    device: str = "cpu",
) -> List[Tuple[str, str, Tuple[int, int, int, int], float, str]]:
    """
    Extract body part detections from the crops directory structure.
    
    Assumes crops are organized in subdirectories by image name, with files
    named like: {image_stem}_{body_part}_{idx}.jpg
    
    Args:
        crops_root: Root directory containing crop images
        image_id: ID of the current image being processed
        device: Device string (for compatibility)
    
    Returns:
        List of (body_part, crop_path, bbox, confidence, image_id) tuples
    """
    detections = []
    
    if not crops_root.exists():
        return detections
    
    # Iterate through crop subdirectories
    for image_dir in crops_root.iterdir():
        if not image_dir.is_dir():
            continue
        
        # Find all crop files (exclude alpha PNGs)
        for crop_file in image_dir.glob("*.jpg"):
            if "_alpha.png" in crop_file.name:
                continue
            
            # Parse body part from filename
            # Format: {stem}_{body_part}_{idx}.jpg
            parts = crop_file.stem.rsplit('_', 1)
            if len(parts) < 2:
                continue
            
            prefix = parts[0]
            prefix_parts = prefix.rsplit('_', 1)
            if len(prefix_parts) < 2:
                continue
            
            body_part = prefix_parts[1]
            
            # Estimate bbox from crop (we don't have exact bbox from just crop file)
            # This is a simplified approach; real implementation should store bbox metadata
            bbox = (0, 0, 0, 0)  # Placeholder
            confidence = 0.95  # High confidence for YOLO detections
            
            detections.append((
                body_part,
                str(crop_file),
                bbox,
                confidence,
                image_id,
            ))
    
    return detections


def load_detection_metadata(
    detection_output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    Load detection metadata if available.
    
    Looks for a metadata JSON file that might contain bbox information.
    
    Args:
        detection_output_dir: Detection output directory
    
    Returns:
        Metadata dictionary or None
    """
    metadata_file = detection_output_dir / "detection_metadata.json"
    
    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[reid] Error loading metadata: {e}")
        return None


def run_person_reid_on_detections(
    crops_root: Path,
    image_id: str,
    tracker: PersonTracker,
    is_reference: bool = False,
) -> Dict[str, Any]:
    """
    Run person re-identification on detected body parts.
    
    Args:
        crops_root: Root directory of detection crops
        image_id: Current image ID
        reid_engine: PersonReIDEngine instance
        is_gallery_registration: If True, registers detections as gallery reference.
                                If False, performs re-identification against gallery.
    
    Returns:
        Dictionary with re-id results
    """
    
    # Extract detections from crops
    detections = extract_detections_from_crops(crops_root, image_id)
    
    if not detections:
        return {
            "success": False,
            "error": "No detections found in crops",
            "person_id": None,
        }
    
    if is_gallery_registration:
        # Create new person in gallery and register all detections as references
        person_id = reid_engine.register_person()
        
        for body_part, crop_path, bbox, confidence, img_id in detections:
            reid_engine.add_reference_detection(
                person_id=person_id,
                body_part=body_part,
                crop_path=crop_path,
                bbox=bbox,
                confidence=confidence,
                image_id=img_id,
            )
        
        print(f"[reid] Registered person {person_id} with {len(detections)} body parts")
        
        return {
            "success": True,
            "person_id": person_id,
            "detections_registered": len(detections),
            "body_parts": [d[0] for d in detections],
        }
    else:
        # Perform re-identification against gallery
        reid_results = reid_engine.reid_detections(detections)
        summary = reid_engine.get_match_summary(reid_results)
        
        print(f"[reid] Re-identification complete")
        print(f"[reid] Identified: {summary['identified_count']}/{summary['total_detections']} body parts")
        
        return {
            "success": True,
            "person_id": None,
            "reid_results": [
                {
                    "body_part": r.body_part,
                    "is_identified": r.is_match,
                    "confidence": r.confidence,
                    "person_id": r.reference_tracking_id,
                }
                for r in reid_results
            ],
            "summary": summary,
        }


def generate_reid_report(
    reid_engine: PersonReIDEngine,
    output_path: Path,
) -> bool:
    """
    Generate a comprehensive re-identification report.
    
    Args:
        reid_engine: PersonReIDEngine instance
        output_path: Path to save the report
    
    Returns:
        True if successful
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        gallery_info = reid_engine.get_gallery()
        
        report = {
            "total_people_in_gallery": len(gallery_info),
            "gallery": gallery_info,
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[reid] Report saved to {output_path}")
        return True
    except Exception as e:
        print(f"[reid] Error generating report: {e}")
        return False


def extract_detections_from_crops(
    crops_root: Path,
    image_id: str,
    device: str = "cpu",
) -> List[Tuple[str, str, Tuple[int, int, int, int], float, str]]:
    """
    Extract body part detections from the crops directory structure.
    
    Assumes crops are organized in subdirectories by image name, with files
    named like: {image_stem}_{body_part}_{idx}.jpg
    
    Args:
        crops_root: Root directory containing crop images
        image_id: ID of the current image being processed
        device: Device string (for compatibility)
    
    Returns:
        List of (body_part, crop_path, bbox, confidence, image_id) tuples
    """
    detections = []
    
    if not crops_root.exists():
        return detections
    
    # Iterate through crop subdirectories
    for image_dir in crops_root.iterdir():
        if not image_dir.is_dir():
            continue
        
        # Find all crop files (exclude alpha PNGs)
        for crop_file in image_dir.glob("*.jpg"):
            if "_alpha.png" in crop_file.name:
                continue
            
            # Parse body part from filename
            # Format: {stem}_{body_part}_{idx}.jpg
            parts = crop_file.stem.rsplit('_', 1)
            if len(parts) < 2:
                continue
            
            prefix = parts[0]
            prefix_parts = prefix.rsplit('_', 1)
            if len(prefix_parts) < 2:
                continue
            
            body_part = prefix_parts[1]
            
            # Estimate bbox from crop (we don't have exact bbox from just crop file)
            # This is a simplified approach; real implementation should store bbox metadata
            bbox = (0, 0, 0, 0)  # Placeholder
            confidence = 0.95  # High confidence for YOLO detections
            
            detections.append((
                body_part,
                str(crop_file),
                bbox,
                confidence,
                image_id,
            ))
    
    return detections


def load_detection_metadata(
    detection_output_dir: Path,
) -> Optional[Dict[str, Any]]:
    """
    Load detection metadata if available.
    
    Looks for a metadata JSON file that might contain bbox information.
    
    Args:
        detection_output_dir: Detection output directory
    
    Returns:
        Metadata dictionary or None
    """
    metadata_file = detection_output_dir / "detection_metadata.json"
    
    if not metadata_file.exists():
        return None
    
    try:
        with open(metadata_file, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"[integration] Error loading metadata: {e}")
        return None


def run_person_tracking_on_detections(
    crops_root: Path,
    image_id: str,
    reid_engine: PersonReIDEngine,
    is_gallery_registration: bool = False,
) -> Dict[str, Any]:
    """
    Run person tracking on detected body parts.
    
    Args:
        crops_root: Root directory of detection crops
        image_id: Current image ID
        tracker: PersonTracker instance
        is_reference: If True, creates new person and adds detections as reference.
                     If False, matches against existing people.
    
    Returns:
        Dictionary with tracking results
    """
    
    # Extract detections from crops
    detections = extract_detections_from_crops(crops_root, image_id)
    
    if not detections:
        return {
            "success": False,
            "error": "No detections found in crops",
            "tracking_id": None,
        }
    
    if is_reference:
        # Create new person and add all detections as reference
        tracking_id = tracker.create_person()
        
        for body_part, crop_path, bbox, confidence, img_id in detections:
            tracker.add_detection(
                tracking_id=tracking_id,
                body_part=body_part,
                crop_path=crop_path,
                bbox=bbox,
                confidence=confidence,
                image_id=img_id,
            )
        
        print(f"[integration] Created reference tracking for {tracking_id}")
        print(f"[integration] Registered {len(detections)} body parts")
        
        return {
            "success": True,
            "tracking_id": tracking_id,
            "detections_added": len(detections),
            "body_parts": [d[0] for d in detections],
        }
    else:
        # Match detections against existing people
        match_results = tracker.match_detections(detections)
        summary = tracker.get_match_summary(match_results)
        
        print(f"[integration] Matching complete")
        print(f"[integration] Matched: {summary['matched_count']}/{summary['total_detections']}")
        
        return {
            "success": True,
            "tracking_id": None,
            "match_results": [
                {
                    "body_part": r.body_part,
                    "is_match": r.is_match,
                    "confidence": r.confidence,
                    "reference_tracking_id": r.reference_tracking_id,
                }
                for r in match_results
            ],
            "summary": summary,
        }


def generate_tracking_report(
    tracker: PersonTracker,
    output_path: Path,
) -> bool:
    """
    Generate a comprehensive tracking report.
    
    Args:
        tracker: PersonTracker instance
        output_path: Path to save the report
    
    Returns:
        True if successful
    """
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        people_info = tracker.get_all_people()
        
        report = {
            "total_people_tracked": len(people_info),
            "people": people_info,
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[integration] Report saved to {output_path}")
        return True
    except Exception as e:
        print(f"[integration] Error generating report: {e}")
        return False
