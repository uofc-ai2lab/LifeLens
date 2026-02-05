"""Integration module for person ReID with detection pipeline.

This module demonstrates how to integrate the PersonReIDEngine service
with the existing body part detection pipeline.
"""

import json
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple

from src_video.services.person_reid_service.tracker import PersonReIDEngine
from src_video.domain.entities import MatchResult


def _select_target_image_dir(crops_root: Path, image_id: str) -> Optional[Path]:
    """Select the crop subdirectory that represents the current image.

    Preference order:
    1) crops_root/image_id if it exists
    2) newest subdirectory under crops_root

    This avoids accidentally running ReID over *all* historical crop folders.
    """

    candidate = crops_root / image_id
    if candidate.exists() and candidate.is_dir():
        return candidate

    subdirs = [p for p in crops_root.iterdir() if p.is_dir()]
    if not subdirs:
        return None

    return max(subdirs, key=lambda p: p.stat().st_mtime)


def extract_detections_from_image_dir(
    image_dir: Path,
    image_id: str,
) -> List[Tuple[str, str, Tuple[int, int, int, int], float, str]]:
    """Extract detections from a single crop folder (one captured image)."""

    detections: List[Tuple[str, str, Tuple[int, int, int, int], float, str]] = []
    if not image_dir.exists() or not image_dir.is_dir():
        return detections

    exts = {".jpg", ".jpeg", ".png"}
    for crop_file in image_dir.iterdir():
        if not crop_file.is_file() or crop_file.suffix.lower() not in exts:
            continue

        # Skip alpha-mask crops (commonly written as *_alpha.png)
        if crop_file.name.lower().endswith("_alpha.png"):
            continue

        # Format: {image_stem}_{body_part}_{idx}.jpg
        parts = crop_file.stem.rsplit('_', 1)
        if len(parts) < 2:
            continue

        prefix = parts[0]
        prefix_parts = prefix.rsplit('_', 1)
        if len(prefix_parts) < 2:
            continue

        body_part = prefix_parts[1]
        bbox = (0, 0, 0, 0)  # Placeholder; can be replaced with real bbox metadata later.
        confidence = 0.95

        detections.append((
            body_part,
            str(crop_file),
            bbox,
            confidence,
            image_id,
        ))

    return detections


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
    
    target_dir = _select_target_image_dir(crops_root, image_id)
    if target_dir is None:
        return detections

    # Use the folder name as the image_id for traceability in reports.
    # (The caller-provided image_id is often "image_00X" while folders are like "captured_img_...".)
    return extract_detections_from_image_dir(target_dir, image_id=target_dir.name)


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
    reid_engine: PersonReIDEngine,
    is_gallery_registration: bool = False,
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
    saved_images_dir: Optional[Path] = None,
    primary_person_id: Optional[str] = None,
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

        report: Dict[str, Any] = {
            "total_people_in_gallery": len(gallery_info),
            "gallery": gallery_info,
            "primary_person_id": primary_person_id,
        }

        # Optional per-image mapping: for each crop in each image folder, show which person_id
        # it matched and which reference crop it matched against.
        detection_output_dir = output_path.parent.parent
        crops_root = detection_output_dir / "crops"
        images_section: Dict[str, Any] = {}

        # If provided, include *every* image currently present in saved_images_dir in the
        # report, even if crops are missing (e.g., detection didn't run yet / failed).
        saved_stems: Optional[set[str]] = None
        saved_images_by_stem: Dict[str, Path] = {}
        if saved_images_dir is not None and saved_images_dir.exists():
            for p in saved_images_dir.iterdir():
                if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}:
                    saved_images_by_stem[p.stem] = p
            saved_stems = set(saved_images_by_stem.keys()) if saved_images_by_stem else None

        def _empty_summary() -> Dict[str, Any]:
            return {
                "total_detections": 0,
                "identified_count": 0,
                "new_count": 0,
                "reid_rate": 0.0,
                "identified_by_part": {},
                "confidence_scores": {},
            }

        # Primary path: saved_images_dir provided → iterate those stems so the report can
        # assess/report on every capture in saved_imgs.
        if saved_stems is not None:
            ordered_stems = sorted(
                saved_stems,
                key=lambda s: saved_images_by_stem.get(s).stat().st_mtime if saved_images_by_stem.get(s) else 0,
            )
            for stem in ordered_stems:
                image_dir = crops_root / stem

                # No crops yet → still include entry.
                if not image_dir.exists() or not image_dir.is_dir():
                    images_section[stem] = {
                        "best_person_id": None,
                        "primary_person_present": False,
                        "summary": _empty_summary(),
                        "detections": [],
                        "has_crops": False,
                        "note": "No crops folder found for this image (detection may not have run or produced no crops).",
                    }
                    continue

                detections = extract_detections_from_image_dir(image_dir, image_id=image_dir.name)
                if not detections:
                    images_section[stem] = {
                        "best_person_id": None,
                        "primary_person_present": False,
                        "summary": _empty_summary(),
                        "detections": [],
                        "has_crops": True,
                        "note": "Crops folder exists but no crop images were found/parsed.",
                    }
                    continue

                match_results = reid_engine.reid_detections(detections)
                summary = reid_engine.get_match_summary(match_results)

                # Majority-vote assignment for the whole image (helpful for quick sanity checks).
                matched = [r for r in match_results if r.is_match and r.reference_tracking_id]
                best_person_id: Optional[str] = None
                if matched:
                    counts: Dict[str, int] = {}
                    conf_sums: Dict[str, float] = {}
                    for r in matched:
                        pid = r.reference_tracking_id
                        counts[pid] = counts.get(pid, 0) + 1
                        conf_sums[pid] = conf_sums.get(pid, 0.0) + float(r.confidence)
                    best_person_id = max(
                        counts.keys(),
                        key=lambda pid: (counts[pid], conf_sums.get(pid, 0.0) / max(1, counts[pid]))
                    )

                detections_out: List[Dict[str, Any]] = []
                # Preserve 1:1 ordering with `detections` list.
                for (body_part, crop_path, bbox, confidence, img_id), r in zip(detections, match_results):
                    ref_crop_path = None
                    if r.is_match and r.reference_tracking_id:
                        person = reid_engine.gallery.get(r.reference_tracking_id)
                        if person:
                            ref_det = person.get_detection(body_part)
                            if ref_det:
                                ref_crop_path = ref_det.crop_path

                    detections_out.append({
                        "image_id": img_id,
                        "body_part": body_part,
                        "crop_path": crop_path,
                        "matched": r.is_match,
                        "person_id": r.reference_tracking_id,
                        "confidence": r.confidence,
                        "reference_crop_path": ref_crop_path,
                    })

                images_section[stem] = {
                    "best_person_id": best_person_id,
                    "primary_person_present": (
                        bool(primary_person_id)
                        and any(
                            d.get("matched") and d.get("person_id") == primary_person_id
                            for d in detections_out
                        )
                    ),
                    "summary": summary,
                    "detections": detections_out,
                    "has_crops": True,
                }

        # Fallback path: no saved_images_dir filter → iterate crop folders.
        elif crops_root.exists():
            image_dirs = [p for p in crops_root.iterdir() if p.is_dir()]
            image_dirs.sort(key=lambda p: p.stat().st_mtime)

            for image_dir in image_dirs:
                detections = extract_detections_from_image_dir(image_dir, image_id=image_dir.name)
                match_results = reid_engine.reid_detections(detections) if detections else []
                summary = reid_engine.get_match_summary(match_results) if detections else _empty_summary()

                matched = [r for r in match_results if r.is_match and r.reference_tracking_id]
                best_person_id: Optional[str] = None
                if matched:
                    counts: Dict[str, int] = {}
                    conf_sums: Dict[str, float] = {}
                    for r in matched:
                        pid = r.reference_tracking_id
                        counts[pid] = counts.get(pid, 0) + 1
                        conf_sums[pid] = conf_sums.get(pid, 0.0) + float(r.confidence)
                    best_person_id = max(
                        counts.keys(),
                        key=lambda pid: (counts[pid], conf_sums.get(pid, 0.0) / max(1, counts[pid]))
                    )

                detections_out: List[Dict[str, Any]] = []
                for (body_part, crop_path, bbox, confidence, img_id), r in zip(detections, match_results):
                    ref_crop_path = None
                    if r.is_match and r.reference_tracking_id:
                        person = reid_engine.gallery.get(r.reference_tracking_id)
                        if person:
                            ref_det = person.get_detection(body_part)
                            if ref_det:
                                ref_crop_path = ref_det.crop_path

                    detections_out.append({
                        "image_id": img_id,
                        "body_part": body_part,
                        "crop_path": crop_path,
                        "matched": r.is_match,
                        "person_id": r.reference_tracking_id,
                        "confidence": r.confidence,
                        "reference_crop_path": ref_crop_path,
                    })

                images_section[image_dir.name] = {
                    "best_person_id": best_person_id,
                    "primary_person_present": (
                        bool(primary_person_id)
                        and any(
                            d.get("matched") and d.get("person_id") == primary_person_id
                            for d in detections_out
                        )
                    ),
                    "summary": summary,
                    "detections": detections_out,
                }

        if images_section:
            report["images"] = images_section

        report["images_filter"] = {
            "saved_images_dir": str(saved_images_dir) if saved_images_dir is not None else None,
            "filtered_to_saved_images": saved_stems is not None,
            "saved_images_count": 0 if saved_stems is None else len(saved_stems),
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"[reid] Report saved to {output_path}")
        return True
    except Exception as e:
        print(f"[reid] Error generating report: {e}")
        return False