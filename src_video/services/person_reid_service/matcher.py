"""Body Part Re-Identification Matcher

Compares detected body parts against reference detections using:
- Image similarity (histogram-based color matching)
- Bounding box spatial proximity
- Structural similarity (shape matching)

Used to determine if a detected body part belongs to a previously identified person.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, Tuple


def calculate_histogram_similarity(crop1_path: str, crop2_path: str) -> float:
    """
    Calculate similarity between two crop images using histogram comparison.
    
    Uses normalized histograms for robustness to lighting variations.
    
    Args:
        crop1_path: Path to first crop image
        crop2_path: Path to second crop image
    
    Returns:
        Similarity score from 0.0 (dissimilar) to 1.0 (identical)
    """
    try:
        img1 = cv2.imread(crop1_path)
        img2 = cv2.imread(crop2_path)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # Convert to HSV for better color matching
        hsv1 = cv2.cvtColor(img1, cv2.COLOR_BGR2HSV)
        hsv2 = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
        
        # Calculate histograms
        hist1 = cv2.calcHist([hsv1], [0, 1], None, [180, 256], [0, 180, 0, 256])
        hist2 = cv2.calcHist([hsv2], [0, 1], None, [180, 256], [0, 180, 0, 256])
        
        # Normalize histograms
        hist1 = cv2.normalize(hist1, hist1).flatten()
        hist2 = cv2.normalize(hist2, hist2).flatten()
        
        # Compare using correlation method (best for full histogram)
        similarity = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)
        
        return float(similarity)
    except Exception as e:
        print(f"[matcher] Error comparing histograms: {e}")
        return 0.0


def calculate_structural_similarity(crop1_path: str, crop2_path: str) -> float:
    """
    Calculate structural similarity between two crop images.
    
    Compares image structure/features rather than just pixel values.
    Useful for detecting the same body part in different poses/lighting.
    
    Args:
        crop1_path: Path to first crop image
        crop2_path: Path to second crop image
    
    Returns:
        Similarity score from 0.0 (dissimilar) to 1.0 (identical)
    """
    try:
        img1 = cv2.imread(crop1_path, cv2.IMREAD_GRAYSCALE)
        img2 = cv2.imread(crop2_path, cv2.IMREAD_GRAYSCALE)
        
        if img1 is None or img2 is None:
            return 0.0
        
        # Resize to same dimensions if needed
        h, w = img1.shape
        if img2.shape != (h, w):
            img2 = cv2.resize(img2, (w, h))
        
        # Use template matching to find best alignment
        # This works well for body part crops which maintain consistent structure
        result = cv2.matchTemplate(img1, img2, cv2.TM_CCOEFF_NORMED)
        
        # If sizes are very different, use MSE instead
        if result.size == 0:
            return calculate_mse_similarity(img1, img2)
        
        return float(np.max(result)) if result.size > 0 else 0.0
    except Exception as e:
        print(f"[matcher] Error calculating structural similarity: {e}")
        return 0.0


def calculate_mse_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """
    Calculate similarity based on Mean Squared Error.
    
    Lower MSE = higher similarity.
    
    Args:
        img1: First image array
        img2: Second image array
    
    Returns:
        Similarity score from 0.0 to 1.0
    """
    try:
        # Resize to same dimensions
        h, w = img1.shape[:2]
        if img2.shape[:2] != (h, w):
            img2 = cv2.resize(img2, (w, h))
        
        mse = np.mean((img1.astype(float) - img2.astype(float)) ** 2)
        
        # Normalize: MSE ranges from 0 to 255^2, convert to similarity
        # Higher MSE = lower similarity
        max_mse = 255.0 ** 2
        similarity = 1.0 - (mse / max_mse)
        
        return max(0.0, min(1.0, float(similarity)))
    except Exception as e:
        print(f"[matcher] Error calculating MSE: {e}")
        return 0.0


def calculate_spatial_distance(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int]
) -> float:
    """
    Calculate spatial distance between two bounding boxes.
    
    Returns the Euclidean distance between box centers.
    
    Args:
        bbox1: First bounding box (x1, y1, x2, y2)
        bbox2: Second bounding box (x1, y1, x2, y2)
    
    Returns:
        Distance in pixels
    """
    x1_c = (bbox1[0] + bbox1[2]) / 2.0
    y1_c = (bbox1[1] + bbox1[3]) / 2.0
    
    x2_c = (bbox2[0] + bbox2[2]) / 2.0
    y2_c = (bbox2[1] + bbox2[3]) / 2.0
    
    distance = np.sqrt((x1_c - x2_c) ** 2 + (y1_c - y2_c) ** 2)
    return float(distance)


def calculate_iou(
    bbox1: Tuple[int, int, int, int],
    bbox2: Tuple[int, int, int, int]
) -> float:
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Measures how much spatial overlap exists between boxes.
    
    Args:
        bbox1: First bounding box (x1, y1, x2, y2)
        bbox2: Second bounding box (x1, y1, x2, y2)
    
    Returns:
        IoU score from 0.0 to 1.0
    """
    x1_inter = max(bbox1[0], bbox2[0])
    y1_inter = max(bbox1[1], bbox2[1])
    x2_inter = min(bbox1[2], bbox2[2])
    y2_inter = min(bbox1[3], bbox2[3])
    
    if x2_inter < x1_inter or y2_inter < y1_inter:
        return 0.0
    
    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    
    box1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    box2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    
    union_area = box1_area + box2_area - inter_area
    
    if union_area == 0:
        return 0.0
    
    return float(inter_area / union_area)


class ReIDMatcher:
    """Matches detected body parts for person re-identification."""
    
    def __init__(
        self,
        histogram_weight: float = 0.4,
        structural_weight: float = 0.4,
        spatial_weight: float = 0.2,
        reid_threshold: float = 0.6,
        max_spatial_distance: Optional[float] = None,
    ):
        """
        Initialize the ReID matcher with weighting parameters.
        
        Args:
            histogram_weight: Weight for histogram similarity (0.0-1.0)
            structural_weight: Weight for structural similarity (0.0-1.0)
            spatial_weight: Weight for spatial proximity (0.0-1.0)
            reid_threshold: Minimum confidence to confirm re-identification (0.0-1.0)
            max_spatial_distance: Max pixel distance to consider re-id (None = unlimited)
        """
        total_weight = histogram_weight + structural_weight + spatial_weight
        if total_weight <= 0:
            raise ValueError("Sum of weights must be positive")
        
        self.histogram_weight = histogram_weight / total_weight
        self.structural_weight = structural_weight / total_weight
        self.spatial_weight = spatial_weight / total_weight
        self.reid_threshold = reid_threshold
        self.max_spatial_distance = max_spatial_distance
    
    def match(
        self,
        detection_crop_path: str,
        detection_bbox: Tuple[int, int, int, int],
        reference_crop_path: str,
        reference_bbox: Tuple[int, int, int, int],
    ) -> Tuple[bool, float]:
        """
        Re-identify a detected body part crop against a reference detection.
        
        Args:
            detection_crop_path: Path to newly detected crop
            detection_bbox: Bounding box of new detection
            reference_crop_path: Path to reference crop
            reference_bbox: Bounding box of reference detection
        
        Returns:
            Tuple of (is_same_person, reid_confidence_score)
        """
        # Check if paths exist
        if not Path(detection_crop_path).exists() or not Path(reference_crop_path).exists():
            return False, 0.0
        
        # Calculate component similarities
        hist_sim = calculate_histogram_similarity(detection_crop_path, reference_crop_path)
        struct_sim = calculate_structural_similarity(detection_crop_path, reference_crop_path)
        
        # Calculate spatial metrics
        spatial_dist = calculate_spatial_distance(detection_bbox, reference_bbox)
        iou = calculate_iou(detection_bbox, reference_bbox)
        
        # Normalize spatial distance to 0-1 range
        # Closer = higher score
        if self.max_spatial_distance is None:
            # Use a default based on box sizes
            ref_size = np.sqrt(
                (reference_bbox[2] - reference_bbox[0]) ** 2 +
                (reference_bbox[3] - reference_bbox[1]) ** 2
            )
            max_dist = ref_size * 2.0  # Allow 2x the reference box diagonal
        else:
            max_dist = self.max_spatial_distance
        
        spatial_sim = max(0.0, 1.0 - (spatial_dist / max_dist))
        
        # Combine with IoU (high IoU should boost confidence)
        spatial_sim = (spatial_sim + iou) / 2.0
        
        # Weighted combination
        confidence = (
            self.histogram_weight * hist_sim +
            self.structural_weight * struct_sim +
            self.spatial_weight * spatial_sim
        )
        
        is_same_person = confidence >= self.reid_threshold
        
        return is_same_person, confidence
