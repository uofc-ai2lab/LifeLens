"""
Person Re-Identification (ReID) Feature Extractor Service

Extracts appearance features from person detections using ResNet50 model.
Integrates with BoxMot for improved tracking accuracy across camera views.
"""

import cv2
import numpy as np
from pathlib import Path
from config.logger import Logger

log = Logger("[video][reid]")


class ResNet50ReIDExtractor:
    """
    ResNet50-based Re-Identification (ReID) feature extractor for person re-identification.
    Integrates with BoxMot for improved tracking accuracy.
    """
    
    def __init__(self, model_path: str = "resnet50_market1501_aicity156.onnx", device: str = "cpu"):
        """
        Initialize ReID extractor with ResNet50 model.
        
        Args:
            model_path: Path to the ONNX model file
            device: Device to run inference on ('cpu' or 'cuda')
        """
        self.model_path = Path(model_path)
        self.device = device
        self.model = None
        self.input_size = (128, 256)  # Standard ReID input size
        
        self._load_model()
    
    def _load_model(self):
        """Load ONNX model for inference."""
        try:
            import onnxruntime as ort
            
            if not self.model_path.exists():
                log.error(f"Model not found: {self.model_path}")
                return False
            
            # Use GPU if available, otherwise CPU
            providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if self.device == 'cuda' else ['CPUExecutionProvider']
            
            self.model = ort.InferenceSession(str(self.model_path), providers=providers)
            log.success(f"ReID model loaded: {self.model_path}")
            return True
            
        except Exception as e:
            log.error(f"Failed to load ReID model: {e}")
            return False
    
    def extract_features(self, frame: np.ndarray, bboxes: list) -> np.ndarray:
        """
        Extract ReID features for each person bounding box.
        
        Args:
            frame: Input frame (BGR)
            bboxes: List of bounding boxes [x1, y1, x2, y2, ...]
        
        Returns:
            Features array of shape (num_detections, feature_dim)
        """
        if self.model is None:
            return np.array([])
        
        features = []
        
        for bbox in bboxes:
            try:
                x1, y1, x2, y2 = map(int, bbox[:4])
                
                # Clip to frame bounds
                x1 = max(0, x1)
                y1 = max(0, y1)
                x2 = min(frame.shape[1], x2)
                y2 = min(frame.shape[0], y2)
                
                if x2 - x1 <= 0 or y2 - y1 <= 0:
                    continue
                
                # Crop person region
                person_img = frame[y1:y2, x1:x2]
                
                # Preprocess
                processed = self._preprocess(person_img)
                
                # Extract feature
                feature = self._infer(processed)
                features.append(feature)
                
            except Exception as e:
                log.warning(f"Failed to extract feature for bbox {bbox}: {e}")
                continue
        
        return np.array(features) if features else np.array([])
    
    def _preprocess(self, img: np.ndarray) -> np.ndarray:
        """Preprocess image for model input."""
        # Resize to model input size
        img = cv2.resize(img, self.input_size)
        
        # Convert BGR to RGB
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Normalize (ImageNet stats)
        img = img.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        img = (img - mean) / std
        
        # Add batch dimension and transpose to CHW
        img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
        img = np.expand_dims(img, 0)  # Add batch
        
        return img.astype(np.float32)
    
    def _infer(self, img: np.ndarray) -> np.ndarray:
        """Run inference and get feature vector."""
        try:
            inputs = {self.model.get_inputs()[0].name: img}
            outputs = self.model.run(None, inputs)
            
            # Get feature vector (typically last layer output)
            feature = outputs[0][0]
            
            # L2 normalize
            feature = feature / (np.linalg.norm(feature) + 1e-5)
            
            return feature
            
        except Exception as e:
            log.error(f"Inference failed: {e}")
            return np.array([])


def cosine_distance(feature1: np.ndarray, feature2: np.ndarray) -> float:
    """Compute cosine distance between two feature vectors."""
    if len(feature1) == 0 or len(feature2) == 0:
        return 1.0
    
    distance = 1 - np.dot(feature1, feature2) / (np.linalg.norm(feature1) * np.linalg.norm(feature2) + 1e-5)
    return float(distance)
