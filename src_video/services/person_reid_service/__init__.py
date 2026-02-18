"""
Person ReID Service Module
"""

from .reid_extractor import ResNet50ReIDExtractor, cosine_distance

__all__ = ["ResNet50ReIDExtractor", "cosine_distance"]
