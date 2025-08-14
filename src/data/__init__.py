"""Data package initialization."""

from .collection.nfl_collector import NFLDataCollector
from .processing.feature_extractor import FeatureExtractor

__all__ = [
    "FeatureExtractor",
    "NFLDataCollector",
]
