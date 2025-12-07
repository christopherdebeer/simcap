"""
SIMCAP Machine Learning Pipeline

Modules:
    schema - Data structures and gesture vocabulary
    data_loader - Load and preprocess sensor data
    model - CNN models for gesture classification
    train - Training script
    label - Data labeling tool
"""

from .schema import Gesture, SessionMetadata, LabeledSegment, SensorSample
from .data_loader import GambitDataset, DatasetStats

__all__ = [
    'Gesture',
    'SessionMetadata',
    'LabeledSegment',
    'SensorSample',
    'GambitDataset',
    'DatasetStats'
]
