"""
SIMCAP Visualization Package

Modular visualization components for SIMCAP sensor data.
"""

from .data_processor import SensorDataProcessor
from .visual_distinction import VisualDistinctionEngine

__all__ = [
    'SensorDataProcessor',
    'VisualDistinctionEngine',
]
