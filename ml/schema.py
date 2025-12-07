"""
SIMCAP Data Schema and Gesture Vocabulary

Defines the standard gesture vocabulary and data structures for ML training.
"""

from enum import IntEnum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
import json


class Gesture(IntEnum):
    """
    Gesture vocabulary for static pose classification (Tier 1).

    These are discrete hand poses that can be held statically.
    Start with a small vocabulary and expand as needed.
    """
    REST = 0           # Hand relaxed, neutral position
    FIST = 1           # All fingers flexed into palm
    OPEN_PALM = 2      # All fingers extended
    INDEX_UP = 3       # Index extended, others flexed (pointing)
    PEACE = 4          # Index + middle extended (peace/victory sign)
    THUMBS_UP = 5      # Thumb extended upward, others flexed
    OK_SIGN = 6        # Thumb-index circle, others extended
    PINCH = 7          # Thumb-index pinch, others relaxed
    GRAB = 8           # Fingers curled as if gripping
    WAVE = 9           # Hand tilted side to side (dynamic, for future)

    @classmethod
    def names(cls) -> List[str]:
        return [g.name.lower() for g in cls]

    @classmethod
    def from_name(cls, name: str) -> 'Gesture':
        return cls[name.upper()]


@dataclass
class SensorSample:
    """Single timestep of sensor data from GAMBIT device."""
    # Accelerometer (raw ADC values)
    ax: int
    ay: int
    az: int
    # Gyroscope (raw ADC values)
    gx: int
    gy: int
    gz: int
    # Magnetometer (raw ADC values)
    mx: int
    my: int
    mz: int
    # Auxiliary sensors
    light: float       # l: Light sensor (0-1)
    temp: int          # t: Magnetometer temperature
    cap: int           # c: Capacitive touch
    state: int         # s: State (0/1)
    battery: int       # b: Battery percentage
    press_count: int   # n: Button press count

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SensorSample':
        return cls(
            ax=d['ax'], ay=d['ay'], az=d['az'],
            gx=d['gx'], gy=d['gy'], gz=d['gz'],
            mx=d['mx'], my=d['my'], mz=d['mz'],
            light=d['l'], temp=d['t'], cap=d['c'],
            state=d['s'], battery=d['b'], press_count=d['n']
        )

    def imu_vector(self) -> List[int]:
        """Return 9-DoF IMU values as a flat list."""
        return [
            self.ax, self.ay, self.az,
            self.gx, self.gy, self.gz,
            self.mx, self.my, self.mz
        ]


@dataclass
class LabeledSegment:
    """A labeled segment within a recording session."""
    start_sample: int
    end_sample: int
    gesture: Gesture
    confidence: str = "high"  # high, medium, low
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "gesture": self.gesture.name.lower(),
            "confidence": self.confidence,
            "notes": self.notes
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LabeledSegment':
        return cls(
            start_sample=d['start_sample'],
            end_sample=d['end_sample'],
            gesture=Gesture.from_name(d['gesture']),
            confidence=d.get('confidence', 'high'),
            notes=d.get('notes', '')
        )


@dataclass
class SessionMetadata:
    """
    Metadata for a recording session.

    Stored alongside data files as {timestamp}.meta.json
    """
    timestamp: str
    subject_id: str = "unknown"
    environment: str = "unknown"
    hand: str = "right"  # left, right
    split: str = "train"  # train, validation, test
    device_id: str = "puck_default"
    labels: List[LabeledSegment] = field(default_factory=list)
    session_notes: str = ""
    sample_rate_hz: int = 50

    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "subject_id": self.subject_id,
            "environment": self.environment,
            "hand": self.hand,
            "split": self.split,
            "device_id": self.device_id,
            "labels": [l.to_dict() for l in self.labels],
            "session_notes": self.session_notes,
            "sample_rate_hz": self.sample_rate_hz
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SessionMetadata':
        labels = [LabeledSegment.from_dict(l) for l in d.get('labels', [])]
        return cls(
            timestamp=d['timestamp'],
            subject_id=d.get('subject_id', 'unknown'),
            environment=d.get('environment', 'unknown'),
            hand=d.get('hand', 'right'),
            split=d.get('split', 'train'),
            device_id=d.get('device_id', 'puck_default'),
            labels=labels,
            session_notes=d.get('session_notes', ''),
            sample_rate_hz=d.get('sample_rate_hz', 50)
        )

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SessionMetadata':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))


# Sensor value ranges (observed from data, for normalization)
SENSOR_RANGES = {
    'ax': (-16384, 16384),   # Accelerometer typical range
    'ay': (-16384, 16384),
    'az': (-16384, 16384),
    'gx': (-32768, 32768),   # Gyroscope typical range
    'gy': (-32768, 32768),
    'gz': (-32768, 32768),
    'mx': (-2048, 2048),     # Magnetometer typical range
    'my': (-2048, 2048),
    'mz': (-2048, 2048),
}

# Feature indices for 9-DoF vector
FEATURE_NAMES = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']
NUM_FEATURES = 9
