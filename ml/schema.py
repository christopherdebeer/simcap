"""
SIMCAP Data Schema and Gesture Vocabulary

Defines the standard gesture vocabulary and data structures for ML training.
Supports multi-label annotations for magnetic finger tracking.
"""

from enum import IntEnum, Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any, Union, Set
import json


# =============================================================================
# ENUMS - Label Categories
# =============================================================================

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


class FingerState(str, Enum):
    """State of individual finger flexion."""
    EXTENDED = "extended"    # Finger fully extended
    PARTIAL = "partial"      # Finger partially flexed
    FLEXED = "flexed"        # Finger fully flexed into palm
    UNKNOWN = "unknown"      # State not specified


class MotionState(str, Enum):
    """Motion state during capture."""
    STATIC = "static"        # Hand stationary in pose
    MOVING = "moving"        # Hand in motion
    TRANSITION = "transition"  # Transitioning between poses


class CalibrationType(str, Enum):
    """Type of calibration marker."""
    NONE = "none"                    # Normal data capture
    EARTH_FIELD = "earth_field"      # Earth field calibration rotation
    HARD_IRON = "hard_iron"          # Hard iron offset calibration
    SOFT_IRON = "soft_iron"          # Soft iron matrix calibration
    FINGER_RANGE = "finger_range"    # Per-finger range of motion capture
    REFERENCE_POSE = "reference_pose"  # Known reference pose for alignment
    MAGNET_BASELINE = "magnet_baseline"  # Baseline with magnets at known positions


class MagnetPolarity(str, Enum):
    """Magnet orientation relative to palm."""
    NORTH_PALM = "north_palm"    # North pole facing palm
    SOUTH_PALM = "south_palm"    # South pole facing palm
    UNKNOWN = "unknown"          # Polarity not specified


class HandRegion(str, Enum):
    """Region of hand for spatial labels."""
    PALM_CENTER = "palm_center"
    THUMB = "thumb"
    INDEX = "index"
    MIDDLE = "middle"
    RING = "ring"
    PINKY = "pinky"
    WRIST = "wrist"


# =============================================================================
# FINGER LABELS - Per-finger state tracking
# =============================================================================

@dataclass
class FingerLabels:
    """
    Per-finger state labels for detailed hand pose tracking.

    Used for magnetic finger tracking where individual finger
    positions need to be labeled independently.
    """
    thumb: FingerState = FingerState.UNKNOWN
    index: FingerState = FingerState.UNKNOWN
    middle: FingerState = FingerState.UNKNOWN
    ring: FingerState = FingerState.UNKNOWN
    pinky: FingerState = FingerState.UNKNOWN

    def to_dict(self) -> Dict[str, str]:
        return {
            "thumb": self.thumb.value,
            "index": self.index.value,
            "middle": self.middle.value,
            "ring": self.ring.value,
            "pinky": self.pinky.value
        }

    @classmethod
    def from_dict(cls, d: Dict[str, str]) -> 'FingerLabels':
        return cls(
            thumb=FingerState(d.get('thumb', 'unknown')),
            index=FingerState(d.get('index', 'unknown')),
            middle=FingerState(d.get('middle', 'unknown')),
            ring=FingerState(d.get('ring', 'unknown')),
            pinky=FingerState(d.get('pinky', 'unknown'))
        )

    def to_binary_string(self) -> str:
        """
        Convert to binary string for pose identification.
        E=0 (extended), P=1 (partial), F=2 (flexed), U=? (unknown)

        Returns: String like "00000" (all extended) or "22222" (all flexed)
        """
        mapping = {
            FingerState.EXTENDED: '0',
            FingerState.PARTIAL: '1',
            FingerState.FLEXED: '2',
            FingerState.UNKNOWN: '?'
        }
        return ''.join([
            mapping[self.thumb],
            mapping[self.index],
            mapping[self.middle],
            mapping[self.ring],
            mapping[self.pinky]
        ])

    @classmethod
    def from_binary_string(cls, s: str) -> 'FingerLabels':
        """Create from binary string like "00000" or "22222"."""
        mapping = {
            '0': FingerState.EXTENDED,
            '1': FingerState.PARTIAL,
            '2': FingerState.FLEXED,
            '?': FingerState.UNKNOWN
        }
        if len(s) != 5:
            raise ValueError(f"Binary string must be 5 characters, got {len(s)}")
        return cls(
            thumb=mapping.get(s[0], FingerState.UNKNOWN),
            index=mapping.get(s[1], FingerState.UNKNOWN),
            middle=mapping.get(s[2], FingerState.UNKNOWN),
            ring=mapping.get(s[3], FingerState.UNKNOWN),
            pinky=mapping.get(s[4], FingerState.UNKNOWN)
        )

    @classmethod
    def all_extended(cls) -> 'FingerLabels':
        return cls(
            thumb=FingerState.EXTENDED,
            index=FingerState.EXTENDED,
            middle=FingerState.EXTENDED,
            ring=FingerState.EXTENDED,
            pinky=FingerState.EXTENDED
        )

    @classmethod
    def all_flexed(cls) -> 'FingerLabels':
        return cls(
            thumb=FingerState.FLEXED,
            index=FingerState.FLEXED,
            middle=FingerState.FLEXED,
            ring=FingerState.FLEXED,
            pinky=FingerState.FLEXED
        )


# =============================================================================
# MAGNET CONFIGURATION
# =============================================================================

@dataclass
class MagnetConfig:
    """
    Magnet configuration for magnetic finger tracking.

    Tracks which fingers have magnets and their polarity.
    Per analysis: alternating polarity recommended (thumb+, index-, middle+, etc.)
    """
    thumb_present: bool = False
    index_present: bool = False
    middle_present: bool = False
    ring_present: bool = False
    pinky_present: bool = False

    thumb_polarity: MagnetPolarity = MagnetPolarity.UNKNOWN
    index_polarity: MagnetPolarity = MagnetPolarity.UNKNOWN
    middle_polarity: MagnetPolarity = MagnetPolarity.UNKNOWN
    ring_polarity: MagnetPolarity = MagnetPolarity.UNKNOWN
    pinky_polarity: MagnetPolarity = MagnetPolarity.UNKNOWN

    magnet_type: str = "unknown"  # e.g., "6x3mm_N48", "5x2mm_N42"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "thumb": {"present": self.thumb_present, "polarity": self.thumb_polarity.value},
            "index": {"present": self.index_present, "polarity": self.index_polarity.value},
            "middle": {"present": self.middle_present, "polarity": self.middle_polarity.value},
            "ring": {"present": self.ring_present, "polarity": self.ring_polarity.value},
            "pinky": {"present": self.pinky_present, "polarity": self.pinky_polarity.value},
            "magnet_type": self.magnet_type
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MagnetConfig':
        return cls(
            thumb_present=d.get('thumb', {}).get('present', False),
            index_present=d.get('index', {}).get('present', False),
            middle_present=d.get('middle', {}).get('present', False),
            ring_present=d.get('ring', {}).get('present', False),
            pinky_present=d.get('pinky', {}).get('present', False),
            thumb_polarity=MagnetPolarity(d.get('thumb', {}).get('polarity', 'unknown')),
            index_polarity=MagnetPolarity(d.get('index', {}).get('polarity', 'unknown')),
            middle_polarity=MagnetPolarity(d.get('middle', {}).get('polarity', 'unknown')),
            ring_polarity=MagnetPolarity(d.get('ring', {}).get('polarity', 'unknown')),
            pinky_polarity=MagnetPolarity(d.get('pinky', {}).get('polarity', 'unknown')),
            magnet_type=d.get('magnet_type', 'unknown')
        )

    @classmethod
    def alternating_all_fingers(cls, magnet_type: str = "6x3mm_N48") -> 'MagnetConfig':
        """
        Recommended configuration: all fingers with alternating polarity.
        Per analysis: thumb+, index-, middle+, ring-, pinky+
        """
        return cls(
            thumb_present=True, thumb_polarity=MagnetPolarity.NORTH_PALM,
            index_present=True, index_polarity=MagnetPolarity.SOUTH_PALM,
            middle_present=True, middle_polarity=MagnetPolarity.NORTH_PALM,
            ring_present=True, ring_polarity=MagnetPolarity.SOUTH_PALM,
            pinky_present=True, pinky_polarity=MagnetPolarity.NORTH_PALM,
            magnet_type=magnet_type
        )

    @classmethod
    def single_finger(cls, finger: str, polarity: MagnetPolarity = MagnetPolarity.NORTH_PALM,
                      magnet_type: str = "6x3mm_N48") -> 'MagnetConfig':
        """Configuration for single-finger testing (Phase 1)."""
        config = cls(magnet_type=magnet_type)
        setattr(config, f"{finger}_present", True)
        setattr(config, f"{finger}_polarity", polarity)
        return config


# =============================================================================
# MULTI-LABEL SYSTEM
# =============================================================================

@dataclass
class MultiLabel:
    """
    Multi-label annotation for a data segment.

    Combines multiple label categories that can be active simultaneously:
    - Hand pose (high-level gesture)
    - Finger states (per-finger detail)
    - Motion state (static/moving/transition)
    - Calibration type (if calibration capture)
    - Custom labels (arbitrary user-defined tags)

    Example: A "peace sign" might have:
        pose="peace",
        fingers={index: extended, middle: extended, others: flexed},
        motion=static,
        custom=["demo_capture", "good_lighting"]
    """
    # High-level pose (optional - can use finger states instead)
    pose: Optional[str] = None

    # Per-finger state detail
    fingers: Optional[FingerLabels] = None

    # Motion state
    motion: MotionState = MotionState.STATIC

    # Calibration marker (if applicable)
    calibration: CalibrationType = CalibrationType.NONE

    # Custom labels (arbitrary strings)
    custom: List[str] = field(default_factory=list)

    # Quality/confidence indicators
    confidence: str = "high"  # high, medium, low
    quality_notes: str = ""   # Notes about data quality

    def to_dict(self) -> Dict[str, Any]:
        result = {
            "motion": self.motion.value,
            "calibration": self.calibration.value,
            "custom": self.custom,
            "confidence": self.confidence,
            "quality_notes": self.quality_notes
        }
        if self.pose is not None:
            result["pose"] = self.pose
        if self.fingers is not None:
            result["fingers"] = self.fingers.to_dict()
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'MultiLabel':
        fingers = None
        if 'fingers' in d:
            fingers = FingerLabels.from_dict(d['fingers'])

        return cls(
            pose=d.get('pose'),
            fingers=fingers,
            motion=MotionState(d.get('motion', 'static')),
            calibration=CalibrationType(d.get('calibration', 'none')),
            custom=d.get('custom', []),
            confidence=d.get('confidence', 'high'),
            quality_notes=d.get('quality_notes', '')
        )

    def all_labels(self) -> Set[str]:
        """Return all active labels as a flat set of strings."""
        labels = set()
        if self.pose:
            labels.add(f"pose:{self.pose}")
        if self.fingers:
            labels.add(f"fingers:{self.fingers.to_binary_string()}")
            # Also add individual finger states
            for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                state = getattr(self.fingers, finger)
                if state != FingerState.UNKNOWN:
                    labels.add(f"{finger}:{state.value}")
        labels.add(f"motion:{self.motion.value}")
        if self.calibration != CalibrationType.NONE:
            labels.add(f"calibration:{self.calibration.value}")
        labels.update(self.custom)
        return labels

    def has_label(self, label: str) -> bool:
        """Check if a specific label is present."""
        return label in self.all_labels()


# =============================================================================
# LABELED SEGMENTS (V1 for backwards compatibility + V2 for multi-label)
# =============================================================================

@dataclass
class LabeledSegment:
    """
    A labeled segment within a recording session (V1 - backwards compatible).
    Uses single gesture label.
    """
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
class LabeledSegmentV2:
    """
    A labeled segment with multi-label support (V2).

    Supports simultaneous labels across multiple categories.
    """
    start_sample: int
    end_sample: int
    labels: MultiLabel
    notes: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "start_sample": self.start_sample,
            "end_sample": self.end_sample,
            "labels": self.labels.to_dict(),
            "notes": self.notes,
            "_version": 2
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'LabeledSegmentV2':
        return cls(
            start_sample=d['start_sample'],
            end_sample=d['end_sample'],
            labels=MultiLabel.from_dict(d['labels']),
            notes=d.get('notes', '')
        )

    @classmethod
    def from_v1(cls, v1: LabeledSegment) -> 'LabeledSegmentV2':
        """Convert V1 LabeledSegment to V2."""
        return cls(
            start_sample=v1.start_sample,
            end_sample=v1.end_sample,
            labels=MultiLabel(
                pose=v1.gesture.name.lower(),
                confidence=v1.confidence
            ),
            notes=v1.notes
        )


# =============================================================================
# SENSOR DATA
# =============================================================================

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


# =============================================================================
# SESSION METADATA
# =============================================================================

@dataclass
class SessionMetadata:
    """
    Metadata for a recording session.

    Stored alongside data files as {timestamp}.meta.json

    Supports both V1 (single-label) and V2 (multi-label) segments.
    """
    timestamp: str
    subject_id: str = "unknown"
    environment: str = "unknown"
    hand: str = "right"  # left, right
    split: str = "train"  # train, validation, test
    device_id: str = "puck_default"
    labels: List[LabeledSegment] = field(default_factory=list)
    labels_v2: List[LabeledSegmentV2] = field(default_factory=list)
    session_notes: str = ""
    sample_rate_hz: int = 50

    # Magnetic finger tracking specific
    magnet_config: Optional[MagnetConfig] = None
    calibration_data: Optional[Dict[str, Any]] = None
    custom_label_definitions: List[str] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        result = {
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
        if self.labels_v2:
            result["labels_v2"] = [l.to_dict() for l in self.labels_v2]
        if self.magnet_config:
            result["magnet_config"] = self.magnet_config.to_dict()
        if self.calibration_data:
            result["calibration_data"] = self.calibration_data
        if self.custom_label_definitions:
            result["custom_label_definitions"] = self.custom_label_definitions
        return result

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'SessionMetadata':
        labels = [LabeledSegment.from_dict(l) for l in d.get('labels', [])]
        labels_v2 = [LabeledSegmentV2.from_dict(l) for l in d.get('labels_v2', [])]

        magnet_config = None
        if 'magnet_config' in d:
            magnet_config = MagnetConfig.from_dict(d['magnet_config'])

        return cls(
            timestamp=d['timestamp'],
            subject_id=d.get('subject_id', 'unknown'),
            environment=d.get('environment', 'unknown'),
            hand=d.get('hand', 'right'),
            split=d.get('split', 'train'),
            device_id=d.get('device_id', 'puck_default'),
            labels=labels,
            labels_v2=labels_v2,
            session_notes=d.get('session_notes', ''),
            sample_rate_hz=d.get('sample_rate_hz', 50),
            magnet_config=magnet_config,
            calibration_data=d.get('calibration_data'),
            custom_label_definitions=d.get('custom_label_definitions', [])
        )

    def save(self, path: str):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'SessionMetadata':
        with open(path, 'r') as f:
            return cls.from_dict(json.load(f))

    def get_all_labels_v2(self) -> List[LabeledSegmentV2]:
        """Get all labels as V2 format, converting V1 if needed."""
        all_labels = list(self.labels_v2)
        for v1_label in self.labels:
            all_labels.append(LabeledSegmentV2.from_v1(v1_label))
        return all_labels


# =============================================================================
# PREDEFINED LABEL SETS
# =============================================================================

# Standard poses for UI buttons
STANDARD_POSES = [
    "rest", "fist", "open_palm", "index_up", "peace",
    "thumbs_up", "ok_sign", "pinch", "grab", "wave"
]

# Finger tracking specific labels
FINGER_TRACKING_LABELS = [
    # Calibration
    "cal:earth_field", "cal:hard_iron", "cal:reference",
    # Single finger (Phase 1)
    "single:thumb", "single:index", "single:middle", "single:ring", "single:pinky",
    # Motion
    "motion:flex", "motion:extend", "motion:spread", "motion:close",
    # Quality
    "quality:good", "quality:noisy", "quality:artifact"
]

# Common gesture transitions
TRANSITION_LABELS = [
    "trans:rest_to_fist", "trans:fist_to_open", "trans:rest_to_point",
    "trans:point_to_peace", "trans:peace_to_open"
]


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
