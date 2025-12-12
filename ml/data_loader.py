"""
SIMCAP Data Loader

Loads raw JSON data files, applies preprocessing, and creates
windowed tensors suitable for ML training.

Supports both V1 (single-label) and V2 (multi-label) formats.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any, Set, Union
from dataclasses import dataclass

from .schema import (
    Gesture, SessionMetadata, LabeledSegment, LabeledSegmentV2,
    MultiLabel, FingerLabels, FingerState,
    SENSOR_RANGES, FEATURE_NAMES, NUM_FEATURES
)
from .calibration import EnvironmentalCalibration, decorate_telemetry_with_calibration
from .filters import KalmanFilter3D, decorate_telemetry_with_filtering


@dataclass
class DatasetStats:
    """Global statistics for normalization."""
    mean: np.ndarray  # Shape: (NUM_FEATURES,)
    std: np.ndarray   # Shape: (NUM_FEATURES,)
    min_val: np.ndarray
    max_val: np.ndarray

    def save(self, path: str):
        np.savez(path, mean=self.mean, std=self.std,
                 min_val=self.min_val, max_val=self.max_val)

    @classmethod
    def load(cls, path: str) -> 'DatasetStats':
        data = np.load(path)
        return cls(
            mean=data['mean'],
            std=data['std'],
            min_val=data['min_val'],
            max_val=data['max_val']
        )


@dataclass
class SessionInfo:
    """Information about a loaded session, including embedded metadata."""
    samples: List[Dict]
    version: str  # '1.0', '2.0', '2.1'
    timestamp: Optional[str] = None
    firmware_version: Optional[str] = None
    labels: Optional[List[Dict]] = None
    metadata: Optional[Dict] = None

    @property
    def has_embedded_metadata(self) -> bool:
        """Check if session has embedded v2.1+ metadata."""
        return self.version >= '2.1' and self.metadata is not None


def load_session_raw(json_path: Path) -> SessionInfo:
    """
    Load raw session data with version detection and metadata extraction.

    Supports three JSON formats:
    - V1 (legacy): Array of samples directly: [{sample1}, {sample2}, ...]
    - V2.0: Wrapper object: {version: "2.0", timestamp: "...", samples: [...]}
    - V2.1: Wrapper with embedded metadata: {version: "2.1", samples: [...], labels: [...], metadata: {...}}

    Args:
        json_path: Path to the .json data file

    Returns:
        SessionInfo with samples, version info, and any embedded metadata
    """
    with open(json_path, 'r') as f:
        raw_json = json.load(f)

    if isinstance(raw_json, list):
        # V1 format: array of samples directly
        return SessionInfo(
            samples=raw_json,
            version='1.0',
            timestamp=json_path.stem
        )
    elif isinstance(raw_json, dict) and 'samples' in raw_json:
        # V2+ format: wrapper object
        version = raw_json.get('version', '2.0')
        metadata = raw_json.get('metadata', {})

        return SessionInfo(
            samples=raw_json['samples'],
            version=version,
            timestamp=raw_json.get('timestamp', json_path.stem),
            firmware_version=metadata.get('firmware_version') if metadata else None,
            labels=raw_json.get('labels'),
            metadata=metadata
        )
    else:
        raise ValueError(f"Unknown JSON format in {json_path}: expected array or object with 'samples' key")


def load_session_data(json_path: Path, apply_calibration: bool = True,
                      apply_filtering: bool = True,
                      calibration_file: Optional[str] = None) -> np.ndarray:
    """
    Load raw sensor data from a JSON file with optional calibration and filtering.

    IMPORTANT: Raw data is always preserved. Calibration and filtering add
    decorated fields (calibrated_mx, filtered_mx, etc.) if available.
    The returned array uses the best available data:
    - filtered > calibrated > raw magnetometer values

    Supports three JSON formats:
    - V1 (legacy): Array of samples directly: [{sample1}, {sample2}, ...]
    - V2.0: Wrapper object: {version: "2.0", timestamp: "...", samples: [...]}
    - V2.1: Wrapper with embedded metadata: {version: "2.1", samples: [...], labels: [...], metadata: {...}}

    Args:
        json_path: Path to the .json data file
        apply_calibration: Apply magnetometer calibration if available
        apply_filtering: Apply Kalman filtering if available
        calibration_file: Path to calibration JSON (default: 'gambit_calibration.json')

    Returns:
        numpy array of shape (N, 9) where N is number of samples
        and 9 is the IMU features [ax, ay, az, gx, gy, gz, mx, my, mz]
        Note: mx, my, mz will be filtered/calibrated if available, otherwise raw
    """
    session_info = load_session_raw(json_path)
    data = session_info.samples

    # Preserve raw data, apply decorations
    if apply_calibration or apply_filtering:
        # Try to load calibration
        calibration = None
        if apply_calibration:
            try:
                if calibration_file is None:
                    # Try default locations
                    cal_paths = [
                        json_path.parent / 'gambit_calibration.json',
                        Path.home() / '.gambit' / 'calibration.json'
                    ]
                else:
                    cal_paths = [Path(calibration_file)]

                for cal_path in cal_paths:
                    if cal_path.exists():
                        calibration = EnvironmentalCalibration()
                        calibration.load(str(cal_path))
                        print(f"Loaded calibration from {cal_path}")
                        break
            except Exception as e:
                print(f"Warning: Failed to load calibration: {e}")

        # Apply calibration decoration
        if calibration is not None:
            data = decorate_telemetry_with_calibration(data, calibration)

        # Apply filtering decoration
        if apply_filtering:
            try:
                mag_filter = KalmanFilter3D(process_noise=1.0, measurement_noise=1.0)
                data = decorate_telemetry_with_filtering(data, mag_filter)
            except Exception as e:
                print(f"Warning: Failed to apply filtering: {e}")

    # Extract IMU features, using best available magnetometer data
    samples = []
    for sample in data:
        # Use filtered > calibrated > raw for magnetometer
        mx = sample.get('filtered_mx', sample.get('calibrated_mx', sample.get('mx', 0)))
        my = sample.get('filtered_my', sample.get('calibrated_my', sample.get('my', 0)))
        mz = sample.get('filtered_mz', sample.get('calibrated_mz', sample.get('mz', 0)))

        row = [
            sample['ax'], sample['ay'], sample['az'],
            sample['gx'], sample['gy'], sample['gz'],
            mx, my, mz
        ]
        samples.append(row)

    return np.array(samples, dtype=np.float32)


def load_session_metadata(json_path: Path) -> Optional[SessionMetadata]:
    """
    Load metadata for a session, checking both embedded v2.1 metadata and separate .meta.json files.

    Priority:
    1. Embedded metadata in v2.1 JSON files (preferred for new data)
    2. Separate .meta.json file (legacy format)

    Args:
        json_path: Path to the .json data file

    Returns:
        SessionMetadata if available, else None
    """
    # First, try to load embedded metadata from v2.1 format
    try:
        session_info = load_session_raw(json_path)
        if session_info.has_embedded_metadata:
            # Build SessionMetadata from embedded data
            meta = session_info.metadata
            embedded_labels = session_info.labels or []

            # Convert embedded labels to LabeledSegmentV2 format
            labels_v2 = []
            for label in embedded_labels:
                if 'labels' in label:
                    # Already in v2 format
                    labels_v2.append(LabeledSegmentV2.from_dict(label))
                elif 'gesture' in label:
                    # v1 format label
                    labels_v2.append(LabeledSegmentV2.from_v1(LabeledSegment.from_dict(label)))

            return SessionMetadata(
                timestamp=session_info.timestamp or json_path.stem,
                subject_id=meta.get('subject_id', 'unknown'),
                environment=meta.get('environment', 'unknown'),
                hand=meta.get('hand', 'unknown'),
                split=meta.get('split', 'train'),
                device_id=meta.get('device', 'GAMBIT'),
                labels=[],  # v1 labels empty for v2.1 sessions
                labels_v2=labels_v2,
                session_notes=meta.get('notes', ''),
                sample_rate_hz=meta.get('sample_rate', 26),
                magnet_config=MagnetConfig.from_dict(meta['magnet_config']) if meta.get('magnet_config') else None,
                calibration_data=meta.get('calibration'),
                custom_label_definitions=meta.get('custom_label_definitions', []),
                # Extended fields for v2.1
                firmware_version=session_info.firmware_version,
                session_type=meta.get('session_type', 'recording')
            )
    except Exception:
        pass  # Fall through to legacy .meta.json loading

    # Fallback: load from separate .meta.json file
    meta_path = json_path.with_suffix('.meta.json')
    if meta_path.exists():
        return SessionMetadata.load(str(meta_path))
    return None


def compute_dataset_stats(data_dir: Path) -> DatasetStats:
    """
    Compute global statistics across all data files for normalization.

    Args:
        data_dir: Directory containing .json data files

    Returns:
        DatasetStats with mean, std, min, max for each feature
    
    TODO: Task 5 - Enhance with magnetic-specific normalization stats
        Current implementation computes stats across ALL sessions (with and without magnets),
        which can lead to poor normalization for magnetic finger tracking.
        
        Enhancement needed:
        1. Add `with_magnets: bool = False` parameter
        2. Filter sessions based on metadata.magnet_config field
        3. Compute separate stats for:
           - Baseline (no magnets): for calibration sessions
           - With magnets: for finger tracking sessions
        4. Save both stat sets: dataset_stats.npz and dataset_stats_magnetic.npz
        5. Update GambitDataset.__init__ to load appropriate stats based on use case
        
        Impact: Currently acceptable for initial training, but may reduce accuracy
                by 5-10% if magnetometer magnitude scales differ significantly between
                calibration and tracking sessions.
        
        Priority: Low (implement only if finger tracking model accuracy < 60%)
        
        Reference: docs/design/magnetic-tracking-pipeline-analysis.md Section 3.4
    """
    all_data = []

    for json_path in data_dir.glob('*.json'):
        # Skip non-session files
        if (json_path.name.endswith('.meta.json') or
            json_path.name.endswith('.full.json') or
            'calibration' in json_path.name.lower()):
            continue
        data = load_session_data(json_path)
        all_data.append(data)

    if not all_data:
        raise ValueError(f"No data files found in {data_dir}")

    combined = np.concatenate(all_data, axis=0)

    return DatasetStats(
        mean=np.mean(combined, axis=0),
        std=np.std(combined, axis=0) + 1e-8,  # Avoid division by zero
        min_val=np.min(combined, axis=0),
        max_val=np.max(combined, axis=0)
    )


def normalize_data(data: np.ndarray, stats: DatasetStats,
                   method: str = 'standardize') -> np.ndarray:
    """
    Normalize sensor data using global statistics.

    Args:
        data: Raw sensor data, shape (N, 9)
        stats: Global dataset statistics
        method: 'standardize' (z-score) or 'minmax' (0-1 range)

    Returns:
        Normalized data, same shape as input
    """
    if method == 'standardize':
        return (data - stats.mean) / stats.std
    elif method == 'minmax':
        range_val = stats.max_val - stats.min_val + 1e-8
        return (data - stats.min_val) / range_val
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def create_windows(data: np.ndarray, labels: np.ndarray,
                   window_size: int = 50, stride: int = 25,
                   require_single_label: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from sequential data.

    Args:
        data: Sensor data, shape (N, 9)
        labels: Per-sample labels, shape (N,)
        window_size: Number of samples per window (50 @ 50Hz = 1 second)
        stride: Step size between windows (25 = 50% overlap)
        require_single_label: If True, only include windows where all samples
                              have the same label

    Returns:
        Tuple of (windows, window_labels):
        - windows: shape (num_windows, window_size, 9)
        - window_labels: shape (num_windows,)
    """
    windows = []
    window_labels = []

    num_samples = len(data)

    for start in range(0, num_samples - window_size + 1, stride):
        end = start + window_size
        window_data = data[start:end]
        window_label_seq = labels[start:end]

        if require_single_label:
            # Only include if all samples have the same label
            unique_labels = np.unique(window_label_seq)
            if len(unique_labels) == 1:
                windows.append(window_data)
                window_labels.append(unique_labels[0])
        else:
            # Use majority vote for window label
            window_labels.append(np.bincount(window_label_seq.astype(int)).argmax())
            windows.append(window_data)

    if not windows:
        return np.array([]).reshape(0, window_size, NUM_FEATURES), np.array([])

    return np.array(windows), np.array(window_labels)


def create_windows_multilabel(
    data: np.ndarray,
    label_matrix: np.ndarray,
    window_size: int = 50,
    stride: int = 25,
    require_consistent: bool = True
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows from sequential data with multi-label support.

    Args:
        data: Sensor data, shape (N, 9)
        label_matrix: Per-sample multi-label matrix, shape (N, num_labels)
        window_size: Number of samples per window
        stride: Step size between windows
        require_consistent: If True, only include windows where all samples
                            have the same label vector

    Returns:
        Tuple of (windows, window_labels):
        - windows: shape (num_windows, window_size, 9)
        - window_labels: shape (num_windows, num_labels)
    """
    windows = []
    window_labels = []

    num_samples = len(data)

    for start in range(0, num_samples - window_size + 1, stride):
        end = start + window_size
        window_data = data[start:end]
        window_label_seq = label_matrix[start:end]

        if require_consistent:
            # Check if all rows are identical
            if np.all(window_label_seq == window_label_seq[0]):
                windows.append(window_data)
                window_labels.append(window_label_seq[0])
        else:
            # Use mode for each label dimension
            mode_labels = []
            for col in range(window_label_seq.shape[1]):
                counts = np.bincount(window_label_seq[:, col].astype(int))
                mode_labels.append(counts.argmax())
            windows.append(window_data)
            window_labels.append(mode_labels)

    if not windows:
        num_labels = label_matrix.shape[1] if len(label_matrix.shape) > 1 else 1
        return (np.array([]).reshape(0, window_size, NUM_FEATURES),
                np.array([]).reshape(0, num_labels))

    return np.array(windows), np.array(window_labels)


def labels_from_segments(num_samples: int, segments: List[LabeledSegment],
                         default_label: Gesture = Gesture.REST) -> np.ndarray:
    """
    Convert labeled segments to per-sample labels array (V1 format).

    Args:
        num_samples: Total number of samples in the session
        segments: List of labeled segments
        default_label: Label for unlabeled samples

    Returns:
        Array of shape (num_samples,) with gesture labels
    """
    labels = np.full(num_samples, default_label.value, dtype=np.int32)

    for seg in segments:
        labels[seg.start_sample:seg.end_sample] = seg.gesture.value

    return labels


def labels_from_segments_v2(
    num_samples: int,
    segments: List[LabeledSegmentV2],
    label_columns: List[str]
) -> np.ndarray:
    """
    Convert V2 labeled segments to per-sample multi-label matrix.

    Args:
        num_samples: Total number of samples in the session
        segments: List of V2 labeled segments
        label_columns: List of label column names to extract

    Returns:
        Array of shape (num_samples, len(label_columns))
    """
    # Initialize with -1 (unlabeled)
    label_matrix = np.full((num_samples, len(label_columns)), -1, dtype=np.int32)

    for seg in segments:
        for i, col in enumerate(label_columns):
            value = _extract_label_value(seg.labels, col)
            if value is not None:
                label_matrix[seg.start_sample:seg.end_sample, i] = value

    return label_matrix


def _extract_label_value(labels: MultiLabel, column: str) -> Optional[int]:
    """
    Extract a numeric label value from a MultiLabel object.

    Supported columns:
    - 'pose': Maps pose name to Gesture enum value
    - 'motion': Maps motion state to 0/1/2
    - 'calibration': Maps calibration type to 0-6
    - 'thumb', 'index', 'middle', 'ring', 'pinky': Finger state 0/1/2
    - 'fingers_binary': Binary encoding of all finger states
    """
    if column == 'pose':
        if labels.pose:
            try:
                return Gesture.from_name(labels.pose).value
            except (KeyError, ValueError):
                return None
        return None

    elif column == 'motion':
        motion_map = {'static': 0, 'moving': 1, 'transition': 2}
        return motion_map.get(labels.motion.value, 0)

    elif column == 'calibration':
        cal_map = {
            'none': 0, 'earth_field': 1, 'hard_iron': 2,
            'soft_iron': 3, 'finger_range': 4, 'reference_pose': 5,
            'magnet_baseline': 6
        }
        return cal_map.get(labels.calibration.value, 0)

    elif column in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        if labels.fingers:
            state = getattr(labels.fingers, column)
            state_map = {'extended': 0, 'partial': 1, 'flexed': 2, 'unknown': -1}
            return state_map.get(state.value, -1)
        return -1

    elif column == 'fingers_binary':
        if labels.fingers:
            # Encode as base-3 number: 00000 to 22222
            values = []
            for f in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                state = getattr(labels.fingers, f)
                if state == FingerState.EXTENDED:
                    values.append(0)
                elif state == FingerState.PARTIAL:
                    values.append(1)
                elif state == FingerState.FLEXED:
                    values.append(2)
                else:
                    return -1  # Unknown state
            # Convert to single integer
            return sum(v * (3 ** (4-i)) for i, v in enumerate(values))
        return -1

    return None


class GambitDataset:
    """
    Dataset class for loading and preparing GAMBIT data for training.

    Supports both V1 (single-label) and V2 (multi-label) formats.
    """

    def __init__(self, data_dir: str, window_size: int = 50, stride: int = 25,
                 normalize_method: str = 'standardize'):
        """
        Initialize the dataset.

        Args:
            data_dir: Path to data/GAMBIT/ directory
            window_size: Samples per window
            stride: Window stride
            normalize_method: 'standardize' or 'minmax'
        """
        self.data_dir = Path(data_dir)
        self.window_size = window_size
        self.stride = stride
        self.normalize_method = normalize_method

        # Compute or load global stats
        self.stats_path = self.data_dir / 'dataset_stats.npz'
        if self.stats_path.exists():
            self.stats = DatasetStats.load(str(self.stats_path))
        else:
            print("Computing dataset statistics...")
            self.stats = compute_dataset_stats(self.data_dir)
            self.stats.save(str(self.stats_path))
            print(f"Saved stats to {self.stats_path}")

    def load_labeled_sessions(self, split: Optional[str] = None
                              ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all labeled sessions and create windowed dataset (V1 format).

        Args:
            split: If specified, only load sessions with this split ('train', 'validation', 'test')

        Returns:
            Tuple of (X, y):
            - X: shape (num_windows, window_size, 9)
            - y: shape (num_windows,)
        """
        all_windows = []
        all_labels = []

        for json_path in sorted(self.data_dir.glob('*.json')):
            if json_path.name.endswith('.meta.json'):
                continue

            meta = load_session_metadata(json_path)
            if meta is None:
                continue

            # Check for V1 or V2 labels
            has_v1_labels = bool(meta.labels)
            has_v2_labels = bool(meta.labels_v2)

            if not has_v1_labels and not has_v2_labels:
                continue  # Skip unlabeled sessions

            if split is not None and meta.split != split:
                continue

            # Load and normalize data
            data = load_session_data(json_path)
            data = normalize_data(data, self.stats, self.normalize_method)

            # Create per-sample labels (prefer V2, fall back to V1)
            if has_v2_labels:
                # Convert V2 to V1-style pose labels
                segments = meta.get_all_labels_v2()
                labels = self._v2_to_pose_labels(len(data), segments)
            else:
                labels = labels_from_segments(len(data), meta.labels)

            # Create windows
            windows, window_labels = create_windows(
                data, labels, self.window_size, self.stride
            )

            if len(windows) > 0:
                all_windows.append(windows)
                all_labels.append(window_labels)

        if not all_windows:
            return (np.array([]).reshape(0, self.window_size, NUM_FEATURES),
                    np.array([]))

        return np.concatenate(all_windows), np.concatenate(all_labels)

    def _v2_to_pose_labels(self, num_samples: int,
                           segments: List[LabeledSegmentV2]) -> np.ndarray:
        """Convert V2 segments to V1-style pose labels."""
        labels = np.full(num_samples, Gesture.REST.value, dtype=np.int32)

        for seg in segments:
            if seg.labels.pose:
                try:
                    gesture = Gesture.from_name(seg.labels.pose)
                    labels[seg.start_sample:seg.end_sample] = gesture.value
                except (KeyError, ValueError):
                    pass  # Unknown pose, keep default

        return labels

    def load_multilabel_sessions(
        self,
        label_columns: List[str] = ['pose', 'motion'],
        split: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load all labeled sessions with multi-label support (V2 format).

        Args:
            label_columns: List of label columns to extract
            split: If specified, only load sessions with this split

        Returns:
            Tuple of (X, y):
            - X: shape (num_windows, window_size, 9)
            - y: shape (num_windows, len(label_columns))
        """
        all_windows = []
        all_labels = []

        for json_path in sorted(self.data_dir.glob('*.json')):
            if json_path.name.endswith('.meta.json'):
                continue

            meta = load_session_metadata(json_path)
            if meta is None:
                continue

            segments = meta.get_all_labels_v2()
            if not segments:
                continue

            if split is not None and meta.split != split:
                continue

            # Load and normalize data
            data = load_session_data(json_path)
            data = normalize_data(data, self.stats, self.normalize_method)

            # Create per-sample label matrix
            label_matrix = labels_from_segments_v2(len(data), segments, label_columns)

            # Create windows
            windows, window_labels = create_windows_multilabel(
                data, label_matrix, self.window_size, self.stride
            )

            if len(windows) > 0:
                all_windows.append(windows)
                all_labels.append(window_labels)

        if not all_windows:
            return (np.array([]).reshape(0, self.window_size, NUM_FEATURES),
                    np.array([]).reshape(0, len(label_columns)))

        return np.concatenate(all_windows), np.concatenate(all_labels)

    def load_finger_tracking_sessions(
        self,
        split: Optional[str] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Load sessions for finger tracking (5-finger state prediction).

        Returns:
            Tuple of (X, y):
            - X: shape (num_windows, window_size, 9)
            - y: shape (num_windows, 5) - one column per finger
        """
        return self.load_multilabel_sessions(
            label_columns=['thumb', 'index', 'middle', 'ring', 'pinky'],
            split=split
        )

    def get_train_val_split(self, val_ratio: float = 0.2
                            ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Load data with train/validation split.

        Args:
            val_ratio: Fraction of data for validation

        Returns:
            Tuple of (X_train, y_train, X_val, y_val)
        """
        X_train, y_train = self.load_labeled_sessions(split='train')
        X_val, y_val = self.load_labeled_sessions(split='validation')

        # If no explicit splits, split randomly
        if len(X_val) == 0 and len(X_train) > 0:
            n = len(X_train)
            indices = np.random.permutation(n)
            val_size = int(n * val_ratio)

            val_idx = indices[:val_size]
            train_idx = indices[val_size:]

            X_val, y_val = X_train[val_idx], y_train[val_idx]
            X_train, y_train = X_train[train_idx], y_train[train_idx]

        return X_train, y_train, X_val, y_val

    def get_all_custom_labels(self) -> Set[str]:
        """Get all unique custom labels used across all sessions."""
        custom_labels = set()

        for json_path in sorted(self.data_dir.glob('*.json')):
            if json_path.name.endswith('.meta.json'):
                continue

            meta = load_session_metadata(json_path)
            if meta is None:
                continue

            # From custom_label_definitions
            custom_labels.update(meta.custom_label_definitions)

            # From actual labels in V2 segments
            for seg in meta.labels_v2:
                custom_labels.update(seg.labels.custom)

        return custom_labels

    def get_available_firmware_versions(self) -> Set[str]:
        """Get all unique firmware versions in the dataset."""
        versions = set()
        for json_path in sorted(self.data_dir.glob('*.json')):
            if json_path.name.endswith('.meta.json'):
                continue
            try:
                session_info = load_session_raw(json_path)
                if session_info.firmware_version:
                    versions.add(session_info.firmware_version)
            except Exception:
                pass
        return versions

    def get_available_session_types(self) -> Set[str]:
        """Get all unique session types in the dataset."""
        types = set()
        for json_path in sorted(self.data_dir.glob('*.json')):
            if json_path.name.endswith('.meta.json'):
                continue
            meta = load_session_metadata(json_path)
            if meta:
                types.add(meta.session_type)
        return types

    def get_calibration_labels(self) -> Set[str]:
        """Get all unique calibration labels in the dataset."""
        cal_labels = set()
        for json_path in sorted(self.data_dir.glob('*.json')):
            if json_path.name.endswith('.meta.json'):
                continue
            meta = load_session_metadata(json_path)
            if meta:
                for seg in meta.labels_v2:
                    if seg.labels.calibration.value != 'none':
                        cal_labels.add(seg.labels.calibration.value)
        return cal_labels

    def list_sessions(
        self,
        firmware_version: Optional[str] = None,
        session_type: Optional[str] = None,
        has_labels: Optional[bool] = None,
        custom_labels: Optional[List[str]] = None,
        calibration_type: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        List sessions with optional filtering.

        Args:
            firmware_version: Filter by firmware version (e.g., '0.3.6')
            session_type: Filter by session type ('recording', 'calibration', 'test')
            has_labels: Filter by whether session has labels (True/False/None for all)
            custom_labels: Filter by presence of specific custom labels
            calibration_type: Filter by calibration type ('earth_field', 'hard_iron', etc.)

        Returns:
            List of session info dicts with filename, metadata, and filter matches
        """
        sessions = []

        for json_path in sorted(self.data_dir.glob('*.json')):
            if (json_path.name.endswith('.meta.json') or
                json_path.name.endswith('.full.json')):
                continue

            try:
                session_info = load_session_raw(json_path)
                meta = load_session_metadata(json_path)

                # Apply filters
                if firmware_version is not None:
                    if session_info.firmware_version != firmware_version:
                        continue

                if session_type is not None and meta:
                    if meta.session_type != session_type:
                        continue

                if has_labels is not None:
                    session_has_labels = meta is not None and (bool(meta.labels) or bool(meta.labels_v2))
                    if session_has_labels != has_labels:
                        continue

                if custom_labels is not None and meta:
                    session_custom = set()
                    for seg in meta.labels_v2:
                        session_custom.update(seg.labels.custom)
                    if not set(custom_labels).intersection(session_custom):
                        continue

                if calibration_type is not None and meta:
                    has_cal = any(
                        seg.labels.calibration.value == calibration_type
                        for seg in meta.labels_v2
                    )
                    if not has_cal:
                        continue

                # Build session info dict
                session_dict = {
                    'filename': json_path.name,
                    'path': str(json_path),
                    'version': session_info.version,
                    'timestamp': session_info.timestamp,
                    'firmware_version': session_info.firmware_version,
                    'sample_count': len(session_info.samples),
                    'has_embedded_metadata': session_info.has_embedded_metadata,
                }

                if meta:
                    session_dict.update({
                        'session_type': meta.session_type,
                        'has_labels': bool(meta.labels) or bool(meta.labels_v2),
                        'label_count': len(meta.labels) + len(meta.labels_v2),
                        'custom_labels': list(set(
                            label for seg in meta.labels_v2
                            for label in seg.labels.custom
                        )),
                        'calibration_types': list(set(
                            seg.labels.calibration.value
                            for seg in meta.labels_v2
                            if seg.labels.calibration.value != 'none'
                        )),
                        'split': meta.split,
                        'subject_id': meta.subject_id,
                    })

                sessions.append(session_dict)

            except Exception as e:
                print(f"Warning: Could not process {json_path}: {e}")

        return sessions

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics about the dataset."""
        labeled_count = 0
        unlabeled_count = 0
        v1_count = 0
        v2_count = 0
        total_samples = 0
        gesture_counts = {g.name: 0 for g in Gesture}
        finger_state_counts = {
            'extended': 0, 'partial': 0, 'flexed': 0, 'unknown': 0
        }
        custom_labels = set()
        firmware_versions = {}  # Count sessions per firmware version
        session_types = {}  # Count sessions per type
        calibration_types = {}  # Count sessions per calibration type

        for json_path in sorted(self.data_dir.glob('*.json')):
            # Skip non-session files
            if (json_path.name.endswith('.meta.json') or
                json_path.name.endswith('.full.json') or
                'calibration' in json_path.name.lower()):
                continue

            data = load_session_data(json_path)
            total_samples += len(data)

            # Load session info for firmware version
            try:
                session_info = load_session_raw(json_path)
                fw_ver = session_info.firmware_version or 'unknown'
                firmware_versions[fw_ver] = firmware_versions.get(fw_ver, 0) + 1
            except Exception:
                firmware_versions['unknown'] = firmware_versions.get('unknown', 0) + 1

            meta = load_session_metadata(json_path)
            if meta:
                has_labels = bool(meta.labels) or bool(meta.labels_v2)
                if has_labels:
                    labeled_count += 1
                    if meta.labels:
                        v1_count += 1
                    if meta.labels_v2:
                        v2_count += 1
                else:
                    unlabeled_count += 1

                # Track session type
                sess_type = meta.session_type
                session_types[sess_type] = session_types.get(sess_type, 0) + 1

                # Count V1 gestures
                for seg in meta.labels:
                    samples = seg.end_sample - seg.start_sample
                    gesture_counts[seg.gesture.name] += samples

                # Count V2 poses and finger states
                for seg in meta.labels_v2:
                    samples = seg.end_sample - seg.start_sample
                    if seg.labels.pose:
                        try:
                            gesture = Gesture.from_name(seg.labels.pose)
                            gesture_counts[gesture.name] += samples
                        except (KeyError, ValueError):
                            pass

                    if seg.labels.fingers:
                        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                            state = getattr(seg.labels.fingers, finger)
                            finger_state_counts[state.value] += samples

                    custom_labels.update(seg.labels.custom)

                    # Track calibration types
                    if seg.labels.calibration.value != 'none':
                        cal_type = seg.labels.calibration.value
                        calibration_types[cal_type] = calibration_types.get(cal_type, 0) + 1
            else:
                unlabeled_count += 1

        return {
            'total_sessions': labeled_count + unlabeled_count,
            'labeled_sessions': labeled_count,
            'unlabeled_sessions': unlabeled_count,
            'v1_sessions': v1_count,
            'v2_sessions': v2_count,
            'total_samples': total_samples,
            'gesture_counts': gesture_counts,
            'finger_state_counts': finger_state_counts,
            'custom_labels': list(custom_labels),
            'firmware_versions': firmware_versions,
            'session_types': session_types,
            'calibration_types': calibration_types,
            'stats': {
                'mean': self.stats.mean.tolist(),
                'std': self.stats.std.tolist()
            }
        }


if __name__ == '__main__':
    # Quick test
    import sys
    data_dir = sys.argv[1] if len(sys.argv) > 1 else 'data/GAMBIT'

    dataset = GambitDataset(data_dir)
    print("\nDataset Summary:")
    print(json.dumps(dataset.summary(), indent=2))

    X_train, y_train, X_val, y_val = dataset.get_train_val_split()
    print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}")

    # Test multi-label loading
    X_ml, y_ml = dataset.load_multilabel_sessions(['pose', 'motion', 'thumb', 'index'])
    print(f"\nMulti-label: X={X_ml.shape}, y={y_ml.shape}")
