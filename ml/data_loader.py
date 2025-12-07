"""
SIMCAP Data Loader

Loads raw JSON data files, applies preprocessing, and creates
windowed tensors suitable for ML training.
"""

import json
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .schema import (
    Gesture, SessionMetadata, LabeledSegment,
    SENSOR_RANGES, FEATURE_NAMES, NUM_FEATURES
)


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


def load_session_data(json_path: Path) -> np.ndarray:
    """
    Load raw sensor data from a JSON file.

    Args:
        json_path: Path to the .json data file

    Returns:
        numpy array of shape (N, 9) where N is number of samples
        and 9 is the IMU features [ax, ay, az, gx, gy, gz, mx, my, mz]
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Extract IMU features
    samples = []
    for sample in data:
        row = [
            sample['ax'], sample['ay'], sample['az'],
            sample['gx'], sample['gy'], sample['gz'],
            sample['mx'], sample['my'], sample['mz']
        ]
        samples.append(row)

    return np.array(samples, dtype=np.float32)


def load_session_metadata(json_path: Path) -> Optional[SessionMetadata]:
    """
    Load metadata for a session if it exists.

    Args:
        json_path: Path to the .json data file

    Returns:
        SessionMetadata if .meta.json exists, else None
    """
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
    """
    all_data = []

    for json_path in data_dir.glob('*.json'):
        if json_path.suffix == '.json' and not json_path.name.endswith('.meta.json'):
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


def labels_from_segments(num_samples: int, segments: List[LabeledSegment],
                         default_label: Gesture = Gesture.REST) -> np.ndarray:
    """
    Convert labeled segments to per-sample labels array.

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


class GambitDataset:
    """
    Dataset class for loading and preparing GAMBIT data for training.
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
        Load all labeled sessions and create windowed dataset.

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
            if meta is None or not meta.labels:
                continue  # Skip unlabeled sessions

            if split is not None and meta.split != split:
                continue

            # Load and normalize data
            data = load_session_data(json_path)
            data = normalize_data(data, self.stats, self.normalize_method)

            # Create per-sample labels
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

    def summary(self) -> Dict[str, Any]:
        """Return summary statistics about the dataset."""
        labeled_count = 0
        unlabeled_count = 0
        total_samples = 0
        gesture_counts = {g.name: 0 for g in Gesture}

        for json_path in sorted(self.data_dir.glob('*.json')):
            if json_path.name.endswith('.meta.json'):
                continue

            data = load_session_data(json_path)
            total_samples += len(data)

            meta = load_session_metadata(json_path)
            if meta and meta.labels:
                labeled_count += 1
                for seg in meta.labels:
                    samples = seg.end_sample - seg.start_sample
                    gesture_counts[seg.gesture.name] += samples
            else:
                unlabeled_count += 1

        return {
            'total_sessions': labeled_count + unlabeled_count,
            'labeled_sessions': labeled_count,
            'unlabeled_sessions': unlabeled_count,
            'total_samples': total_samples,
            'gesture_counts': gesture_counts,
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
