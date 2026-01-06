#!/usr/bin/env python3
"""
Point Cloud ($P/$Q) Ablation Study for Finger State Classification

Tests whether treating magnetometer windows as point clouds (unordered sets)
helps with orientation invariance compared to sequential processing.

$-family variants tested:
1. Sequential (baseline): Windows processed as time series
2. Point Cloud ($P): Windows as unordered point sets with cloud distance
3. Sorted by magnitude: Points sorted by distance from mean
4. Cloud features: Statistical features of the point cloud distribution

Author: Claude
Date: January 2026
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from scipy.spatial.distance import cdist
from scipy.stats import skew, kurtosis


# ============================================================================
# DATA LOADING
# ============================================================================

@dataclass
class FingerStateData:
    combo: str
    samples: np.ndarray  # (n, 9)
    pitch_angles: np.ndarray


def load_session_with_pitch() -> Dict[str, FingerStateData]:
    """Load session data with pitch angles."""
    session_path = Path('data/GAMBIT/2025-12-31T14_06_18.270Z.json')
    if not session_path.exists():
        session_path = Path('.worktrees/data/GAMBIT/2025-12-31T14_06_18.270Z.json')

    with open(session_path) as f:
        data = json.load(f)

    samples = data['samples']
    labels = data['labels']
    combo_data = {}

    for lbl in labels:
        if 'labels' in lbl and isinstance(lbl['labels'], dict):
            fingers = lbl['labels'].get('fingers', {})
            start = lbl.get('start_sample', 0)
            end = lbl.get('end_sample', 0)
        else:
            fingers = lbl.get('fingers', {})
            start = lbl.get('startIndex', 0)
            end = lbl.get('endIndex', 0)

        if not fingers or all(v == 'unknown' for v in fingers.values()):
            continue

        segment_samples = samples[start:end]
        if len(segment_samples) < 5:
            continue

        sensor_data = []
        pitch_data = []

        for s in segment_samples:
            ax = s.get('ax', 0) / 8192.0
            ay = s.get('ay', 0) / 8192.0
            az = s.get('az', 0) / 8192.0
            gx = s.get('gx', 0) / 114.28
            gy = s.get('gy', 0) / 114.28
            gz = s.get('gz', 0) / 114.28

            if 'mx_ut' in s:
                mx, my, mz = s['mx_ut'], s['my_ut'], s['mz_ut']
            else:
                mx = s.get('mx', 0) / 10.24
                my = s.get('my', 0) / 10.24
                mz = s.get('mz', 0) / 10.24

            pitch = s.get('euler_pitch', 0)
            sensor_data.append([ax, ay, az, gx, gy, gz, mx, my, mz])
            pitch_data.append(pitch)

        if not sensor_data:
            continue

        combo = ''.join([
            'e' if fingers.get(f, '?') == 'extended' else
            'f' if fingers.get(f, '?') == 'flexed' else '?'
            for f in ['thumb', 'index', 'middle', 'ring', 'pinky']
        ])

        if '?' in combo:
            continue

        if combo not in combo_data:
            combo_data[combo] = FingerStateData(
                combo=combo,
                samples=np.array(sensor_data),
                pitch_angles=np.array(pitch_data)
            )
        else:
            existing = combo_data[combo]
            combo_data[combo] = FingerStateData(
                combo=combo,
                samples=np.vstack([existing.samples, sensor_data]),
                pitch_angles=np.concatenate([existing.pitch_angles, pitch_data])
            )

    return combo_data


# ============================================================================
# SYNTHETIC GENERATOR
# ============================================================================

class SyntheticGenerator:
    def __init__(self, real_data: Dict[str, FingerStateData]):
        self.real_data = real_data
        self.baseline = real_data.get('eeeee')
        self._compute_finger_effects()

    def _compute_finger_effects(self):
        self.finger_effects = {}
        if not self.baseline:
            return

        baseline_mag = self.baseline.samples[:, 6:9].mean(axis=0)
        single_finger_combos = {
            'thumb': 'feeee', 'index': 'efeee', 'middle': 'eefee',
            'ring': 'eeefe', 'pinky': 'eeeef'
        }

        for finger, combo in single_finger_combos.items():
            if combo in self.real_data:
                data = self.real_data[combo]
                self.finger_effects[finger] = {
                    'mag_delta': data.samples[:, 6:9].mean(axis=0) - baseline_mag,
                    'mag_std': data.samples[:, 6:9].std(axis=0),
                }
            else:
                self.finger_effects[finger] = {
                    'mag_delta': np.array([200, 200, 200]),
                    'mag_std': np.array([50, 50, 50]),
                }

    def generate_combo(self, combo: str, n_samples: int) -> np.ndarray:
        if combo in self.real_data:
            real = self.real_data[combo]
            mag_mean = real.samples[:, 6:9].mean(axis=0)
            mag_std = real.samples[:, 6:9].std(axis=0)
        else:
            if self.baseline:
                mag_mean = self.baseline.samples[:, 6:9].mean(axis=0)
                mag_std = self.baseline.samples[:, 6:9].std(axis=0)
            else:
                mag_mean = np.array([46, -46, 31])
                mag_std = np.array([25, 40, 50])

            fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
            for i, state in enumerate(combo):
                if state == 'f':
                    finger = fingers[i]
                    if finger in self.finger_effects:
                        mag_mean = mag_mean + self.finger_effects[finger]['mag_delta']

        samples = []
        for _ in range(n_samples):
            mag_sample = mag_mean + np.random.randn(3) * mag_std
            ax = np.random.normal(0, 0.05)
            ay = np.random.normal(0, 0.05)
            az = np.random.normal(-1, 0.05)
            gx = np.random.normal(0, 2.0)
            gy = np.random.normal(0, 2.0)
            gz = np.random.normal(0, 2.0)
            samples.append([ax, ay, az, gx, gy, gz, mag_sample[0], mag_sample[1], mag_sample[2]])

        return np.array(samples)


# ============================================================================
# WINDOW CREATION VARIANTS
# ============================================================================

def create_sequential_windows(samples: np.ndarray, window_size: int, stride: int = None) -> np.ndarray:
    """Standard sequential windows (baseline)."""
    if stride is None:
        stride = max(1, window_size // 2)

    n_samples = len(samples)
    if n_samples < window_size:
        padding = np.zeros((window_size - n_samples, samples.shape[1]))
        samples = np.vstack([samples, padding])
        n_samples = window_size

    windows = []
    for i in range(0, n_samples - window_size + 1, stride):
        windows.append(samples[i:i+window_size])

    if not windows:
        windows.append(samples[:window_size])

    return np.array(windows)


def create_sorted_windows(samples: np.ndarray, window_size: int, stride: int = None) -> np.ndarray:
    """
    Sort points within each window by magnitude from centroid.
    This makes the representation order-invariant within the window.
    """
    sequential = create_sequential_windows(samples, window_size, stride)

    sorted_windows = []
    for window in sequential:
        # Compute centroid
        centroid = window.mean(axis=0)
        # Sort by distance from centroid
        distances = np.linalg.norm(window - centroid, axis=1)
        sorted_idx = np.argsort(distances)
        sorted_windows.append(window[sorted_idx])

    return np.array(sorted_windows)


def create_cloud_features(samples: np.ndarray, window_size: int, stride: int = None) -> np.ndarray:
    """
    Extract point cloud distribution features from each window.
    Returns statistical features instead of raw points.
    """
    sequential = create_sequential_windows(samples, window_size, stride)

    features_list = []
    for window in sequential:
        features = []

        # Mean (centroid)
        mean = window.mean(axis=0)
        features.extend(mean)

        # Standard deviation (spread)
        std = window.std(axis=0)
        features.extend(std)

        # Min and max (bounding box)
        features.extend(window.min(axis=0))
        features.extend(window.max(axis=0))

        # Median
        features.extend(np.median(window, axis=0))

        # Interquartile range
        q75 = np.percentile(window, 75, axis=0)
        q25 = np.percentile(window, 25, axis=0)
        features.extend(q75 - q25)

        features_list.append(features)

    return np.array(features_list)


def create_pairwise_distance_features(samples: np.ndarray, window_size: int, stride: int = None) -> np.ndarray:
    """
    Extract pairwise distance matrix features - truly order invariant.
    Uses eigenvalues of distance matrix as features.
    """
    sequential = create_sequential_windows(samples, window_size, stride)

    features_list = []
    for window in sequential:
        # Compute pairwise distance matrix
        dist_matrix = cdist(window, window, metric='euclidean')

        # Extract features from distance matrix
        features = []

        # Mean pairwise distance
        upper_tri = dist_matrix[np.triu_indices(len(window), k=1)]
        features.append(upper_tri.mean())
        features.append(upper_tri.std())
        features.append(upper_tri.max())

        # Eigenvalues (sorted, top k)
        eigenvalues = np.linalg.eigvalsh(dist_matrix)
        eigenvalues = np.sort(eigenvalues)[::-1]
        features.extend(eigenvalues[:min(5, len(eigenvalues))])

        # Pad if needed
        while len(features) < 8:
            features.append(0)

        features_list.append(features[:8])

    return np.array(features_list)


# ============================================================================
# MODEL BUILDING
# ============================================================================

def combo_to_label(combo: str) -> np.ndarray:
    return np.array([0 if c == 'e' else 1 for c in combo], dtype=np.float32)


def build_sequential_model(window_size: int, n_features: int) -> keras.Model:
    """Model for sequential windows (CNN-LSTM)."""
    inputs = keras.layers.Input(shape=(window_size, n_features))

    x = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(5, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def build_dense_model(n_features: int) -> keras.Model:
    """Model for flat feature vectors (cloud features)."""
    inputs = keras.layers.Input(shape=(n_features,))

    x = keras.layers.Dense(64, activation='relu')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    x = keras.layers.Dropout(0.3)(x)
    outputs = keras.layers.Dense(5, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def prepare_cross_orientation_data(
    real_data: Dict[str, FingerStateData],
    window_fn,
    window_size: int = 10,
    synthetic_ratio: float = 0.5,
    is_flat: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Prepare data with cross-orientation split."""

    all_pitches = []
    for cd in real_data.values():
        all_pitches.extend(cd.pitch_angles.tolist())
    q1 = np.percentile(all_pitches, 25)
    q3 = np.percentile(all_pitches, 75)

    generator = SyntheticGenerator(real_data) if synthetic_ratio > 0 else None

    train_data = []
    train_labels = []
    test_data = []
    test_labels = []

    for combo, combo_data in real_data.items():
        label = combo_to_label(combo)
        mag_features = combo_data.samples[:, 6:9]

        high_pitch_mask = combo_data.pitch_angles >= q3
        low_pitch_mask = combo_data.pitch_angles <= q1

        high_pitch_samples = mag_features[high_pitch_mask]
        low_pitch_samples = mag_features[low_pitch_mask]

        # Add synthetic
        if synthetic_ratio > 0 and generator:
            n_synth = int(150 * synthetic_ratio)
            synth_samples = generator.generate_combo(combo, n_synth)[:, 6:9]
            if len(high_pitch_samples) > 0:
                high_pitch_samples = np.vstack([high_pitch_samples, synth_samples])
            else:
                high_pitch_samples = synth_samples

        # Create windows/features
        if len(high_pitch_samples) >= window_size:
            windows = window_fn(high_pitch_samples, window_size)
            for w in windows:
                train_data.append(w)
                train_labels.append(label)

        if len(low_pitch_samples) >= window_size:
            windows = window_fn(low_pitch_samples, window_size)
            for w in windows:
                test_data.append(w)
                test_labels.append(label)

    X_train = np.array(train_data)
    y_train = np.array(train_labels)
    X_test = np.array(test_data)
    y_test = np.array(test_labels)

    # Normalize
    if is_flat:
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
    else:
        mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
        std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    info = {
        'n_train': len(X_train),
        'n_test': len(X_test),
        'shape': X_train.shape,
    }

    return X_train, y_train, X_test, y_test, info


def train_and_evaluate(X_train, y_train, X_test, y_test, model_fn, epochs=30) -> Dict:
    """Train and evaluate model."""
    model = model_fn()

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    # Split train for validation
    val_size = int(0.15 * len(X_train))
    X_val = X_train[:val_size]
    y_val = y_train[:val_size]
    X_train = X_train[val_size:]
    y_train = y_train[val_size:]

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluate
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)
    test_acc = np.mean(y_pred == y_test)

    # Per-finger
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger = {f: float(np.mean(y_pred[:, i] == y_test[:, i])) for i, f in enumerate(fingers)}

    tf.keras.backend.clear_session()

    return {
        'test_acc': float(test_acc),
        'per_finger': per_finger,
    }


# ============================================================================
# MAIN ABLATION
# ============================================================================

def main():
    print("=" * 80)
    print("POINT CLOUD ($P/$Q) ABLATION STUDY")
    print("=" * 80)

    # Load data
    print("\nLoading data...")
    real_data = load_session_with_pitch()
    print(f"Loaded {len(real_data)} finger state combinations")

    window_size = 10
    results = {}

    # =========================================================================
    # CONFIG 1: Sequential (V3 baseline)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONFIG 1: Sequential Windows (V3 baseline)")
    print("=" * 70)

    X_train, y_train, X_test, y_test, info = prepare_cross_orientation_data(
        real_data, create_sequential_windows, window_size=window_size
    )
    print(f"Shape: {info['shape']}, Train: {info['n_train']}, Test: {info['n_test']}")

    def model_fn():
        return build_sequential_model(window_size, 3)

    result = train_and_evaluate(X_train, y_train, X_test, y_test, model_fn)
    results['sequential'] = result
    print(f"Cross-orientation accuracy: {result['test_acc']:.1%}")

    # =========================================================================
    # CONFIG 2: Sorted Windows (Order Invariant)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONFIG 2: Sorted Windows (by distance from centroid)")
    print("=" * 70)

    X_train, y_train, X_test, y_test, info = prepare_cross_orientation_data(
        real_data, create_sorted_windows, window_size=window_size
    )
    print(f"Shape: {info['shape']}, Train: {info['n_train']}, Test: {info['n_test']}")

    result = train_and_evaluate(X_train, y_train, X_test, y_test, model_fn)
    results['sorted'] = result
    print(f"Cross-orientation accuracy: {result['test_acc']:.1%}")

    # =========================================================================
    # CONFIG 3: Cloud Statistical Features
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONFIG 3: Cloud Statistical Features (mean, std, min, max, median, IQR)")
    print("=" * 70)

    X_train, y_train, X_test, y_test, info = prepare_cross_orientation_data(
        real_data, create_cloud_features, window_size=window_size, is_flat=True
    )
    print(f"Shape: {info['shape']}, Train: {info['n_train']}, Test: {info['n_test']}")

    n_features = X_train.shape[-1]
    def cloud_model_fn():
        return build_dense_model(n_features)

    result = train_and_evaluate(X_train, y_train, X_test, y_test, cloud_model_fn)
    results['cloud_features'] = result
    print(f"Cross-orientation accuracy: {result['test_acc']:.1%}")

    # =========================================================================
    # CONFIG 4: Pairwise Distance Features (truly order invariant)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONFIG 4: Pairwise Distance Features (eigenvalues of dist matrix)")
    print("=" * 70)

    X_train, y_train, X_test, y_test, info = prepare_cross_orientation_data(
        real_data, create_pairwise_distance_features, window_size=window_size, is_flat=True
    )
    print(f"Shape: {info['shape']}, Train: {info['n_train']}, Test: {info['n_test']}")

    n_features = X_train.shape[-1]
    def dist_model_fn():
        return build_dense_model(n_features)

    result = train_and_evaluate(X_train, y_train, X_test, y_test, dist_model_fn)
    results['pairwise_distance'] = result
    print(f"Cross-orientation accuracy: {result['test_acc']:.1%}")

    # =========================================================================
    # CONFIG 5: Mean only (single point - simplest)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONFIG 5: Mean Only (window collapsed to single point)")
    print("=" * 70)

    def mean_only_windows(samples, ws, stride=None):
        seq = create_sequential_windows(samples, ws, stride)
        return seq.mean(axis=1)  # (n_windows, 3)

    X_train, y_train, X_test, y_test, info = prepare_cross_orientation_data(
        real_data, mean_only_windows, window_size=window_size, is_flat=True
    )
    print(f"Shape: {info['shape']}, Train: {info['n_train']}, Test: {info['n_test']}")

    def mean_model_fn():
        return build_dense_model(3)

    result = train_and_evaluate(X_train, y_train, X_test, y_test, mean_model_fn)
    results['mean_only'] = result
    print(f"Cross-orientation accuracy: {result['test_acc']:.1%}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY: POINT CLOUD ABLATION RESULTS")
    print("=" * 80)

    print(f"\n{'Config':<25} {'Accuracy':>10} {'Notes'}")
    print("-" * 60)

    for config, r in sorted(results.items(), key=lambda x: -x[1]['test_acc']):
        notes = ""
        if config == 'sequential':
            notes = "(V3 baseline)"
        elif config == 'sorted':
            notes = "(order invariant)"
        elif config == 'cloud_features':
            notes = "(statistical features)"
        elif config == 'pairwise_distance':
            notes = "(eigenvalues - truly order invariant)"
        elif config == 'mean_only':
            notes = "(simplest possible)"

        print(f"{config:<25} {r['test_acc']:>9.1%} {notes}")

    # Save results
    output_path = Path("ml/point_cloud_ablation.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Analysis
    print("\n" + "=" * 80)
    print("ANALYSIS")
    print("=" * 80)

    best = max(results.items(), key=lambda x: x[1]['test_acc'])
    print(f"""
Best configuration: {best[0]} ({best[1]['test_acc']:.1%})

Key insights:
- Point cloud methods treat windows as unordered sets of points
- This tests if ORDER within window matters for orientation invariance
- If sorted/cloud features beat sequential, it suggests temporal order
  is an orientation-dependent artifact that hurts generalization
""")

    return results


if __name__ == "__main__":
    main()
