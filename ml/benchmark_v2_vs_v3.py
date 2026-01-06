#!/usr/bin/env python3
"""
Benchmark V2 vs V3 Models with Held-Out Validation

Tests re-training on smaller subset of data and compares:
1. Inference latency (ms per prediction)
2. Accuracy improvements
3. Model complexity metrics

Author: Claude
Date: January 2026
"""

import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FingerStateData:
    combo: str
    samples: np.ndarray  # (n, 9) ax,ay,az,gx,gy,gz,mx,my,mz
    pitch_angles: np.ndarray


# ============================================================================
# DATA LOADING
# ============================================================================

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
# SYNTHETIC DATA
# ============================================================================

class SyntheticGenerator:
    """Generate synthetic samples with tight distribution."""

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
        """Generate synthetic samples."""
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
# DATA PREPROCESSING
# ============================================================================

def extract_features(samples: np.ndarray, feature_set: str) -> np.ndarray:
    """Extract feature set."""
    if feature_set == '9dof':
        return samples
    elif feature_set == 'mag_only':
        return samples[:, 6:9]
    return samples


def create_windows(samples: np.ndarray, window_size: int, stride: int = None) -> np.ndarray:
    """Create sliding windows."""
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


def combo_to_label(combo: str) -> np.ndarray:
    """Convert combo string to binary label."""
    return np.array([0 if c == 'e' else 1 for c in combo], dtype=np.float32)


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

def build_v2_model(window_size: int = 50, n_features: int = 9) -> keras.Model:
    """Build V2 model: 9-DoF, window=50."""
    inputs = keras.layers.Input(shape=(window_size, n_features))

    x = keras.layers.Conv1D(32, 5, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(5, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_v3_model(window_size: int = 10, n_features: int = 3) -> keras.Model:
    """Build V3 model: mag_only, window=10."""
    inputs = keras.layers.Input(shape=(window_size, n_features))

    x = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(5, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================================
# DATA PREPARATION WITH 3-WAY SPLIT
# ============================================================================

def prepare_data_with_heldout(
    real_data: Dict[str, FingerStateData],
    feature_set: str,
    window_size: int,
    synthetic_ratio: float,
    samples_per_combo: int = 300,
    subset_ratio: float = 0.5,  # Use only 50% of training data
    heldout_ratio: float = 0.2   # 20% held out for final test
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Prepare data with 3-way split:
    - Training: subset_ratio of high-pitch data + synthetic
    - Validation: remaining high-pitch data
    - Held-out Test: low-pitch data (cross-orientation)
    """
    # Calculate pitch quartiles
    all_pitches = []
    for cd in real_data.values():
        all_pitches.extend(cd.pitch_angles.tolist())
    q1 = np.percentile(all_pitches, 25)
    q3 = np.percentile(all_pitches, 75)

    generator = SyntheticGenerator(real_data) if synthetic_ratio > 0 else None

    train_windows = []
    train_labels = []
    val_windows = []
    val_labels = []
    test_windows = []
    test_labels = []

    for combo, combo_data in real_data.items():
        label = combo_to_label(combo)
        features = extract_features(combo_data.samples, feature_set)

        # Split by pitch
        high_pitch_mask = combo_data.pitch_angles >= q3
        low_pitch_mask = combo_data.pitch_angles <= q1

        high_pitch_samples = features[high_pitch_mask]
        low_pitch_samples = features[low_pitch_mask]

        # Further split high_pitch into train/val based on subset_ratio
        if len(high_pitch_samples) > 0:
            n_samples = len(high_pitch_samples)
            n_train = int(n_samples * subset_ratio)

            indices = np.random.permutation(n_samples)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]

            train_samples = high_pitch_samples[train_indices]
            val_samples = high_pitch_samples[val_indices]

            # Add synthetic to training
            if synthetic_ratio > 0 and generator and len(train_samples) > 0:
                n_synth = int(len(train_samples) * synthetic_ratio)
                if n_synth > 0:
                    synth_samples = generator.generate_combo(combo, n_synth)
                    synth_features = extract_features(synth_samples, feature_set)
                    if len(synth_features) > 0 and synth_features.shape[1] == train_samples.shape[1]:
                        train_samples = np.vstack([train_samples, synth_features])

            # Create windows for train
            if len(train_samples) >= window_size:
                windows = create_windows(train_samples, window_size)
                for w in windows:
                    train_windows.append(w)
                    train_labels.append(label)

            # Create windows for val
            if len(val_samples) >= window_size:
                windows = create_windows(val_samples, window_size)
                for w in windows:
                    val_windows.append(w)
                    val_labels.append(label)

        # Low pitch for held-out test
        if len(low_pitch_samples) >= window_size:
            windows = create_windows(low_pitch_samples, window_size)
            for w in windows:
                test_windows.append(w)
                test_labels.append(label)

    X_train = np.array(train_windows)
    y_train = np.array(train_labels)
    X_val = np.array(val_windows)
    y_val = np.array(val_labels)
    X_test = np.array(test_windows)
    y_test = np.array(test_labels)

    # Compute global normalization stats from training data only
    n_features = X_train.shape[-1]
    mean = X_train.reshape(-1, n_features).mean(axis=0)
    std = X_train.reshape(-1, n_features).std(axis=0) + 1e-8

    # Apply normalization
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    info = {
        'n_train': len(X_train),
        'n_val': len(X_val),
        'n_test': len(X_test),
        'n_features': n_features,
        'window_size': window_size,
        'mean': mean.tolist(),
        'std': std.tolist(),
        'q1_pitch': q1,
        'q3_pitch': q3,
        'subset_ratio': subset_ratio,
    }

    return X_train, y_train, X_val, y_val, X_test, y_test, info


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def train_model(
    X_train, y_train, X_val, y_val,
    model_fn, epochs=30, verbose=0
) -> keras.Model:
    """Train model with early stopping."""
    window_size = X_train.shape[1]
    n_features = X_train.shape[2]

    model = model_fn(window_size, n_features)

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=32,
        callbacks=[early_stop],
        verbose=verbose
    )

    return model


def evaluate_model(model: keras.Model, X: np.ndarray, y: np.ndarray) -> Dict:
    """Evaluate model accuracy."""
    y_pred = model.predict(X, verbose=0)
    y_pred_bin = (y_pred > 0.5).astype(int)

    overall_acc = np.mean(y_pred_bin == y)

    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger = {}
    for i, f in enumerate(fingers):
        per_finger[f] = float(np.mean(y_pred_bin[:, i] == y[:, i]))

    return {
        'overall_acc': float(overall_acc),
        'per_finger': per_finger
    }


def benchmark_inference_latency(
    model: keras.Model, X: np.ndarray, n_iterations: int = 100
) -> Dict:
    """Benchmark inference latency."""
    # Warmup
    for _ in range(10):
        _ = model.predict(X[:1], verbose=0)

    # Single sample latency
    single_times = []
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = model.predict(X[:1], verbose=0)
        end = time.perf_counter()
        single_times.append((end - start) * 1000)  # Convert to ms

    # Batch latency
    batch_times = []
    batch_size = min(32, len(X))
    for _ in range(n_iterations):
        start = time.perf_counter()
        _ = model.predict(X[:batch_size], verbose=0)
        end = time.perf_counter()
        batch_times.append((end - start) * 1000 / batch_size)  # ms per sample

    return {
        'single_sample_ms': {
            'mean': float(np.mean(single_times)),
            'std': float(np.std(single_times)),
            'min': float(np.min(single_times)),
            'max': float(np.max(single_times)),
        },
        'batch_ms_per_sample': {
            'mean': float(np.mean(batch_times)),
            'std': float(np.std(batch_times)),
            'min': float(np.min(batch_times)),
            'max': float(np.max(batch_times)),
        }
    }


def get_model_complexity(model: keras.Model) -> Dict:
    """Get model complexity metrics."""
    total_params = model.count_params()

    trainable_params = sum([
        tf.keras.backend.count_params(w)
        for w in model.trainable_weights
    ])

    return {
        'total_params': int(total_params),
        'trainable_params': int(trainable_params),
        'layers': len(model.layers),
    }


# ============================================================================
# MAIN BENCHMARK
# ============================================================================

def main():
    print("=" * 80)
    print("V2 vs V3 BENCHMARK - HELD-OUT VALIDATION & INFERENCE LATENCY")
    print("=" * 80)

    # Load data
    print("\n--- Loading Data ---")
    real_data = load_session_with_pitch()
    print(f"Loaded {len(real_data)} finger state combinations")

    results = {}

    # =========================================================================
    # V2 Model: 9-DoF, window=50
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING V2 MODEL (9-DoF, window=50)")
    print("=" * 70)

    X_train_v2, y_train_v2, X_val_v2, y_val_v2, X_test_v2, y_test_v2, info_v2 = prepare_data_with_heldout(
        real_data, feature_set='9dof', window_size=50,
        synthetic_ratio=0.5, samples_per_combo=300,
        subset_ratio=0.5  # Use 50% of training data
    )

    print(f"Data split: Train={info_v2['n_train']}, Val={info_v2['n_val']}, Test={info_v2['n_test']}")

    model_v2 = train_model(X_train_v2, y_train_v2, X_val_v2, y_val_v2, build_v2_model, verbose=1)

    # Evaluate on all splits
    print("\n--- Evaluating V2 Model ---")
    train_metrics_v2 = evaluate_model(model_v2, X_train_v2, y_train_v2)
    val_metrics_v2 = evaluate_model(model_v2, X_val_v2, y_val_v2)
    test_metrics_v2 = evaluate_model(model_v2, X_test_v2, y_test_v2)

    print(f"Train Accuracy: {train_metrics_v2['overall_acc']:.1%}")
    print(f"Val Accuracy:   {val_metrics_v2['overall_acc']:.1%}")
    print(f"Test Accuracy:  {test_metrics_v2['overall_acc']:.1%} (held-out cross-orientation)")

    # Benchmark latency
    print("\n--- Benchmarking V2 Inference Latency ---")
    latency_v2 = benchmark_inference_latency(model_v2, X_test_v2, n_iterations=100)
    print(f"Single sample: {latency_v2['single_sample_ms']['mean']:.2f} ± {latency_v2['single_sample_ms']['std']:.2f} ms")
    print(f"Batch (per sample): {latency_v2['batch_ms_per_sample']['mean']:.2f} ± {latency_v2['batch_ms_per_sample']['std']:.2f} ms")

    # Model complexity
    complexity_v2 = get_model_complexity(model_v2)
    print(f"Model params: {complexity_v2['total_params']:,}")

    results['v2'] = {
        'train': train_metrics_v2,
        'val': val_metrics_v2,
        'test': test_metrics_v2,
        'latency': latency_v2,
        'complexity': complexity_v2,
        'config': info_v2,
    }

    tf.keras.backend.clear_session()

    # =========================================================================
    # V3 Model: mag_only, window=10
    # =========================================================================
    print("\n" + "=" * 70)
    print("TRAINING V3 MODEL (mag_only, window=10)")
    print("=" * 70)

    X_train_v3, y_train_v3, X_val_v3, y_val_v3, X_test_v3, y_test_v3, info_v3 = prepare_data_with_heldout(
        real_data, feature_set='mag_only', window_size=10,
        synthetic_ratio=0.5, samples_per_combo=300,
        subset_ratio=0.5  # Use 50% of training data
    )

    print(f"Data split: Train={info_v3['n_train']}, Val={info_v3['n_val']}, Test={info_v3['n_test']}")

    model_v3 = train_model(X_train_v3, y_train_v3, X_val_v3, y_val_v3, build_v3_model, verbose=1)

    # Evaluate on all splits
    print("\n--- Evaluating V3 Model ---")
    train_metrics_v3 = evaluate_model(model_v3, X_train_v3, y_train_v3)
    val_metrics_v3 = evaluate_model(model_v3, X_val_v3, y_val_v3)
    test_metrics_v3 = evaluate_model(model_v3, X_test_v3, y_test_v3)

    print(f"Train Accuracy: {train_metrics_v3['overall_acc']:.1%}")
    print(f"Val Accuracy:   {val_metrics_v3['overall_acc']:.1%}")
    print(f"Test Accuracy:  {test_metrics_v3['overall_acc']:.1%} (held-out cross-orientation)")

    # Benchmark latency
    print("\n--- Benchmarking V3 Inference Latency ---")
    latency_v3 = benchmark_inference_latency(model_v3, X_test_v3, n_iterations=100)
    print(f"Single sample: {latency_v3['single_sample_ms']['mean']:.2f} ± {latency_v3['single_sample_ms']['std']:.2f} ms")
    print(f"Batch (per sample): {latency_v3['batch_ms_per_sample']['mean']:.2f} ± {latency_v3['batch_ms_per_sample']['std']:.2f} ms")

    # Model complexity
    complexity_v3 = get_model_complexity(model_v3)
    print(f"Model params: {complexity_v3['total_params']:,}")

    results['v3'] = {
        'train': train_metrics_v3,
        'val': val_metrics_v3,
        'test': test_metrics_v3,
        'latency': latency_v3,
        'complexity': complexity_v3,
        'config': info_v3,
    }

    tf.keras.backend.clear_session()

    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY: V2 vs V3")
    print("=" * 80)

    # Accuracy comparison
    print("\n--- Accuracy Metrics ---")
    print(f"{'Metric':<25} {'V2':>12} {'V3':>12} {'Improvement':>15}")
    print("-" * 70)

    v2_test = results['v2']['test']['overall_acc']
    v3_test = results['v3']['test']['overall_acc']
    print(f"{'Train Accuracy':<25} {results['v2']['train']['overall_acc']:>11.1%} {results['v3']['train']['overall_acc']:>11.1%} {(results['v3']['train']['overall_acc'] - results['v2']['train']['overall_acc'])*100:>+13.1f}%")
    print(f"{'Val Accuracy':<25} {results['v2']['val']['overall_acc']:>11.1%} {results['v3']['val']['overall_acc']:>11.1%} {(results['v3']['val']['overall_acc'] - results['v2']['val']['overall_acc'])*100:>+13.1f}%")
    print(f"{'Test Accuracy (Held-out)':<25} {v2_test:>11.1%} {v3_test:>11.1%} {(v3_test - v2_test)*100:>+13.1f}%")

    # Inference latency comparison
    print("\n--- Inference Latency ---")
    print(f"{'Metric':<25} {'V2':>12} {'V3':>12} {'Speedup':>15}")
    print("-" * 70)

    v2_single = results['v2']['latency']['single_sample_ms']['mean']
    v3_single = results['v3']['latency']['single_sample_ms']['mean']
    v2_batch = results['v2']['latency']['batch_ms_per_sample']['mean']
    v3_batch = results['v3']['latency']['batch_ms_per_sample']['mean']

    print(f"{'Single Sample (ms)':<25} {v2_single:>11.2f} {v3_single:>11.2f} {v2_single/v3_single:>14.2f}x")
    print(f"{'Batch (ms/sample)':<25} {v2_batch:>11.2f} {v3_batch:>11.2f} {v2_batch/v3_batch:>14.2f}x")

    # Model complexity comparison
    print("\n--- Model Complexity ---")
    print(f"{'Metric':<25} {'V2':>12} {'V3':>12} {'Reduction':>15}")
    print("-" * 70)

    v2_params = results['v2']['complexity']['total_params']
    v3_params = results['v3']['complexity']['total_params']
    v2_window = info_v2['window_size']
    v3_window = info_v3['window_size']
    v2_features = info_v2['n_features']
    v3_features = info_v3['n_features']

    print(f"{'Parameters':<25} {v2_params:>12,} {v3_params:>12,} {v2_params/v3_params:>14.2f}x")
    print(f"{'Window Size':<25} {v2_window:>12} {v3_window:>12} {v2_window/v3_window:>14.2f}x")
    print(f"{'Features':<25} {v2_features:>12} {v3_features:>12} {v2_features/v3_features:>14.2f}x")

    # Per-finger accuracy comparison
    print("\n--- Per-Finger Test Accuracy ---")
    print(f"{'Finger':<15} {'V2':>10} {'V3':>10} {'Improvement':>15}")
    print("-" * 55)

    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        v2_acc = results['v2']['test']['per_finger'][finger]
        v3_acc = results['v3']['test']['per_finger'][finger]
        print(f"{finger.capitalize():<15} {v2_acc:>9.1%} {v3_acc:>9.1%} {(v3_acc - v2_acc)*100:>+13.1f}%")

    # Key improvements
    print("\n" + "=" * 80)
    print("KEY IMPROVEMENTS IN V3")
    print("=" * 80)

    acc_improvement = (v3_test - v2_test) * 100
    latency_speedup = v2_single / v3_single
    param_reduction = v2_params / v3_params

    print(f"\n1. Accuracy:  {'+' if acc_improvement >= 0 else ''}{acc_improvement:.1f}% improvement on held-out data")
    print(f"2. Latency:   {latency_speedup:.1f}x faster inference")
    print(f"3. Complexity: {param_reduction:.1f}x fewer parameters")
    print(f"4. Window:    {v2_window} → {v3_window} samples ({v2_window/v3_window:.0f}x smaller)")
    print(f"5. Features:  {v2_features} → {v3_features} features ({v2_features/v3_features:.0f}x fewer)")

    # Save results
    results_path = Path("ml/v2_v3_benchmark_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nFull results saved to: {results_path}")

    return results


if __name__ == '__main__':
    main()
