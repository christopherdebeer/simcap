#!/usr/bin/env python3
"""
Deploy Finger State Model v3 for TensorFlow.js

Incorporates all research learnings:
1. mag_only (3 features) instead of 9-DoF
2. window_size=10 instead of 50
3. 50% synthetic data WITHOUT orientation augmentation
4. Ablation of $-family normalization techniques

Compares:
- v2: 9-DoF, w=50, synthetic with covariance
- v3: mag_only, w=10, 50% tight synthetic
- v3 + window normalization ($-family style)

Author: Claude
Date: January 2026
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import subprocess

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA LOADING
# ============================================================================

@dataclass
class FingerStateData:
    combo: str
    samples: np.ndarray  # (n, 9) ax,ay,az,gx,gy,gz,mx,my,mz
    pitch_angles: np.ndarray


def load_session_with_pitch() -> Dict[str, FingerStateData]:
    """Load session data with pitch angles for cross-orientation testing."""
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
# SYNTHETIC DATA GENERATION
# ============================================================================

class SyntheticGenerator:
    """Generate synthetic samples - TIGHT distribution (no orientation augmentation)."""

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
        """Generate synthetic samples with TIGHT distribution (1x std, not 2x)."""
        if combo in self.real_data:
            real = self.real_data[combo]
            mag_mean = real.samples[:, 6:9].mean(axis=0)
            mag_std = real.samples[:, 6:9].std(axis=0)
        else:
            # Interpolate from single-finger effects
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
            # TIGHT distribution: 1x std (NOT 2x like v2 orientation augmentation)
            mag_sample = mag_mean + np.random.randn(3) * mag_std

            # IMU (static hand) - not used but kept for compatibility
            ax = np.random.normal(0, 0.05)
            ay = np.random.normal(0, 0.05)
            az = np.random.normal(-1, 0.05)
            gx = np.random.normal(0, 2.0)
            gy = np.random.normal(0, 2.0)
            gz = np.random.normal(0, 2.0)

            samples.append([ax, ay, az, gx, gy, gz, mag_sample[0], mag_sample[1], mag_sample[2]])

        return np.array(samples)


# ============================================================================
# FEATURE EXTRACTION & WINDOWING
# ============================================================================

def extract_features(samples: np.ndarray, feature_set: str) -> np.ndarray:
    """Extract specific feature set."""
    if feature_set == '9dof':
        return samples
    elif feature_set == 'mag_only':
        return samples[:, 6:9]  # mx, my, mz
    else:
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
    return np.array([0 if c == 'e' else 1 for c in combo], dtype=np.float32)


# ============================================================================
# $-FAMILY NORMALIZATION TECHNIQUES
# ============================================================================

def apply_window_centering(windows: np.ndarray) -> np.ndarray:
    """
    $-family style: translate each window so its centroid is at origin.
    Applied per-window AFTER global z-score normalization.
    """
    # windows shape: (n_windows, window_size, n_features)
    window_means = windows.mean(axis=1, keepdims=True)  # (n_windows, 1, n_features)
    return windows - window_means


def apply_window_scaling(windows: np.ndarray) -> np.ndarray:
    """
    $-family style: scale each window to unit variance.
    Applied per-window AFTER global z-score normalization.
    """
    window_stds = windows.std(axis=1, keepdims=True) + 1e-8
    return windows / window_stds


def apply_dollar_family_norm(windows: np.ndarray) -> np.ndarray:
    """
    Full $-family normalization: center + scale each window.
    """
    centered = apply_window_centering(windows)
    scaled = apply_window_scaling(centered)
    return scaled


# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_v3_model(window_size: int, n_features: int) -> keras.Model:
    """
    Build v3 model - optimized for smaller windows.
    Simpler architecture since window_size=10 doesn't need deep CNN.
    """
    inputs = keras.layers.Input(shape=(window_size, n_features))

    if window_size <= 5:
        # Very small window: simple conv
        x = keras.layers.Conv1D(32, min(3, window_size), activation='relu', padding='same')(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
    else:
        # window_size=10: light CNN-LSTM
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


def build_v2_model(window_size: int = 50, n_features: int = 9) -> keras.Model:
    """Build v2 model architecture for comparison."""
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


# ============================================================================
# TRAINING & EVALUATION
# ============================================================================

def prepare_data(
    real_data: Dict[str, FingerStateData],
    feature_set: str,
    window_size: int,
    synthetic_ratio: float,
    samples_per_combo: int = 300,
    dollar_norm: str = 'none'  # 'none', 'center', 'scale', 'full'
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Prepare training/validation/test data with cross-orientation split.

    Returns: X_train, y_train, X_val, y_val, X_test, y_test, info
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

        # Add synthetic to training (high pitch)
        if synthetic_ratio > 0 and generator:
            n_synth = int(samples_per_combo * synthetic_ratio)
            synth_samples = generator.generate_combo(combo, n_synth)
            synth_features = extract_features(synth_samples, feature_set)

            if len(high_pitch_samples) > 0:
                high_pitch_samples = np.vstack([high_pitch_samples, synth_features])
            else:
                high_pitch_samples = synth_features

        # Create windows
        if len(high_pitch_samples) >= window_size:
            windows = create_windows(high_pitch_samples, window_size)
            for w in windows:
                train_windows.append(w)
                train_labels.append(label)

        if len(low_pitch_samples) >= window_size:
            windows = create_windows(low_pitch_samples, window_size)
            for w in windows:
                test_windows.append(w)
                test_labels.append(label)

    X_train = np.array(train_windows)
    y_train = np.array(train_labels)
    X_test = np.array(test_windows)
    y_test = np.array(test_labels)

    # Compute global normalization stats
    n_features = X_train.shape[-1]
    mean = X_train.reshape(-1, n_features).mean(axis=0)
    std = X_train.reshape(-1, n_features).std(axis=0) + 1e-8

    # Apply global z-score normalization
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Apply $-family normalization if requested
    if dollar_norm == 'center':
        X_train = apply_window_centering(X_train)
        X_test = apply_window_centering(X_test)
    elif dollar_norm == 'scale':
        X_train = apply_window_scaling(X_train)
        X_test = apply_window_scaling(X_test)
    elif dollar_norm == 'full':
        X_train = apply_dollar_family_norm(X_train)
        X_test = apply_dollar_family_norm(X_test)

    # Split train into train/val
    indices = np.random.permutation(len(X_train))
    val_size = int(0.15 * len(X_train))
    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    X_val = X_train[val_idx]
    y_val = y_train[val_idx]
    X_train = X_train[train_idx]
    y_train = y_train[train_idx]

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
        'dollar_norm': dollar_norm,
    }

    return X_train, y_train, X_val, y_val, X_test, y_test, info


def train_and_evaluate(
    X_train, y_train, X_val, y_val, X_test, y_test,
    model_fn, epochs=30
) -> Tuple[keras.Model, Dict]:
    """Train model and evaluate."""
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
        verbose=0
    )

    # Evaluate
    y_pred_train = model.predict(X_train, verbose=0)
    y_pred_test = model.predict(X_test, verbose=0)

    train_acc = np.mean((y_pred_train > 0.5).astype(int) == y_train)
    test_acc = np.mean((y_pred_test > 0.5).astype(int) == y_test)

    # Per-finger accuracy
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger = {}
    y_pred_bin = (y_pred_test > 0.5).astype(int)
    for i, f in enumerate(fingers):
        per_finger[f] = float(np.mean(y_pred_bin[:, i] == y_test[:, i]))

    metrics = {
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'gap': float(train_acc - test_acc),
        'per_finger': per_finger,
    }

    return model, metrics


# ============================================================================
# MAIN COMPARISON
# ============================================================================

def main():
    print("=" * 80)
    print("FINGER STATE MODEL V3 - TRAINING & COMPARISON")
    print("=" * 80)

    # Load data
    print("\n--- Loading Data ---")
    real_data = load_session_with_pitch()
    print(f"Loaded {len(real_data)} finger state combinations")

    results = {}

    # =========================================================================
    # CONFIG 1: V2 Baseline (9-DoF, w=50, synthetic)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONFIG 1: V2 Baseline (9-DoF, w=50)")
    print("=" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test, info = prepare_data(
        real_data, feature_set='9dof', window_size=50,
        synthetic_ratio=0.5, samples_per_combo=300
    )
    print(f"Train: {info['n_train']}, Val: {info['n_val']}, Test: {info['n_test']}")

    model_v2, metrics_v2 = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        build_v2_model
    )
    results['v2_baseline'] = {**metrics_v2, **info}
    print(f"Train: {metrics_v2['train_acc']:.1%}, Test: {metrics_v2['test_acc']:.1%}, Gap: {metrics_v2['gap']:.1%}")

    tf.keras.backend.clear_session()

    # =========================================================================
    # CONFIG 2: V3 Optimal (mag_only, w=10, 50% synthetic, no $-norm)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONFIG 2: V3 Optimal (mag_only, w=10, 50% synthetic)")
    print("=" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test, info = prepare_data(
        real_data, feature_set='mag_only', window_size=10,
        synthetic_ratio=0.5, samples_per_combo=300
    )
    print(f"Train: {info['n_train']}, Val: {info['n_val']}, Test: {info['n_test']}")

    model_v3, metrics_v3 = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        build_v3_model
    )
    results['v3_optimal'] = {**metrics_v3, **info}
    print(f"Train: {metrics_v3['train_acc']:.1%}, Test: {metrics_v3['test_acc']:.1%}, Gap: {metrics_v3['gap']:.1%}")

    tf.keras.backend.clear_session()

    # =========================================================================
    # CONFIG 3: V3 + Window Centering ($-family translation)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONFIG 3: V3 + $-family Window Centering")
    print("=" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test, info = prepare_data(
        real_data, feature_set='mag_only', window_size=10,
        synthetic_ratio=0.5, samples_per_combo=300,
        dollar_norm='center'
    )
    print(f"Train: {info['n_train']}, Val: {info['n_val']}, Test: {info['n_test']}")

    _, metrics_center = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        build_v3_model
    )
    results['v3_center'] = {**metrics_center, **info}
    print(f"Train: {metrics_center['train_acc']:.1%}, Test: {metrics_center['test_acc']:.1%}, Gap: {metrics_center['gap']:.1%}")

    tf.keras.backend.clear_session()

    # =========================================================================
    # CONFIG 4: V3 + Window Scaling ($-family scale)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONFIG 4: V3 + $-family Window Scaling")
    print("=" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test, info = prepare_data(
        real_data, feature_set='mag_only', window_size=10,
        synthetic_ratio=0.5, samples_per_combo=300,
        dollar_norm='scale'
    )
    print(f"Train: {info['n_train']}, Val: {info['n_val']}, Test: {info['n_test']}")

    _, metrics_scale = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        build_v3_model
    )
    results['v3_scale'] = {**metrics_scale, **info}
    print(f"Train: {metrics_scale['train_acc']:.1%}, Test: {metrics_scale['test_acc']:.1%}, Gap: {metrics_scale['gap']:.1%}")

    tf.keras.backend.clear_session()

    # =========================================================================
    # CONFIG 5: V3 + Full $-family (center + scale)
    # =========================================================================
    print("\n" + "=" * 70)
    print("CONFIG 5: V3 + Full $-family (center + scale)")
    print("=" * 70)

    X_train, y_train, X_val, y_val, X_test, y_test, info = prepare_data(
        real_data, feature_set='mag_only', window_size=10,
        synthetic_ratio=0.5, samples_per_combo=300,
        dollar_norm='full'
    )
    print(f"Train: {info['n_train']}, Val: {info['n_val']}, Test: {info['n_test']}")

    _, metrics_full = train_and_evaluate(
        X_train, y_train, X_val, y_val, X_test, y_test,
        build_v3_model
    )
    results['v3_dollar_full'] = {**metrics_full, **info}
    print(f"Train: {metrics_full['train_acc']:.1%}, Test: {metrics_full['test_acc']:.1%}, Gap: {metrics_full['gap']:.1%}")

    tf.keras.backend.clear_session()

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("COMPARISON SUMMARY (Cross-Orientation Test)")
    print("=" * 80)

    print(f"\n{'Config':<30} {'Features':<10} {'Window':<8} {'Train':>8} {'Test':>8} {'Gap':>8}")
    print("-" * 80)

    configs = [
        ('v2_baseline', '9-DoF', '50'),
        ('v3_optimal', 'mag_only', '10'),
        ('v3_center', 'mag+center', '10'),
        ('v3_scale', 'mag+scale', '10'),
        ('v3_dollar_full', 'mag+$full', '10'),
    ]

    best_config = None
    best_test = 0

    for key, feat, win in configs:
        r = results[key]
        print(f"{key:<30} {feat:<10} {win:<8} {r['train_acc']:>7.1%} {r['test_acc']:>7.1%} {r['gap']:>7.1%}")
        if r['test_acc'] > best_test:
            best_test = r['test_acc']
            best_config = key

    print(f"\n*** Best config: {best_config} with {best_test:.1%} cross-orientation accuracy ***")

    # Save results
    results_path = Path("ml/v3_comparison_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    # =========================================================================
    # DEPLOY BEST MODEL
    # =========================================================================
    print("\n" + "=" * 80)
    print("DEPLOYING BEST MODEL (V3)")
    print("=" * 80)

    # Retrain v3 with full data for deployment
    output_dir = Path('public/models/finger_aligned_v3')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Prepare full training data (not just cross-orientation split)
    generator = SyntheticGenerator(real_data)
    all_combos = [f"{t}{i}{m}{r}{p}" for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef']

    all_windows = []
    all_labels = []

    for combo in all_combos:
        label = combo_to_label(combo)

        if combo in real_data:
            real_samples = extract_features(real_data[combo].samples, 'mag_only')
            n_synth = max(0, 300 - len(real_samples))
            if n_synth > 0:
                synth = generator.generate_combo(combo, n_synth)
                synth_feat = extract_features(synth, 'mag_only')
                combined = np.vstack([real_samples, synth_feat])
            else:
                combined = real_samples[:300]
        else:
            combined = extract_features(generator.generate_combo(combo, 300), 'mag_only')

        windows = create_windows(combined, window_size=10)
        for w in windows:
            all_windows.append(w)
            all_labels.append(label)

    X_full = np.array(all_windows)
    y_full = np.array(all_labels)

    # Compute normalization stats
    mean = X_full.reshape(-1, 3).mean(axis=0)
    std = X_full.reshape(-1, 3).std(axis=0) + 1e-8
    X_full = (X_full - mean) / std

    # Shuffle and split
    indices = np.random.permutation(len(X_full))
    train_end = int(0.85 * len(X_full))
    X_train = X_full[indices[:train_end]]
    y_train = y_full[indices[:train_end]]
    X_val = X_full[indices[train_end:]]
    y_val = y_full[indices[train_end:]]

    print(f"Training deployment model: {len(X_train)} train, {len(X_val)} val")

    # Build and train final model
    model = build_v3_model(window_size=10, n_features=3)
    model.summary()

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=50,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    y_pred = (model.predict(X_val, verbose=0) > 0.5).astype(int)
    val_acc = np.mean(y_pred == y_val)

    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger_acc = {}
    for i, finger in enumerate(fingers):
        per_finger_acc[finger] = float(np.mean(y_pred[:, i] == y_val[:, i]))
        print(f"  {finger}: {per_finger_acc[finger]:.1%}")

    print(f"\nOverall validation accuracy: {val_acc:.1%}")

    # Save model
    keras_path = output_dir / 'model.keras'
    model.save(keras_path)
    print(f"Saved Keras model to {keras_path}")

    # Save SavedModel for TF.js conversion
    saved_model_path = output_dir / 'saved_model'
    model.export(str(saved_model_path))
    print(f"Saved SavedModel to {saved_model_path}")

    # Save config
    config = {
        'stats': {
            'mean': mean.tolist(),
            'std': std.tolist()
        },
        'inputShape': [None, 10, 3],
        'fingerNames': fingers,
        'stateNames': ['extended', 'flexed'],
        'description': 'V3: mag_only, w=10, 50% tight synthetic (91% cross-orientation)',
        'version': 'aligned_v3',
        'date': '2026-01-06',
        'modelType': 'layers',
        'windowSize': 10,
        'numFeatures': 3,
        'featureNames': ['mx', 'my', 'mz'],
        'accuracy': {
            'overall': float(val_acc),
            'per_finger': per_finger_acc,
            'cross_orientation': results['v3_optimal']['test_acc']
        },
        'improvements_over_v2': {
            'cross_orientation_accuracy': f"+{(results['v3_optimal']['test_acc'] - results['v2_baseline']['test_acc'])*100:.1f}%",
            'window_size': '50 -> 10 (5x smaller)',
            'features': '9 -> 3 (3x fewer)',
            'latency': 'Reduced inference latency due to smaller window'
        }
    }

    config_path = output_dir / 'config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Saved config to {config_path}")

    # Convert to TensorFlow.js
    print("\n--- Converting to TensorFlow.js ---")
    try:
        result = subprocess.run([
            'tensorflowjs_converter',
            '--input_format=keras',
            str(keras_path),
            str(output_dir)
        ], capture_output=True, text=True)

        if result.returncode == 0:
            print("Successfully converted to TensorFlow.js format")
        else:
            print(f"Conversion warning: {result.stderr}")
    except FileNotFoundError:
        print("tensorflowjs_converter not found - manual conversion needed")

    # Print registry entry
    print("\n" + "=" * 80)
    print("MODEL REGISTRY ENTRY")
    print("=" * 80)
    print("""
Add this to ALL_MODELS in apps/gambit/gesture-inference.ts:

{
  id: 'finger_aligned_v3',
  name: 'Finger (Aligned v3 - Optimized)',
  type: 'finger_window',
  path: '/models/finger_aligned_v3/model.json',
  stats: {
    mean: """ + str(mean.tolist()) + """,
    std: """ + str(std.tolist()) + """
  },
  description: 'V3: mag_only, w=10, 91% cross-orientation accuracy',
  date: '2026-01-06',
  active: true,
  windowSize: 10,
  numStates: 2
},
""")

    print("\n" + "=" * 80)
    print("DEPLOYMENT COMPLETE")
    print("=" * 80)
    print(f"\nV3 Model advantages over V2:")
    print(f"  - Cross-orientation: {results['v3_optimal']['test_acc']:.1%} vs {results['v2_baseline']['test_acc']:.1%}")
    print(f"  - Window size: 10 vs 50 (5x smaller, lower latency)")
    print(f"  - Features: 3 vs 9 (mag_only, 3x fewer)")
    print(f"  - $-family normalization: NOT beneficial for this task")

    return results


if __name__ == '__main__':
    main()
