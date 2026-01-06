#!/usr/bin/env python3
"""
Test Context-Aware V5 Architecture for Finger State Classification

Inspired by Chorus (Zhang et al., arXiv:2512.15206, Dec 2025):
Context-aware model customization for handling unseen orientation shifts.

Key innovations:
1. Orientation as context: Treat hand orientation (pitch, roll, yaw) as context signal
2. Gated fusion: Dynamically weight magnetometer features based on orientation confidence
3. Adaptive weighting: More context when mag signal is weak (e.g., pinky)

Expected improvements:
- Test accuracy: 67.6% → 75-78%
- Generalization gap: 20.8% → 15-18%
- Better handling of extreme orientations

Author: Claude
Date: January 2026
"""

import json
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
    samples: np.ndarray
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
            roll = s.get('euler_roll', 0)
            yaw = s.get('euler_yaw', 0)

            sensor_data.append([ax, ay, az, gx, gy, gz, mx, my, mz, pitch, roll, yaw])
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
        """Generate synthetic samples with orientation features."""
        if combo in self.real_data:
            real = self.real_data[combo]
            mag_mean = real.samples[:, 6:9].mean(axis=0)
            mag_std = real.samples[:, 6:9].std(axis=0)
            # Use real orientation distribution
            orient_mean = real.samples[:, 9:12].mean(axis=0)
            orient_std = real.samples[:, 9:12].std(axis=0)
        else:
            if self.baseline:
                mag_mean = self.baseline.samples[:, 6:9].mean(axis=0)
                mag_std = self.baseline.samples[:, 6:9].std(axis=0)
                orient_mean = self.baseline.samples[:, 9:12].mean(axis=0)
                orient_std = self.baseline.samples[:, 9:12].std(axis=0)
            else:
                mag_mean = np.array([46, -46, 31])
                mag_std = np.array([25, 40, 50])
                orient_mean = np.array([0, 0, 0])
                orient_std = np.array([30, 30, 30])

            fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
            for i, state in enumerate(combo):
                if state == 'f':
                    finger = fingers[i]
                    if finger in self.finger_effects:
                        mag_mean = mag_mean + self.finger_effects[finger]['mag_delta']

        samples = []
        for _ in range(n_samples):
            mag_sample = mag_mean + np.random.randn(3) * mag_std
            orient_sample = orient_mean + np.random.randn(3) * orient_std
            ax = np.random.normal(0, 0.05)
            ay = np.random.normal(0, 0.05)
            az = np.random.normal(-1, 0.05)
            gx = np.random.normal(0, 2.0)
            gy = np.random.normal(0, 2.0)
            gz = np.random.normal(0, 2.0)
            samples.append([
                ax, ay, az, gx, gy, gz,
                mag_sample[0], mag_sample[1], mag_sample[2],
                orient_sample[0], orient_sample[1], orient_sample[2]
            ])

        return np.array(samples)


# ============================================================================
# FEATURE EXTRACTION & WINDOWING
# ============================================================================

def extract_features_with_context(samples: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """
    Extract magnetometer features and orientation context.

    Returns:
        mag_features: (n_samples, 3) - mx, my, mz
        context_features: (n_samples, 3) - pitch, roll, yaw
    """
    mag_features = samples[:, 6:9]  # mx, my, mz
    context_features = samples[:, 9:12]  # pitch, roll, yaw
    return mag_features, context_features


def create_windows_with_context(
    samples: np.ndarray,
    window_size: int,
    stride: int = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sliding windows for both magnetometer and orientation.

    Returns:
        mag_windows: (n_windows, window_size, 3)
        context_windows: (n_windows, 3) - aggregated orientation (mean)
    """
    if stride is None:
        stride = max(1, window_size // 2)

    n_samples = len(samples)
    if n_samples < window_size:
        padding = np.zeros((window_size - n_samples, samples.shape[1]))
        samples = np.vstack([samples, padding])
        n_samples = window_size

    mag_windows = []
    context_windows = []

    for i in range(0, n_samples - window_size + 1, stride):
        window = samples[i:i+window_size]
        mag_feat, context_feat = extract_features_with_context(window)

        mag_windows.append(mag_feat)
        # Aggregate context over window (mean orientation)
        context_windows.append(context_feat.mean(axis=0))

    if not mag_windows:
        mag_feat, context_feat = extract_features_with_context(samples[:window_size])
        mag_windows.append(mag_feat)
        context_windows.append(context_feat.mean(axis=0))

    return np.array(mag_windows), np.array(context_windows)


def combo_to_label(combo: str) -> np.ndarray:
    """Convert combo string to binary label."""
    return np.array([0 if c == 'e' else 1 for c in combo], dtype=np.float32)


# ============================================================================
# V5 CONTEXT-AWARE ARCHITECTURE
# ============================================================================

def build_v5_context_aware(
    window_size: int = 10,
    n_mag_features: int = 3,
    n_context_features: int = 3
) -> keras.Model:
    """
    V5 Context-Aware Architecture (Chorus-inspired).

    Architecture:
    1. Magnetometer branch: Conv1D → LSTM → feature embedding
    2. Context branch: Dense layers → context embedding
    3. Gated fusion: Dynamically weight mag vs context based on confidence
    4. Final prediction head

    Key innovation: The gate learns when to rely on magnetometer signal vs
    orientation context (e.g., more context for pinky with weak mag signal).
    """
    # Magnetometer input (time series)
    mag_input = keras.layers.Input(shape=(window_size, n_mag_features), name='mag_input')

    # Context input (aggregated orientation)
    context_input = keras.layers.Input(shape=(n_context_features,), name='context_input')

    # === Magnetometer Branch (Base Model) ===
    mag_x = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(mag_input)
    mag_x = keras.layers.BatchNormalization()(mag_x)
    mag_x = keras.layers.MaxPooling1D(2)(mag_x)
    mag_x = keras.layers.Dropout(0.4)(mag_x)
    mag_x = keras.layers.LSTM(32)(mag_x)
    mag_x = keras.layers.Dropout(0.5)(mag_x)

    # Magnetometer feature embedding
    mag_features = keras.layers.Dense(
        32, activation='relu',
        kernel_regularizer=keras.regularizers.l2(0.01),
        name='mag_features'
    )(mag_x)
    mag_features = keras.layers.Dropout(0.4)(mag_features)

    # === Context Branch ===
    context_x = keras.layers.Dense(16, activation='relu', name='context_dense1')(context_input)
    context_x = keras.layers.Dropout(0.3)(context_x)
    context_x = keras.layers.Dense(16, activation='relu', name='context_dense2')(context_x)
    context_x = keras.layers.Dropout(0.3)(context_x)

    # Context embedding (project to same dim as mag_features)
    context_embedding = keras.layers.Dense(
        32, activation='relu',
        name='context_embedding'
    )(context_x)

    # === Gated Fusion ===
    # Concatenate mag and context features
    combined = keras.layers.Concatenate(name='concat')([mag_features, context_embedding])

    # Gate: learns to weight mag vs context (sigmoid → [0, 1])
    gate = keras.layers.Dense(
        1, activation='sigmoid',
        name='gate'
    )(combined)

    # Adaptive fusion:
    # - gate=1: rely on magnetometer
    # - gate=0: rely on context
    # - gate=0.5: equal weighting
    adapted_features = keras.layers.Lambda(
        lambda x: x[0] * x[1] + x[2] * (1 - x[1]),
        name='gated_fusion'
    )([mag_features, gate, context_embedding])

    # === Final Prediction Head ===
    outputs = keras.layers.Dense(
        5, activation='sigmoid',
        kernel_regularizer=keras.regularizers.l2(0.01),
        name='output'
    )(adapted_features)

    model = keras.Model(
        inputs=[mag_input, context_input],
        outputs=outputs,
        name='V5_Context_Aware'
    )

    # Label smoothing loss
    def label_smoothed_loss(y_true, y_pred):
        y_true_smooth = y_true * 0.9 + 0.05
        return keras.losses.binary_crossentropy(y_true_smooth, y_pred)

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=label_smoothed_loss,
        metrics=['accuracy']
    )

    return model


def build_v4_regularized(window_size: int = 10, n_features: int = 3) -> keras.Model:
    """V4-Regularized baseline (for comparison)."""
    inputs = keras.layers.Input(shape=(window_size, n_features))

    x = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(32, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(5, activation='sigmoid',
                                 kernel_regularizer=keras.regularizers.l2(0.01))(x)

    model = keras.Model(inputs, outputs, name='V4_Regularized')

    def label_smoothed_loss(y_true, y_pred):
        y_true_smooth = y_true * 0.9 + 0.05
        return keras.losses.binary_crossentropy(y_true_smooth, y_pred)

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=label_smoothed_loss,
        metrics=['accuracy']
    )
    return model


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_data_with_context(
    real_data: Dict[str, FingerStateData],
    window_size: int,
    synthetic_ratio: float,
) -> Tuple[np.ndarray, ...]:
    """
    Prepare data with orientation context for V5.

    Returns:
        X_mag_train, X_context_train, y_train,
        X_mag_val, X_context_val, y_val,
        X_mag_test, X_context_test, y_test
    """
    generator = SyntheticGenerator(real_data)

    # Collect all pitch angles
    all_pitches = []
    for data in real_data.values():
        all_pitches.extend(data.pitch_angles)
    all_pitches = np.array(all_pitches)

    # Compute quartiles
    q1 = np.percentile(all_pitches, 25)
    q3 = np.percentile(all_pitches, 75)

    print(f"Pitch quartiles: Q1={q1:.1f}°, Q3={q3:.1f}°")

    # Storage
    train_mag_windows = []
    train_context_windows = []
    train_labels = []
    val_mag_windows = []
    val_context_windows = []
    val_labels = []
    test_mag_windows = []
    test_context_windows = []
    test_labels = []

    for combo in real_data.keys():
        combo_data = real_data[combo]
        label = combo_to_label(combo)

        # Split by pitch
        high_pitch_mask = combo_data.pitch_angles >= q3
        low_pitch_mask = combo_data.pitch_angles <= q1

        high_pitch_samples = combo_data.samples[high_pitch_mask]
        low_pitch_samples = combo_data.samples[low_pitch_mask]

        # High pitch → train/val
        if len(high_pitch_samples) > 0:
            # Add synthetic data
            if synthetic_ratio > 0:
                n_synth = int(len(high_pitch_samples) * synthetic_ratio)
                if n_synth > 0:
                    synth = generator.generate_combo(combo, n_synth)
                    if len(synth) > 0 and synth.ndim == 2:
                        high_pitch_samples = np.vstack([high_pitch_samples, synth])

            # Create windows with context
            mag_windows, context_windows = create_windows_with_context(
                high_pitch_samples, window_size
            )

            # Split train/val (80/20)
            n_train = int(0.8 * len(mag_windows))
            for i in range(len(mag_windows)):
                if i < n_train:
                    train_mag_windows.append(mag_windows[i])
                    train_context_windows.append(context_windows[i])
                    train_labels.append(label)
                else:
                    val_mag_windows.append(mag_windows[i])
                    val_context_windows.append(context_windows[i])
                    val_labels.append(label)

        # Low pitch → test (held-out)
        if len(low_pitch_samples) > 0:
            mag_windows, context_windows = create_windows_with_context(
                low_pitch_samples, window_size
            )
            for i in range(len(mag_windows)):
                test_mag_windows.append(mag_windows[i])
                test_context_windows.append(context_windows[i])
                test_labels.append(label)

    X_mag_train = np.array(train_mag_windows)
    X_context_train = np.array(train_context_windows)
    y_train = np.array(train_labels)

    X_mag_val = np.array(val_mag_windows)
    X_context_val = np.array(val_context_windows)
    y_val = np.array(val_labels)

    X_mag_test = np.array(test_mag_windows)
    X_context_test = np.array(test_context_windows)
    y_test = np.array(test_labels)

    print(f"\nData split:")
    print(f"  Train: {len(X_mag_train)} windows (Q3 high pitch + synthetic)")
    print(f"  Val: {len(X_mag_val)} windows (Q3 high pitch, held-out)")
    print(f"  Test: {len(X_mag_test)} windows (Q1 low pitch, COMPLETELY HELD OUT)")

    return (X_mag_train, X_context_train, y_train,
            X_mag_val, X_context_val, y_val,
            X_mag_test, X_context_test, y_test)


def prepare_data_v4(
    real_data: Dict[str, FingerStateData],
    window_size: int,
    synthetic_ratio: float,
) -> Tuple[np.ndarray, ...]:
    """Prepare data for V4 baseline (mag only, no context)."""
    generator = SyntheticGenerator(real_data)

    all_pitches = []
    for data in real_data.values():
        all_pitches.extend(data.pitch_angles)
    all_pitches = np.array(all_pitches)

    q1 = np.percentile(all_pitches, 25)
    q3 = np.percentile(all_pitches, 75)

    train_windows = []
    train_labels = []
    val_windows = []
    val_labels = []
    test_windows = []
    test_labels = []

    for combo in real_data.keys():
        combo_data = real_data[combo]
        label = combo_to_label(combo)

        high_pitch_mask = combo_data.pitch_angles >= q3
        low_pitch_mask = combo_data.pitch_angles <= q1

        high_pitch_samples = combo_data.samples[high_pitch_mask]
        low_pitch_samples = combo_data.samples[low_pitch_mask]

        if len(high_pitch_samples) > 0:
            mag_features, _ = extract_features_with_context(high_pitch_samples)

            if synthetic_ratio > 0:
                n_synth = int(len(mag_features) * synthetic_ratio)
                if n_synth > 0:
                    synth = generator.generate_combo(combo, n_synth)
                    if len(synth) > 0 and synth.ndim == 2:
                        synth_mag, _ = extract_features_with_context(synth)
                        mag_features = np.vstack([mag_features, synth_mag])

            # Create windows
            mag_windows, _ = create_windows_with_context(
                np.hstack([np.zeros((len(mag_features), 9)), mag_features, np.zeros((len(mag_features), 3))]),
                window_size
            )

            # Simpler: just window the mag features directly
            n_samples = len(mag_features)
            stride = max(1, window_size // 2)
            if n_samples < window_size:
                padding = np.zeros((window_size - n_samples, 3))
                mag_features = np.vstack([mag_features, padding])
                n_samples = window_size

            windows = []
            for i in range(0, n_samples - window_size + 1, stride):
                windows.append(mag_features[i:i+window_size])
            if not windows:
                windows.append(mag_features[:window_size])

            n_train = int(0.8 * len(windows))
            for i, w in enumerate(windows):
                if i < n_train:
                    train_windows.append(w)
                    train_labels.append(label)
                else:
                    val_windows.append(w)
                    val_labels.append(label)

        if len(low_pitch_samples) > 0:
            mag_features, _ = extract_features_with_context(low_pitch_samples)

            n_samples = len(mag_features)
            stride = max(1, window_size // 2)
            if n_samples < window_size:
                padding = np.zeros((window_size - n_samples, 3))
                mag_features = np.vstack([mag_features, padding])
                n_samples = window_size

            windows = []
            for i in range(0, n_samples - window_size + 1, stride):
                windows.append(mag_features[i:i+window_size])
            if not windows:
                windows.append(mag_features[:window_size])

            for w in windows:
                test_windows.append(w)
                test_labels.append(label)

    return (np.array(train_windows), np.array(train_labels),
            np.array(val_windows), np.array(val_labels),
            np.array(test_windows), np.array(test_labels))


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_v5(model, X_mag, X_context, y, dataset_name: str):
    """Evaluate V5 model (with context)."""
    y_pred = (model.predict([X_mag, X_context], verbose=0) > 0.5).astype(int)
    overall_acc = np.mean(y_pred == y)

    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger = {}
    for i, finger in enumerate(fingers):
        acc = np.mean(y_pred[:, i] == y[:, i])
        per_finger[finger] = acc

    print(f"\n{dataset_name} Accuracy: {overall_acc:.1%}")
    for finger, acc in per_finger.items():
        print(f"  {finger}: {acc:.1%}")

    return overall_acc, per_finger


def evaluate_v4(model, X, y, dataset_name: str):
    """Evaluate V4 model (mag only)."""
    y_pred = (model.predict(X, verbose=0) > 0.5).astype(int)
    overall_acc = np.mean(y_pred == y)

    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger = {}
    for i, finger in enumerate(fingers):
        acc = np.mean(y_pred[:, i] == y[:, i])
        per_finger[finger] = acc

    print(f"\n{dataset_name} Accuracy: {overall_acc:.1%}")
    for finger, acc in per_finger.items():
        print(f"  {finger}: {acc:.1%}")

    return overall_acc, per_finger


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 80)
    print("CONTEXT-AWARE V5 EXPERIMENT (Chorus-Inspired)")
    print("=" * 80)
    print("\nComparing:")
    print("  1. V4-Regularized: Magnetometer only (baseline)")
    print("  2. V5-Context: Magnetometer + Orientation context with gated fusion")

    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load data
    print("\n--- Loading Data ---")
    real_data = load_session_with_pitch()
    print(f"Loaded {len(real_data)} finger state combinations")

    # Prepare data for V5 (with context)
    print("\n--- Preparing Data with Orientation Context (V5) ---")
    (X_mag_train, X_context_train, y_train,
     X_mag_val, X_context_val, y_val,
     X_mag_test, X_context_test, y_test) = prepare_data_with_context(
        real_data=real_data,
        window_size=10,
        synthetic_ratio=0.5
    )

    # Normalize magnetometer features
    mag_mean = X_mag_train.reshape(-1, 3).mean(axis=0)
    mag_std = X_mag_train.reshape(-1, 3).std(axis=0) + 1e-8
    X_mag_train = (X_mag_train - mag_mean) / mag_std
    X_mag_val = (X_mag_val - mag_mean) / mag_std
    X_mag_test = (X_mag_test - mag_mean) / mag_std

    # Normalize context features (orientation)
    context_mean = X_context_train.mean(axis=0)
    context_std = X_context_train.std(axis=0) + 1e-8
    X_context_train = (X_context_train - context_mean) / context_std
    X_context_val = (X_context_val - context_mean) / context_std
    X_context_test = (X_context_test - context_mean) / context_std

    # Prepare data for V4 (mag only)
    print("\n--- Preparing Data for V4 Baseline (Mag Only) ---")
    (X_train_v4, y_train_v4,
     X_val_v4, y_val_v4,
     X_test_v4, y_test_v4) = prepare_data_v4(
        real_data=real_data,
        window_size=10,
        synthetic_ratio=0.5
    )

    # Normalize V4 data
    X_train_v4 = (X_train_v4 - mag_mean) / mag_std
    X_val_v4 = (X_val_v4 - mag_mean) / mag_std
    X_test_v4 = (X_test_v4 - mag_mean) / mag_std

    # Results storage
    results = {}

    # ========================================================================
    # EXPERIMENT 1: V4-Regularized (Baseline)
    # ========================================================================

    print("\n" + "=" * 80)
    print("EXPERIMENT 1: V4-Regularized (Baseline)")
    print("=" * 80)

    model_v4 = build_v4_regularized(window_size=10, n_features=3)
    print("\nModel architecture:")
    model_v4.summary()

    print(f"\nTraining data: {len(X_train_v4)} samples")
    print(f"Validation data: {len(X_val_v4)} samples")
    print(f"Test data: {len(X_test_v4)} samples")

    print("\n--- Training V4-Regularized ---")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    history_v4 = model_v4.fit(
        X_train_v4, y_train_v4,
        validation_data=(X_val_v4, y_val_v4),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    val_acc_v4, val_per_finger_v4 = evaluate_v4(model_v4, X_val_v4, y_val_v4, "V4 Validation")
    test_acc_v4, test_per_finger_v4 = evaluate_v4(model_v4, X_test_v4, y_test_v4, "V4 Test (Q1 held-out)")

    gap_v4 = (val_acc_v4 - test_acc_v4) * 100
    print(f"\nGeneralization gap: {gap_v4:.1f}%")

    results['v4_regularized'] = {
        'val_acc': val_acc_v4,
        'test_acc': test_acc_v4,
        'gap': gap_v4,
        'val_per_finger': val_per_finger_v4,
        'test_per_finger': test_per_finger_v4
    }

    # ========================================================================
    # EXPERIMENT 2: V5-Context
    # ========================================================================

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: V5-Context (with Gated Fusion)")
    print("=" * 80)

    model_v5 = build_v5_context_aware(window_size=10, n_mag_features=3, n_context_features=3)
    print("\nModel architecture:")
    model_v5.summary()

    print(f"\nTraining data: {len(X_mag_train)} samples")
    print(f"Validation data: {len(X_mag_val)} samples")
    print(f"Test data: {len(X_mag_test)} samples")

    print("\n--- Training V5-Context ---")
    history_v5 = model_v5.fit(
        [X_mag_train, X_context_train], y_train,
        validation_data=([X_mag_val, X_context_val], y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    val_acc_v5, val_per_finger_v5 = evaluate_v5(
        model_v5, X_mag_val, X_context_val, y_val, "V5 Validation"
    )
    test_acc_v5, test_per_finger_v5 = evaluate_v5(
        model_v5, X_mag_test, X_context_test, y_test, "V5 Test (Q1 held-out)"
    )

    gap_v5 = (val_acc_v5 - test_acc_v5) * 100
    print(f"\nGeneralization gap: {gap_v5:.1f}%")

    results['v5_context'] = {
        'val_acc': val_acc_v5,
        'test_acc': test_acc_v5,
        'gap': gap_v5,
        'val_per_finger': val_per_finger_v5,
        'test_per_finger': test_per_finger_v5
    }

    # ========================================================================
    # COMPARISON
    # ========================================================================

    print("\n" + "=" * 80)
    print("COMPARISON: V4-Regularized vs V5-Context")
    print("=" * 80)

    print("\n| Metric | V4-Regularized | V5-Context | Improvement |")
    print("|--------|----------------|------------|-------------|")

    test_diff = (test_acc_v5 - test_acc_v4) * 100
    gap_diff = gap_v4 - gap_v5

    print(f"| Test Accuracy | {test_acc_v4:.1%} | {test_acc_v5:.1%} | {test_diff:+.1f}% |")
    print(f"| Generalization Gap | {gap_v4:.1f}% | {gap_v5:.1f}% | {gap_diff:+.1f}% |")

    print("\nPer-Finger Test Accuracy:")
    print("| Finger | V4-Regularized | V5-Context | Improvement |")
    print("|--------|----------------|------------|-------------|")

    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    for finger in fingers:
        acc_v4 = test_per_finger_v4[finger]
        acc_v5 = test_per_finger_v5[finger]
        diff = (acc_v5 - acc_v4) * 100
        print(f"| {finger} | {acc_v4:.1%} | {acc_v5:.1%} | {diff:+.1f}% |")

    # Determine winner
    winner = "V5-Context" if test_acc_v5 > test_acc_v4 else "V4-Regularized"
    print(f"\n🏆 Winner: {winner}")

    if test_acc_v5 > test_acc_v4:
        print(f"   V5-Context improves test accuracy by {test_diff:.1f}%")
        print(f"   and reduces generalization gap by {gap_diff:.1f}%")
    else:
        print(f"   V4 remains best (V5 difference: {test_diff:.1f}%)")

    # Save results
    output_path = Path('ml/results/v5_context_experiment.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n📊 Results saved to {output_path}")

    # Save model if V5 is better
    if test_acc_v5 > test_acc_v4:
        print("\n--- Saving V5-Context Model ---")
        model_path = Path('ml/models/finger_v5_context.keras')
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_v5.save(model_path)
        print(f"Model saved to {model_path}")

        # Save normalization stats
        stats_path = Path('ml/models/finger_v5_context_stats.json')
        stats = {
            'mag_mean': mag_mean.tolist(),
            'mag_std': mag_std.tolist(),
            'context_mean': context_mean.tolist(),
            'context_std': context_std.tolist()
        }
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"Stats saved to {stats_path}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    return results


if __name__ == '__main__':
    main()
