#!/usr/bin/env python3
"""
Test DropDim Regularization for Finger State Classification

DropDim drops entire feature dimensions instead of random neurons,
forcing the model to encode information redundantly across dimensions.
This should improve robustness to corrupted/weak magnetometer axes.

Comparison:
- V4-Regularized (baseline): Standard dropout on neurons
- V4-DropDim: Drop entire feature dimensions after Conv1D

Based on: Zhang, H., et al. "DropDim: A Regularization Method for
Transformer Networks." arXiv:2304.10321, Apr 2023.

Author: Claude
Date: January 2026
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass
import time

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
# FEATURE EXTRACTION & WINDOWING
# ============================================================================

def extract_features(samples: np.ndarray, feature_set: str = 'mag_only') -> np.ndarray:
    """Extract magnetometer features."""
    if feature_set == 'mag_only':
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
# DROPDIM LAYER
# ============================================================================

class DropDim(keras.layers.Layer):
    """
    DropDim regularization layer.

    Drops entire feature dimensions instead of random neurons.
    Forces the model to encode information redundantly across dimensions.

    For magnetometer data (mx, my, mz → 32 Conv1D features), this encourages
    the model to use all available information and be robust when one axis
    is corrupted or weak (e.g., pinky with low signal).
    """

    def __init__(self, drop_rate=0.3, **kwargs):
        super().__init__(**kwargs)
        self.drop_rate = drop_rate

    def call(self, inputs, training=None):
        if not training:
            return inputs

        # inputs shape: (batch, time, features) or (batch, features)
        if len(inputs.shape) == 3:
            # Time series: (batch, time, features)
            n_features = int(inputs.shape[-1])
            n_drop = int(n_features * self.drop_rate)

            if n_drop == 0:
                return inputs

            # Create mask: randomly select dimensions to keep
            mask = tf.ones([n_features], dtype=inputs.dtype)
            indices = tf.random.shuffle(tf.range(n_features))[:n_drop]
            mask = tf.tensor_scatter_nd_update(
                mask,
                tf.reshape(indices, [-1, 1]),
                tf.zeros([n_drop], dtype=inputs.dtype)
            )

            # Reshape mask for broadcasting
            mask = tf.reshape(mask, [1, 1, n_features])

            # Scale remaining dimensions to maintain expected value
            scale = 1.0 / (1.0 - self.drop_rate)
            return inputs * mask * scale
        else:
            # Dense: (batch, features)
            n_features = int(inputs.shape[-1])
            n_drop = int(n_features * self.drop_rate)

            if n_drop == 0:
                return inputs

            mask = tf.ones([n_features], dtype=inputs.dtype)
            indices = tf.random.shuffle(tf.range(n_features))[:n_drop]
            mask = tf.tensor_scatter_nd_update(
                mask,
                tf.reshape(indices, [-1, 1]),
                tf.zeros([n_drop], dtype=inputs.dtype)
            )

            mask = tf.reshape(mask, [1, n_features])
            scale = 1.0 / (1.0 - self.drop_rate)
            return inputs * mask * scale

    def get_config(self):
        config = super().get_config()
        config.update({'drop_rate': self.drop_rate})
        return config


# ============================================================================
# MODEL ARCHITECTURES
# ============================================================================

def build_v4_regularized(window_size: int = 10, n_features: int = 3) -> keras.Model:
    """
    V4-Regularized (Baseline): Standard dropout on neurons.
    """
    inputs = keras.layers.Input(shape=(window_size, n_features))

    x = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.4)(x)  # Standard dropout
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dropout(0.5)(x)  # Standard dropout
    x = keras.layers.Dense(32, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.Dropout(0.4)(x)  # Standard dropout
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


def build_v4_dropdim(window_size: int = 10, n_features: int = 3) -> keras.Model:
    """
    V4-DropDim: DropDim on feature dimensions after Conv1D.

    Key difference: After Conv1D produces 32 feature maps, DropDim drops
    entire feature dimensions (e.g., drop 10 of the 32 features completely)
    instead of random neurons within each feature.
    """
    inputs = keras.layers.Input(shape=(window_size, n_features))

    x = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = DropDim(0.3)(x)  # DropDim: drop 30% of feature dimensions
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dropout(0.5)(x)  # Keep standard dropout on LSTM
    x = keras.layers.Dense(32, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = DropDim(0.3)(x)  # DropDim on dense features
    outputs = keras.layers.Dense(5, activation='sigmoid',
                                 kernel_regularizer=keras.regularizers.l2(0.01))(x)

    model = keras.Model(inputs, outputs, name='V4_DropDim')

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

def prepare_data_with_heldout(
    real_data: Dict[str, FingerStateData],
    feature_set: str,
    window_size: int,
    synthetic_ratio: float,
    samples_per_combo: int = 200,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare data with cross-orientation split:
    - Train: Q3 pitch angles (high pitch) + synthetic
    - Val: Q3 pitch angles (high pitch), held-out portion
    - Test: Q1 pitch angles (low pitch) - COMPLETELY HELD OUT
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

    # Process each combo
    train_windows = []
    train_labels = []
    val_windows = []
    val_labels = []
    test_windows = []
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
            features = extract_features(high_pitch_samples, feature_set)

            # Add synthetic data (50% ratio)
            if synthetic_ratio > 0 and len(features) > 0:
                n_synth = int(len(features) * synthetic_ratio)
                if n_synth > 0:
                    synth = generator.generate_combo(combo, n_synth)
                    if len(synth) > 0 and synth.ndim == 2:
                        synth_feat = extract_features(synth, feature_set)
                        if len(synth_feat) > 0 and synth_feat.shape[1] == features.shape[1]:
                            features = np.vstack([features, synth_feat])

            # Create windows
            windows = create_windows(features, window_size)

            # Split train/val (80/20)
            n_train = int(0.8 * len(windows))
            for i, w in enumerate(windows):
                if i < n_train:
                    train_windows.append(w)
                    train_labels.append(label)
                else:
                    val_windows.append(w)
                    val_labels.append(label)

        # Low pitch → test (held-out)
        if len(low_pitch_samples) > 0:
            features = extract_features(low_pitch_samples, feature_set)
            windows = create_windows(features, window_size)
            for w in windows:
                test_windows.append(w)
                test_labels.append(label)

    X_train = np.array(train_windows)
    y_train = np.array(train_labels)
    X_val = np.array(val_windows)
    y_val = np.array(val_labels)
    X_test = np.array(test_windows)
    y_test = np.array(test_labels)

    print(f"\nData split:")
    print(f"  Train: {len(X_train)} windows (Q3 high pitch + synthetic)")
    print(f"  Val: {len(X_val)} windows (Q3 high pitch, held-out)")
    print(f"  Test: {len(X_test)} windows (Q1 low pitch, COMPLETELY HELD OUT)")

    return X_train, y_train, X_val, y_val, X_test, y_test


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X, y, dataset_name: str):
    """Evaluate model on dataset."""
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
    print("DROPDIM REGULARIZATION EXPERIMENT")
    print("=" * 80)
    print("\nComparing:")
    print("  1. V4-Regularized (baseline): Standard dropout on neurons")
    print("  2. V4-DropDim: Drop entire feature dimensions")

    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load data
    print("\n--- Loading Data ---")
    real_data = load_session_with_pitch()
    print(f"Loaded {len(real_data)} finger state combinations")

    # Prepare data
    print("\n--- Preparing Data with Cross-Orientation Split ---")
    X_train, y_train, X_val, y_val, X_test, y_test = prepare_data_with_heldout(
        real_data=real_data,
        feature_set='mag_only',
        window_size=10,
        synthetic_ratio=0.5,
        samples_per_combo=200
    )

    # Normalize
    mean = X_train.reshape(-1, 3).mean(axis=0)
    std = X_train.reshape(-1, 3).std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

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

    print("\n--- Training V4-Regularized ---")
    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    history_v4 = model_v4.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    val_acc_v4, val_per_finger_v4 = evaluate_model(model_v4, X_val, y_val, "V4-Regularized Validation")
    test_acc_v4, test_per_finger_v4 = evaluate_model(model_v4, X_test, y_test, "V4-Regularized Test (Q1 held-out)")

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
    # EXPERIMENT 2: V4-DropDim
    # ========================================================================

    print("\n" + "=" * 80)
    print("EXPERIMENT 2: V4-DropDim")
    print("=" * 80)

    model_dropdim = build_v4_dropdim(window_size=10, n_features=3)
    print("\nModel architecture:")
    model_dropdim.summary()

    print("\n--- Training V4-DropDim ---")
    history_dropdim = model_dropdim.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate
    val_acc_dropdim, val_per_finger_dropdim = evaluate_model(model_dropdim, X_val, y_val, "V4-DropDim Validation")
    test_acc_dropdim, test_per_finger_dropdim = evaluate_model(model_dropdim, X_test, y_test, "V4-DropDim Test (Q1 held-out)")

    gap_dropdim = (val_acc_dropdim - test_acc_dropdim) * 100
    print(f"\nGeneralization gap: {gap_dropdim:.1f}%")

    results['v4_dropdim'] = {
        'val_acc': val_acc_dropdim,
        'test_acc': test_acc_dropdim,
        'gap': gap_dropdim,
        'val_per_finger': val_per_finger_dropdim,
        'test_per_finger': test_per_finger_dropdim
    }

    # ========================================================================
    # COMPARISON
    # ========================================================================

    print("\n" + "=" * 80)
    print("COMPARISON: V4-Regularized vs V4-DropDim")
    print("=" * 80)

    print("\n| Metric | V4-Regularized | V4-DropDim | Improvement |")
    print("|--------|----------------|------------|-------------|")

    test_diff = (test_acc_dropdim - test_acc_v4) * 100
    gap_diff = gap_v4 - gap_dropdim

    print(f"| Test Accuracy | {test_acc_v4:.1%} | {test_acc_dropdim:.1%} | {test_diff:+.1f}% |")
    print(f"| Generalization Gap | {gap_v4:.1f}% | {gap_dropdim:.1f}% | {gap_diff:+.1f}% |")

    print("\nPer-Finger Test Accuracy:")
    print("| Finger | V4-Regularized | V4-DropDim | Improvement |")
    print("|--------|----------------|------------|-------------|")

    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    for finger in fingers:
        acc_v4 = test_per_finger_v4[finger]
        acc_dropdim = test_per_finger_dropdim[finger]
        diff = (acc_dropdim - acc_v4) * 100
        print(f"| {finger} | {acc_v4:.1%} | {acc_dropdim:.1%} | {diff:+.1f}% |")

    # Determine winner
    winner = "V4-DropDim" if test_acc_dropdim > test_acc_v4 else "V4-Regularized"
    print(f"\n🏆 Winner: {winner}")

    if test_acc_dropdim > test_acc_v4:
        print(f"   DropDim improves test accuracy by {test_diff:.1f}%")
        print(f"   and reduces generalization gap by {gap_diff:.1f}%")
    else:
        print(f"   Baseline remains best (DropDim: {test_diff:.1f}%)")

    # Save results
    output_path = Path('ml/results/dropdim_experiment.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n📊 Results saved to {output_path}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    return results


if __name__ == '__main__':
    main()
