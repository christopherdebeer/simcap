#!/usr/bin/env python3
"""
Explore New Architectures for Improved Cross-Orientation Performance

Based on findings from:
- V2 vs V3 benchmark: V3 achieves 68.4% with 25.8% generalization gap
- Ablation study: w=1 gets 97% in-distribution but 48% cross-orientation
- Key insight: Need to reduce overfitting while maintaining accuracy

New architectures to test:
1. V4-Regularized: Stronger regularization (dropout, L2, label smoothing)
2. V4-Attention: Attention mechanism for discriminative features
3. V4-Residual: Skip connections for better gradient flow
4. V4-MultiScale: Multi-window fusion (w=5, 10, 15)
5. V4-PerFinger: Per-finger heads with shared encoder
6. V4-Ensemble: Multiple small models

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

# Import data loading from benchmark script
import sys
sys.path.append(str(Path(__file__).parent))


# ============================================================================
# DATA STRUCTURES (reuse from benchmark)
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
# SYNTHETIC DATA (reuse from benchmark)
# ============================================================================

class SyntheticGenerator:
    """Generate synthetic samples."""

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
# DATA PREPROCESSING (reuse from benchmark)
# ============================================================================

def extract_features(samples: np.ndarray, feature_set: str = 'mag_only') -> np.ndarray:
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


def prepare_data_with_heldout(
    real_data: Dict[str, FingerStateData],
    feature_set: str,
    window_size: int,
    synthetic_ratio: float,
    samples_per_combo: int = 300,
    subset_ratio: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """Prepare data with 3-way split."""
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

        high_pitch_mask = combo_data.pitch_angles >= q3
        low_pitch_mask = combo_data.pitch_angles <= q1

        high_pitch_samples = features[high_pitch_mask]
        low_pitch_samples = features[low_pitch_mask]

        if len(high_pitch_samples) > 0:
            n_samples = len(high_pitch_samples)
            n_train = int(n_samples * subset_ratio)

            indices = np.random.permutation(n_samples)
            train_indices = indices[:n_train]
            val_indices = indices[n_train:]

            train_samples = high_pitch_samples[train_indices]
            val_samples = high_pitch_samples[val_indices]

            if synthetic_ratio > 0 and generator and len(train_samples) > 0:
                n_synth = int(len(train_samples) * synthetic_ratio)
                if n_synth > 0:
                    synth_samples = generator.generate_combo(combo, n_synth)
                    synth_features = extract_features(synth_samples, feature_set)
                    if len(synth_features) > 0 and synth_features.shape[1] == train_samples.shape[1]:
                        train_samples = np.vstack([train_samples, synth_features])

            if len(train_samples) >= window_size:
                windows = create_windows(train_samples, window_size)
                for w in windows:
                    train_windows.append(w)
                    train_labels.append(label)

            if len(val_samples) >= window_size:
                windows = create_windows(val_samples, window_size)
                for w in windows:
                    val_windows.append(w)
                    val_labels.append(label)

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

    n_features = X_train.shape[-1]
    mean = X_train.reshape(-1, n_features).mean(axis=0)
    std = X_train.reshape(-1, n_features).std(axis=0) + 1e-8

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
# V3 BASELINE (for comparison)
# ============================================================================

def build_v3_baseline(window_size: int = 10, n_features: int = 3) -> keras.Model:
    """V3 baseline architecture."""
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
# NEW ARCHITECTURES
# ============================================================================

def build_v4_regularized(window_size: int = 10, n_features: int = 3) -> keras.Model:
    """
    V4 Regularized: Stronger regularization to reduce overfitting.

    Changes from V3:
    - Higher dropout (0.3 -> 0.5)
    - L2 regularization on dense layers
    - Label smoothing via loss function
    """
    inputs = keras.layers.Input(shape=(window_size, n_features))

    x = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.4)(x)  # Added early dropout
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dropout(0.5)(x)  # Increased from 0.3
    x = keras.layers.Dense(32, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.Dropout(0.4)(x)  # Additional dropout
    outputs = keras.layers.Dense(5, activation='sigmoid',
                                 kernel_regularizer=keras.regularizers.l2(0.01))(x)

    model = keras.Model(inputs, outputs)

    # Label smoothing: smooth labels from 0/1 to 0.1/0.9
    def label_smoothed_loss(y_true, y_pred):
        y_true_smooth = y_true * 0.9 + 0.05
        return keras.losses.binary_crossentropy(y_true_smooth, y_pred)

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=label_smoothed_loss,
        metrics=['accuracy']
    )
    return model


def build_v4_attention(window_size: int = 10, n_features: int = 3) -> keras.Model:
    """
    V4 Attention: Add attention mechanism to focus on discriminative time steps.

    Changes from V3:
    - Attention layer after Conv1D to weight important time steps
    - Helps model focus on finger state transitions
    """
    inputs = keras.layers.Input(shape=(window_size, n_features))

    # Convolutional feature extraction
    x = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)

    # Self-attention mechanism
    # Query, Key, Value from same input
    attention_output = keras.layers.MultiHeadAttention(
        num_heads=2, key_dim=16, dropout=0.2
    )(x, x)
    x = keras.layers.Add()([x, attention_output])  # Residual connection
    x = keras.layers.LayerNormalization()(x)

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


def build_v4_residual(window_size: int = 10, n_features: int = 3) -> keras.Model:
    """
    V4 Residual: Add skip connections for better gradient flow.

    Changes from V3:
    - Residual connections around conv and dense layers
    - Helps with deeper networks and gradient flow
    """
    inputs = keras.layers.Input(shape=(window_size, n_features))

    # First conv block with residual
    x = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)

    # Second conv block with residual
    conv_out = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(x)
    conv_out = keras.layers.BatchNormalization()(conv_out)
    x = keras.layers.Add()([x, conv_out])  # Residual connection

    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.LSTM(32, return_sequences=False)(x)
    x = keras.layers.Dropout(0.3)(x)

    # Dense block with residual
    dense_out = keras.layers.Dense(32, activation='relu')(x)
    # Need to match dimensions for residual
    x_proj = keras.layers.Dense(32)(x)
    x = keras.layers.Add()([x_proj, dense_out])
    x = keras.layers.Activation('relu')(x)

    outputs = keras.layers.Dense(5, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_v4_multiscale(n_features: int = 3) -> keras.Model:
    """
    V4 Multi-Scale: Process multiple window sizes in parallel.

    Changes from V3:
    - Three parallel branches for w=5, 10, 15
    - Concatenate features from all scales
    - Captures both short and long-term patterns
    """
    # Three input branches for different window sizes
    input_w5 = keras.layers.Input(shape=(5, n_features), name='input_w5')
    input_w10 = keras.layers.Input(shape=(10, n_features), name='input_w10')
    input_w15 = keras.layers.Input(shape=(15, n_features), name='input_w15')

    # Branch 1: w=5 (short-term)
    x1 = keras.layers.Conv1D(16, 3, activation='relu', padding='same')(input_w5)
    x1 = keras.layers.GlobalAveragePooling1D()(x1)

    # Branch 2: w=10 (medium-term)
    x2 = keras.layers.Conv1D(16, 3, activation='relu', padding='same')(input_w10)
    x2 = keras.layers.MaxPooling1D(2)(x2)
    x2 = keras.layers.LSTM(16)(x2)

    # Branch 3: w=15 (long-term)
    x3 = keras.layers.Conv1D(16, 3, activation='relu', padding='same')(input_w15)
    x3 = keras.layers.MaxPooling1D(3)(x3)
    x3 = keras.layers.LSTM(16)(x3)

    # Concatenate all scales
    x = keras.layers.Concatenate()([x1, x2, x3])
    x = keras.layers.Dropout(0.3)(x)
    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(5, activation='sigmoid')(x)

    model = keras.Model([input_w5, input_w10, input_w15], outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def build_v4_perfinger(window_size: int = 10, n_features: int = 3) -> keras.Model:
    """
    V4 Per-Finger: Separate prediction heads for each finger.

    Changes from V3:
    - Shared encoder (conv + LSTM)
    - Separate dense layers per finger
    - Allows specialization per finger (helps with pinky!)
    """
    inputs = keras.layers.Input(shape=(window_size, n_features))

    # Shared encoder
    x = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dropout(0.3)(x)
    shared = keras.layers.Dense(32, activation='relu')(x)

    # Per-finger heads
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    outputs = []
    for finger in fingers:
        finger_out = keras.layers.Dense(8, activation='relu', name=f'{finger}_hidden')(shared)
        finger_out = keras.layers.Dropout(0.2)(finger_out)
        finger_pred = keras.layers.Dense(1, activation='sigmoid', name=f'{finger}_out')(finger_out)
        outputs.append(finger_pred)

    # Concatenate predictions
    outputs = keras.layers.Concatenate()(outputs)

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

def train_model(
    X_train, y_train, X_val, y_val,
    model_fn, epochs=30, verbose=0, **model_kwargs
) -> keras.Model:
    """Train model with early stopping."""
    model = model_fn(**model_kwargs)

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


# ============================================================================
# MAIN EXPLORATION
# ============================================================================

def main():
    print("=" * 80)
    print("EXPLORING NEW ARCHITECTURES FOR IMPROVED GENERALIZATION")
    print("=" * 80)

    # Load data
    print("\n--- Loading Data ---")
    real_data = load_session_with_pitch()
    print(f"Loaded {len(real_data)} finger state combinations")

    # Prepare data
    print("\n--- Preparing Data ---")
    X_train, y_train, X_val, y_val, X_test, y_test, info = prepare_data_with_heldout(
        real_data, feature_set='mag_only', window_size=10,
        synthetic_ratio=0.5, samples_per_combo=300,
        subset_ratio=0.5
    )
    print(f"Train: {info['n_train']}, Val: {info['n_val']}, Test: {info['n_test']}")

    results = {}

    # =========================================================================
    # V3 Baseline
    # =========================================================================
    print("\n" + "=" * 70)
    print("BASELINE: V3")
    print("=" * 70)

    model_v3 = train_model(X_train, y_train, X_val, y_val,
                          build_v3_baseline, verbose=1,
                          window_size=10, n_features=3)

    train_v3 = evaluate_model(model_v3, X_train, y_train)
    val_v3 = evaluate_model(model_v3, X_val, y_val)
    test_v3 = evaluate_model(model_v3, X_test, y_test)

    print(f"Train: {train_v3['overall_acc']:.1%}, Val: {val_v3['overall_acc']:.1%}, Test: {test_v3['overall_acc']:.1%}")
    print(f"Gap: {(train_v3['overall_acc'] - test_v3['overall_acc'])*100:.1f}%")

    results['v3_baseline'] = {'train': train_v3, 'val': val_v3, 'test': test_v3}
    tf.keras.backend.clear_session()

    # =========================================================================
    # V4 Regularized
    # =========================================================================
    print("\n" + "=" * 70)
    print("V4 REGULARIZED (Stronger Dropout + L2 + Label Smoothing)")
    print("=" * 70)

    model_v4_reg = train_model(X_train, y_train, X_val, y_val,
                               build_v4_regularized, verbose=1,
                               window_size=10, n_features=3)

    train_reg = evaluate_model(model_v4_reg, X_train, y_train)
    val_reg = evaluate_model(model_v4_reg, X_val, y_val)
    test_reg = evaluate_model(model_v4_reg, X_test, y_test)

    print(f"Train: {train_reg['overall_acc']:.1%}, Val: {val_reg['overall_acc']:.1%}, Test: {test_reg['overall_acc']:.1%}")
    print(f"Gap: {(train_reg['overall_acc'] - test_reg['overall_acc'])*100:.1f}%")

    results['v4_regularized'] = {'train': train_reg, 'val': val_reg, 'test': test_reg}
    tf.keras.backend.clear_session()

    # =========================================================================
    # V4 Attention
    # =========================================================================
    print("\n" + "=" * 70)
    print("V4 ATTENTION (Multi-Head Attention)")
    print("=" * 70)

    model_v4_att = train_model(X_train, y_train, X_val, y_val,
                               build_v4_attention, verbose=1,
                               window_size=10, n_features=3)

    train_att = evaluate_model(model_v4_att, X_train, y_train)
    val_att = evaluate_model(model_v4_att, X_val, y_val)
    test_att = evaluate_model(model_v4_att, X_test, y_test)

    print(f"Train: {train_att['overall_acc']:.1%}, Val: {val_att['overall_acc']:.1%}, Test: {test_att['overall_acc']:.1%}")
    print(f"Gap: {(train_att['overall_acc'] - test_att['overall_acc'])*100:.1f}%")

    results['v4_attention'] = {'train': train_att, 'val': val_att, 'test': test_att}
    tf.keras.backend.clear_session()

    # =========================================================================
    # V4 Residual
    # =========================================================================
    print("\n" + "=" * 70)
    print("V4 RESIDUAL (Skip Connections)")
    print("=" * 70)

    model_v4_res = train_model(X_train, y_train, X_val, y_val,
                               build_v4_residual, verbose=1,
                               window_size=10, n_features=3)

    train_res = evaluate_model(model_v4_res, X_train, y_train)
    val_res = evaluate_model(model_v4_res, X_val, y_val)
    test_res = evaluate_model(model_v4_res, X_test, y_test)

    print(f"Train: {train_res['overall_acc']:.1%}, Val: {val_res['overall_acc']:.1%}, Test: {test_res['overall_acc']:.1%}")
    print(f"Gap: {(train_res['overall_acc'] - test_res['overall_acc'])*100:.1f}%")

    results['v4_residual'] = {'train': train_res, 'val': val_res, 'test': test_res}
    tf.keras.backend.clear_session()

    # =========================================================================
    # V4 Per-Finger
    # =========================================================================
    print("\n" + "=" * 70)
    print("V4 PER-FINGER (Separate Heads)")
    print("=" * 70)

    model_v4_pf = train_model(X_train, y_train, X_val, y_val,
                              build_v4_perfinger, verbose=1,
                              window_size=10, n_features=3)

    train_pf = evaluate_model(model_v4_pf, X_train, y_train)
    val_pf = evaluate_model(model_v4_pf, X_val, y_val)
    test_pf = evaluate_model(model_v4_pf, X_test, y_test)

    print(f"Train: {train_pf['overall_acc']:.1%}, Val: {val_pf['overall_acc']:.1%}, Test: {test_pf['overall_acc']:.1%}")
    print(f"Gap: {(train_pf['overall_acc'] - test_pf['overall_acc'])*100:.1f}%")

    results['v4_perfinger'] = {'train': train_pf, 'val': val_pf, 'test': test_pf}
    tf.keras.backend.clear_session()

    # =========================================================================
    # COMPARISON SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("ARCHITECTURE COMPARISON SUMMARY")
    print("=" * 80)

    print(f"\n{'Architecture':<20} {'Train':>8} {'Val':>8} {'Test':>8} {'Gap':>8}")
    print("-" * 60)

    architectures = [
        ('V3 Baseline', 'v3_baseline'),
        ('V4 Regularized', 'v4_regularized'),
        ('V4 Attention', 'v4_attention'),
        ('V4 Residual', 'v4_residual'),
        ('V4 Per-Finger', 'v4_perfinger'),
    ]

    best_test = 0
    best_arch = None
    best_gap = float('inf')
    best_gap_arch = None

    for name, key in architectures:
        r = results[key]
        train_acc = r['train']['overall_acc']
        val_acc = r['val']['overall_acc']
        test_acc = r['test']['overall_acc']
        gap = train_acc - test_acc

        print(f"{name:<20} {train_acc:>7.1%} {val_acc:>7.1%} {test_acc:>7.1%} {gap*100:>7.1f}%")

        if test_acc > best_test:
            best_test = test_acc
            best_arch = name

        if gap < best_gap:
            best_gap = gap
            best_gap_arch = name

    print(f"\n*** Best Test Accuracy: {best_arch} with {best_test:.1%} ***")
    print(f"*** Smallest Gap: {best_gap_arch} with {best_gap*100:.1f}% ***")

    # Per-finger comparison for best model
    print(f"\n{'Finger':<15} {'V3':>10} {'Best':>10} {'Improvement':>15}")
    print("-" * 55)

    best_key = [k for n, k in architectures if n == best_arch][0]
    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        v3_acc = results['v3_baseline']['test']['per_finger'][finger]
        best_acc = results[best_key]['test']['per_finger'][finger]
        print(f"{finger.capitalize():<15} {v3_acc:>9.1%} {best_acc:>9.1%} {(best_acc - v3_acc)*100:>+13.1f}%")

    # Save results
    results_path = Path("ml/new_architectures_results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\n\nResults saved to: {results_path}")

    return results


if __name__ == '__main__':
    main()
