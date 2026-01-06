#!/usr/bin/env python3
"""
Deep Dive: CNN-LSTM Cross-Orientation Study

Comprehensive ablation study comparing:
1. Deployed model (full 9-DoF, w=50) vs mag-only variants
2. With/without synthetic data augmentation
3. Various data mix ratios
4. Different window sizes

All evaluated using the SAME cross-orientation test (train high pitch → test low pitch).

Author: Claude
Date: January 2026
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras


# ============================================================================
# DATA LOADING
# ============================================================================

@dataclass
class FingerStateData:
    combo: str
    samples: np.ndarray  # (n, 9) ax,ay,az,gx,gy,gz,mx,my,mz
    pitch_angles: np.ndarray  # (n,) euler pitch for each sample


def load_session_with_pitch() -> Dict[str, FingerStateData]:
    """Load session data including pitch angles for orientation splitting."""
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
# SYNTHETIC DATA GENERATION (from deploy script)
# ============================================================================

class SyntheticGenerator:
    """Generate synthetic samples matching observed real data patterns."""

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

    def generate_combo(self, combo: str, n_samples: int,
                       include_orientation_variance: bool = True) -> np.ndarray:
        """Generate synthetic 9-DoF samples for a finger combo."""
        if combo in self.real_data:
            return self._generate_from_real(combo, n_samples, include_orientation_variance)
        else:
            return self._generate_interpolated(combo, n_samples, include_orientation_variance)

    def _generate_from_real(self, combo: str, n_samples: int,
                            include_orientation_variance: bool) -> np.ndarray:
        real = self.real_data[combo]
        mag_mean = real.samples[:, 6:9].mean(axis=0)
        mag_std = real.samples[:, 6:9].std(axis=0)

        samples = []
        for _ in range(n_samples):
            # Magnetometer with noise
            if include_orientation_variance:
                # Add extra variance to simulate different orientations
                mag_sample = mag_mean + np.random.randn(3) * mag_std * 2.0
            else:
                mag_sample = mag_mean + np.random.randn(3) * mag_std

            # IMU (static hand)
            ax = np.random.normal(0, 0.05)
            ay = np.random.normal(0, 0.05)
            az = np.random.normal(-1, 0.05)
            gx = np.random.normal(0, 2.0)
            gy = np.random.normal(0, 2.0)
            gz = np.random.normal(0, 2.0)

            samples.append([ax, ay, az, gx, gy, gz, mag_sample[0], mag_sample[1], mag_sample[2]])

        return np.array(samples)

    def _generate_interpolated(self, combo: str, n_samples: int,
                               include_orientation_variance: bool) -> np.ndarray:
        if self.baseline:
            base_mag = self.baseline.samples[:, 6:9].mean(axis=0)
            base_std = self.baseline.samples[:, 6:9].std(axis=0)
        else:
            base_mag = np.array([46, -46, 31])
            base_std = np.array([25, 40, 50])

        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        total_delta = np.zeros(3)
        total_var = base_std ** 2

        for i, state in enumerate(combo):
            if state == 'f':
                finger = fingers[i]
                if finger in self.finger_effects:
                    total_delta += self.finger_effects[finger]['mag_delta']
                    total_var += self.finger_effects[finger]['mag_std'] ** 2

        mag_mean = base_mag + total_delta
        mag_std = np.sqrt(total_var)

        samples = []
        for _ in range(n_samples):
            if include_orientation_variance:
                mag_sample = mag_mean + np.random.randn(3) * mag_std * 2.0
            else:
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
# FEATURE EXTRACTION
# ============================================================================

def extract_features(samples: np.ndarray, feature_set: str) -> np.ndarray:
    """Extract specific feature set from 9-DoF samples."""
    if feature_set == '9dof':
        return samples  # All 9 features
    elif feature_set == 'mag_only':
        return samples[:, 6:9]  # mx, my, mz
    elif feature_set == 'accel_gyro':
        return samples[:, 0:6]  # ax, ay, az, gx, gy, gz
    elif feature_set == 'accel_only':
        return samples[:, 0:3]  # ax, ay, az
    else:
        return samples


# ============================================================================
# WINDOWING
# ============================================================================

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


# ============================================================================
# MODEL BUILDING
# ============================================================================

def build_model(window_size: int, n_features: int, model_type: str = 'cnn_lstm') -> keras.Model:
    """Build model based on window size and type."""

    if window_size == 1:
        # Single sample: Dense only
        inputs = keras.layers.Input(shape=(1, n_features))
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(5, activation='sigmoid')(x)

    elif window_size <= 5:
        # Small window: Simple Conv
        inputs = keras.layers.Input(shape=(window_size, n_features))
        x = keras.layers.Conv1D(32, min(3, window_size), activation='relu', padding='same')(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(5, activation='sigmoid')(x)

    elif model_type == 'cnn_lstm':
        # Deployed model architecture
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

    else:
        # Simple CNN
        inputs = keras.layers.Input(shape=(window_size, n_features))
        x = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(5, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================================
# TRAINING AND EVALUATION
# ============================================================================

def combo_to_label(combo: str) -> np.ndarray:
    return np.array([0 if c == 'e' else 1 for c in combo], dtype=np.float32)


def prepare_cross_orientation_data(
    real_data: Dict[str, FingerStateData],
    feature_set: str,
    window_size: int,
    synthetic_ratio: float = 0.0,  # 0 = real only, 1 = synthetic only
    samples_per_combo: int = 500,
    orientation_augment: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict]:
    """
    Prepare data with cross-orientation split.

    Returns:
        X_train: High-pitch training data
        y_train: Training labels
        X_test: Low-pitch test data
        y_test: Test labels
        info: Metadata about the split
    """
    all_combos = list(real_data.keys())

    # Collect all samples with their pitch angles
    train_windows = []
    train_labels = []
    test_windows = []
    test_labels = []

    # Calculate pitch quartiles across all data
    all_pitches = []
    for combo_data in real_data.values():
        all_pitches.extend(combo_data.pitch_angles.tolist())
    q1 = np.percentile(all_pitches, 25)
    q3 = np.percentile(all_pitches, 75)

    generator = SyntheticGenerator(real_data) if synthetic_ratio > 0 else None

    for combo, combo_data in real_data.items():
        label = combo_to_label(combo)

        # Extract features
        features = extract_features(combo_data.samples, feature_set)
        n_features = features.shape[1]

        # Split by pitch
        high_pitch_mask = combo_data.pitch_angles >= q3
        low_pitch_mask = combo_data.pitch_angles <= q1

        high_pitch_samples = features[high_pitch_mask]
        low_pitch_samples = features[low_pitch_mask]

        # Generate synthetic if needed
        if synthetic_ratio > 0 and generator:
            n_synth = int(samples_per_combo * synthetic_ratio)
            synth_samples = generator.generate_combo(combo, n_synth, orientation_augment)
            synth_features = extract_features(synth_samples, feature_set)

            # Add synthetic to training
            high_pitch_samples = np.vstack([high_pitch_samples, synth_features]) if len(high_pitch_samples) > 0 else synth_features

        # Create windows for training (high pitch + synthetic)
        if len(high_pitch_samples) >= window_size:
            windows = create_windows(high_pitch_samples, window_size)
            for w in windows:
                train_windows.append(w)
                train_labels.append(label)

        # Create windows for testing (low pitch only - no synthetic)
        if len(low_pitch_samples) >= window_size:
            windows = create_windows(low_pitch_samples, window_size)
            for w in windows:
                test_windows.append(w)
                test_labels.append(label)

    X_train = np.array(train_windows)
    y_train = np.array(train_labels)
    X_test = np.array(test_windows)
    y_test = np.array(test_labels)

    # Normalize
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    info = {
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_features': X_train.shape[-1],
        'window_size': window_size,
        'q1_pitch': q1,
        'q3_pitch': q3,
    }

    return X_train, y_train, X_test, y_test, info


def train_and_evaluate(
    X_train: np.ndarray, y_train: np.ndarray,
    X_test: np.ndarray, y_test: np.ndarray,
    model_type: str = 'cnn_lstm',
    epochs: int = 30
) -> Dict:
    """Train model and return metrics."""
    window_size = X_train.shape[1]
    n_features = X_train.shape[2]

    model = build_model(window_size, n_features, model_type)

    early_stop = keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=epochs,
        batch_size=min(32, len(X_train) // 4),
        callbacks=[early_stop],
        verbose=0
    )

    # Evaluate
    y_pred_train = (model.predict(X_train, verbose=0) > 0.5).astype(int)
    y_pred_test = (model.predict(X_test, verbose=0) > 0.5).astype(int)

    train_acc = np.mean(y_pred_train == y_train)
    test_acc = np.mean(y_pred_test == y_test)

    # Per-finger accuracy
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger = {}
    for i, f in enumerate(fingers):
        per_finger[f] = float(np.mean(y_pred_test[:, i] == y_test[:, i]))

    tf.keras.backend.clear_session()

    return {
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'gap': float(train_acc - test_acc),
        'per_finger': per_finger,
    }


# ============================================================================
# MAIN ABLATION STUDY
# ============================================================================

def run_ablation():
    """Run comprehensive ablation study."""
    print("=" * 80)
    print("CNN-LSTM CROSS-ORIENTATION ABLATION STUDY")
    print("=" * 80)

    # Load data
    print("\nLoading session data...")
    real_data = load_session_with_pitch()
    print(f"Loaded {len(real_data)} finger state combinations")
    for combo, data in sorted(real_data.items()):
        print(f"  {combo}: {len(data.samples)} samples, pitch range [{data.pitch_angles.min():.0f}°, {data.pitch_angles.max():.0f}°]")

    results = {}

    # =========================================================================
    # PART 1: Feature Set Comparison (fixed window=50, no synthetic)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 1: FEATURE SET COMPARISON (window=50, real data only)")
    print("=" * 80)

    feature_sets = ['9dof', 'mag_only', 'accel_gyro']

    for fs in feature_sets:
        print(f"\n{fs}:")
        X_train, y_train, X_test, y_test, info = prepare_cross_orientation_data(
            real_data, fs, window_size=50, synthetic_ratio=0.0
        )
        print(f"  Train: {info['n_train']}, Test: {info['n_test']}")

        if info['n_train'] < 20 or info['n_test'] < 20:
            print("  Skipped: insufficient data")
            continue

        result = train_and_evaluate(X_train, y_train, X_test, y_test)
        results[f'{fs}_w50_real'] = result
        print(f"  Train: {result['train_acc']:.1%}, Test: {result['test_acc']:.1%}, Gap: {result['gap']:.1%}")

    # =========================================================================
    # PART 2: Window Size Comparison (mag_only, no synthetic)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 2: WINDOW SIZE COMPARISON (mag_only, real data only)")
    print("=" * 80)

    window_sizes = [1, 2, 5, 10, 25, 50]

    for ws in window_sizes:
        print(f"\nwindow={ws}:")
        X_train, y_train, X_test, y_test, info = prepare_cross_orientation_data(
            real_data, 'mag_only', window_size=ws, synthetic_ratio=0.0
        )
        print(f"  Train: {info['n_train']}, Test: {info['n_test']}")

        if info['n_train'] < 20 or info['n_test'] < 20:
            print("  Skipped: insufficient data")
            continue

        result = train_and_evaluate(X_train, y_train, X_test, y_test)
        results[f'mag_only_w{ws}_real'] = result
        print(f"  Train: {result['train_acc']:.1%}, Test: {result['test_acc']:.1%}, Gap: {result['gap']:.1%}")

    # =========================================================================
    # PART 3: Synthetic Data Ratio (mag_only, w=10)
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 3: SYNTHETIC DATA RATIO (mag_only, window=10)")
    print("=" * 80)

    synth_ratios = [0.0, 0.25, 0.5, 0.75, 1.0]

    for ratio in synth_ratios:
        label = f"{int(ratio*100)}% synth"
        print(f"\n{label}:")

        X_train, y_train, X_test, y_test, info = prepare_cross_orientation_data(
            real_data, 'mag_only', window_size=10,
            synthetic_ratio=ratio, samples_per_combo=300,
            orientation_augment=True
        )
        print(f"  Train: {info['n_train']}, Test: {info['n_test']}")

        if info['n_train'] < 20 or info['n_test'] < 20:
            print("  Skipped: insufficient data")
            continue

        result = train_and_evaluate(X_train, y_train, X_test, y_test)
        results[f'mag_w10_synth{int(ratio*100)}'] = result
        print(f"  Train: {result['train_acc']:.1%}, Test: {result['test_acc']:.1%}, Gap: {result['gap']:.1%}")

    # =========================================================================
    # PART 4: Best Config with Orientation Augmentation
    # =========================================================================
    print("\n" + "=" * 80)
    print("PART 4: ORIENTATION AUGMENTATION ABLATION")
    print("=" * 80)

    for aug in [False, True]:
        label = "with" if aug else "without"
        print(f"\n{label} orientation variance in synthetic:")

        X_train, y_train, X_test, y_test, info = prepare_cross_orientation_data(
            real_data, 'mag_only', window_size=10,
            synthetic_ratio=0.5, samples_per_combo=300,
            orientation_augment=aug
        )

        result = train_and_evaluate(X_train, y_train, X_test, y_test)
        results[f'mag_w10_synth50_aug{aug}'] = result
        print(f"  Train: {result['train_acc']:.1%}, Test: {result['test_acc']:.1%}, Gap: {result['gap']:.1%}")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("\n### Cross-Orientation Test Accuracy (sorted)")
    print(f"{'Config':<35} {'Train':>8} {'Test':>8} {'Gap':>8}")
    print("-" * 65)

    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True)
    for key, r in sorted_results:
        print(f"{key:<35} {r['train_acc']:>7.1%} {r['test_acc']:>7.1%} {r['gap']:>7.1%}")

    if sorted_results:
        best = sorted_results[0]
        print(f"\n✓ Best config: {best[0]}")
        print(f"  Test accuracy: {best[1]['test_acc']:.1%}")
        print(f"  Orientation gap: {best[1]['gap']:.1%}")

    # Save results
    output_path = Path("ml/cross_orientation_ablation.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    run_ablation()
