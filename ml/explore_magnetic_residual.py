#!/usr/bin/env python3
"""
Explore training on magnetic residual instead of raw magnetometer values.

Hypothesis: Using residual (mag - baseline) instead of raw mag values may:
1. Improve generalization across different sensor orientations
2. Reduce dependence on absolute calibration
3. Focus model on finger-induced field changes

Approach:
- Baseline: Open palm (eeeee) magnetic field
- Residual: current_mag - baseline_mag
- Train models on both raw and residual to compare
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass

# TensorFlow
import tensorflow as tf
from tensorflow import keras

print("=" * 70)
print("MAGNETIC RESIDUAL TRAINING EXPLORATION")
print("=" * 70)


# ============================================================================
# DATA LOADING
# ============================================================================

@dataclass
class FingerStateData:
    combo: str
    samples: np.ndarray  # (n, 9) ax,ay,az,gx,gy,gz,mx,my,mz
    magnitudes: np.ndarray
    mag_vectors: np.ndarray  # (n, 3) mx, my, mz


def load_dec31_session() -> Dict[str, FingerStateData]:
    """Load the December 31 labeled session data."""
    session_path = Path(__file__).parent.parent / 'data' / 'GAMBIT' / '2025-12-31T14_06_18.270Z.json'

    with open(session_path, 'r') as f:
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
        mag_data = []
        mag_vectors = []

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

            sensor_data.append([ax, ay, az, gx, gy, gz, mx, my, mz])
            mag_data.append(np.sqrt(mx**2 + my**2 + mz**2))
            mag_vectors.append([mx, my, mz])

        if not sensor_data:
            continue

        combo = ''.join([
            'e' if fingers.get(f, '?') == 'extended' else
            'f' if fingers.get(f, '?') == 'flexed' else '?'
            for f in ['thumb', 'index', 'middle', 'ring', 'pinky']
        ])

        if combo not in combo_data:
            combo_data[combo] = FingerStateData(
                combo=combo,
                samples=np.array(sensor_data),
                magnitudes=np.array(mag_data),
                mag_vectors=np.array(mag_vectors)
            )
        else:
            existing = combo_data[combo]
            combo_data[combo] = FingerStateData(
                combo=combo,
                samples=np.vstack([existing.samples, sensor_data]),
                magnitudes=np.concatenate([existing.magnitudes, mag_data]),
                mag_vectors=np.vstack([existing.mag_vectors, mag_vectors])
            )

    return combo_data


# ============================================================================
# RESIDUAL COMPUTATION
# ============================================================================

def compute_baseline(real_data: Dict[str, FingerStateData]) -> np.ndarray:
    """Compute baseline magnetic field from open palm (eeeee) state."""
    if 'eeeee' in real_data:
        baseline = real_data['eeeee'].mag_vectors.mean(axis=0)
        print(f"Baseline from eeeee: [{baseline[0]:.1f}, {baseline[1]:.1f}, {baseline[2]:.1f}] μT")
        return baseline
    else:
        # Fallback: use overall mean
        all_mags = np.vstack([d.mag_vectors for d in real_data.values()])
        baseline = all_mags.mean(axis=0)
        print(f"Baseline from overall mean: [{baseline[0]:.1f}, {baseline[1]:.1f}, {baseline[2]:.1f}] μT")
        return baseline


def convert_to_residual(samples: np.ndarray, baseline: np.ndarray) -> np.ndarray:
    """Convert raw samples to residual by subtracting baseline from mag values."""
    # samples shape: (n, 9) - ax,ay,az,gx,gy,gz,mx,my,mz
    residual = samples.copy()
    residual[:, 6] -= baseline[0]  # mx residual
    residual[:, 7] -= baseline[1]  # my residual
    residual[:, 8] -= baseline[2]  # mz residual
    return residual


# ============================================================================
# ANALYSIS
# ============================================================================

def analyze_raw_vs_residual(real_data: Dict[str, FingerStateData], baseline: np.ndarray):
    """Compare raw vs residual magnetic field distributions."""
    print("\n" + "-" * 70)
    print("RAW vs RESIDUAL ANALYSIS")
    print("-" * 70)

    print(f"\n{'Combo':<8} {'Raw Mag (μT)':<20} {'Residual Mag (μT)':<20} {'Direction':<20}")
    print("-" * 70)

    for combo in sorted(real_data.keys()):
        data = real_data[combo]
        raw_mag = data.mag_vectors.mean(axis=0)
        residual_mag = raw_mag - baseline

        raw_norm = np.linalg.norm(raw_mag)
        res_norm = np.linalg.norm(residual_mag)

        # Direction: normalized residual vector
        if res_norm > 1:
            direction = residual_mag / res_norm
            dir_str = f"[{direction[0]:+.2f},{direction[1]:+.2f},{direction[2]:+.2f}]"
        else:
            dir_str = "[~0, ~0, ~0]"

        print(f"{combo:<8} {raw_norm:>8.1f}            {res_norm:>8.1f}            {dir_str}")

    # Analyze separability
    print("\n" + "-" * 70)
    print("FINGER EFFECT ANALYSIS (Residual)")
    print("-" * 70)

    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    single_finger_combos = {
        'thumb': 'feeee', 'index': 'efeee', 'middle': 'eefee',
        'ring': 'eeefe', 'pinky': 'eeeef'
    }

    print(f"\n{'Finger':<10} {'Δx':<12} {'Δy':<12} {'Δz':<12} {'|Δ|':<12}")
    print("-" * 60)

    for finger, combo in single_finger_combos.items():
        if combo in real_data:
            raw_mag = real_data[combo].mag_vectors.mean(axis=0)
            delta = raw_mag - baseline
            delta_norm = np.linalg.norm(delta)
            print(f"{finger:<10} {delta[0]:>+10.1f}  {delta[1]:>+10.1f}  {delta[2]:>+10.1f}  {delta_norm:>10.1f}")
        else:
            print(f"{finger:<10} (no data)")

    # Signal-to-noise ratio
    print("\n" + "-" * 70)
    print("SIGNAL-TO-NOISE ANALYSIS")
    print("-" * 70)

    if 'eeeee' in real_data:
        baseline_std = real_data['eeeee'].mag_vectors.std(axis=0)
        noise_norm = np.linalg.norm(baseline_std)
        print(f"\nBaseline noise (std): [{baseline_std[0]:.1f}, {baseline_std[1]:.1f}, {baseline_std[2]:.1f}] μT")
        print(f"Baseline noise magnitude: {noise_norm:.1f} μT")

        print(f"\n{'Finger':<10} {'Signal |Δ|':<12} {'SNR':<12}")
        print("-" * 40)

        for finger, combo in single_finger_combos.items():
            if combo in real_data:
                raw_mag = real_data[combo].mag_vectors.mean(axis=0)
                delta = raw_mag - baseline
                signal = np.linalg.norm(delta)
                snr = signal / noise_norm if noise_norm > 0 else float('inf')
                print(f"{finger:<10} {signal:>10.1f}  {snr:>10.1f}")


# ============================================================================
# SYNTHETIC DATA GENERATION (with residual)
# ============================================================================

class ResidualSyntheticGenerator:
    """Generate synthetic samples using residual-based approach."""

    def __init__(self, real_data: Dict[str, FingerStateData], baseline: np.ndarray):
        self.real_data = real_data
        self.baseline = baseline
        self._compute_finger_effects()

    def _compute_finger_effects(self):
        """Compute per-finger magnetic field deltas from baseline."""
        self.finger_effects = {}

        single_finger_combos = {
            'thumb': 'feeee', 'index': 'efeee', 'middle': 'eefee',
            'ring': 'eeefe', 'pinky': 'eeeef'
        }

        for finger, combo in single_finger_combos.items():
            if combo in self.real_data:
                data = self.real_data[combo]
                raw_mean = data.mag_vectors.mean(axis=0)
                raw_std = data.mag_vectors.std(axis=0)
                self.finger_effects[finger] = {
                    'delta': raw_mean - self.baseline,
                    'std': raw_std
                }
            else:
                # Default fallback
                self.finger_effects[finger] = {
                    'delta': np.array([200, 200, 200]),
                    'std': np.array([50, 50, 50])
                }

    def generate_combo(self, combo: str, n_samples: int) -> np.ndarray:
        """Generate synthetic samples for a finger combo using residual approach."""
        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']

        samples = []
        for _ in range(n_samples):
            # Start with baseline + noise
            if 'eeeee' in self.real_data:
                base_std = self.real_data['eeeee'].mag_vectors.std(axis=0)
            else:
                base_std = np.array([20, 20, 20])

            # Compute total magnetic field as baseline + sum of finger deltas
            total_delta = np.zeros(3)
            total_std = base_std.copy()

            for i, state in enumerate(combo):
                if state == 'f':
                    finger = fingers[i]
                    if finger in self.finger_effects:
                        total_delta += self.finger_effects[finger]['delta']
                        total_std = np.sqrt(total_std**2 + self.finger_effects[finger]['std']**2)

            # Sample magnetic field
            mag = self.baseline + total_delta + np.random.normal(0, total_std)

            # Generate IMU values (stationary hand assumption)
            ax = np.random.normal(0, 0.02)
            ay = np.random.normal(0, 0.02)
            az = np.random.normal(-1, 0.02)
            gx = np.random.normal(0, 1.0)
            gy = np.random.normal(0, 1.0)
            gz = np.random.normal(0, 1.0)

            samples.append([ax, ay, az, gx, gy, gz, mag[0], mag[1], mag[2]])

        return np.array(samples)


# ============================================================================
# DATASET PREPARATION
# ============================================================================

def create_windows(samples: np.ndarray, window_size: int = 50, stride: int = 25) -> np.ndarray:
    """Create sliding windows from sample data."""
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
    """Convert combo string to binary label array."""
    return np.array([0 if c == 'e' else 1 for c in combo], dtype=np.float32)


def prepare_dataset(real_data: Dict[str, FingerStateData],
                   baseline: np.ndarray,
                   use_residual: bool = True,
                   samples_per_combo: int = 500,
                   window_size: int = 50) -> Tuple:
    """Prepare dataset with optional residual transformation."""

    all_combos = [f"{t}{i}{m}{r}{p}" for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef']
    generator = ResidualSyntheticGenerator(real_data, baseline)

    all_windows = []
    all_labels = []

    for combo in all_combos:
        if combo in real_data:
            real_samples = real_data[combo].samples
            n_real = len(real_samples)
            n_synth = max(0, samples_per_combo - n_real)

            if n_synth > 0:
                synth_samples = generator.generate_combo(combo, n_synth)
                combined = np.vstack([real_samples, synth_samples])
            else:
                combined = real_samples[:samples_per_combo]
        else:
            combined = generator.generate_combo(combo, samples_per_combo)

        # Apply residual transformation if requested
        if use_residual:
            combined = convert_to_residual(combined, baseline)

        windows = create_windows(combined, window_size)
        label = combo_to_label(combo)

        for w in windows:
            all_windows.append(w)
            all_labels.append(label)

    X = np.array(all_windows)
    y = np.array(all_labels)

    # Compute normalization stats
    norm_stats = {
        'mean': X.reshape(-1, 9).mean(axis=0).tolist(),
        'std': X.reshape(-1, 9).std(axis=0).tolist()
    }

    # Shuffle and split
    indices = np.arange(len(X))
    np.random.shuffle(indices)

    train_end = int(0.7 * len(X))
    val_end = int(0.85 * len(X))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Normalize
    mean = np.array(norm_stats['mean'])
    std = np.array(norm_stats['std'])
    std[std < 1e-6] = 1  # Prevent division by zero

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_val, y_val, X_test, y_test, norm_stats


# ============================================================================
# MODEL
# ============================================================================

def build_model(input_shape: Tuple[int, int], n_outputs: int = 5) -> keras.Model:
    """Build CNN-LSTM hybrid model."""
    inputs = keras.layers.Input(shape=input_shape)

    x = keras.layers.Conv1D(32, 5, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)

    x = keras.layers.Conv1D(64, 5, activation='relu', padding='same')(x)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)

    x = keras.layers.LSTM(32, return_sequences=False)(x)
    x = keras.layers.Dropout(0.3)(x)

    x = keras.layers.Dense(32, activation='relu')(x)
    outputs = keras.layers.Dense(n_outputs, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================================
# TRAINING AND COMPARISON
# ============================================================================

def train_and_evaluate(X_train, y_train, X_val, y_val, X_test, y_test, name: str):
    """Train model and return evaluation metrics."""
    print(f"\n--- Training: {name} ---")

    model = build_model(input_shape=(50, 9), n_outputs=5)

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5)
    ]

    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=0
    )

    # Evaluate
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)

    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger_acc = {}
    for i, finger in enumerate(fingers):
        acc = np.mean(y_pred[:, i] == y_test[:, i])
        per_finger_acc[finger] = float(acc)

    overall_acc = np.mean(y_pred == y_test)

    return {
        'name': name,
        'overall_acc': overall_acc,
        'per_finger': per_finger_acc,
        'epochs': len(history.history['loss']),
        'best_val_loss': min(history.history['val_loss'])
    }


def main():
    # Load data
    print("\n--- Loading Data ---")
    real_data = load_dec31_session()
    print(f"Loaded {len(real_data)} finger state combinations")

    # Compute baseline
    print("\n--- Computing Baseline ---")
    baseline = compute_baseline(real_data)

    # Analyze raw vs residual
    analyze_raw_vs_residual(real_data, baseline)

    # Prepare datasets
    print("\n" + "=" * 70)
    print("TRAINING COMPARISON: RAW vs RESIDUAL")
    print("=" * 70)

    print("\n--- Preparing Raw Dataset ---")
    X_train_raw, y_train_raw, X_val_raw, y_val_raw, X_test_raw, y_test_raw, stats_raw = \
        prepare_dataset(real_data, baseline, use_residual=False, samples_per_combo=500)
    print(f"Train: {len(X_train_raw)}, Val: {len(X_val_raw)}, Test: {len(X_test_raw)}")

    print("\n--- Preparing Residual Dataset ---")
    X_train_res, y_train_res, X_val_res, y_val_res, X_test_res, y_test_res, stats_res = \
        prepare_dataset(real_data, baseline, use_residual=True, samples_per_combo=500)
    print(f"Train: {len(X_train_res)}, Val: {len(X_val_res)}, Test: {len(X_test_res)}")

    # Compare normalization stats
    print("\n--- Normalization Stats Comparison ---")
    print(f"\n{'Feature':<6} {'Raw Mean':>12} {'Raw Std':>12} {'Res Mean':>12} {'Res Std':>12}")
    print("-" * 60)
    feature_names = ['ax', 'ay', 'az', 'gx', 'gy', 'gz', 'mx', 'my', 'mz']
    for i, feat in enumerate(feature_names):
        print(f"{feat:<6} {stats_raw['mean'][i]:>12.2f} {stats_raw['std'][i]:>12.2f} "
              f"{stats_res['mean'][i]:>12.2f} {stats_res['std'][i]:>12.2f}")

    # Train and compare
    results_raw = train_and_evaluate(X_train_raw, y_train_raw, X_val_raw, y_val_raw,
                                     X_test_raw, y_test_raw, "Raw Magnetometer")

    results_res = train_and_evaluate(X_train_res, y_train_res, X_val_res, y_val_res,
                                     X_test_res, y_test_res, "Magnetic Residual")

    # Print comparison
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    print(f"\n{'Metric':<20} {'Raw':>15} {'Residual':>15} {'Δ':>12}")
    print("-" * 65)

    print(f"{'Overall Accuracy':<20} {results_raw['overall_acc']:>14.1%} "
          f"{results_res['overall_acc']:>14.1%} "
          f"{(results_res['overall_acc'] - results_raw['overall_acc']):>+11.1%}")

    print(f"{'Best Val Loss':<20} {results_raw['best_val_loss']:>15.4f} "
          f"{results_res['best_val_loss']:>15.4f} "
          f"{(results_res['best_val_loss'] - results_raw['best_val_loss']):>+12.4f}")

    print(f"{'Epochs Trained':<20} {results_raw['epochs']:>15} "
          f"{results_res['epochs']:>15}")

    print(f"\n{'Finger':<20} {'Raw':>15} {'Residual':>15} {'Δ':>12}")
    print("-" * 65)

    for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        raw_acc = results_raw['per_finger'][finger]
        res_acc = results_res['per_finger'][finger]
        delta = res_acc - raw_acc
        print(f"{finger:<20} {raw_acc:>14.1%} {res_acc:>14.1%} {delta:>+11.1%}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results_res['overall_acc'] > results_raw['overall_acc']:
        print(f"\n✓ Residual approach IMPROVES accuracy by "
              f"{(results_res['overall_acc'] - results_raw['overall_acc'])*100:.1f}%")
    elif results_res['overall_acc'] < results_raw['overall_acc']:
        print(f"\n✗ Residual approach DECREASES accuracy by "
              f"{(results_raw['overall_acc'] - results_res['overall_acc'])*100:.1f}%")
    else:
        print("\n≈ Residual approach has SIMILAR accuracy")

    print(f"\nKey insight: Magnetic residual stats show:")
    print(f"  - Baseline: [{baseline[0]:.1f}, {baseline[1]:.1f}, {baseline[2]:.1f}] μT")
    print(f"  - Residual mean mx: {stats_res['mean'][6]:.1f} μT (vs raw {stats_raw['mean'][6]:.1f})")
    print(f"  - Residual mean my: {stats_res['mean'][7]:.1f} μT (vs raw {stats_raw['mean'][7]:.1f})")
    print(f"  - Residual mean mz: {stats_res['mean'][8]:.1f} μT (vs raw {stats_raw['mean'][8]:.1f})")


if __name__ == '__main__':
    main()
