#!/usr/bin/env python3
"""
Train Finger State Classifier with Synthetic Data Augmentation

This script:
1. Loads labeled data from the Dec 31 session
2. Calibrates simulation to match observed magnetometer characteristics
3. Generates synthetic data for missing finger state combinations
4. Trains a model on combined real + synthetic data
5. Evaluates accuracy on both synthetic and real ground truth

Usage:
    python -m ml.train_with_synthetic
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from collections import defaultdict

# Check for tensorflow
try:
    import tensorflow as tf
    from tensorflow import keras
    HAS_TF = True
except ImportError:
    HAS_TF = False
    print("TensorFlow not available - will generate data only")


# ============================================================================
# DATA LOADING
# ============================================================================

@dataclass
class LabeledSegment:
    """A segment of labeled sensor data."""
    combo: str  # e.g., 'eeeee' for all extended
    fingers: Dict[str, str]  # {'thumb': 'extended', ...}
    samples: np.ndarray  # Shape (n, 9) - ax,ay,az,gx,gy,gz,mx,my,mz
    magnitudes: np.ndarray  # Magnetometer magnitudes


def load_dec31_session() -> Tuple[List[LabeledSegment], Dict]:
    """Load and parse the December 31 labeled session."""
    session_path = Path(__file__).parent.parent / 'data' / 'GAMBIT' / '2025-12-31T14_06_18.270Z.json'

    with open(session_path, 'r') as f:
        data = json.load(f)

    samples = data['samples']
    labels = data['labels']

    segments = []

    for lbl in labels:
        # Handle both label formats
        if 'labels' in lbl and isinstance(lbl['labels'], dict):
            fingers = lbl['labels'].get('fingers', {})
            start = lbl.get('start_sample', 0)
            end = lbl.get('end_sample', 0)
        else:
            fingers = lbl.get('fingers', {})
            start = lbl.get('startIndex', 0)
            end = lbl.get('endIndex', 0)

        # Skip if no valid labels
        if not fingers or all(v == 'unknown' for v in fingers.values()):
            continue

        segment_samples = samples[start:end]
        if len(segment_samples) < 5:
            continue

        # Extract sensor data
        sensor_data = []
        mag_data = []

        for s in segment_samples:
            # Get IMU data
            ax = s.get('ax', 0) / 8192.0  # Convert to g
            ay = s.get('ay', 0) / 8192.0
            az = s.get('az', 0) / 8192.0
            gx = s.get('gx', 0) / 114.28  # Convert to dps
            gy = s.get('gy', 0) / 114.28
            gz = s.get('gz', 0) / 114.28

            # Get magnetometer data
            if 'mx_ut' in s:
                mx, my, mz = s['mx_ut'], s['my_ut'], s['mz_ut']
            else:
                mx = s.get('mx', 0) / 10.24
                my = s.get('my', 0) / 10.24
                mz = s.get('mz', 0) / 10.24

            sensor_data.append([ax, ay, az, gx, gy, gz, mx, my, mz])
            mag_data.append(np.sqrt(mx**2 + my**2 + mz**2))

        if not sensor_data:
            continue

        # Create combo string
        combo = ''.join([
            'e' if fingers.get(f, '?') == 'extended' else
            'f' if fingers.get(f, '?') == 'flexed' else '?'
            for f in ['thumb', 'index', 'middle', 'ring', 'pinky']
        ])

        segments.append(LabeledSegment(
            combo=combo,
            fingers=fingers,
            samples=np.array(sensor_data),
            magnitudes=np.array(mag_data)
        ))

    # Compute statistics
    stats = {}
    for seg in segments:
        if seg.combo not in stats:
            stats[seg.combo] = {'n': 0, 'mags': []}
        stats[seg.combo]['n'] += len(seg.magnitudes)
        stats[seg.combo]['mags'].extend(seg.magnitudes.tolist())

    return segments, stats


# ============================================================================
# SYNTHETIC DATA GENERATION
# ============================================================================

def generate_synthetic_samples(
    combo: str,
    n_samples: int,
    baseline_mean: float = 74.0,
    baseline_std: float = 20.0,
    flexed_scale: float = 10.0,  # How much each flexed finger adds
) -> np.ndarray:
    """
    Generate synthetic sensor samples for a finger state combination.

    The model assumes:
    - Baseline (all extended) is near Earth field (~74 μT based on real data)
    - Each flexed finger adds field due to magnet proximity
    - Multiple flexed fingers combine (with some interaction)

    Args:
        combo: Finger state code (e.g., 'eefff')
        n_samples: Number of samples to generate
        baseline_mean: Baseline magnetometer magnitude
        baseline_std: Baseline standard deviation
        flexed_scale: Additional field per flexed finger

    Returns:
        Array of shape (n_samples, 9) with sensor data
    """
    # Count flexed fingers
    n_flexed = combo.count('f')

    # Model the magnitude based on flexed fingers
    # Use a nonlinear model since multiple magnets interact
    if n_flexed == 0:
        mag_mean = baseline_mean
        mag_std = baseline_std
    else:
        # Each flexed finger adds field, with diminishing returns
        mag_mean = baseline_mean + flexed_scale * n_flexed * (1 + 0.5 * n_flexed)
        mag_std = baseline_std * (1 + 0.3 * n_flexed)

    # Generate magnetometer data with realistic noise
    magnitudes = np.random.normal(mag_mean, mag_std, n_samples)
    magnitudes = np.clip(magnitudes, 30, 500)  # Reasonable range

    # Random direction for field (varies with hand orientation)
    theta = np.random.uniform(0, 2*np.pi, n_samples)
    phi = np.random.uniform(0, np.pi, n_samples)

    mx = magnitudes * np.sin(phi) * np.cos(theta)
    my = magnitudes * np.sin(phi) * np.sin(theta)
    mz = magnitudes * np.cos(phi)

    # Generate IMU data (static hand assumption)
    # Accelerometer shows gravity (~1g in Z)
    ax = np.random.normal(0, 0.02, n_samples)
    ay = np.random.normal(0, 0.02, n_samples)
    az = np.random.normal(-1, 0.02, n_samples)  # Gravity

    # Gyroscope (mostly zero for static)
    gx = np.random.normal(0, 1.0, n_samples)
    gy = np.random.normal(0, 1.0, n_samples)
    gz = np.random.normal(0, 1.0, n_samples)

    return np.column_stack([ax, ay, az, gx, gy, gz, mx, my, mz])


def calibrate_and_generate(
    real_stats: Dict,
    missing_combos: List[str],
    samples_per_combo: int = 200
) -> Dict[str, np.ndarray]:
    """
    Calibrate synthetic generation from real data and generate missing combos.

    Args:
        real_stats: Statistics from real labeled data
        missing_combos: List of missing finger state combinations
        samples_per_combo: How many samples to generate per combo

    Returns:
        Dict mapping combo to synthetic samples array
    """
    # Get baseline from real data (all extended)
    if 'eeeee' in real_stats:
        baseline_mags = np.array(real_stats['eeeee']['mags'])
        baseline_mean = np.mean(baseline_mags)
        baseline_std = np.std(baseline_mags)
    else:
        baseline_mean = 74.0
        baseline_std = 20.0

    print(f"Calibration baseline (eeeee): mean={baseline_mean:.1f}, std={baseline_std:.1f}")

    # Estimate flexed_scale from real data
    # Compare baseline to single-finger-flexed states
    single_flexed = ['feeee', 'efeee', 'eefee', 'eeefe', 'eeeef']
    single_mags = []
    for combo in single_flexed:
        if combo in real_stats:
            single_mags.extend(real_stats[combo]['mags'])

    if single_mags:
        single_mean = np.mean(single_mags)
        flexed_scale = (single_mean - baseline_mean) / 1.5  # Adjust for model
    else:
        flexed_scale = 200.0  # Default based on observed data

    print(f"Calibrated flexed_scale: {flexed_scale:.1f}")

    # Generate synthetic data for missing combos
    synthetic = {}
    for combo in missing_combos:
        samples = generate_synthetic_samples(
            combo, samples_per_combo,
            baseline_mean, baseline_std, flexed_scale
        )
        synthetic[combo] = samples
        print(f"  Generated {samples_per_combo} samples for {combo}")

    return synthetic


# ============================================================================
# MODEL TRAINING
# ============================================================================

def create_windows(samples: np.ndarray, window_size: int = 50, stride: int = 25) -> np.ndarray:
    """Create sliding windows from sample data."""
    n_samples = len(samples)
    if n_samples < window_size:
        # Pad if too short
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
    """Convert combo string to binary label array (5 values, 0=extended, 1=flexed)."""
    return np.array([0 if c == 'e' else 1 for c in combo], dtype=np.float32)


def prepare_dataset(
    real_segments: List[LabeledSegment],
    synthetic_data: Dict[str, np.ndarray],
    window_size: int = 50
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare training and test datasets.

    Returns:
        X_train, y_train, X_test, y_test
    """
    all_windows = []
    all_labels = []
    is_synthetic = []

    # Process real data
    for seg in real_segments:
        windows = create_windows(seg.samples, window_size)
        label = combo_to_label(seg.combo)

        for w in windows:
            all_windows.append(w)
            all_labels.append(label)
            is_synthetic.append(False)

    # Process synthetic data
    for combo, samples in synthetic_data.items():
        windows = create_windows(samples, window_size)
        label = combo_to_label(combo)

        for w in windows:
            all_windows.append(w)
            all_labels.append(label)
            is_synthetic.append(True)

    X = np.array(all_windows)
    y = np.array(all_labels)
    synthetic_mask = np.array(is_synthetic)

    # Split: use 80% for training, 20% for test
    # Stratify by ensuring real data is in test set
    n_total = len(X)
    indices = np.arange(n_total)
    np.random.shuffle(indices)

    split_idx = int(0.8 * n_total)
    train_idx = indices[:split_idx]
    test_idx = indices[split_idx:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    # Track which test samples are real vs synthetic
    test_synthetic = synthetic_mask[test_idx]

    return X_train, y_train, X_test, y_test, test_synthetic


def build_model(input_shape: Tuple[int, int], n_outputs: int = 5) -> 'keras.Model':
    """Build a 1D CNN model for finger state classification."""
    model = keras.Sequential([
        keras.layers.Conv1D(32, 5, activation='relu', input_shape=input_shape),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),

        keras.layers.Conv1D(64, 5, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.MaxPooling1D(2),

        keras.layers.Conv1D(64, 3, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.GlobalAveragePooling1D(),

        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dense(n_outputs, activation='sigmoid')  # Multi-label
    ])

    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    return model


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 70)
    print("FINGER STATE CLASSIFIER WITH SYNTHETIC DATA AUGMENTATION")
    print("=" * 70)

    # Load real data
    print("\n--- Loading Dec 31 Labeled Session ---")
    segments, stats = load_dec31_session()

    print(f"Loaded {len(segments)} labeled segments")
    print(f"Unique finger state combos: {len(stats)}")

    for combo, s in sorted(stats.items()):
        mags = np.array(s['mags'])
        print(f"  {combo}: n={s['n']:4d}, mag_mean={np.mean(mags):.1f}")

    # Find missing combos
    all_combos = [f"{t}{i}{m}{r}{p}" for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef']
    present = set(stats.keys())
    missing = [c for c in all_combos if c not in present]

    print(f"\n--- Missing Finger State Combinations ---")
    print(f"Present: {len(present)}/32")
    print(f"Missing: {len(missing)}")

    # Generate synthetic data
    print("\n--- Generating Synthetic Data ---")
    synthetic = calibrate_and_generate(stats, missing, samples_per_combo=200)

    total_synthetic = sum(len(v) for v in synthetic.values())
    print(f"\nGenerated {total_synthetic} synthetic samples for {len(missing)} combos")

    if not HAS_TF:
        print("\n[TensorFlow not available - saving data only]")

        # Save the synthetic data
        output = {
            'real_stats': {k: {'n': v['n'], 'mag_mean': float(np.mean(v['mags']))} for k, v in stats.items()},
            'synthetic_combos': list(synthetic.keys()),
            'n_synthetic_per_combo': 200,
        }

        output_path = Path(__file__).parent / 'synthetic_data_summary.json'
        with open(output_path, 'w') as f:
            json.dump(output, f, indent=2)
        print(f"Saved summary to {output_path}")
        return

    # Prepare dataset
    print("\n--- Preparing Dataset ---")
    X_train, y_train, X_test, y_test, test_synthetic = prepare_dataset(
        segments, synthetic, window_size=50
    )

    print(f"Training: {len(X_train)} windows")
    print(f"Test: {len(X_test)} windows ({sum(~test_synthetic)} real, {sum(test_synthetic)} synthetic)")

    # Build and train model
    print("\n--- Training Model ---")
    model = build_model(input_shape=(50, 9), n_outputs=5)
    model.summary()

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=30,
        batch_size=32,
        verbose=1
    )

    # Evaluate
    print("\n--- Evaluation ---")

    # Overall test accuracy
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"Overall test accuracy: {acc:.3f}")

    # Separate evaluation on real vs synthetic
    real_mask = ~test_synthetic
    synth_mask = test_synthetic

    if sum(real_mask) > 0:
        loss_real, acc_real = model.evaluate(X_test[real_mask], y_test[real_mask], verbose=0)
        print(f"Real data accuracy: {acc_real:.3f} (n={sum(real_mask)})")

    if sum(synth_mask) > 0:
        loss_synth, acc_synth = model.evaluate(X_test[synth_mask], y_test[synth_mask], verbose=0)
        print(f"Synthetic data accuracy: {acc_synth:.3f} (n={sum(synth_mask)})")

    # Per-finger accuracy
    print("\n--- Per-Finger Accuracy ---")
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)

    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    for i, finger in enumerate(fingers):
        acc_i = np.mean(y_pred[:, i] == y_test[:, i])
        print(f"  {finger}: {acc_i:.3f}")

    # Save model
    model_path = Path(__file__).parent / 'models' / 'finger_synthetic_v1'
    model_path.mkdir(parents=True, exist_ok=True)
    model.save(model_path / 'model.keras')
    print(f"\nModel saved to {model_path}")


if __name__ == '__main__':
    main()
