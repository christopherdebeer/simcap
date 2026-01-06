#!/usr/bin/env python3
"""
Orientation-Aware Training with Calibrated Residuals.

Improvements over train_improved_hybrid.py:
1. Uses pre-calibrated residual_mx/my/mz (iron + earth subtracted)
2. Transforms residuals to world frame for orientation-invariance
3. Compares sensor-frame vs world-frame approaches

Key insight from GAMBIT calibration research:
- residual = iron_corrected - R × earth_world (already orientation-compensated for earth)
- But magnet signal ALSO rotates with orientation
- World-frame transform: residual_world = R^T × residual_sensor

Builds on: train_improved_hybrid.py (95.9% baseline)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import tensorflow as tf
from tensorflow import keras

print("=" * 70)
print("ORIENTATION-AWARE TRAINING")
print("Using calibrated residuals + world-frame transformation")
print("=" * 70)


def quaternion_to_rotation_matrix(q: np.ndarray) -> np.ndarray:
    """Convert quaternion [w,x,y,z] to 3x3 rotation matrix."""
    w, x, y, z = q

    # Rotation matrix from quaternion
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    return R


def sensor_to_world_frame(residual: np.ndarray, quaternion: np.ndarray) -> np.ndarray:
    """Transform sensor-frame residual to world frame using quaternion."""
    R = quaternion_to_rotation_matrix(quaternion)
    # R^T transforms from sensor frame to world frame
    return R.T @ residual


def load_session_data_with_orientation() -> Tuple[Dict, Dict, np.ndarray]:
    """Load session data with calibrated residuals and orientation."""
    session_path = Path(__file__).parent.parent / 'data' / 'GAMBIT' / '2025-12-31T14_06_18.270Z.json'
    with open(session_path, 'r') as f:
        data = json.load(f)

    # Compute baseline from open palm (eeeee) samples
    baseline_mags = []
    for lbl in data['labels']:
        if 'labels' in lbl and isinstance(lbl['labels'], dict):
            fingers = lbl['labels'].get('fingers', {})
            start, end = lbl.get('start_sample', 0), lbl.get('end_sample', 0)
        else:
            fingers = lbl.get('fingers', {})
            start, end = lbl.get('startIndex', 0), lbl.get('endIndex', 0)

        combo = ''.join(['e' if fingers.get(f, '?') == 'extended' else 'f' if fingers.get(f, '?') == 'flexed' else '?'
                        for f in ['thumb', 'index', 'middle', 'ring', 'pinky']])

        if combo == 'eeeee':
            for s in data['samples'][start:end]:
                if 'mx_ut' in s:
                    baseline_mags.append([s['mx_ut'], s['my_ut'], s['mz_ut']])

    baseline = np.mean(baseline_mags, axis=0) if baseline_mags else np.array([46.0, -45.8, 31.3])
    print(f"Baseline (open palm): [{baseline[0]:.2f}, {baseline[1]:.2f}, {baseline[2]:.2f}] μT")

    # Extract per-combo samples with multiple residual types
    combo_data = {}  # combo -> {'sensor': [], 'world': [], 'calibrated': []}

    for lbl in data['labels']:
        if 'labels' in lbl and isinstance(lbl['labels'], dict):
            fingers = lbl['labels'].get('fingers', {})
            start, end = lbl.get('start_sample', 0), lbl.get('end_sample', 0)
        else:
            fingers = lbl.get('fingers', {})
            start, end = lbl.get('startIndex', 0), lbl.get('endIndex', 0)

        if not fingers or all(v == 'unknown' for v in fingers.values()):
            continue

        combo = ''.join(['e' if fingers.get(f, '?') == 'extended' else 'f' if fingers.get(f, '?') == 'flexed' else '?'
                        for f in ['thumb', 'index', 'middle', 'ring', 'pinky']])

        if combo not in combo_data:
            combo_data[combo] = {'sensor': [], 'world': [], 'calibrated': [], 'raw': []}

        for s in data['samples'][start:end]:
            # Get quaternion
            if 'orientation_w' not in s:
                continue
            quat = np.array([s['orientation_w'], s['orientation_x'],
                           s['orientation_y'], s['orientation_z']])

            # Method 1: Raw residual (current approach)
            if 'mx_ut' in s:
                raw = np.array([s['mx_ut'], s['my_ut'], s['mz_ut']])
                raw_residual = raw - baseline
                combo_data[combo]['raw'].append(raw_residual)

                # Transform raw to world frame
                world_raw = sensor_to_world_frame(raw_residual, quat)
                combo_data[combo]['sensor'].append(raw_residual)
                combo_data[combo]['world'].append(world_raw)

            # Method 2: Pre-calibrated residual
            if 'residual_mx' in s and s['residual_mx'] is not None:
                cal = np.array([s['residual_mx'], s['residual_my'], s['residual_mz']])
                combo_data[combo]['calibrated'].append(cal)

    # Convert to arrays and compute stats
    combo_stats = {}
    for combo, samples in combo_data.items():
        combo_stats[combo] = {}
        for key in ['sensor', 'world', 'calibrated', 'raw']:
            if samples[key]:
                arr = np.array(samples[key])
                combo_stats[combo][key] = {
                    'mean': arr.mean(axis=0),
                    'std': arr.std(axis=0),
                    'n': len(arr)
                }

    return combo_data, combo_stats, baseline


def print_residual_comparison(combo_stats: Dict):
    """Compare residual approaches per combo."""
    print("\n" + "=" * 70)
    print("RESIDUAL COMPARISON: Sensor-frame vs World-frame vs Calibrated")
    print("=" * 70)

    print(f"\n{'Combo':<8} {'N':>5} | {'Sensor (raw-base)':<30} | {'World-frame':<30} | {'Calibrated':<30}")
    print("-" * 110)

    for combo in sorted(combo_stats.keys()):
        stats = combo_stats[combo]
        n = stats.get('sensor', {}).get('n', 0)

        sensor_mean = stats.get('sensor', {}).get('mean', np.zeros(3))
        world_mean = stats.get('world', {}).get('mean', np.zeros(3))
        cal_mean = stats.get('calibrated', {}).get('mean', np.zeros(3))

        sensor_str = f"[{sensor_mean[0]:6.1f}, {sensor_mean[1]:6.1f}, {sensor_mean[2]:6.1f}]"
        world_str = f"[{world_mean[0]:6.1f}, {world_mean[1]:6.1f}, {world_mean[2]:6.1f}]"
        cal_str = f"[{cal_mean[0]:6.1f}, {cal_mean[1]:6.1f}, {cal_mean[2]:6.1f}]"

        print(f"{combo:<8} {n:>5} | {sensor_str:<30} | {world_str:<30} | {cal_str:<30}")

    # Compute separability metric for each approach
    print("\n" + "=" * 70)
    print("SEPARABILITY ANALYSIS")
    print("=" * 70)

    for approach in ['sensor', 'world', 'calibrated']:
        means = []
        for combo in sorted(combo_stats.keys()):
            if approach in combo_stats[combo]:
                means.append(combo_stats[combo][approach]['mean'])

        if len(means) > 1:
            means = np.array(means)
            # Compute pairwise distances
            distances = []
            for i in range(len(means)):
                for j in range(i+1, len(means)):
                    dist = np.linalg.norm(means[i] - means[j])
                    distances.append(dist)

            mean_dist = np.mean(distances)
            min_dist = np.min(distances)
            print(f"  {approach:12}: Mean pairwise distance = {mean_dist:.2f} μT, Min = {min_dist:.2f} μT")


def prepare_training_data(combo_data: Dict, approach: str = 'sensor') -> Tuple[np.ndarray, np.ndarray]:
    """Prepare X, y for training with specified approach."""
    X_list = []
    y_list = []

    for combo, samples in combo_data.items():
        if approach not in samples or not samples[approach]:
            continue

        residuals = np.array(samples[approach])
        labels = np.array([[1.0 if c == 'f' else 0.0 for c in combo]] * len(residuals))

        X_list.append(residuals)
        y_list.append(labels)

    X = np.vstack(X_list)
    y = np.vstack(y_list)

    return X, y


def create_model(input_dim: int = 3) -> keras.Model:
    """Create simple MLP for residual classification."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(5, activation='sigmoid')
    ])
    model.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def train_and_evaluate(X: np.ndarray, y: np.ndarray, name: str) -> Dict:
    """Train model and return metrics."""
    # Split data
    n = len(X)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = indices[:split], indices[split:]

    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]

    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    # Train
    model = create_model(X.shape[1])
    model.fit(X_train_norm, y_train, epochs=50, batch_size=32,
              validation_split=0.1, verbose=0)

    # Evaluate
    y_pred = (model.predict(X_test_norm, verbose=0) > 0.5).astype(int)

    # Per-finger accuracy
    finger_acc = {}
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    for i, name_f in enumerate(finger_names):
        finger_acc[name_f] = (y_pred[:, i] == y_test[:, i]).mean()

    # Full combo accuracy
    combo_acc = (y_pred == y_test).all(axis=1).mean()

    return {
        'name': name,
        'combo_accuracy': float(combo_acc),
        'per_finger': finger_acc,
        'n_train': len(X_train),
        'n_test': len(X_test)
    }


def load_interaction_model() -> Tuple[Dict[str, np.ndarray], float]:
    """Load fitted interaction model from prior analysis."""
    model_path = Path(__file__).parent / 'per_finger_fit_results.json'
    with open(model_path, 'r') as f:
        data = json.load(f)

    fitted = data['interaction_model']['fitted_effects']
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

    if isinstance(fitted, list):
        effects = {name: np.array(fitted[i]) for i, name in enumerate(finger_names)}
    else:
        effects = {name: np.array(fitted[name]) for name in finger_names}

    interaction = data['interaction_model']['interaction_strength']
    return effects, interaction


def generate_synthetic_world_frame(
    effects: Dict[str, np.ndarray],
    interaction: float,
    combo_stats: Dict,
    n_samples: int = 200
) -> Tuple[np.ndarray, np.ndarray]:
    """Generate synthetic samples in world frame with calibrated noise."""
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    observed_combos = set(combo_stats.keys())

    X_list = []
    y_list = []

    # Generate all 32 combos
    for i in range(32):
        combo = ''.join(['f' if (i >> (4-j)) & 1 else 'e' for j in range(5)])

        # Skip observed combos (use real data instead)
        if combo in observed_combos:
            continue

        # Compute synthetic mean using interaction model
        n_flexed = combo.count('f')
        if n_flexed == 0:
            mean = np.zeros(3)
        else:
            scaling = 1.0 + interaction * (n_flexed - 1) / 4
            mean = np.zeros(3)
            for j, finger in enumerate(finger_names):
                if combo[j] == 'f':
                    mean += effects[finger] * scaling

        # Get noise from nearest observed combo
        min_dist = float('inf')
        nearest_combo = 'eeeee'
        for obs_combo in observed_combos:
            dist = sum(c1 != c2 for c1, c2 in zip(combo, obs_combo))
            if dist < min_dist:
                min_dist = dist
                nearest_combo = obs_combo

        # Use world-frame stats if available, else sensor
        if 'world' in combo_stats[nearest_combo]:
            std = combo_stats[nearest_combo]['world']['std']
        else:
            std = combo_stats[nearest_combo]['sensor']['std']

        # Scale noise by distance (more uncertainty for distant combos)
        std = std * (1.0 + 0.2 * min_dist)

        # Generate samples
        samples = np.random.normal(mean, std, size=(n_samples, 3))
        labels = np.array([[1.0 if c == 'f' else 0.0 for c in combo]] * n_samples)

        X_list.append(samples)
        y_list.append(labels)

    if X_list:
        return np.vstack(X_list), np.vstack(y_list)
    return np.array([]).reshape(0, 3), np.array([]).reshape(0, 5)


def main():
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load data
    combo_data, combo_stats, baseline = load_session_data_with_orientation()

    print(f"\nObserved combos: {len(combo_stats)}")
    for combo in sorted(combo_stats.keys()):
        n = combo_stats[combo].get('sensor', {}).get('n', 0)
        print(f"  {combo}: {n} samples")

    # Compare residual approaches
    print_residual_comparison(combo_stats)

    # Train models with different approaches
    print("\n" + "=" * 70)
    print("TRAINING COMPARISON")
    print("=" * 70)

    results = {}

    for approach in ['sensor', 'world', 'calibrated']:
        X, y = prepare_training_data(combo_data, approach)
        if len(X) > 0:
            result = train_and_evaluate(X, y, f"Real-only ({approach})")
            results[approach] = result
            print(f"\n{approach.upper()} FRAME (Real Data Only)")
            print(f"  Combo Accuracy: {result['combo_accuracy']*100:.1f}%")
            print(f"  Per-finger: " + ", ".join(f"{k}={v*100:.1f}%" for k, v in result['per_finger'].items()))

    # Hybrid training with world-frame synthetic
    print("\n" + "-" * 70)
    print("HYBRID: Real (world-frame) + Synthetic (world-frame)")
    print("-" * 70)

    effects, interaction = load_interaction_model()
    X_real, y_real = prepare_training_data(combo_data, 'world')
    X_synth, y_synth = generate_synthetic_world_frame(effects, interaction, combo_stats)

    if len(X_synth) > 0:
        X_hybrid = np.vstack([X_real, X_synth])
        y_hybrid = np.vstack([y_real, y_synth])

        print(f"  Real samples: {len(X_real)}")
        print(f"  Synthetic samples: {len(X_synth)}")

        result = train_and_evaluate(X_hybrid, y_hybrid, "Hybrid (world-frame)")
        results['hybrid_world'] = result
        print(f"\n  Combo Accuracy: {result['combo_accuracy']*100:.1f}%")
        print(f"  Per-finger: " + ", ".join(f"{k}={v*100:.1f}%" for k, v in result['per_finger'].items()))

    # Save results
    output = {
        'baseline': baseline.tolist(),
        'observed_combos': list(combo_stats.keys()),
        'results': results,
        'comparison': {
            'previous_baseline': 0.959,  # From train_improved_hybrid.py
            'sensor_vs_world': (results.get('world', {}).get('combo_accuracy', 0) -
                               results.get('sensor', {}).get('combo_accuracy', 0))
        }
    }

    output_path = Path(__file__).parent / 'orientation_aware_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=lambda x: x.tolist() if hasattr(x, 'tolist') else x)

    print(f"\n✓ Results saved to {output_path}")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Previous baseline (calibrated hybrid): 95.9%")
    if 'sensor' in results:
        print(f"Sensor-frame (real only):              {results['sensor']['combo_accuracy']*100:.1f}%")
    if 'world' in results:
        print(f"World-frame (real only):               {results['world']['combo_accuracy']*100:.1f}%")
    if 'calibrated' in results:
        print(f"Calibrated residual (real only):       {results['calibrated']['combo_accuracy']*100:.1f}%")
    if 'hybrid_world' in results:
        print(f"Hybrid world-frame:                    {results['hybrid_world']['combo_accuracy']*100:.1f}%")


if __name__ == '__main__':
    main()
