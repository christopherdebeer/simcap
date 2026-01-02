#!/usr/bin/env python3
"""
Hybrid Training: Real Data + Nearest-Neighbor Synthetic Data.

Goal: Maximize real-world accuracy by:
1. Using real data for observed combos (ground truth)
2. Using NN-interpolated synthetic for unseen combos (domain coverage)
3. Proper train/test splits for honest evaluation
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
import tensorflow as tf
from tensorflow import keras


def train_test_split_simple(indices: np.ndarray, test_size: float = 0.2,
                           random_state: int = 42) -> Tuple[np.ndarray, np.ndarray]:
    """Simple train/test split without sklearn."""
    np.random.seed(random_state)
    shuffled = np.random.permutation(indices)
    split_idx = int(len(shuffled) * (1 - test_size))
    return shuffled[:split_idx], shuffled[split_idx:]

print("=" * 70)
print("HYBRID TRAINING: Real + Nearest-Neighbor Synthetic")
print("=" * 70)


@dataclass
class ComboData:
    """Raw samples for a finger combination."""
    combo: str
    residuals: np.ndarray  # Shape: (n_samples, 3)
    is_observed: bool = True


def hamming_distance(a: str, b: str) -> int:
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def load_real_data() -> Tuple[Dict[str, ComboData], np.ndarray]:
    """Load real data with raw samples (not just statistics)."""
    session_path = Path(__file__).parent.parent / 'data' / 'GAMBIT' / '2025-12-31T14_06_18.270Z.json'
    with open(session_path, 'r') as f:
        data = json.load(f)

    baseline_mags = []
    combo_samples = {}

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

        segment = data['samples'][start:end]
        if len(segment) < 5:
            continue

        mags = []
        for s in segment:
            if 'mx_ut' in s:
                mx, my, mz = s['mx_ut'], s['my_ut'], s['mz_ut']
            else:
                mx, my, mz = s.get('mx', 0)/10.24, s.get('my', 0)/10.24, s.get('mz', 0)/10.24
            mags.append([mx, my, mz])

        if combo not in combo_samples:
            combo_samples[combo] = []
        combo_samples[combo].extend(mags)

        if combo == 'eeeee':
            baseline_mags.extend(mags)

    baseline = np.mean(baseline_mags, axis=0)

    # Convert to residuals
    result = {}
    for combo, mags in combo_samples.items():
        mags = np.array(mags)
        residuals = mags - baseline
        result[combo] = ComboData(combo=combo, residuals=residuals, is_observed=True)

    return result, baseline


def generate_synthetic_for_missing(observed: Dict[str, ComboData], n_per_combo: int = 200) -> Dict[str, ComboData]:
    """Generate synthetic data for unobserved combos using NN interpolation."""
    all_combos = [f"{t}{i}{m}{r}{p}" for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef']
    observed_combos = list(observed.keys())

    result = {}
    for combo in all_combos:
        if combo in observed:
            continue

        # Find k nearest observed combos
        distances = [(obs, hamming_distance(combo, obs)) for obs in observed_combos]
        distances.sort(key=lambda x: x[1])

        k = min(3, len(distances))
        nearest = distances[:k]

        # Inverse distance weighting
        weights = [1.0 / (d + 0.5) for _, d in nearest]
        total_weight = sum(weights)
        weights = [w / total_weight for w in weights]

        # Compute weighted mean and std
        mean = np.zeros(3)
        std = np.zeros(3)
        for (obs_combo, _), w in zip(nearest, weights):
            obs = observed[obs_combo]
            mean += w * obs.residuals.mean(axis=0)
            std += w * obs.residuals.std(axis=0)

        # Add extra uncertainty for interpolated combos
        uncertainty = 1.0 + 0.3 * min(nearest, key=lambda x: x[1])[1]
        std *= uncertainty

        # Generate samples
        samples = np.random.normal(mean, std, size=(n_per_combo, 3))
        result[combo] = ComboData(combo=combo, residuals=samples, is_observed=False)

    return result


def combo_to_labels(combo: str) -> np.ndarray:
    """Convert combo string to binary labels."""
    return np.array([1.0 if c == 'f' else 0.0 for c in combo])


def prepare_dataset(observed: Dict[str, ComboData],
                   synthetic: Dict[str, ComboData],
                   test_size: float = 0.2) -> Tuple:
    """Prepare train/test splits from real and synthetic data."""

    # Split observed data into train/test
    X_train_real = []
    y_train_real = []
    X_test_real = []
    y_test_real = []

    for combo, data in observed.items():
        labels = combo_to_labels(combo)
        n = len(data.residuals)

        # Split per combo
        train_idx, test_idx = train_test_split_simple(
            np.arange(n), test_size=test_size, random_state=42
        )

        X_train_real.append(data.residuals[train_idx])
        y_train_real.append(np.tile(labels, (len(train_idx), 1)))
        X_test_real.append(data.residuals[test_idx])
        y_test_real.append(np.tile(labels, (len(test_idx), 1)))

    # Add synthetic data to training only
    X_train_synth = []
    y_train_synth = []
    for combo, data in synthetic.items():
        labels = combo_to_labels(combo)
        X_train_synth.append(data.residuals)
        y_train_synth.append(np.tile(labels, (len(data.residuals), 1)))

    # Combine
    X_train = np.vstack(X_train_real + X_train_synth)
    y_train = np.vstack(y_train_real + y_train_synth)
    X_test = np.vstack(X_test_real)
    y_test = np.vstack(y_test_real)

    # Shuffle training
    idx = np.random.permutation(len(X_train))
    X_train = X_train[idx]
    y_train = y_train[idx]

    return X_train, y_train, X_test, y_test


def build_model(input_dim: int = 3, output_dim: int = 5) -> keras.Model:
    """Build the classifier model."""
    model = keras.Sequential([
        keras.layers.Input(shape=(input_dim,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(output_dim, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


def evaluate_model(model: keras.Model, X: np.ndarray, y: np.ndarray,
                  mean: np.ndarray, std: np.ndarray, name: str) -> Dict:
    """Evaluate model and return metrics."""
    X_norm = (X - mean) / std
    pred = model.predict(X_norm, verbose=0)
    binary = (pred > 0.5).astype(int)

    overall = np.mean((binary == y).all(axis=1))

    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger = {f: np.mean(binary[:, i] == y[:, i]) for i, f in enumerate(fingers)}

    print(f"\n{name}:")
    print(f"  Overall: {overall*100:.1f}%")
    print(f"  Per-finger: {', '.join(f'{f}={a*100:.0f}%' for f, a in per_finger.items())}")

    return {'overall': overall, 'per_finger': per_finger}


def run_experiment(name: str, use_synthetic: bool, synth_weight: float = 1.0):
    """Run a training experiment."""
    print(f"\n{'='*60}")
    print(f"EXPERIMENT: {name}")
    print(f"{'='*60}")

    # Load data
    observed, baseline = load_real_data()
    print(f"Loaded {len(observed)} observed combos")

    if use_synthetic:
        synthetic = generate_synthetic_for_missing(observed, n_per_combo=int(200 * synth_weight))
        print(f"Generated synthetic for {len(synthetic)} missing combos")
    else:
        synthetic = {}

    # Prepare dataset
    X_train, y_train, X_test, y_test = prepare_dataset(observed, synthetic)
    print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")

    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    # Train model
    model = build_model()
    print("\nTraining...")
    history = model.fit(
        X_train_norm, y_train,
        epochs=100,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )

    # Evaluate
    train_results = evaluate_model(model, X_train[:len(y_test)], y_train[:len(y_test)], mean, std, "Training (subset)")
    test_results = evaluate_model(model, X_test, y_test, mean, std, "Test (Real Data)")

    return {
        'name': name,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'test_accuracy': test_results['overall'],
        'per_finger': test_results['per_finger']
    }


def main():
    results = []

    # Experiment 1: Real data only
    results.append(run_experiment(
        "Real Data Only",
        use_synthetic=False
    ))

    # Experiment 2: Real + NN Synthetic (1x)
    results.append(run_experiment(
        "Real + NN Synthetic (1x)",
        use_synthetic=True,
        synth_weight=1.0
    ))

    # Experiment 3: Real + NN Synthetic (2x)
    results.append(run_experiment(
        "Real + NN Synthetic (2x)",
        use_synthetic=True,
        synth_weight=2.0
    ))

    # Experiment 4: Real + NN Synthetic (0.5x)
    results.append(run_experiment(
        "Real + NN Synthetic (0.5x)",
        use_synthetic=True,
        synth_weight=0.5
    ))

    # Summary
    print("\n" + "=" * 70)
    print("EXPERIMENT SUMMARY")
    print("=" * 70)
    print(f"\n{'Experiment':<30} {'Train':>10} {'Test Acc':>12}")
    print("-" * 55)
    for r in results:
        print(f"{r['name']:<30} {r['train_samples']:>10} {r['test_accuracy']*100:>11.1f}%")

    # Find best
    best = max(results, key=lambda x: x['test_accuracy'])
    print(f"\nBest: {best['name']} with {best['test_accuracy']*100:.1f}% test accuracy")

    # Save results
    output_path = Path(__file__).parent / 'hybrid_training_results.json'
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
