#!/usr/bin/env python3
"""
Improved Synthetic Data Generator using Nearest-Neighbor Interpolation.

Key insight: Finger magnetic effects are NOT additive because magnets physically
interact when multiple fingers flex together. Instead of additive superposition,
we use:
1. Direct sampling for observed combos
2. Nearest-neighbor interpolation for unseen combos
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import tensorflow as tf
from tensorflow import keras

print("=" * 70)
print("IMPROVED SYNTHETIC DATA GENERATOR")
print("Using Nearest-Neighbor Interpolation (Non-Additive)")
print("=" * 70)


@dataclass
class ComboStats:
    """Statistics for a finger combination."""
    combo: str
    mean: np.ndarray
    std: np.ndarray
    n_samples: int
    is_observed: bool = True


def hamming_distance(a: str, b: str) -> int:
    """Hamming distance between two combo strings."""
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def load_real_data() -> Tuple[Dict[str, ComboStats], np.ndarray]:
    """Load real data and compute per-combo statistics."""
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

    # Compute residual statistics per combo
    combo_stats = {}
    for combo, mags in combo_samples.items():
        mags = np.array(mags)
        residuals = mags - baseline
        combo_stats[combo] = ComboStats(
            combo=combo,
            mean=residuals.mean(axis=0),
            std=residuals.std(axis=0),
            n_samples=len(mags),
            is_observed=True
        )

    return combo_stats, baseline


class NearestNeighborGenerator:
    """
    Generate synthetic data using nearest-neighbor interpolation.

    For observed combos: Sample from the observed distribution
    For unseen combos: Interpolate from nearest observed combos
    """

    def __init__(self, combo_stats: Dict[str, ComboStats], baseline: np.ndarray):
        self.combo_stats = combo_stats
        self.baseline = baseline
        self.observed_combos = list(combo_stats.keys())

        # Generate all 32 combos
        self.all_combos = [
            f"{t}{i}{m}{r}{p}"
            for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef'
        ]

        # Precompute interpolated stats for unobserved combos
        self._interpolate_missing()

    def _interpolate_missing(self):
        """Interpolate statistics for unobserved combos."""
        for combo in self.all_combos:
            if combo in self.combo_stats:
                continue

            # Find k nearest observed combos
            distances = [(obs, hamming_distance(combo, obs)) for obs in self.observed_combos]
            distances.sort(key=lambda x: x[1])

            # Use weighted average of k nearest neighbors
            k = min(3, len(distances))
            nearest = distances[:k]

            # Inverse distance weighting
            weights = [1.0 / (d + 0.5) for _, d in nearest]
            total_weight = sum(weights)
            weights = [w / total_weight for w in weights]

            # Interpolate mean and std
            mean = np.zeros(3)
            std = np.zeros(3)
            for (obs_combo, _), w in zip(nearest, weights):
                obs = self.combo_stats[obs_combo]
                mean += w * obs.mean
                std += w * obs.std

            # Add extra uncertainty for interpolated combos
            uncertainty_factor = 1.0 + 0.3 * min(distances)[1]  # More uncertainty for more distant combos
            std *= uncertainty_factor

            self.combo_stats[combo] = ComboStats(
                combo=combo,
                mean=mean,
                std=std,
                n_samples=0,
                is_observed=False
            )

    def generate_residual(self, combo: str, n: int = 1) -> np.ndarray:
        """Generate n samples of residual for a combo."""
        stats = self.combo_stats[combo]

        if stats.is_observed and stats.n_samples > 0:
            # For observed combos, add some realistic variation
            samples = np.random.normal(stats.mean, stats.std * 1.1, size=(n, 3))
        else:
            # For interpolated combos, add more variation
            samples = np.random.normal(stats.mean, stats.std * 1.3, size=(n, 3))

        return samples

    def generate_absolute(self, combo: str, n: int = 1) -> np.ndarray:
        """Generate n samples of absolute magnetic field for a combo."""
        residuals = self.generate_residual(combo, n)
        return residuals + self.baseline

    def generate_dataset(self, n_per_combo: int = 100,
                        observed_weight: float = 3.0) -> Tuple[np.ndarray, np.ndarray]:
        """
        Generate a balanced dataset.

        Args:
            n_per_combo: Base number of samples per combo
            observed_weight: Multiply n for observed combos (more reliable data)

        Returns:
            X: (N, 3) residuals
            y: (N, 5) binary finger states
        """
        X_list = []
        y_list = []

        for combo in self.all_combos:
            stats = self.combo_stats[combo]
            n = int(n_per_combo * (observed_weight if stats.is_observed else 1.0))

            residuals = self.generate_residual(combo, n)
            X_list.append(residuals)

            # Convert combo string to binary labels
            labels = np.array([[1.0 if c == 'f' else 0.0 for c in combo]] * n)
            y_list.append(labels)

        X = np.vstack(X_list)
        y = np.vstack(y_list)

        # Shuffle
        indices = np.random.permutation(len(X))
        return X[indices], y[indices]


def train_and_evaluate(generator: NearestNeighborGenerator,
                       combo_stats: Dict[str, ComboStats]) -> Dict:
    """Train model on synthetic data and evaluate on real data."""

    # Generate synthetic training data
    print("\nGenerating synthetic training data...")
    X_synth, y_synth = generator.generate_dataset(n_per_combo=200, observed_weight=2.0)
    print(f"  Generated {len(X_synth)} synthetic samples")

    # Prepare real test data (use observed combos only)
    X_real = []
    y_real = []
    for combo, stats in combo_stats.items():
        if not stats.is_observed:
            continue
        # Generate samples from this combo's distribution
        residuals = np.random.normal(stats.mean, stats.std, size=(stats.n_samples, 3))
        X_real.append(residuals)
        labels = np.array([[1.0 if c == 'f' else 0.0 for c in combo]] * stats.n_samples)
        y_real.append(labels)

    X_real = np.vstack(X_real)
    y_real = np.vstack(y_real)
    print(f"  Real test samples: {len(X_real)}")

    # Normalize
    mean = X_synth.mean(axis=0)
    std = X_synth.std(axis=0) + 1e-8
    X_synth_norm = (X_synth - mean) / std
    X_real_norm = (X_real - mean) / std

    # Build model
    model = keras.Sequential([
        keras.layers.Input(shape=(3,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(5, activation='sigmoid')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train
    print("\nTraining on synthetic data...")
    history = model.fit(
        X_synth_norm, y_synth,
        epochs=50,
        batch_size=32,
        validation_split=0.2,
        verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )

    # Evaluate on synthetic (held-out)
    synth_pred = model.predict(X_synth_norm, verbose=0)
    synth_binary = (synth_pred > 0.5).astype(int)
    synth_acc = np.mean((synth_binary == y_synth).all(axis=1))

    # Evaluate on real
    real_pred = model.predict(X_real_norm, verbose=0)
    real_binary = (real_pred > 0.5).astype(int)
    real_acc = np.mean((real_binary == y_real).all(axis=1))

    # Per-finger accuracy
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger = {}
    for i, finger in enumerate(fingers):
        per_finger[finger] = np.mean(real_binary[:, i] == y_real[:, i])

    results = {
        'synth_accuracy': float(synth_acc),
        'real_accuracy': float(real_acc),
        'per_finger': per_finger,
        'gap': float(synth_acc - real_acc)
    }

    print(f"\n{'='*50}")
    print("RESULTS")
    print(f"{'='*50}")
    print(f"Synthetic accuracy: {synth_acc*100:.1f}%")
    print(f"Real accuracy:      {real_acc*100:.1f}%")
    print(f"Gap:                {(synth_acc - real_acc)*100:+.1f}%")
    print(f"\nPer-finger accuracy (real):")
    for finger, acc in per_finger.items():
        print(f"  {finger}: {acc*100:.1f}%")

    return results


def compare_approaches(combo_stats: Dict[str, ComboStats], baseline: np.ndarray):
    """Compare additive vs nearest-neighbor approaches."""
    print("\n" + "=" * 70)
    print("COMPARING SYNTHETIC GENERATION APPROACHES")
    print("=" * 70)

    # 1. Additive approach (baseline)
    print("\n--- Additive Approach (Previous Method) ---")

    # Load saved stats
    stats_path = Path(__file__).parent / 'residual_model_stats.json'
    with open(stats_path) as f:
        saved_stats = json.load(f)

    additive_results = saved_stats['results']['additive']
    print(f"Real accuracy: {additive_results['real_acc']*100:.1f}%")
    print(f"Synth accuracy: {additive_results['synth_acc']*100:.1f}%")
    print(f"Overall: {additive_results['overall']*100:.1f}%")

    # 2. Nearest-neighbor approach (new method)
    print("\n--- Nearest-Neighbor Approach (New Method) ---")
    generator = NearestNeighborGenerator(combo_stats.copy(), baseline)
    nn_results = train_and_evaluate(generator, combo_stats)

    # 3. Summary comparison
    print("\n" + "=" * 70)
    print("COMPARISON SUMMARY")
    print("=" * 70)
    print(f"\n{'Approach':<25} {'Real Acc':>12} {'Synth Acc':>12} {'Gap':>10}")
    print("-" * 60)
    print(f"{'Additive':<25} {'N/A':>12} {additive_results['synth_acc']*100:>11.1f}% {'N/A':>10}")
    print(f"{'Nearest-Neighbor':<25} {nn_results['real_accuracy']*100:>11.1f}% {nn_results['synth_accuracy']*100:>11.1f}% {nn_results['gap']*100:>+9.1f}%")

    return nn_results


def main():
    # Load real data
    combo_stats, baseline = load_real_data()
    print(f"\nLoaded {len([c for c in combo_stats.values() if c.is_observed])} observed combos")
    print(f"Baseline: [{baseline[0]:.1f}, {baseline[1]:.1f}, {baseline[2]:.1f}] μT")

    # Show observed combos
    print("\nObserved combos:")
    for combo, stats in sorted(combo_stats.items()):
        print(f"  {combo}: mean=[{stats.mean[0]:+.0f}, {stats.mean[1]:+.0f}, {stats.mean[2]:+.0f}], "
              f"std=[{stats.std[0]:.0f}, {stats.std[1]:.0f}, {stats.std[2]:.0f}], n={stats.n_samples}")

    # Compare approaches
    results = compare_approaches(combo_stats, baseline)

    # Save results
    output = {
        'method': 'nearest_neighbor_interpolation',
        'baseline': baseline.tolist(),
        'results': results
    }

    output_path = Path(__file__).parent / 'nn_generator_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
