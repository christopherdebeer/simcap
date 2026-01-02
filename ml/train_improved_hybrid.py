#!/usr/bin/env python3
"""
Improved Hybrid Training with Calibrated Synthetic Data.

Key improvements over prior approaches:
1. Real data ONLY for observed combos (no synthetic mixing)
2. Synthetic ONLY for missing combos, calibrated from nearest observed
3. Confidence-weighted synthetic based on prediction quality
4. Uses existing per_finger_fit_results.json for interaction model

This builds on:
- train_physics_synthetic.py (baseline hybrid approach)
- per_finger_fit_results.json (fitted effects)
- physics_synthetic_training_results.json (per-combo accuracy)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import tensorflow as tf
from tensorflow import keras

print("=" * 70)
print("IMPROVED HYBRID TRAINING")
print("Real for observed, Calibrated synthetic for missing")
print("=" * 70)


def load_session_data() -> Tuple[Dict[str, np.ndarray], np.ndarray, Dict]:
    """Load session data and compute per-combo residuals with stats."""
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

    # Convert to residuals and compute stats
    result = {}
    stats = {}
    for combo, mags in combo_samples.items():
        mags = np.array(mags)
        residuals = mags - baseline
        result[combo] = residuals
        stats[combo] = {
            'mean': residuals.mean(axis=0),
            'std': residuals.std(axis=0),
            'n': len(residuals)
        }

    return result, baseline, stats


def load_interaction_model() -> Tuple[Dict[str, np.ndarray], float]:
    """Load fitted interaction model from prior analysis."""
    model_path = Path(__file__).parent / 'per_finger_fit_results.json'
    with open(model_path, 'r') as f:
        data = json.load(f)

    fitted = data['interaction_model']['fitted_effects']
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

    # Handle both list and dict formats
    if isinstance(fitted, list):
        effects = {name: np.array(fitted[i]) for i, name in enumerate(finger_names)}
    else:
        effects = {name: np.array(fitted[name]) for name in finger_names}

    interaction = data['interaction_model']['interaction_strength']
    return effects, interaction


def hamming_distance(a: str, b: str) -> int:
    """Hamming distance between two combo strings."""
    return sum(c1 != c2 for c1, c2 in zip(a, b))


def find_nearest_observed(combo: str, observed_combos: List[str]) -> Tuple[str, int]:
    """Find nearest observed combo by Hamming distance."""
    distances = [(obs, hamming_distance(combo, obs)) for obs in observed_combos]
    distances.sort(key=lambda x: x[1])
    return distances[0]


class CalibratedSyntheticGenerator:
    """
    Generate synthetic data calibrated from nearest observed combo.

    Key improvements:
    1. Uses noise std from nearest observed combo (not magnitude-based)
    2. Applies confidence weighting based on distance
    3. Blends multiple nearest neighbors for better interpolation
    """

    def __init__(self, finger_effects: Dict[str, np.ndarray],
                 interaction: float,
                 observed_stats: Dict):
        self.effects = finger_effects
        self.interaction = interaction
        self.observed_stats = observed_stats
        self.observed_combos = list(observed_stats.keys())
        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

    def predict_mean(self, combo: str) -> np.ndarray:
        """Predict mean residual using interaction model."""
        if combo == 'eeeee':
            return np.zeros(3)

        flexed = [self.finger_names[i] for i, c in enumerate(combo) if c == 'f']
        n_flexed = len(flexed)

        linear = sum(self.effects[f] for f in flexed)
        scale = 1.0 + self.interaction * (n_flexed - 1) / 4 if n_flexed > 1 else 1.0

        return linear * scale

    def get_calibrated_noise(self, combo: str) -> Tuple[np.ndarray, float]:
        """Get noise std calibrated from nearest observed combo."""
        if combo in self.observed_stats:
            stats = self.observed_stats[combo]
            return stats['std'], 1.0  # High confidence for observed

        # Find k nearest neighbors
        k = min(2, len(self.observed_combos))
        distances = [(obs, hamming_distance(combo, obs)) for obs in self.observed_combos]
        distances.sort(key=lambda x: x[1])
        nearest = distances[:k]

        # Weighted average of neighbor noise
        total_weight = 0
        weighted_std = np.zeros(3)

        for obs, dist in nearest:
            weight = 1.0 / (dist + 0.5)
            weighted_std += weight * self.observed_stats[obs]['std']
            total_weight += weight

        std = weighted_std / total_weight

        # Confidence based on distance (closer = more confident)
        min_dist = nearest[0][1]
        confidence = 1.0 / (1.0 + min_dist * 0.5)  # 1.0 for dist=0, 0.67 for dist=1, etc.

        # Add extra uncertainty for interpolated combos
        uncertainty_factor = 1.0 + 0.2 * min_dist
        std *= uncertainty_factor

        return std, confidence

    def generate_samples(self, combo: str, n_samples: int) -> Tuple[np.ndarray, float]:
        """Generate synthetic samples with confidence score."""
        mean = self.predict_mean(combo)
        std, confidence = self.get_calibrated_noise(combo)

        samples = np.random.normal(mean, std, size=(n_samples, 3))
        return samples, confidence


def combo_to_labels(combo: str) -> np.ndarray:
    return np.array([1.0 if c == 'f' else 0.0 for c in combo])


def prepare_improved_dataset(combo_samples: Dict[str, np.ndarray],
                            generator: CalibratedSyntheticGenerator,
                            synth_per_missing: int = 150,
                            test_ratio: float = 0.2) -> Tuple:
    """
    Prepare dataset with strict separation:
    - Real data only for observed combos
    - Synthetic only for missing combos
    """
    all_combos = [f"{t}{i}{m}{r}{p}"
                  for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef']

    X_train, y_train, weights_train = [], [], []
    X_test, y_test = [], []

    for combo in all_combos:
        labels = combo_to_labels(combo)

        if combo in combo_samples:
            # REAL DATA ONLY for observed combos
            samples = combo_samples[combo]
            n = len(samples)

            np.random.seed(42)
            idx = np.random.permutation(n)
            split = int(n * (1 - test_ratio))

            X_train.append(samples[idx[:split]])
            y_train.append(np.tile(labels, (split, 1)))
            weights_train.append(np.ones(split))  # Full weight for real

            X_test.append(samples[idx[split:]])
            y_test.append(np.tile(labels, (n - split, 1)))
        else:
            # SYNTHETIC for missing combos
            samples, confidence = generator.generate_samples(combo, synth_per_missing)
            X_train.append(samples)
            y_train.append(np.tile(labels, (synth_per_missing, 1)))
            weights_train.append(np.ones(synth_per_missing) * confidence)  # Weighted by confidence

    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)
    weights_train = np.concatenate(weights_train)
    X_test = np.vstack(X_test)
    y_test = np.vstack(y_test)

    # Shuffle training
    idx = np.random.permutation(len(X_train))
    return X_train[idx], y_train[idx], weights_train[idx], X_test, y_test


def build_model() -> keras.Model:
    """Build classifier model."""
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
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def train_and_evaluate(X_train, y_train, weights, X_test, y_test, name: str) -> Dict:
    """Train with sample weights and evaluate."""
    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    model = build_model()

    model.fit(
        X_train_norm, y_train,
        sample_weight=weights,
        epochs=100, batch_size=32,
        validation_split=0.2, verbose=0,
        callbacks=[
            keras.callbacks.EarlyStopping(patience=15, restore_best_weights=True),
            keras.callbacks.ReduceLROnPlateau(factor=0.5, patience=5)
        ]
    )

    # Evaluate
    pred = model.predict(X_test_norm, verbose=0)
    pred_binary = (pred > 0.5).astype(int)

    overall = np.mean((pred_binary == y_test).all(axis=1))
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger = {f: float(np.mean(pred_binary[:, i] == y_test[:, i])) for i, f in enumerate(fingers)}

    return {
        'name': name,
        'overall_accuracy': float(overall),
        'per_finger': per_finger,
        'normalization': {'mean': mean.tolist(), 'std': std.tolist()}
    }


def main():
    # Load data
    combo_samples, baseline, stats = load_session_data()
    effects, interaction = load_interaction_model()

    print(f"\nObserved combos: {len(combo_samples)}")
    print(f"Interaction strength: {interaction:.3f}")

    # Create generator
    generator = CalibratedSyntheticGenerator(effects, interaction, stats)

    # Compare approaches
    results = []

    # 1. Real data only (baseline)
    print("\n--- Training: Real Data Only ---")
    X_real = np.vstack([combo_samples[c] for c in combo_samples])
    y_real = np.vstack([np.tile(combo_to_labels(c), (len(combo_samples[c]), 1)) for c in combo_samples])

    np.random.seed(42)
    idx = np.random.permutation(len(X_real))
    split = int(len(X_real) * 0.8)
    X_train_r, y_train_r = X_real[idx[:split]], y_real[idx[:split]]
    X_test_r, y_test_r = X_real[idx[split:]], y_real[idx[split:]]
    w_r = np.ones(len(X_train_r))

    r1 = train_and_evaluate(X_train_r, y_train_r, w_r, X_test_r, y_test_r, "Real Data Only")
    results.append(r1)
    print(f"Overall: {r1['overall_accuracy']*100:.1f}%")

    # 2. Improved Hybrid
    print("\n--- Training: Improved Hybrid (Calibrated) ---")
    X_train, y_train, weights, X_test, y_test = prepare_improved_dataset(
        combo_samples, generator, synth_per_missing=150
    )
    r2 = train_and_evaluate(X_train, y_train, weights, X_test, y_test, "Improved Hybrid (Calibrated)")
    results.append(r2)
    print(f"Overall: {r2['overall_accuracy']*100:.1f}%")

    # 3. Improved Hybrid with more synthetic
    print("\n--- Training: Improved Hybrid (More Synthetic) ---")
    X_train, y_train, weights, X_test, y_test = prepare_improved_dataset(
        combo_samples, generator, synth_per_missing=300
    )
    r3 = train_and_evaluate(X_train, y_train, weights, X_test, y_test, "Improved Hybrid (300/combo)")
    results.append(r3)
    print(f"Overall: {r3['overall_accuracy']*100:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)
    print(f"\n{'Approach':<40} {'Overall':>10} {'Thumb':>8} {'Index':>8} {'Middle':>8} {'Ring':>8} {'Pinky':>8}")
    print("-" * 105)

    for r in results:
        pf = r['per_finger']
        print(f"{r['name']:<40} {r['overall_accuracy']*100:>9.1f}% "
              f"{pf['thumb']*100:>7.1f}% {pf['index']*100:>7.1f}% {pf['middle']*100:>7.1f}% "
              f"{pf['ring']*100:>7.1f}% {pf['pinky']*100:>7.1f}%")

    # Load prior results for comparison
    prior_path = Path(__file__).parent / 'physics_synthetic_training_results.json'
    if prior_path.exists():
        with open(prior_path) as f:
            prior = json.load(f)

        print("\n" + "-" * 105)
        print("PRIOR RESULTS (for comparison):")
        for exp in prior['experiments']:
            pf = exp['per_finger_accuracy']
            print(f"{exp['name']:<40} {exp['overall_accuracy']*100:>9.1f}% "
                  f"{pf['thumb']*100:>7.1f}% {pf['index']*100:>7.1f}% {pf['middle']*100:>7.1f}% "
                  f"{pf['ring']*100:>7.1f}% {pf['pinky']*100:>7.1f}%")

    # Save results
    output = {
        'experiments': results,
        'improvements': [
            'Real data only for observed combos (no synthetic mixing)',
            'Calibrated noise from nearest observed combo',
            'Confidence weighting based on distance to observed',
            'Strict separation of real vs synthetic'
        ]
    }

    output_path = Path(__file__).parent / 'improved_hybrid_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
