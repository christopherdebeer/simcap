#!/usr/bin/env python3
"""
Train Finger State Model using Physics-Based Synthetic Data.

Uses the interaction model discovered from physics simulation:
- Per-finger residual effects (from single-finger observations)
- Interaction suppression: each additional finger scales effect by 0.30x

Compares against:
1. Real data only (baseline)
2. Additive synthetic (prior approach)
3. Interaction-based synthetic (new approach)
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from dataclasses import dataclass
import tensorflow as tf
from tensorflow import keras

print("=" * 70)
print("PHYSICS-BASED SYNTHETIC TRAINING")
print("Using Interaction Model (0.30x per additional finger)")
print("=" * 70)


# ===== Data Loading =====

def load_session_data() -> Tuple[Dict[str, np.ndarray], np.ndarray]:
    """Load raw session data and compute per-combo samples."""
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
        result[combo] = mags - baseline

    return result, baseline


def load_interaction_model() -> Tuple[Dict[str, np.ndarray], float]:
    """Load the fitted interaction model parameters."""
    model_path = Path(__file__).parent / 'per_finger_fit_results.json'
    with open(model_path, 'r') as f:
        data = json.load(f)

    # Use fitted effects from interaction model
    fitted = np.array(data['interaction_model']['fitted_effects'])
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

    effects = {name: fitted[i] for i, name in enumerate(finger_names)}
    interaction = data['interaction_model']['interaction_strength']

    return effects, interaction


# ===== Synthetic Data Generation =====

class InteractionSyntheticGenerator:
    """
    Generate synthetic data using the interaction model.

    Key insight: When multiple fingers flex, the total effect is NOT additive.
    Instead, each additional finger suppresses the total by a factor.
    """

    def __init__(self, finger_effects: Dict[str, np.ndarray],
                 interaction_strength: float,
                 noise_scale: float = 0.15):
        self.finger_effects = finger_effects
        self.interaction = interaction_strength
        self.noise_scale = noise_scale
        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

    def predict_residual(self, combo: str) -> np.ndarray:
        """Predict mean residual for a combo using interaction model."""
        if combo == 'eeeee':
            return np.zeros(3)

        # Get flexed fingers
        flexed = [self.finger_names[i] for i, c in enumerate(combo) if c == 'f']
        n_flexed = len(flexed)

        # Linear sum of effects
        linear = sum(self.finger_effects[f] for f in flexed)

        # Apply interaction scaling
        if n_flexed > 1:
            scale = 1.0 + self.interaction * (n_flexed - 1) / 4
        else:
            scale = 1.0

        return linear * scale

    def generate_samples(self, combo: str, n_samples: int) -> np.ndarray:
        """Generate n synthetic samples for a combo."""
        mean = self.predict_residual(combo)
        magnitude = np.linalg.norm(mean) + 1e-6

        # Noise proportional to signal magnitude
        noise_std = magnitude * self.noise_scale

        samples = np.random.normal(mean, noise_std, size=(n_samples, 3))
        return samples


class AdditiveSyntheticGenerator:
    """Generate synthetic data using simple additive model (for comparison)."""

    def __init__(self, finger_effects: Dict[str, np.ndarray], noise_scale: float = 0.15):
        self.finger_effects = finger_effects
        self.noise_scale = noise_scale
        self.finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

    def predict_residual(self, combo: str) -> np.ndarray:
        if combo == 'eeeee':
            return np.zeros(3)

        flexed = [self.finger_names[i] for i, c in enumerate(combo) if c == 'f']
        return sum(self.finger_effects[f] for f in flexed)

    def generate_samples(self, combo: str, n_samples: int) -> np.ndarray:
        mean = self.predict_residual(combo)
        magnitude = np.linalg.norm(mean) + 1e-6
        noise_std = magnitude * self.noise_scale
        samples = np.random.normal(mean, noise_std, size=(n_samples, 3))
        return samples


def combo_to_labels(combo: str) -> np.ndarray:
    """Convert combo string to binary labels."""
    return np.array([1.0 if c == 'f' else 0.0 for c in combo])


# ===== Dataset Preparation =====

def prepare_real_dataset(combo_samples: Dict[str, np.ndarray],
                        test_ratio: float = 0.2) -> Tuple:
    """Prepare train/test split from real data only."""
    X_train, y_train = [], []
    X_test, y_test = [], []

    for combo, samples in combo_samples.items():
        labels = combo_to_labels(combo)
        n = len(samples)

        # Random split
        np.random.seed(42)
        idx = np.random.permutation(n)
        split = int(n * (1 - test_ratio))

        train_samples = samples[idx[:split]]
        test_samples = samples[idx[split:]]

        X_train.append(train_samples)
        y_train.append(np.tile(labels, (len(train_samples), 1)))
        X_test.append(test_samples)
        y_test.append(np.tile(labels, (len(test_samples), 1)))

    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)
    X_test = np.vstack(X_test)
    y_test = np.vstack(y_test)

    # Shuffle training
    idx = np.random.permutation(len(X_train))
    X_train = X_train[idx]
    y_train = y_train[idx]

    return X_train, y_train, X_test, y_test


def prepare_synthetic_dataset(generator, n_per_combo: int = 200) -> Tuple:
    """Generate synthetic dataset for all 32 combos."""
    all_combos = [f"{t}{i}{m}{r}{p}"
                  for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef']

    X, y = [], []
    for combo in all_combos:
        samples = generator.generate_samples(combo, n_per_combo)
        labels = combo_to_labels(combo)
        X.append(samples)
        y.append(np.tile(labels, (n_per_combo, 1)))

    X = np.vstack(X)
    y = np.vstack(y)

    # Shuffle
    idx = np.random.permutation(len(X))
    return X[idx], y[idx]


def prepare_hybrid_dataset(combo_samples: Dict[str, np.ndarray],
                          generator,
                          synth_per_unseen: int = 200,
                          test_ratio: float = 0.2) -> Tuple:
    """
    Hybrid dataset: Real data for observed combos, synthetic for unseen.
    """
    all_combos = [f"{t}{i}{m}{r}{p}"
                  for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef']

    X_train, y_train = [], []
    X_test, y_test = [], []

    for combo in all_combos:
        labels = combo_to_labels(combo)

        if combo in combo_samples:
            # Use real data
            samples = combo_samples[combo]
            n = len(samples)

            np.random.seed(42)
            idx = np.random.permutation(n)
            split = int(n * (1 - test_ratio))

            X_train.append(samples[idx[:split]])
            y_train.append(np.tile(labels, (split, 1)))
            X_test.append(samples[idx[split:]])
            y_test.append(np.tile(labels, (n - split, 1)))
        else:
            # Use synthetic for training only (no test data for unseen)
            samples = generator.generate_samples(combo, synth_per_unseen)
            X_train.append(samples)
            y_train.append(np.tile(labels, (synth_per_unseen, 1)))

    X_train = np.vstack(X_train)
    y_train = np.vstack(y_train)
    X_test = np.vstack(X_test)
    y_test = np.vstack(y_test)

    idx = np.random.permutation(len(X_train))
    return X_train[idx], y_train[idx], X_test, y_test


# ===== Model Training =====

def build_model(input_dim: int = 3, output_dim: int = 5) -> keras.Model:
    """Build classifier model."""
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


def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray,
                      X_test: np.ndarray, y_test: np.ndarray,
                      name: str) -> Dict:
    """Train model and evaluate."""
    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-8
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    # Build and train
    model = build_model()

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
    pred = model.predict(X_test_norm, verbose=0)
    pred_binary = (pred > 0.5).astype(int)

    # Overall accuracy (all 5 fingers correct)
    overall_acc = np.mean((pred_binary == y_test).all(axis=1))

    # Per-finger accuracy
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger = {f: np.mean(pred_binary[:, i] == y_test[:, i]) for i, f in enumerate(fingers)}

    result = {
        'name': name,
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'overall_accuracy': float(overall_acc),
        'per_finger_accuracy': per_finger,
        'normalization': {'mean': mean.tolist(), 'std': std.tolist()}
    }

    return result, model


def analyze_per_combo_accuracy(model, X_test: np.ndarray, y_test: np.ndarray,
                               combo_samples: Dict[str, np.ndarray],
                               mean: np.ndarray, std: np.ndarray) -> Dict:
    """Analyze accuracy per combo."""
    results = {}

    for combo, samples in combo_samples.items():
        labels = combo_to_labels(combo)
        samples_norm = (samples - mean) / std

        pred = model.predict(samples_norm, verbose=0)
        pred_binary = (pred > 0.5).astype(int)

        expected = np.tile(labels, (len(samples), 1))
        acc = np.mean((pred_binary == expected).all(axis=1))

        results[combo] = {
            'accuracy': float(acc),
            'n_samples': len(samples),
            'n_flexed': sum(1 for c in combo if c == 'f')
        }

    return results


# ===== Main Experiment =====

def main():
    # Load data
    combo_samples, baseline = load_session_data()
    print(f"\nLoaded {len(combo_samples)} observed combinations")
    print(f"Baseline: [{baseline[0]:.1f}, {baseline[1]:.1f}, {baseline[2]:.1f}] μT")

    # Load interaction model
    finger_effects, interaction = load_interaction_model()
    print(f"\nInteraction strength: {interaction:.3f}")
    print(f"Scaling per additional finger: {1 + interaction:.2f}x")

    # Create generators
    interaction_gen = InteractionSyntheticGenerator(finger_effects, interaction)
    additive_gen = AdditiveSyntheticGenerator(finger_effects)

    # Prepare datasets
    print("\n" + "=" * 70)
    print("PREPARING DATASETS")
    print("=" * 70)

    # 1. Real data only
    X_train_real, y_train_real, X_test, y_test = prepare_real_dataset(combo_samples)
    print(f"Real only: {len(X_train_real)} train, {len(X_test)} test")

    # 2. Synthetic only (additive)
    X_synth_add, y_synth_add = prepare_synthetic_dataset(additive_gen, n_per_combo=200)
    print(f"Synthetic (additive): {len(X_synth_add)} samples")

    # 3. Synthetic only (interaction)
    X_synth_int, y_synth_int = prepare_synthetic_dataset(interaction_gen, n_per_combo=200)
    print(f"Synthetic (interaction): {len(X_synth_int)} samples")

    # 4. Hybrid (real + interaction synthetic)
    X_hybrid, y_hybrid, _, _ = prepare_hybrid_dataset(combo_samples, interaction_gen)
    print(f"Hybrid (real + interaction): {len(X_hybrid)} train")

    # Train and evaluate
    print("\n" + "=" * 70)
    print("TRAINING MODELS")
    print("=" * 70)

    results = []

    # 1. Real data only
    print("\n--- Training: Real Data Only ---")
    r1, m1 = train_and_evaluate(X_train_real, y_train_real, X_test, y_test, "Real Data Only")
    results.append(r1)
    print(f"Overall accuracy: {r1['overall_accuracy']*100:.1f}%")

    # 2. Synthetic (additive) -> test on real
    print("\n--- Training: Additive Synthetic ---")
    r2, m2 = train_and_evaluate(X_synth_add, y_synth_add, X_test, y_test, "Additive Synthetic")
    results.append(r2)
    print(f"Overall accuracy: {r2['overall_accuracy']*100:.1f}%")

    # 3. Synthetic (interaction) -> test on real
    print("\n--- Training: Interaction Synthetic ---")
    r3, m3 = train_and_evaluate(X_synth_int, y_synth_int, X_test, y_test, "Interaction Synthetic")
    results.append(r3)
    print(f"Overall accuracy: {r3['overall_accuracy']*100:.1f}%")

    # 4. Hybrid -> test on real
    print("\n--- Training: Hybrid (Real + Interaction) ---")
    r4, m4 = train_and_evaluate(X_hybrid, y_hybrid, X_test, y_test, "Hybrid (Real + Interaction)")
    results.append(r4)
    print(f"Overall accuracy: {r4['overall_accuracy']*100:.1f}%")

    # Summary
    print("\n" + "=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    print(f"\n{'Approach':<35} {'Train':>8} {'Overall':>10} {'Thumb':>8} {'Index':>8} {'Middle':>8} {'Ring':>8} {'Pinky':>8}")
    print("-" * 110)

    for r in results:
        pf = r['per_finger_accuracy']
        print(f"{r['name']:<35} {r['train_samples']:>8} {r['overall_accuracy']*100:>9.1f}% "
              f"{pf['thumb']*100:>7.1f}% {pf['index']*100:>7.1f}% {pf['middle']*100:>7.1f}% "
              f"{pf['ring']*100:>7.1f}% {pf['pinky']*100:>7.1f}%")

    # Per-combo analysis for best model
    print("\n" + "=" * 70)
    print("PER-COMBO ACCURACY (Real Data Only Model)")
    print("=" * 70)

    mean = np.array(r1['normalization']['mean'])
    std = np.array(r1['normalization']['std'])
    combo_acc = analyze_per_combo_accuracy(m1, X_test, y_test, combo_samples, mean, std)

    print(f"\n{'Combo':<10} {'Accuracy':>10} {'N Flexed':>10} {'Samples':>10}")
    print("-" * 45)
    for combo in sorted(combo_acc.keys()):
        ca = combo_acc[combo]
        print(f"{combo:<10} {ca['accuracy']*100:>9.1f}% {ca['n_flexed']:>10} {ca['n_samples']:>10}")

    # Generalization analysis
    print("\n" + "=" * 70)
    print("GENERALIZATION ANALYSIS")
    print("=" * 70)

    # Compare accuracy by number of flexed fingers
    flex_groups = {}
    for combo, ca in combo_acc.items():
        n = ca['n_flexed']
        if n not in flex_groups:
            flex_groups[n] = []
        flex_groups[n].append(ca['accuracy'])

    print(f"\n{'N Flexed':<10} {'Mean Acc':>12} {'Std':>10} {'N Combos':>10}")
    print("-" * 45)
    for n in sorted(flex_groups.keys()):
        accs = flex_groups[n]
        print(f"{n:<10} {np.mean(accs)*100:>11.1f}% {np.std(accs)*100:>9.1f}% {len(accs):>10}")

    # Save results
    output = {
        'experiments': results,
        'per_combo_accuracy': combo_acc,
        'interaction_model': {
            'finger_effects': {k: v.tolist() for k, v in finger_effects.items()},
            'interaction_strength': interaction,
        },
        'baseline': baseline.tolist(),
    }

    output_path = Path(__file__).parent / 'physics_synthetic_training_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
