#!/usr/bin/env python3
"""
Single-Sample Magnetic Residual Training

Train on individual samples (no windowing) using only 3-feature magnetic residual.
Focus on matching synthetic data generation to real residual distributions.

Key questions:
1. What accuracy can we achieve with single-sample 3-feature residual?
2. How do real residual distributions look per finger state?
3. How can synthetic generation better match these distributions?
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy import stats

import tensorflow as tf
from tensorflow import keras

print("=" * 70)
print("SINGLE-SAMPLE MAGNETIC RESIDUAL TRAINING")
print("=" * 70)


@dataclass
class ResidualData:
    """Store residual magnetic data for a finger combo."""
    combo: str
    residuals: np.ndarray  # (n, 3) - mx-bx, my-by, mz-bz
    magnitudes: np.ndarray  # (n,) - |residual|


def load_and_compute_residuals() -> Tuple[Dict[str, ResidualData], np.ndarray]:
    """Load session data and compute magnetic residuals."""
    session_path = Path(__file__).parent.parent / 'data' / 'GAMBIT' / '2025-12-31T14_06_18.270Z.json'
    with open(session_path, 'r') as f:
        data = json.load(f)

    # First pass: collect all open palm (eeeee) samples for baseline
    baseline_mags = []
    combo_raw = {}

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

        if combo not in combo_raw:
            combo_raw[combo] = []
        combo_raw[combo].extend(mags)

        if combo == 'eeeee':
            baseline_mags.extend(mags)

    # Compute baseline
    baseline = np.mean(baseline_mags, axis=0)
    print(f"\nBaseline (eeeee): [{baseline[0]:.1f}, {baseline[1]:.1f}, {baseline[2]:.1f}] μT")
    print(f"Baseline std: {np.std(baseline_mags, axis=0)}")

    # Convert to residuals
    residual_data = {}
    for combo, mags in combo_raw.items():
        mags = np.array(mags)
        residuals = mags - baseline
        magnitudes = np.linalg.norm(residuals, axis=1)
        residual_data[combo] = ResidualData(
            combo=combo,
            residuals=residuals,
            magnitudes=magnitudes
        )

    return residual_data, baseline


def analyze_residual_distributions(data: Dict[str, ResidualData]):
    """Analyze how residual distributions differ per finger state."""
    print("\n" + "=" * 70)
    print("RESIDUAL DISTRIBUTION ANALYSIS")
    print("=" * 70)

    # Per-combo statistics
    print(f"\n{'Combo':<8} {'N':>6} {'|Δ| mean':>10} {'|Δ| std':>10} {'Δx mean':>10} {'Δy mean':>10} {'Δz mean':>10}")
    print("-" * 70)

    for combo in sorted(data.keys()):
        d = data[combo]
        n = len(d.residuals)
        mag_mean = np.mean(d.magnitudes)
        mag_std = np.std(d.magnitudes)
        mean = d.residuals.mean(axis=0)
        print(f"{combo:<8} {n:>6} {mag_mean:>10.1f} {mag_std:>10.1f} {mean[0]:>+10.1f} {mean[1]:>+10.1f} {mean[2]:>+10.1f}")

    # Finger effect decomposition
    print("\n" + "-" * 70)
    print("INDIVIDUAL FINGER EFFECTS")
    print("-" * 70)

    single_finger = {'thumb': 'feeee', 'index': 'efeee', 'middle': 'eefee', 'ring': 'eeefe', 'pinky': 'eeeef'}

    print(f"\n{'Finger':<10} {'Δx':>12} {'Δy':>12} {'Δz':>12} {'|Δ|':>12} {'σx':>8} {'σy':>8} {'σz':>8}")
    print("-" * 90)

    finger_effects = {}
    for finger, combo in single_finger.items():
        if combo in data:
            d = data[combo]
            mean = d.residuals.mean(axis=0)
            std = d.residuals.std(axis=0)
            mag = np.mean(d.magnitudes)
            finger_effects[finger] = {'mean': mean, 'std': std, 'mag': mag}
            print(f"{finger:<10} {mean[0]:>+12.1f} {mean[1]:>+12.1f} {mean[2]:>+12.1f} {mag:>12.1f} "
                  f"{std[0]:>8.1f} {std[1]:>8.1f} {std[2]:>8.1f}")
        else:
            print(f"{finger:<10} (no data)")

    # Test additivity hypothesis
    print("\n" + "-" * 70)
    print("ADDITIVITY TEST: Does thumb + index ≈ ffeee?")
    print("-" * 70)

    if 'thumb' in finger_effects and 'index' in finger_effects and 'ffeee' in data:
        predicted = finger_effects['thumb']['mean'] + finger_effects['index']['mean']
        actual = data['ffeee'].residuals.mean(axis=0)
        error = actual - predicted
        print(f"\nPredicted (thumb + index): [{predicted[0]:+.1f}, {predicted[1]:+.1f}, {predicted[2]:+.1f}]")
        print(f"Actual (ffeee):            [{actual[0]:+.1f}, {actual[1]:+.1f}, {actual[2]:+.1f}]")
        print(f"Error:                     [{error[0]:+.1f}, {error[1]:+.1f}, {error[2]:+.1f}]")
        print(f"Error magnitude: {np.linalg.norm(error):.1f} μT ({np.linalg.norm(error)/np.linalg.norm(actual)*100:.1f}%)")

    return finger_effects


class ResidualSyntheticGenerator:
    """Generate synthetic residual samples matching observed distributions."""

    def __init__(self, real_data: Dict[str, ResidualData], finger_effects: Dict):
        self.real_data = real_data
        self.finger_effects = finger_effects

        # Get baseline noise from eeeee
        if 'eeeee' in real_data:
            # eeeee residual should be ~0 with some noise
            self.baseline_std = real_data['eeeee'].residuals.std(axis=0)
        else:
            self.baseline_std = np.array([15, 15, 25])

        print(f"\nBaseline noise (σ): [{self.baseline_std[0]:.1f}, {self.baseline_std[1]:.1f}, {self.baseline_std[2]:.1f}] μT")

    def generate_additive(self, combo: str, n: int) -> np.ndarray:
        """Generate using simple additive model: residual = sum(finger_effects)."""
        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']

        samples = []
        for _ in range(n):
            total_mean = np.zeros(3)
            total_var = self.baseline_std ** 2

            for i, state in enumerate(combo):
                if state == 'f':
                    f = fingers[i]
                    if f in self.finger_effects:
                        total_mean += self.finger_effects[f]['mean']
                        total_var += self.finger_effects[f]['std'] ** 2

            total_std = np.sqrt(total_var)
            sample = np.random.normal(total_mean, total_std)
            samples.append(sample)

        return np.array(samples)

    def generate_from_real(self, combo: str, n: int) -> np.ndarray:
        """Generate by sampling from real distribution if available."""
        if combo in self.real_data:
            real = self.real_data[combo].residuals
            # Resample with added noise for augmentation
            indices = np.random.choice(len(real), n, replace=True)
            samples = real[indices] + np.random.normal(0, self.baseline_std * 0.1, (n, 3))
            return samples
        else:
            return self.generate_additive(combo, n)

    def generate_mixture(self, combo: str, n: int, real_frac: float = 0.7) -> np.ndarray:
        """Mix real samples (if available) with synthetic additive samples."""
        if combo in self.real_data:
            n_real = int(n * real_frac)
            n_synth = n - n_real
            real_samples = self.generate_from_real(combo, n_real)
            synth_samples = self.generate_additive(combo, n_synth)
            return np.vstack([real_samples, synth_samples])
        else:
            return self.generate_additive(combo, n)


def prepare_dataset(real_data: Dict[str, ResidualData],
                   generator: ResidualSyntheticGenerator,
                   samples_per_combo: int = 500,
                   gen_mode: str = 'mixture') -> Tuple:
    """Prepare training dataset."""
    all_combos = [f"{t}{i}{m}{r}{p}" for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef']

    all_samples, all_labels = [], []
    real_mask = []  # Track which samples are from real data

    for combo in all_combos:
        if gen_mode == 'additive':
            samples = generator.generate_additive(combo, samples_per_combo)
            is_real = [False] * len(samples)
        elif gen_mode == 'real_only' and combo in real_data:
            samples = generator.generate_from_real(combo, samples_per_combo)
            is_real = [True] * len(samples)
        elif gen_mode == 'mixture':
            samples = generator.generate_mixture(combo, samples_per_combo)
            n_real = int(samples_per_combo * 0.7) if combo in real_data else 0
            is_real = [True] * n_real + [False] * (len(samples) - n_real)
        else:
            samples = generator.generate_additive(combo, samples_per_combo)
            is_real = [False] * len(samples)

        label = np.array([0 if c == 'e' else 1 for c in combo], dtype=np.float32)

        for s, r in zip(samples, is_real):
            all_samples.append(s)
            all_labels.append(label)
            real_mask.append(r)

    X = np.array(all_samples)
    y = np.array(all_labels)
    real_mask = np.array(real_mask)

    # Normalize
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std[std < 1e-6] = 1
    X = (X - mean) / std

    # Shuffle and split
    idx = np.arange(len(X))
    np.random.shuffle(idx)
    t1, t2 = int(0.7 * len(X)), int(0.85 * len(X))

    return (X[idx[:t1]], y[idx[:t1]], real_mask[idx[:t1]],
            X[idx[t1:t2]], y[idx[t1:t2]], real_mask[idx[t1:t2]],
            X[idx[t2:]], y[idx[t2:]], real_mask[idx[t2:]],
            {'mean': mean.tolist(), 'std': std.tolist()})


def build_model(n_features: int = 3):
    """Build simple dense model for single-sample prediction."""
    model = keras.Sequential([
        keras.layers.Input(shape=(n_features,)),
        keras.layers.Dense(64, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.BatchNormalization(),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(5, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model


def evaluate_model(model, X_test, y_test, real_mask_test, name: str):
    """Evaluate model on test set, separated by real/synthetic."""
    y_pred = (model.predict(X_test, verbose=0) > 0.5).astype(int)

    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']

    # Overall metrics
    overall = float(np.mean(y_pred == y_test))
    per_finger = {f: float(np.mean(y_pred[:, i] == y_test[:, i])) for i, f in enumerate(fingers)}

    # Split by real/synthetic
    real_idx = real_mask_test
    synth_idx = ~real_mask_test

    real_acc = float(np.mean(y_pred[real_idx] == y_test[real_idx])) if real_idx.sum() > 0 else 0
    synth_acc = float(np.mean(y_pred[synth_idx] == y_test[synth_idx])) if synth_idx.sum() > 0 else 0

    print(f"\n{name}:")
    print(f"  Overall: {overall:.1%}")
    print(f"  Real samples: {real_acc:.1%} (n={real_idx.sum()})")
    print(f"  Synthetic samples: {synth_acc:.1%} (n={synth_idx.sum()})")
    print(f"  Per-finger: {', '.join(f'{f}:{per_finger[f]:.0%}' for f in fingers)}")

    return {'overall': overall, 'real_acc': real_acc, 'synth_acc': synth_acc, 'per_finger': per_finger}


def main():
    # Load data and compute residuals
    print("\n--- Loading Data ---")
    real_data, baseline = load_and_compute_residuals()
    print(f"Loaded {len(real_data)} finger state combinations")

    # Analyze distributions
    finger_effects = analyze_residual_distributions(real_data)

    # Create generator
    generator = ResidualSyntheticGenerator(real_data, finger_effects)

    # Compare different generation strategies
    print("\n" + "=" * 70)
    print("TRAINING COMPARISON: DIFFERENT SYNTHETIC STRATEGIES")
    print("=" * 70)

    strategies = ['additive', 'mixture']
    results = {}

    for strategy in strategies:
        print(f"\n{'='*70}\nStrategy: {strategy.upper()}\n{'='*70}")

        X_train, y_train, rm_train, X_val, y_val, rm_val, X_test, y_test, rm_test, stats = \
            prepare_dataset(real_data, generator, samples_per_combo=500, gen_mode=strategy)

        print(f"Train: {len(X_train)} (real: {rm_train.sum()})")
        print(f"Test: {len(X_test)} (real: {rm_test.sum()})")

        model = build_model(3)
        callbacks = [keras.callbacks.EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]

        model.fit(X_train, y_train, validation_data=(X_val, y_val),
                 epochs=100, batch_size=32, callbacks=callbacks, verbose=0)

        results[strategy] = evaluate_model(model, X_test, y_test, rm_test, strategy)

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY: SINGLE-SAMPLE 3-FEATURE RESIDUAL")
    print("=" * 70)

    print(f"\n{'Strategy':<15} {'Overall':>10} {'Real Acc':>10} {'Synth Acc':>10} {'Gap':>10}")
    print("-" * 60)

    for strategy, r in results.items():
        gap = r['real_acc'] - r['synth_acc']
        print(f"{strategy:<15} {r['overall']:>9.1%} {r['real_acc']:>9.1%} {r['synth_acc']:>9.1%} {gap:>+9.1%}")

    print("\n" + "=" * 70)
    print("CONCLUSIONS")
    print("=" * 70)

    best = max(results.items(), key=lambda x: x[1]['real_acc'])
    print(f"\nBest strategy for real data: {best[0]} ({best[1]['real_acc']:.1%})")

    print(f"\nKey insights:")
    print(f"  - Single-sample 3-feature residual achieves ~{best[1]['overall']*100:.0f}% accuracy")
    print(f"  - Real sample accuracy: {best[1]['real_acc']:.1%}")
    print(f"  - Synthetic samples: {best[1]['synth_acc']:.1%}")

    # Save stats for deployment
    output_path = Path(__file__).parent / 'residual_model_stats.json'
    with open(output_path, 'w') as f:
        json.dump({
            'baseline': baseline.tolist(),
            'finger_effects': {k: {'mean': v['mean'].tolist(), 'std': v['std'].tolist()}
                              for k, v in finger_effects.items()},
            'results': {k: {kk: vv if not isinstance(vv, dict) else vv
                           for kk, vv in v.items()} for k, v in results.items()}
        }, f, indent=2)
    print(f"\nSaved stats to {output_path}")


if __name__ == '__main__':
    main()
