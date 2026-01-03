#!/usr/bin/env python3
"""
Synthetic Data Generation from Observed Residuals.

New approach: Instead of physics simulation, use structured interpolation
based on observed combos and finger structure.

Key insight: We have ALL 5 single-finger effects as ground truth,
plus 4 multi-finger combos. This gives us:
- Single-finger effects: EXACT
- Multi-finger effects: Interpolated from observed + scaled singles

For unobserved combo C:
1. Start with scaled sum of single-finger effects
2. Find nearest observed multi-finger combos
3. Interpolate correction from their corrections
4. Add calibrated noise from nearest observed
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import tensorflow as tf
from tensorflow import keras
from scipy.spatial.distance import hamming

np.random.seed(42)
tf.random.set_seed(42)

print("=" * 70)
print("SYNTHETIC DATA FROM STRUCTURED INTERPOLATION")
print("=" * 70)


def load_observed_data() -> Tuple[Dict, np.ndarray]:
    """Load observed residuals and compute stats."""
    session_path = Path(__file__).parent.parent / 'data' / 'GAMBIT' / '2025-12-31T14_06_18.270Z.json'
    with open(session_path) as f:
        data = json.load(f)

    combo_samples = {}
    baseline_mags = []

    for lbl in data['labels']:
        if 'labels' in lbl and isinstance(lbl['labels'], dict):
            fingers = lbl['labels'].get('fingers', {})
            start, end = lbl.get('start_sample', 0), lbl.get('end_sample', 0)
        else:
            fingers = lbl.get('fingers', {})
            start, end = lbl.get('startIndex', 0), lbl.get('endIndex', 0)

        if not fingers or all(v == 'unknown' for v in fingers.values()):
            continue

        combo = ''.join(['e' if fingers.get(f, '?') == 'extended' else 'f'
                        if fingers.get(f, '?') == 'flexed' else '?'
                        for f in ['thumb', 'index', 'middle', 'ring', 'pinky']])

        if combo not in combo_samples:
            combo_samples[combo] = []

        for s in data['samples'][start:end]:
            if 'mx_ut' in s:
                combo_samples[combo].append([s['mx_ut'], s['my_ut'], s['mz_ut']])

    baseline = np.mean(combo_samples.get('eeeee', [[46, -46, 31]]), axis=0)

    observed = {}
    for combo, samples in combo_samples.items():
        residuals = np.array(samples) - baseline
        observed[combo] = {
            'mean': residuals.mean(axis=0),
            'std': residuals.std(axis=0),
            'cov': np.cov(residuals.T) if len(residuals) > 3 else np.diag(residuals.std(axis=0)**2),
            'samples': residuals,
            'n': len(residuals)
        }

    return observed, baseline


def hamming_distance(c1: str, c2: str) -> int:
    """Hamming distance between two combo strings."""
    return sum(a != b for a, b in zip(c1, c2))


def finger_overlap(c1: str, c2: str) -> int:
    """Number of fingers in same state."""
    return sum(a == b for a, b in zip(c1, c2))


class StructuredSyntheticGenerator:
    """
    Generate synthetic data using structural interpolation.

    Model:
    - Single-finger combos: Use observed directly
    - Multi-finger combos: scaled_sum + interpolated_correction
    - Noise: From nearest observed combo's covariance
    """

    FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']

    def __init__(self, observed: Dict):
        self.observed = observed
        self.observed_combos = set(observed.keys())

        # Extract single-finger effects
        self.single_effects = {}
        single_combos = ['feeee', 'efeee', 'eefee', 'eeefe', 'eeeef']
        for combo, finger in zip(single_combos, self.FINGER_NAMES):
            if combo in observed:
                self.single_effects[finger] = observed[combo]['mean'].copy()

        # Compute multi-finger scaling and corrections
        self._compute_interaction_model()

    def _compute_interaction_model(self):
        """Compute scaling factors and corrections for multi-finger combos."""
        # Scaling by finger count (fitted from observed)
        self.scaling = {0: 1.0, 1: 1.0, 2: 0.245, 3: 0.269, 4: 0.35, 5: 0.167}

        # Corrections for observed multi-finger combos
        self.corrections = {}
        multi_combos = [c for c in self.observed_combos if c.count('f') > 1]

        for combo in multi_combos:
            n = combo.count('f')
            scaled_sum = self._scaled_sum(combo)
            correction = self.observed[combo]['mean'] - scaled_sum
            self.corrections[combo] = correction

        print(f"\nLoaded {len(self.single_effects)} single-finger effects")
        print(f"Computed corrections for {len(self.corrections)} multi-finger combos")

    def _scaled_sum(self, combo: str) -> np.ndarray:
        """Compute scaled sum of single-finger effects."""
        n = combo.count('f')
        if n == 0:
            return np.zeros(3)

        total = np.zeros(3)
        for i, c in enumerate(combo):
            if c == 'f':
                finger = self.FINGER_NAMES[i]
                if finger in self.single_effects:
                    total += self.single_effects[finger]

        return total * self.scaling.get(n, 0.2)

    def _interpolate_correction(self, combo: str) -> Tuple[np.ndarray, float]:
        """
        Interpolate correction from observed multi-finger combos.

        Returns correction vector and confidence (0-1).
        """
        if combo in self.corrections:
            return self.corrections[combo], 1.0

        n = combo.count('f')
        if n <= 1:
            return np.zeros(3), 1.0

        # Find observed multi-finger combos with similar structure
        candidates = []
        for obs_combo, correction in self.corrections.items():
            obs_n = obs_combo.count('f')
            dist = hamming_distance(combo, obs_combo)
            overlap = finger_overlap(combo, obs_combo)

            # Prefer combos with similar finger count and high overlap
            weight = 1.0 / (dist + 0.5) * (overlap / 5.0)
            if obs_n == n:
                weight *= 2.0  # Bonus for same number of fingers

            candidates.append((obs_combo, correction, weight, dist))

        if not candidates:
            return np.zeros(3), 0.0

        # Weighted interpolation
        total_weight = sum(c[2] for c in candidates)
        if total_weight < 0.01:
            return np.zeros(3), 0.0

        interpolated = np.zeros(3)
        for _, correction, weight, _ in candidates:
            interpolated += correction * (weight / total_weight)

        # Confidence based on distance to nearest
        min_dist = min(c[3] for c in candidates)
        confidence = 1.0 / (1.0 + min_dist * 0.3)

        return interpolated, confidence

    def predict_mean(self, combo: str) -> np.ndarray:
        """Predict mean residual for a combo."""
        if combo == 'eeeee':
            return np.zeros(3)

        if combo in self.observed:
            return self.observed[combo]['mean'].copy()

        n = combo.count('f')
        if n == 1:
            # Single finger - exact from observations
            for i, c in enumerate(combo):
                if c == 'f':
                    finger = self.FINGER_NAMES[i]
                    return self.single_effects.get(finger, np.zeros(3)).copy()

        # Multi-finger: scaled sum + interpolated correction
        scaled_sum = self._scaled_sum(combo)
        correction, confidence = self._interpolate_correction(combo)

        return scaled_sum + correction * confidence

    def get_noise_params(self, combo: str) -> Tuple[np.ndarray, float]:
        """Get noise covariance and confidence for a combo."""
        if combo in self.observed:
            return self.observed[combo]['cov'], 1.0

        # Find nearest observed combo
        min_dist = 10
        nearest = 'eeeee'
        for obs_combo in self.observed_combos:
            dist = hamming_distance(combo, obs_combo)
            if dist < min_dist:
                min_dist = dist
                nearest = obs_combo

        # Scale covariance by distance (more uncertainty for distant)
        base_cov = self.observed[nearest]['cov']
        scale = 1.0 + 0.3 * min_dist
        confidence = 1.0 / (1.0 + min_dist * 0.5)

        return base_cov * scale**2, confidence

    def generate_samples(self, combo: str, n_samples: int) -> np.ndarray:
        """Generate synthetic samples for a combo."""
        mean = self.predict_mean(combo)
        cov, _ = self.get_noise_params(combo)

        # Ensure positive definite
        eigvals = np.linalg.eigvalsh(cov)
        if np.any(eigvals < 0):
            cov = cov + np.eye(3) * (abs(eigvals.min()) + 1)

        samples = np.random.multivariate_normal(mean, cov, n_samples)
        return samples

    def validate(self) -> Dict:
        """Validate predictions against observed combos."""
        errors = []
        for combo in sorted(self.observed_combos):
            obs = self.observed[combo]['mean']
            pred = self.predict_mean(combo)
            error = np.linalg.norm(pred - obs)
            rel_error = error / (np.linalg.norm(obs) + 1e-6) * 100

            errors.append({
                'combo': combo,
                'observed': obs.tolist(),
                'predicted': pred.tolist(),
                'error': error,
                'rel_error': rel_error
            })

        return errors


def train_and_evaluate(X_train, y_train, X_test, y_test, name: str) -> float:
    """Train model and return accuracy."""
    # Normalize
    mean = X_train.mean(axis=0)
    std = X_train.std(axis=0) + 1e-6
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    model = keras.Sequential([
        keras.layers.Input(shape=(3,)),
        keras.layers.Dense(32, activation='relu'),
        keras.layers.Dropout(0.2),
        keras.layers.Dense(16, activation='relu'),
        keras.layers.Dense(5, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    model.fit(X_train_norm, y_train, epochs=50, batch_size=32, validation_split=0.1, verbose=0)

    y_pred = (model.predict(X_test_norm, verbose=0) > 0.5).astype(int)
    return (y_pred == y_test).all(axis=1).mean()


def main():
    # Load data
    observed, baseline = load_observed_data()
    print(f"\nBaseline: [{baseline[0]:.1f}, {baseline[1]:.1f}, {baseline[2]:.1f}] μT")
    print(f"Observed combos: {len(observed)}")

    # Create generator
    generator = StructuredSyntheticGenerator(observed)

    # Validate on observed
    print("\n" + "=" * 70)
    print("VALIDATION ON OBSERVED COMBOS")
    print("=" * 70)

    errors = generator.validate()
    for e in sorted(errors, key=lambda x: x['combo']):
        status = "✓" if e['rel_error'] < 5 else "~" if e['rel_error'] < 30 else "✗"
        print(f"{e['combo']}: error = {e['rel_error']:.1f}% {status}")

    mean_error = np.mean([e['rel_error'] for e in errors])
    print(f"\nMean relative error: {mean_error:.1f}%")

    # Generate predictions for all 32 combos
    print("\n" + "=" * 70)
    print("PREDICTIONS FOR ALL 32 COMBOS")
    print("=" * 70)

    all_preds = {}
    for i in range(32):
        combo = ''.join(['f' if (i >> (4-j)) & 1 else 'e' for j in range(5)])
        pred = generator.predict_mean(combo)
        all_preds[combo] = pred

        status = "observed" if combo in observed else "synthetic"
        mag = np.linalg.norm(pred)
        print(f"{combo}: [{pred[0]:>7.0f}, {pred[1]:>7.0f}, {pred[2]:>7.0f}] mag={mag:>6.0f} ({status})")

    # Training experiment
    print("\n" + "=" * 70)
    print("TRAINING EXPERIMENT")
    print("=" * 70)

    # Prepare real data
    X_real = []
    y_real = []
    for combo, data in observed.items():
        samples = data['samples']
        labels = np.array([[1.0 if c == 'f' else 0.0 for c in combo]] * len(samples))
        X_real.append(samples)
        y_real.append(labels)

    X_real = np.vstack(X_real)
    y_real = np.vstack(y_real)

    # Generate synthetic for unobserved
    X_synth = []
    y_synth = []
    for combo in all_preds:
        if combo in observed:
            continue
        samples = generator.generate_samples(combo, 100)
        labels = np.array([[1.0 if c == 'f' else 0.0 for c in combo]] * len(samples))
        X_synth.append(samples)
        y_synth.append(labels)

    X_synth = np.vstack(X_synth)
    y_synth = np.vstack(y_synth)

    print(f"Real samples: {len(X_real)}")
    print(f"Synthetic samples: {len(X_synth)}")

    # Split for testing
    n = len(X_real)
    indices = np.random.permutation(n)
    split = int(0.8 * n)
    train_idx, test_idx = indices[:split], indices[split:]

    # Test 1: Real only
    acc_real = train_and_evaluate(
        X_real[train_idx], y_real[train_idx],
        X_real[test_idx], y_real[test_idx],
        "Real only"
    )
    print(f"\nReal only accuracy: {acc_real*100:.1f}%")

    # Test 2: Hybrid
    X_hybrid = np.vstack([X_real[train_idx], X_synth])
    y_hybrid = np.vstack([y_real[train_idx], y_synth])
    acc_hybrid = train_and_evaluate(
        X_hybrid, y_hybrid,
        X_real[test_idx], y_real[test_idx],
        "Hybrid"
    )
    print(f"Hybrid accuracy: {acc_hybrid*100:.1f}%")

    # Test 3: Hybrid with less synthetic
    X_synth_small = X_synth[::2]  # Half
    y_synth_small = y_synth[::2]
    X_hybrid_small = np.vstack([X_real[train_idx], X_synth_small])
    y_hybrid_small = np.vstack([y_real[train_idx], y_synth_small])
    acc_hybrid_small = train_and_evaluate(
        X_hybrid_small, y_hybrid_small,
        X_real[test_idx], y_real[test_idx],
        "Hybrid (50% synth)"
    )
    print(f"Hybrid (50% synth) accuracy: {acc_hybrid_small*100:.1f}%")

    # Save results
    output = {
        'baseline': baseline.tolist(),
        'validation_errors': errors,
        'mean_validation_error': mean_error,
        'predictions': {k: v.tolist() for k, v in all_preds.items()},
        'training_results': {
            'real_only': acc_real,
            'hybrid': acc_hybrid,
            'hybrid_50pct': acc_hybrid_small
        }
    }

    output_path = Path(__file__).parent / 'synthetic_interpolation_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    main()
