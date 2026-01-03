#!/usr/bin/env python3
"""
Physics Model with Alternating Magnet Polarity.

Key insight from user: Magnets have ALTERNATING polarity between fingers
to help differentiate signals.

Observed pattern:
- Ring has OPPOSITE X,Y signs compared to all other fingers
- This suggests polarity pattern: +, -, +, -, + or similar
- Adjacent fingers with opposite polarity have fields that partially cancel

This explains the strong sub-additivity (0.24x for ring+pinky).

Model:
1. Single-finger effects: Ground truth (observed)
2. Polarity pattern: Alternating, inferred from data
3. Pairwise coupling: Adjacent fingers with opposite polarity cancel partially
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import minimize
import magpylib as magpy

np.random.seed(42)

print("=" * 70)
print("PHYSICS MODEL WITH ALTERNATING POLARITY")
print("=" * 70)


def load_data():
    """Load observed residuals."""
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
            'n': len(residuals)
        }

    return observed, baseline


class PolarityAwareModel:
    """
    Model that accounts for alternating magnet polarity.

    The model uses:
    1. Single-finger effects as base (ground truth)
    2. Polarity factors for each finger
    3. Pairwise coupling matrix for adjacent fingers
    4. Magnetic shielding factor for multiple magnets
    """

    FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']

    def __init__(self):
        # Polarity: inferred from single-finger X,Y patterns
        # Ring is opposite, so likely: +, +, +, -, + or +, -, +, -, +
        # From Z-axis: thumb is opposite (negative), others positive
        # Combine: thumb has unique polarity, ring is opposite in XY
        self.polarity = np.array([1, 1, 1, -1, 1])  # Initial guess

        # Single-finger effects (will be loaded from data)
        self.single_effects = {}

        # Pairwise coupling: how much adjacent fingers affect each other
        # Values < 1 mean cancellation, > 1 means reinforcement
        # [0,1], [1,2], [2,3], [3,4] for adjacent pairs
        self.adjacent_coupling = np.array([0.6, 0.7, 0.5, 0.3])  # Initial guess

        # Global multi-finger attenuation (beyond pairwise)
        self.global_atten = np.array([1.0, 1.0, 0.8, 0.6, 0.5, 0.4])  # 0-5 fingers

    def load_single_effects(self, observed):
        """Load single-finger effects from observations."""
        single_combos = ['feeee', 'efeee', 'eefee', 'eeefe', 'eeeef']
        for combo, finger in zip(single_combos, self.FINGER_NAMES):
            if combo in observed:
                self.single_effects[finger] = observed[combo]['mean'].copy()

    def predict(self, combo):
        """Predict residual for a combo."""
        if combo == 'eeeee':
            return np.zeros(3)

        flexed = [i for i, c in enumerate(combo) if c == 'f']
        n = len(flexed)

        if n == 0:
            return np.zeros(3)

        if n == 1:
            finger = self.FINGER_NAMES[flexed[0]]
            return self.single_effects.get(finger, np.zeros(3)).copy()

        # Multi-finger: Apply pairwise coupling and global attenuation
        result = np.zeros(3)

        for i in flexed:
            finger = self.FINGER_NAMES[i]
            effect = self.single_effects.get(finger, np.zeros(3)).copy()

            # Apply pairwise coupling with other flexed fingers
            for j in flexed:
                if i == j:
                    continue

                # Check if adjacent
                if abs(i - j) == 1:
                    pair_idx = min(i, j)
                    coupling = self.adjacent_coupling[pair_idx]

                    # If opposite polarity, coupling reduces the effect
                    if self.polarity[i] != self.polarity[j]:
                        effect *= coupling

            result += effect

        # Apply global attenuation
        result *= self.global_atten[min(n, 5)]

        return result

    def fit(self, observed):
        """Fit model parameters to observed multi-finger combos."""
        multi_combos = [c for c in observed.keys() if c.count('f') > 1]

        if not multi_combos:
            return

        print(f"\nFitting to {len(multi_combos)} multi-finger combos...")

        def objective(params):
            # params: 4 adjacent couplings + 4 global attenuations (2-5 fingers)
            self.adjacent_coupling = np.clip(params[:4], 0.1, 1.5)
            self.global_atten[2:6] = np.clip(params[4:8], 0.1, 1.0)

            error = 0
            for combo in multi_combos:
                obs = observed[combo]['mean']
                pred = self.predict(combo)
                error += np.sum((pred - obs) ** 2)

            return error

        # Initial guess
        x0 = list(self.adjacent_coupling) + list(self.global_atten[2:6])
        bounds = [(0.1, 1.5)] * 4 + [(0.1, 1.0)] * 4

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
        objective(result.x)  # Apply best params

        print(f"Optimization converged: {result.success}")
        print(f"Final MSE: {result.fun:.1f}")

        print("\nFitted parameters:")
        print(f"  Adjacent coupling (T-I, I-M, M-R, R-P): {self.adjacent_coupling}")
        print(f"  Global attenuation (2-5 fingers): {self.global_atten[2:6]}")

    def validate(self, observed):
        """Validate on all observed combos."""
        print("\n" + "=" * 70)
        print("VALIDATION")
        print("=" * 70)

        errors = []
        for combo in sorted(observed.keys()):
            obs = observed[combo]['mean']
            pred = self.predict(combo)
            error = np.linalg.norm(pred - obs)
            rel_error = error / (np.linalg.norm(obs) + 1e-6) * 100

            errors.append((combo, error, rel_error))

            status = "✓" if rel_error < 15 else "~" if rel_error < 40 else "✗"
            obs_str = f"[{obs[0]:>6.0f}, {obs[1]:>6.0f}, {obs[2]:>6.0f}]"
            pred_str = f"[{pred[0]:>6.0f}, {pred[1]:>6.0f}, {pred[2]:>6.0f}]"
            print(f"{combo}: obs={obs_str} pred={pred_str} err={rel_error:>5.1f}% {status}")

        mean_rel = np.mean([e[2] for e in errors])
        print(f"\nMean relative error: {mean_rel:.1f}%")

        return errors


def analyze_polarity_pattern():
    """Analyze the actual polarity pattern from data."""
    print("\n" + "=" * 70)
    print("POLARITY PATTERN ANALYSIS")
    print("=" * 70)

    observed, _ = load_data()

    # Single-finger effects
    effects = {}
    single_combos = ['feeee', 'efeee', 'eefee', 'eeefe', 'eeeef']
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

    for combo, finger in zip(single_combos, finger_names):
        if combo in observed:
            effects[finger] = observed[combo]['mean']

    print("\nInferred polarity from X-axis sign:")
    for finger in finger_names:
        e = effects[finger]
        pol = "+" if e[0] > 0 else "-"
        print(f"  {finger}: X={e[0]:>7.0f} → polarity {pol}")

    # The pattern appears to be: +, +, +, -, +
    # Only ring is negative in X

    print("\nPolarity pattern: [+, +, +, -, +] (ring is opposite)")

    # Test: do ring+pinky have opposite polarity?
    ring_e = effects['ring']
    pinky_e = effects['pinky']

    print(f"\nRing-Pinky interaction (adjacent, opposite polarity):")
    print(f"  Ring:  [{ring_e[0]:>6.0f}, {ring_e[1]:>6.0f}, {ring_e[2]:>6.0f}]")
    print(f"  Pinky: [{pinky_e[0]:>6.0f}, {pinky_e[1]:>6.0f}, {pinky_e[2]:>6.0f}]")

    dot = np.dot(ring_e, pinky_e)
    mag_r = np.linalg.norm(ring_e)
    mag_p = np.linalg.norm(pinky_e)
    cos_angle = dot / (mag_r * mag_p)
    angle = np.arccos(np.clip(cos_angle, -1, 1)) * 180 / np.pi

    print(f"  Angle between: {angle:.1f}° (180° = opposite, 0° = same direction)")
    print(f"  Dot product: {dot:.0f} ({'opposite' if dot < 0 else 'same'} direction)")


def main():
    # Analyze polarity pattern
    analyze_polarity_pattern()

    # Load data
    observed, baseline = load_data()
    print(f"\nBaseline: [{baseline[0]:.1f}, {baseline[1]:.1f}, {baseline[2]:.1f}] μT")

    # Create and fit model
    model = PolarityAwareModel()
    model.load_single_effects(observed)

    print("\n" + "=" * 70)
    print("BEFORE FITTING")
    print("=" * 70)
    model.validate(observed)

    # Fit to multi-finger observations
    model.fit(observed)

    # Validate after fitting
    errors = model.validate(observed)

    # Generate all 32 predictions
    print("\n" + "=" * 70)
    print("ALL 32 COMBO PREDICTIONS")
    print("=" * 70)

    predictions = {}
    for i in range(32):
        combo = ''.join(['f' if (i >> (4-j)) & 1 else 'e' for j in range(5)])
        pred = model.predict(combo)
        predictions[combo] = pred

        status = "OBS" if combo in observed else "SYN"
        mag = np.linalg.norm(pred)
        print(f"{combo}: [{pred[0]:>7.0f}, {pred[1]:>7.0f}, {pred[2]:>7.0f}] mag={mag:>6.0f} [{status}]")

    # Save results
    output = {
        'polarity': model.polarity.tolist(),
        'adjacent_coupling': model.adjacent_coupling.tolist(),
        'global_attenuation': model.global_atten.tolist(),
        'single_effects': {k: v.tolist() for k, v in model.single_effects.items()},
        'predictions': {k: v.tolist() for k, v in predictions.items()},
        'validation_errors': [(c, e, r) for c, e, r in errors],
        'mean_error': np.mean([e[2] for e in errors])
    }

    output_path = Path(__file__).parent / 'polarity_model_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")


if __name__ == '__main__':
    main()
