#!/usr/bin/env python3
"""
Physics Model Fitting to Observed Residuals.

Strategy:
1. Fit per-finger magnet parameters to EXACTLY match single-finger observations
2. Model multi-finger combinations with learned interaction terms
3. Validate on observed multi-finger combos
4. Generate synthetic for unobserved combos

Key insight: We have ALL 5 single-finger observations as ground truth!
- feeee (thumb): [341.3, -66.7, -389.4] μT
- efeee (index): [583.8, -158.6, 1030.1] μT
- eefee (middle): [657.2, -563.0, 746.4] μT
- eeefe (ring): [-663.4, 416.5, 247.9] μT
- eeeef (pinky): [543.8, -872.6, 2720.0] μT

Multi-finger combos show sub-additive behavior (0.18x - 0.65x of sum).
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
from scipy.optimize import minimize, differential_evolution
import magpylib as magpy

np.random.seed(42)

print("=" * 70)
print("PHYSICS MODEL FITTING TO OBSERVATIONS")
print("=" * 70)


def load_observed_data() -> Dict[str, Dict]:
    """Load all observed combo residuals."""
    session_path = Path(__file__).parent.parent / 'data' / 'GAMBIT' / '2025-12-31T14_06_18.270Z.json'
    with open(session_path) as f:
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
            'n': len(residuals),
            'cov': np.cov(residuals.T) if len(residuals) > 3 else np.eye(3) * residuals.std(axis=0)**2
        }

    return observed, baseline


class PerFingerPhysicsModel:
    """
    Physics model with per-finger dipole parameters.

    Each finger has:
    - dipole_moment: 3D vector [mx, my, mz] in μT·mm³
    - position_extended: 3D position when extended
    - position_flexed: 3D position when flexed

    For simplicity, we fit the EFFECTIVE dipole moment that produces
    the observed field at the sensor, rather than physical magnet params.
    """

    FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']

    def __init__(self):
        # Initialize with single-finger observations as effective dipoles
        self.single_effects = {}

        # Pairwise interaction matrix: interaction[i,j] = scaling when both i,j flexed
        # Values < 1 mean sub-additive, > 1 means super-additive
        self.pairwise_interaction = np.ones((5, 5))

        # Global multi-finger scaling: additional attenuation based on count
        self.multi_scaling = [1.0, 1.0, 0.7, 0.5, 0.35, 0.25]  # 0, 1, 2, 3, 4, 5 fingers

    def set_single_effects(self, observed: Dict):
        """Set single-finger effects from observations."""
        single_combos = {
            'feeee': 'thumb', 'efeee': 'index', 'eefee': 'middle',
            'eeefe': 'ring', 'eeeef': 'pinky'
        }

        for combo, finger in single_combos.items():
            if combo in observed:
                self.single_effects[finger] = observed[combo]['mean'].copy()

        print("\nSingle-finger effects loaded:")
        for finger in self.FINGER_NAMES:
            if finger in self.single_effects:
                e = self.single_effects[finger]
                print(f"  {finger}: [{e[0]:>7.1f}, {e[1]:>7.1f}, {e[2]:>7.1f}]")

    def predict_combo(self, combo: str) -> np.ndarray:
        """Predict residual for a combo using current model."""
        if combo == 'eeeee':
            return np.zeros(3)

        # Get flexed fingers
        flexed_indices = [i for i, c in enumerate(combo) if c == 'f']
        n_flexed = len(flexed_indices)

        if n_flexed == 0:
            return np.zeros(3)

        if n_flexed == 1:
            # Single finger - return exact effect
            finger = self.FINGER_NAMES[flexed_indices[0]]
            return self.single_effects.get(finger, np.zeros(3)).copy()

        # Multi-finger: sum with pairwise interactions and global scaling
        result = np.zeros(3)

        for i in flexed_indices:
            finger = self.FINGER_NAMES[i]
            effect = self.single_effects.get(finger, np.zeros(3))

            # Apply pairwise interaction scaling
            pair_scale = 1.0
            for j in flexed_indices:
                if i != j:
                    pair_scale *= self.pairwise_interaction[i, j]

            # Geometric mean for multiple interactions
            if len(flexed_indices) > 1:
                pair_scale = pair_scale ** (1.0 / (len(flexed_indices) - 1))

            result += effect * pair_scale

        # Apply global multi-finger scaling
        result *= self.multi_scaling[min(n_flexed, 5)]

        return result

    def fit_interactions(self, observed: Dict):
        """Fit pairwise and global interactions to multi-finger observations."""
        multi_combos = ['ffeee', 'eeeff', 'eefff', 'fffff']
        available = [c for c in multi_combos if c in observed]

        if not available:
            print("No multi-finger observations to fit!")
            return

        print(f"\nFitting interactions to {len(available)} multi-finger combos...")

        def objective(params):
            # params: 5 global scaling values for 1-5 fingers, then pairwise
            self.multi_scaling = [1.0] + list(params[:5])

            # Pairwise: upper triangle (10 values)
            idx = 5
            for i in range(5):
                for j in range(i+1, 5):
                    self.pairwise_interaction[i, j] = params[idx]
                    self.pairwise_interaction[j, i] = params[idx]
                    idx += 1

            # Compute error
            total_error = 0
            for combo in available:
                obs = observed[combo]['mean']
                pred = self.predict_combo(combo)
                error = np.linalg.norm(pred - obs)
                total_error += error ** 2

            return total_error

        # Initial guess: current values
        x0 = list(self.multi_scaling[1:6])  # 5 values
        x0 += [1.0] * 10  # 10 pairwise values

        # Bounds
        bounds = [(0.1, 1.0)] * 5 + [(0.1, 2.0)] * 10

        result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)

        # Apply best params
        objective(result.x)

        print(f"Optimization converged: {result.success}")
        print(f"Final error: {np.sqrt(result.fun):.1f} μT")

        print("\nFitted multi-finger scaling:")
        for i in range(1, 6):
            print(f"  {i} fingers: {self.multi_scaling[i]:.3f}")

        print("\nFitted pairwise interactions:")
        for i in range(5):
            row = [f"{self.pairwise_interaction[i,j]:.2f}" for j in range(5)]
            print(f"  {self.FINGER_NAMES[i]}: {row}")

    def validate(self, observed: Dict):
        """Validate model against all observations."""
        print("\n" + "=" * 70)
        print("VALIDATION")
        print("=" * 70)
        print(f"{'Combo':<8} | {'Observed':<30} | {'Predicted':<30} | {'Error':>8}")
        print("-" * 85)

        errors = []
        for combo in sorted(observed.keys()):
            obs = observed[combo]['mean']
            pred = self.predict_combo(combo)

            error = np.linalg.norm(pred - obs)
            rel_error = error / (np.linalg.norm(obs) + 1e-6) * 100
            errors.append((combo, error, rel_error))

            obs_str = f"[{obs[0]:>7.0f}, {obs[1]:>7.0f}, {obs[2]:>7.0f}]"
            pred_str = f"[{pred[0]:>7.0f}, {pred[1]:>7.0f}, {pred[2]:>7.0f}]"
            status = "✓" if rel_error < 25 else "~" if rel_error < 50 else "✗"
            print(f"{combo:<8} | {obs_str} | {pred_str} | {rel_error:>6.1f}% {status}")

        mean_error = np.mean([e[1] for e in errors])
        mean_rel_error = np.mean([e[2] for e in errors])
        print(f"\nMean absolute error: {mean_error:.1f} μT")
        print(f"Mean relative error: {mean_rel_error:.1f}%")

        return errors

    def generate_all_combos(self) -> Dict[str, np.ndarray]:
        """Generate predictions for all 32 combos."""
        predictions = {}
        for i in range(32):
            combo = ''.join(['f' if (i >> (4-j)) & 1 else 'e' for j in range(5)])
            predictions[combo] = self.predict_combo(combo)
        return predictions


def main():
    # Load data
    observed, baseline = load_observed_data()
    print(f"\nLoaded {len(observed)} observed combos")
    print(f"Baseline: [{baseline[0]:.1f}, {baseline[1]:.1f}, {baseline[2]:.1f}] μT")

    # Create model
    model = PerFingerPhysicsModel()

    # Set single-finger effects from observations
    model.set_single_effects(observed)

    # Before fitting
    print("\n" + "=" * 70)
    print("BEFORE FITTING (using default interactions)")
    print("=" * 70)
    model.validate(observed)

    # Fit interactions
    model.fit_interactions(observed)

    # After fitting
    print("\n" + "=" * 70)
    print("AFTER FITTING")
    print("=" * 70)
    errors = model.validate(observed)

    # Generate all predictions
    print("\n" + "=" * 70)
    print("PREDICTIONS FOR ALL 32 COMBOS")
    print("=" * 70)

    predictions = model.generate_all_combos()

    observed_combos = set(observed.keys())
    print(f"\n{'Combo':<8} | {'Predicted':<30} | {'Magnitude':>10} | Status")
    print("-" * 65)

    for combo in sorted(predictions.keys()):
        pred = predictions[combo]
        mag = np.linalg.norm(pred)
        status = "observed ✓" if combo in observed_combos else "synthetic"
        pred_str = f"[{pred[0]:>7.0f}, {pred[1]:>7.0f}, {pred[2]:>7.0f}]"
        print(f"{combo:<8} | {pred_str} | {mag:>10.1f} | {status}")

    # Save model
    output = {
        'baseline': baseline.tolist(),
        'single_effects': {k: v.tolist() for k, v in model.single_effects.items()},
        'multi_scaling': model.multi_scaling,
        'pairwise_interaction': model.pairwise_interaction.tolist(),
        'predictions': {k: v.tolist() for k, v in predictions.items()},
        'validation_errors': [(c, e, r) for c, e, r in errors],
        'observed_combos': list(observed_combos)
    }

    output_path = Path(__file__).parent / 'physics_fit_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    # Compute improvement over simple additive
    print("\n" + "=" * 70)
    print("IMPROVEMENT OVER ADDITIVE MODEL")
    print("=" * 70)

    additive_errors = []
    fitted_errors = []

    for combo in observed.keys():
        if combo == 'eeeee':
            continue

        obs = observed[combo]['mean']

        # Additive prediction
        additive = np.zeros(3)
        for i, c in enumerate(combo):
            if c == 'f':
                finger = model.FINGER_NAMES[i]
                if finger in model.single_effects:
                    additive += model.single_effects[finger]

        additive_error = np.linalg.norm(additive - obs) / (np.linalg.norm(obs) + 1e-6) * 100
        additive_errors.append(additive_error)

        # Fitted prediction
        pred = model.predict_combo(combo)
        fitted_error = np.linalg.norm(pred - obs) / (np.linalg.norm(obs) + 1e-6) * 100
        fitted_errors.append(fitted_error)

    print(f"Additive model mean error: {np.mean(additive_errors):.1f}%")
    print(f"Fitted model mean error:   {np.mean(fitted_errors):.1f}%")
    print(f"Improvement: {np.mean(additive_errors) - np.mean(fitted_errors):.1f}% reduction")


if __name__ == '__main__':
    main()
