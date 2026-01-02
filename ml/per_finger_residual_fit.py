#!/usr/bin/env python3
"""
Per-Finger Residual Analysis and Fitting.

Goal: Understand what each finger contributes to the residual field.
Then use this to predict combinations.

Key insight from earlier: Additivity fails (76% error for thumb+index).
So we need to capture interactions somehow.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from scipy.optimize import minimize

print("=" * 70)
print("PER-FINGER RESIDUAL ANALYSIS")
print("=" * 70)


def load_observed_residuals() -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Load observed residual means and stds from real data."""
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

    result = {}
    for combo, mags in combo_samples.items():
        mags = np.array(mags)
        residuals = mags - baseline
        result[combo] = (residuals.mean(axis=0), residuals.std(axis=0))

    return result


def extract_single_finger_effects(observed: Dict) -> Dict[str, np.ndarray]:
    """Extract residual effect of each single finger from data."""
    single_finger_combos = {
        'thumb': 'feeee',
        'index': 'efeee',
        'middle': 'eefee',
        'ring': 'eeefe',
        'pinky': 'eeeef',
    }

    effects = {}
    for finger, combo in single_finger_combos.items():
        if combo in observed:
            effects[finger] = observed[combo][0]
        else:
            effects[finger] = np.zeros(3)

    return effects


def print_single_finger_analysis(observed: Dict):
    """Analyze single finger residuals."""
    print("\n" + "=" * 70)
    print("SINGLE FINGER RESIDUAL EFFECTS")
    print("=" * 70)

    effects = extract_single_finger_effects(observed)

    print(f"\n{'Finger':<10} {'Residual [X, Y, Z] (μT)':<35} {'Magnitude':<12} {'Dominant'}")
    print("-" * 75)

    for finger, residual in effects.items():
        mag = np.linalg.norm(residual)
        dominant = ['X', 'Y', 'Z'][np.argmax(np.abs(residual))]
        res_str = f"[{residual[0]:+7.0f}, {residual[1]:+7.0f}, {residual[2]:+7.0f}]"
        print(f"{finger:<10} {res_str:<35} {mag:>10.0f}  {dominant}")

    return effects


def test_additivity(observed: Dict, effects: Dict[str, np.ndarray]):
    """Test if finger effects are additive."""
    print("\n" + "=" * 70)
    print("ADDITIVITY TEST")
    print("=" * 70)

    # Test combinations we have observed
    multi_finger_combos = {
        'ffeee': ['thumb', 'index'],
        'eeeff': ['ring', 'pinky'],
        'eefff': ['middle', 'ring', 'pinky'],
        'fffff': ['thumb', 'index', 'middle', 'ring', 'pinky'],
    }

    print(f"\n{'Combo':<8} {'Fingers':<25} {'Predicted':<30} {'Observed':<30} {'Error%'}")
    print("-" * 100)

    errors = []
    for combo, fingers in multi_finger_combos.items():
        if combo not in observed:
            continue

        # Additive prediction
        predicted = sum(effects[f] for f in fingers)
        obs_mean = observed[combo][0]

        error = np.linalg.norm(predicted - obs_mean)
        rel_error = error / (np.linalg.norm(obs_mean) + 1e-6) * 100
        errors.append(rel_error)

        pred_str = f"[{predicted[0]:+7.0f}, {predicted[1]:+7.0f}, {predicted[2]:+7.0f}]"
        obs_str = f"[{obs_mean[0]:+7.0f}, {obs_mean[1]:+7.0f}, {obs_mean[2]:+7.0f}]"
        finger_str = '+'.join(fingers)

        status = "✓" if rel_error < 30 else "✗"
        print(f"{combo:<8} {finger_str:<25} {pred_str:<30} {obs_str:<30} {rel_error:5.1f}% {status}")

    if errors:
        print(f"\nMean additivity error: {np.mean(errors):.1f}%")


def fit_with_interactions(observed: Dict):
    """
    Fit a model with pairwise interaction terms.

    residual(combo) = sum(finger_effects) + sum(pairwise_interactions)
    """
    print("\n" + "=" * 70)
    print("FITTING WITH INTERACTION TERMS")
    print("=" * 70)

    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    n_fingers = 5
    n_components = 3

    # Parameters:
    # - 5 finger effects (3 components each) = 15
    # - 10 pairwise interactions (3 components each) = 30
    # Total: 45 parameters

    # But we only have 9 combos × 3 = 27 data points
    # So let's use regularization or simpler interaction model

    # Simpler approach: Just use scaling factors for interactions
    # residual = sum(finger_effects) * interaction_scale

    def combo_to_vec(combo: str) -> np.ndarray:
        """Convert combo string to binary vector."""
        return np.array([1 if c == 'f' else 0 for c in combo])

    def predict(params: np.ndarray, combo: str) -> np.ndarray:
        """Predict residual for a combo."""
        # params[0:15]: finger effects (5 × 3)
        # params[15]: interaction strength

        finger_effects = params[0:15].reshape(5, 3)
        interaction = params[15]

        vec = combo_to_vec(combo)
        n_flexed = vec.sum()

        # Linear sum of finger effects
        linear = sum(vec[i] * finger_effects[i] for i in range(5))

        # Interaction term: scales with number of flexed fingers
        if n_flexed > 1:
            interaction_factor = 1.0 + interaction * (n_flexed - 1) / 4
        else:
            interaction_factor = 1.0

        return linear * interaction_factor

    def objective(params: np.ndarray) -> float:
        """Compute fitting error."""
        total_error = 0
        for combo, (obs_mean, obs_std) in observed.items():
            if combo == 'eeeee':
                continue
            pred = predict(params, combo)
            weights = 1.0 / (obs_std + 50)
            error = np.sum(weights * (pred - obs_mean) ** 2)
            total_error += error
        return total_error

    # Initialize with single-finger effects
    effects = extract_single_finger_effects(observed)
    x0 = np.zeros(16)
    for i, finger in enumerate(finger_names):
        if finger in effects:
            x0[i*3:(i+1)*3] = effects[finger]
    x0[15] = -0.3  # Initial interaction (negative = suppression)

    # Optimize
    result = minimize(objective, x0, method='L-BFGS-B',
                     options={'maxiter': 500})

    fitted_effects = result.x[0:15].reshape(5, 3)
    interaction = result.x[15]

    print(f"\nFitted interaction strength: {interaction:.3f}")
    print(f"Interpretation: Each additional finger scales effect by {1+interaction:.2f}x")

    print(f"\n{'Finger':<10} {'Original Effect':<30} {'Fitted Effect'}")
    print("-" * 70)
    for i, finger in enumerate(finger_names):
        orig = effects.get(finger, np.zeros(3))
        fitted = fitted_effects[i]
        orig_str = f"[{orig[0]:+7.0f}, {orig[1]:+7.0f}, {orig[2]:+7.0f}]"
        fit_str = f"[{fitted[0]:+7.0f}, {fitted[1]:+7.0f}, {fitted[2]:+7.0f}]"
        print(f"{finger:<10} {orig_str:<30} {fit_str}")

    # Validate
    print(f"\n{'Combo':<8} {'Observed':<35} {'Predicted':<35} {'Error%'}")
    print("-" * 90)

    errors = []
    for combo in sorted(observed.keys()):
        obs_mean, obs_std = observed[combo]
        if combo == 'eeeee':
            pred = np.zeros(3)
        else:
            pred = predict(result.x, combo)

        error = np.linalg.norm(pred - obs_mean)
        rel_error = error / (np.linalg.norm(obs_mean) + 1e-6) * 100
        errors.append(rel_error)

        obs_str = f"[{obs_mean[0]:+7.0f}, {obs_mean[1]:+7.0f}, {obs_mean[2]:+7.0f}]"
        pred_str = f"[{pred[0]:+7.0f}, {pred[1]:+7.0f}, {pred[2]:+7.0f}]"

        status = "✓" if rel_error < 30 else "✗"
        print(f"{combo:<8} {obs_str:<35} {pred_str:<35} {rel_error:5.1f}% {status}")

    print(f"\nMean error: {np.mean(errors):.1f}%")

    # Generate all 32 predictions
    print("\n" + "=" * 70)
    print("ALL 32 COMBO PREDICTIONS")
    print("=" * 70)

    all_combos = [f"{t}{i}{m}{r}{p}"
                  for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef']

    print(f"\n{'Combo':<8} {'Predicted (μT)':<35} {'Status'}")
    print("-" * 55)

    predictions = {}
    for combo in sorted(all_combos):
        if combo == 'eeeee':
            pred = np.zeros(3)
        else:
            pred = predict(result.x, combo)
        predictions[combo] = pred.tolist()
        status = "OBSERVED" if combo in observed else "PREDICTED"
        pred_str = f"[{pred[0]:+7.0f}, {pred[1]:+7.0f}, {pred[2]:+7.0f}]"
        print(f"{combo:<8} {pred_str:<35} {status}")

    return result.x, predictions


def main():
    observed = load_observed_residuals()
    print(f"\nLoaded {len(observed)} observed combinations")

    # Single finger analysis
    effects = print_single_finger_analysis(observed)

    # Test additivity
    test_additivity(observed, effects)

    # Fit with interactions
    params, predictions = fit_with_interactions(observed)

    # Save results
    output = {
        'finger_effects': {
            finger: effects[finger].tolist()
            for finger in effects
        },
        'interaction_model': {
            'fitted_effects': params[0:15].reshape(5, 3).tolist(),
            'interaction_strength': float(params[15]),
        },
        'predictions': predictions,
    }

    output_path = Path(__file__).parent / 'per_finger_fit_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
