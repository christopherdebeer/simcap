#!/usr/bin/env python3
"""
Fast physics simulation fitting using analytical gradient and simpler optimization.

Sensor: Palm of hand
Magnets: Mid-finger (middle phalanx), palmar/curl side
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
import magpylib as magpy
from scipy.optimize import minimize

print("=" * 70)
print("PHYSICS SIMULATION - FAST FITTING")
print("Sensor on palm, magnets on mid-finger palmar side")
print("Extended = far from sensor, Flexed = curls toward sensor (closer)")
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


def compute_field_from_params(params: np.ndarray, combo: str) -> np.ndarray:
    """
    Compute magnetic field given parameters.

    Parameters (50 total):
    - Per finger (5 fingers × 10 params each):
      - Extended pos: x, y, z (3)
      - Flexed pos: x, y, z (3)
      - Orientation: ox, oy, oz (3) - will be normalized
      - Magnet strength scale (1)
    """
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

    magnets = []
    for i, (name, state) in enumerate(zip(finger_names, combo)):
        base_idx = i * 10

        if state == 'e':
            pos = params[base_idx:base_idx+3]
        else:
            pos = params[base_idx+3:base_idx+6]

        orientation = params[base_idx+6:base_idx+9]
        orientation = orientation / (np.linalg.norm(orientation) + 1e-9)

        strength = params[base_idx+9]

        # Create magnet with polarization along orientation
        pol = 1400 * strength * orientation  # mT

        magnet = magpy.magnet.Cylinder(
            polarization=tuple(pol),
            dimension=(6, 3),  # 6mm diameter, 3mm height
            position=pos,
        )
        magnets.append(magnet)

    collection = magpy.Collection(*magnets)
    B = collection.getB([0, 0, 0])  # Sensor at origin

    return B * 1000  # Convert to μT


def compute_residual_from_params(params: np.ndarray, combo: str) -> np.ndarray:
    """Compute residual field (relative to eeeee baseline)."""
    field = compute_field_from_params(params, combo)
    baseline = compute_field_from_params(params, 'eeeee')
    return field - baseline


def create_initial_params() -> np.ndarray:
    """Create initial parameter vector based on hand geometry."""
    # Per finger: extended_pos(3), flexed_pos(3), orientation(3), strength(1)
    params = []

    # Finger configurations [x, y] lateral positions
    finger_lateral = {
        'thumb': -20,
        'index': -8,
        'middle': 0,
        'ring': 8,
        'pinky': 18,
    }

    finger_x_ext = {
        'thumb': 25,
        'index': 45,
        'middle': 50,
        'ring': 45,
        'pinky': 35,
    }

    for name in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        y = finger_lateral[name]
        x_ext = finger_x_ext[name]

        # Extended position: fingers stretched OUT, magnet FURTHER from palm sensor
        # (mid-finger is high above palm when finger is straight)
        ext_pos = [x_ext, y, 35]  # Far from palm in Z

        # Flexed position: finger CURLS TOWARD palm sensor, magnet CLOSER
        # (mid-finger comes down toward palm as finger curls)
        flx_pos = [x_ext * 0.5, y * 0.8, 12]  # Close to palm, pulled back in X

        # Orientation: pointing toward palm (downward in our coord system)
        # Actually, let's try pointing upward first
        orientation = [0, 0, 1]

        # Strength scale (can be adjusted per finger)
        strength = 1.0

        params.extend(ext_pos)
        params.extend(flx_pos)
        params.extend(orientation)
        params.append(strength)

    return np.array(params)


def objective(params: np.ndarray, observed: Dict) -> float:
    """Compute fitting error."""
    total_error = 0

    for combo, (obs_mean, obs_std) in observed.items():
        if combo == 'eeeee':
            continue

        try:
            sim_residual = compute_residual_from_params(params, combo)
            # Weighted MSE
            weights = 1.0 / (obs_std + 50)
            error = np.sum(weights * (sim_residual - obs_mean) ** 2)
            total_error += error
        except Exception:
            total_error += 1e10

    return total_error


def validate(params: np.ndarray, observed: Dict):
    """Validate fitted model."""
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    print(f"\n{'Combo':<8} {'Observed (μT)':<35} {'Simulated (μT)':<35} {'Error'}")
    print("-" * 90)

    errors = []
    for combo in sorted(observed.keys()):
        obs_mean, obs_std = observed[combo]

        if combo == 'eeeee':
            sim = np.zeros(3)
        else:
            sim = compute_residual_from_params(params, combo)

        error = np.linalg.norm(sim - obs_mean)
        rel_error = error / (np.linalg.norm(obs_mean) + 1e-6) * 100
        errors.append(error)

        obs_str = f"[{obs_mean[0]:+6.0f}, {obs_mean[1]:+6.0f}, {obs_mean[2]:+6.0f}]"
        sim_str = f"[{sim[0]:+6.0f}, {sim[1]:+6.0f}, {sim[2]:+6.0f}]"

        print(f"{combo:<8} {obs_str:<35} {sim_str:<35} {rel_error:5.1f}%")

    print(f"\nMean absolute error: {np.mean(errors):.1f} μT")
    return np.mean(errors)


def generate_all_predictions(params: np.ndarray) -> Dict[str, np.ndarray]:
    """Generate predictions for all 32 combos."""
    all_combos = [f"{t}{i}{m}{r}{p}"
                  for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef']

    predictions = {}
    for combo in all_combos:
        if combo == 'eeeee':
            predictions[combo] = np.zeros(3)
        else:
            predictions[combo] = compute_residual_from_params(params, combo)

    return predictions


def main():
    # Load observed data
    observed = load_observed_residuals()
    print(f"\nLoaded {len(observed)} observed combinations")

    print("\nObserved residuals (μT):")
    for combo, (mean, std) in sorted(observed.items()):
        print(f"  {combo}: [{mean[0]:+6.0f}, {mean[1]:+6.0f}, {mean[2]:+6.0f}] ± "
              f"[{std[0]:4.0f}, {std[1]:4.0f}, {std[2]:4.0f}]")

    # Initial parameters
    x0 = create_initial_params()
    print(f"\nParameter vector size: {len(x0)}")

    # Test initial model
    print("\n--- INITIAL MODEL (before fitting) ---")
    validate(x0, observed)

    # Optimize
    print("\n" + "=" * 70)
    print("OPTIMIZING...")
    print("=" * 70)

    # Bounds
    bounds = []
    for i in range(len(x0)):
        if i % 10 < 6:  # Position parameters
            bounds.append((-100, 100))
        elif i % 10 < 9:  # Orientation parameters
            bounds.append((-1, 1))
        else:  # Strength
            bounds.append((0.1, 5.0))

    result = minimize(
        objective,
        x0,
        args=(observed,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 500, 'disp': True}
    )

    print(f"\nOptimization finished. Final error: {result.fun:.2f}")

    # Validate fitted model
    print("\n--- FITTED MODEL ---")
    mean_error = validate(result.x, observed)

    # Generate all 32 predictions
    print("\n" + "=" * 70)
    print("PREDICTIONS FOR ALL 32 COMBINATIONS")
    print("=" * 70)

    predictions = generate_all_predictions(result.x)

    print(f"\n{'Combo':<8} {'Predicted Residual (μT)':<35} {'Status'}")
    print("-" * 60)

    for combo in sorted(predictions.keys()):
        pred = predictions[combo]
        status = "OBSERVED" if combo in observed else "PREDICTED"
        pred_str = f"[{pred[0]:+6.0f}, {pred[1]:+6.0f}, {pred[2]:+6.0f}]"
        print(f"{combo:<8} {pred_str:<35} {status}")

    # Save results
    output = {
        'parameters': result.x.tolist(),
        'mean_error_uT': float(mean_error),
        'predictions': {
            combo: pred.tolist() for combo, pred in predictions.items()
        },
        'observed': {
            combo: {'mean': mean.tolist(), 'std': std.tolist()}
            for combo, (mean, std) in observed.items()
        }
    }

    output_path = Path(__file__).parent / 'physics_sim_fitted.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
