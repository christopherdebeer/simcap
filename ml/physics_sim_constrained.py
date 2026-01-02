#!/usr/bin/env python3
"""
Physics Simulation with Anatomical Constraints.

Key simplifications:
1. Hand anatomy fixes relative finger positions
2. Only fit global parameters: sensor offset, magnet strength, orientation
3. Finger curl follows anatomical trajectory (not arbitrary positions)

Physical setup (per user clarification):
- Sensor on PALM
- Magnets on MID-FINGER (middle phalanx), PALMAR side
- Fingers CURL TOWARD sensor when flexed
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import magpylib as magpy
from scipy.optimize import minimize

print("=" * 70)
print("PHYSICS SIMULATION - ANATOMICALLY CONSTRAINED")
print("Sensor on palm, magnets on mid-finger, flexed = closer to sensor")
print("=" * 70)


# ===== Anatomical Hand Model =====

class AnatomicalHand:
    """
    Hand model based on anatomical measurements.

    Coordinate system:
    - Origin: Sensor on palm (center of palm)
    - X: Toward fingertips
    - Y: Across palm (positive toward pinky side)
    - Z: Up from palm surface

    All distances in mm.
    """

    # Finger metacarpal lengths (base of finger to MCP joint)
    META_LENGTHS = {
        'thumb': 40,   # Shorter, lateral
        'index': 70,
        'middle': 75,
        'ring': 70,
        'pinky': 60,
    }

    # Finger phalanx lengths (MCP to mid-finger magnet location)
    PROX_LENGTHS = {
        'thumb': 25,
        'index': 35,
        'middle': 40,
        'ring': 35,
        'pinky': 28,
    }

    # Lateral (Y) positions of finger bases relative to palm center
    LATERAL_POS = {
        'thumb': -25,  # Thumb is on the side
        'index': -12,
        'middle': 0,
        'ring': 12,
        'pinky': 22,
    }

    def __init__(self, sensor_offset: np.ndarray = None,
                 magnet_strength: float = 1.0,
                 extended_height: float = 30.0,
                 flexed_height: float = 10.0):
        """
        Initialize hand model.

        Args:
            sensor_offset: [dx, dy, dz] offset of sensor from palm center
            magnet_strength: Global scaling for magnet strength
            extended_height: Z height of magnets when extended (far from palm)
            flexed_height: Z height of magnets when flexed (close to palm)
        """
        self.sensor_offset = sensor_offset if sensor_offset is not None else np.array([0, 0, 0])
        self.magnet_strength = magnet_strength
        self.extended_height = extended_height
        self.flexed_height = flexed_height

    def get_magnet_position(self, finger: str, flexed: bool) -> np.ndarray:
        """Get magnet position for a finger in given state."""
        # Base position (metacarpal)
        x_base = self.META_LENGTHS[finger] * 0.7  # Roughly where MCP is
        y = self.LATERAL_POS[finger]

        if flexed:
            # Finger curls toward palm
            # - X moves back (finger curls, mid-phalanx gets closer to palm center)
            # - Z decreases (closer to palm surface)
            x = x_base - self.PROX_LENGTHS[finger] * 0.3
            z = self.flexed_height
        else:
            # Finger extended
            # - X extends forward
            # - Z is higher (fingers extended, mid-phalanx above palm)
            x = x_base + self.PROX_LENGTHS[finger] * 0.5
            z = self.extended_height

        return np.array([x, y, z]) - self.sensor_offset

    def get_magnet_orientation(self, finger: str, flexed: bool) -> np.ndarray:
        """
        Get magnet polarization direction.

        Key insight from data:
        - Thumb has OPPOSITE Z polarity effect compared to other fingers
        - This is because thumb curls in opposite direction (toward palm center)

        When fingers flex toward palm:
        - Other fingers: magnet approaches from above, field increases in +Z
        - Thumb: magnet approaches from the side, field has -Z component
        """
        y = self.LATERAL_POS[finger]

        if finger == 'thumb':
            # Thumb is oriented differently - curls sideways and down
            # Magnet polarization points toward palm but at an angle
            if flexed:
                # When flexed, thumb magnet is positioned differently
                return np.array([0.3, 0.5, -0.8])  # Points inward and down
            else:
                return np.array([0.2, 0.3, -0.9])  # More vertical when extended
        else:
            # Other fingers have magnets on palmar side pointing toward palm (-Z)
            # When they flex, they approach the sensor from above
            tilt_y = -y / 100  # Slight tilt toward palm center
            return np.array([0.1, tilt_y, -1.0])


def compute_field(hand: AnatomicalHand, combo: str,
                  magnet_diameter: float = 6.0,
                  magnet_height: float = 3.0,
                  polarization: float = 1400) -> np.ndarray:
    """Compute total magnetic field at sensor from all magnets."""
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

    magnets = []
    for name, state in zip(finger_names, combo):
        is_flexed = (state == 'f')
        pos = hand.get_magnet_position(name, is_flexed)
        orientation = hand.get_magnet_orientation(name, is_flexed)
        orientation = orientation / np.linalg.norm(orientation)

        # Scale polarization by magnet strength
        pol = polarization * hand.magnet_strength * orientation

        magnet = magpy.magnet.Cylinder(
            polarization=tuple(pol),
            dimension=(magnet_diameter, magnet_height),
            position=pos,
        )
        magnets.append(magnet)

    collection = magpy.Collection(*magnets)
    B = collection.getB([0, 0, 0])  # Sensor at origin

    return B * 1000  # Convert to μT


def compute_residual(hand: AnatomicalHand, combo: str) -> np.ndarray:
    """Compute residual field relative to baseline (eeeee)."""
    field = compute_field(hand, combo)
    baseline = compute_field(hand, 'eeeee')
    return field - baseline


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


def objective(params: np.ndarray, observed: Dict) -> float:
    """
    Compute fitting error.

    Parameters:
    - params[0:3]: sensor offset (dx, dy, dz)
    - params[3]: magnet strength scale
    - params[4]: extended height
    - params[5]: flexed height
    """
    sensor_offset = params[0:3]
    magnet_strength = params[3]
    extended_height = params[4]
    flexed_height = params[5]

    hand = AnatomicalHand(
        sensor_offset=sensor_offset,
        magnet_strength=magnet_strength,
        extended_height=extended_height,
        flexed_height=flexed_height,
    )

    total_error = 0
    for combo, (obs_mean, obs_std) in observed.items():
        if combo == 'eeeee':
            continue

        try:
            sim_residual = compute_residual(hand, combo)
            # Weighted MSE (weight by inverse std, with floor)
            weights = 1.0 / (obs_std + 50)
            error = np.sum(weights * (sim_residual - obs_mean) ** 2)
            total_error += error
        except Exception as e:
            print(f"Error computing {combo}: {e}")
            total_error += 1e10

    return total_error


def validate(hand: AnatomicalHand, observed: Dict) -> float:
    """Validate model against observed data."""
    print("\n" + "=" * 70)
    print("VALIDATION")
    print("=" * 70)
    print(f"\n{'Combo':<8} {'Observed (μT)':<35} {'Simulated (μT)':<35} {'Error'}")
    print("-" * 95)

    errors = []
    for combo in sorted(observed.keys()):
        obs_mean, obs_std = observed[combo]

        if combo == 'eeeee':
            sim = np.zeros(3)
        else:
            sim = compute_residual(hand, combo)

        error = np.linalg.norm(sim - obs_mean)
        rel_error = error / (np.linalg.norm(obs_mean) + 1e-6) * 100
        errors.append(error)

        obs_str = f"[{obs_mean[0]:+7.0f}, {obs_mean[1]:+7.0f}, {obs_mean[2]:+7.0f}]"
        sim_str = f"[{sim[0]:+7.0f}, {sim[1]:+7.0f}, {sim[2]:+7.0f}]"

        match_status = "✓" if rel_error < 30 else "✗"
        print(f"{combo:<8} {obs_str:<35} {sim_str:<35} {rel_error:5.1f}% {match_status}")

    mean_error = np.mean(errors)
    print(f"\nMean absolute error: {mean_error:.1f} μT")
    return mean_error


def main():
    # Load observed data
    observed = load_observed_residuals()
    print(f"\nLoaded {len(observed)} observed combinations")

    print("\nObserved residuals (μT):")
    for combo, (mean, std) in sorted(observed.items()):
        mag = np.linalg.norm(mean)
        print(f"  {combo}: [{mean[0]:+7.0f}, {mean[1]:+7.0f}, {mean[2]:+7.0f}] "
              f"(|{mag:.0f}|) ± [{std[0]:4.0f}, {std[1]:4.0f}, {std[2]:4.0f}]")

    # Initial parameters
    # [sensor_offset_x, y, z, magnet_strength, extended_height, flexed_height]
    x0 = np.array([0.0, 0.0, 0.0, 1.0, 30.0, 10.0])

    # Test initial model
    print("\n--- INITIAL MODEL (before fitting) ---")
    initial_hand = AnatomicalHand(
        sensor_offset=x0[0:3],
        magnet_strength=x0[3],
        extended_height=x0[4],
        flexed_height=x0[5],
    )
    validate(initial_hand, observed)

    # Optimize with sensible bounds
    print("\n" + "=" * 70)
    print("OPTIMIZING (6 parameters)...")
    print("=" * 70)

    bounds = [
        (-30, 30),   # sensor_offset_x
        (-30, 30),   # sensor_offset_y
        (-20, 20),   # sensor_offset_z
        (0.5, 3.0),  # magnet_strength
        (15, 50),    # extended_height
        (5, 25),     # flexed_height
    ]

    result = minimize(
        objective,
        x0,
        args=(observed,),
        method='L-BFGS-B',
        bounds=bounds,
        options={'maxiter': 200, 'disp': True}
    )

    print(f"\nOptimization finished. Final error: {result.fun:.2f}")

    # Extract fitted parameters
    fitted_params = result.x
    print(f"\nFitted parameters:")
    print(f"  Sensor offset: [{fitted_params[0]:.1f}, {fitted_params[1]:.1f}, {fitted_params[2]:.1f}] mm")
    print(f"  Magnet strength: {fitted_params[3]:.2f}x")
    print(f"  Extended height: {fitted_params[4]:.1f} mm")
    print(f"  Flexed height: {fitted_params[5]:.1f} mm")

    # Create fitted model
    fitted_hand = AnatomicalHand(
        sensor_offset=fitted_params[0:3],
        magnet_strength=fitted_params[3],
        extended_height=fitted_params[4],
        flexed_height=fitted_params[5],
    )

    # Validate fitted model
    print("\n--- FITTED MODEL ---")
    mean_error = validate(fitted_hand, observed)

    # Generate all 32 predictions
    print("\n" + "=" * 70)
    print("PREDICTIONS FOR ALL 32 COMBINATIONS")
    print("=" * 70)

    all_combos = [f"{t}{i}{m}{r}{p}"
                  for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef']

    print(f"\n{'Combo':<8} {'Predicted Residual (μT)':<35} {'Status'}")
    print("-" * 60)

    predictions = {}
    for combo in sorted(all_combos):
        if combo == 'eeeee':
            pred = np.zeros(3)
        else:
            pred = compute_residual(fitted_hand, combo)

        predictions[combo] = pred.tolist()
        status = "OBSERVED" if combo in observed else "PREDICTED"
        pred_str = f"[{pred[0]:+7.0f}, {pred[1]:+7.0f}, {pred[2]:+7.0f}]"
        print(f"{combo:<8} {pred_str:<35} {status}")

    # Save results
    output = {
        'fitted_parameters': {
            'sensor_offset': fitted_params[0:3].tolist(),
            'magnet_strength': float(fitted_params[3]),
            'extended_height': float(fitted_params[4]),
            'flexed_height': float(fitted_params[5]),
        },
        'mean_error_uT': float(mean_error),
        'predictions': predictions,
        'observed': {
            combo: {'mean': mean.tolist(), 'std': std.tolist()}
            for combo, (mean, std) in observed.items()
        },
        'anatomical_model': {
            'meta_lengths': AnatomicalHand.META_LENGTHS,
            'prox_lengths': AnatomicalHand.PROX_LENGTHS,
            'lateral_pos': AnatomicalHand.LATERAL_POS,
        }
    }

    output_path = Path(__file__).parent / 'physics_sim_constrained_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
