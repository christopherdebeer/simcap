#!/usr/bin/env python3
"""
Physics-based Magnetic Field Simulation for Finger State Inference.

Uses magpylib to simulate magnetic fields from finger-mounted magnets.
Goal: Match simulation to observed residuals, then extrapolate to all 32 combos.

Coordinate System:
- Origin: Magnetometer sensor (Puck.js on back of hand)
- X: Toward fingers (along hand)
- Y: Across hand (thumb to pinky direction)
- Z: Up from back of hand

Physical Setup:
- Small neodymium disc magnets on each fingertip
- Puck.js v2 magnetometer on back of hand near wrist
- Magnets positioned ~50-100mm from sensor depending on finger
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
import magpylib as magpy
from scipy.optimize import minimize, differential_evolution

print("=" * 70)
print("PHYSICS-BASED MAGNETIC FIELD SIMULATION")
print("Using magpylib for accurate dipole modeling")
print("=" * 70)


@dataclass
class FingerConfig:
    """Configuration for a single finger's magnet."""
    name: str
    # Position when extended (relative to sensor origin, in mm)
    extended_pos: np.ndarray
    # Position when flexed (curled toward palm)
    flexed_pos: np.ndarray
    # Magnet orientation (magnetization direction)
    orientation: np.ndarray = field(default_factory=lambda: np.array([0, 0, 1]))
    # Magnet strength (magnetic moment in mT*mm³)
    moment: float = 1000.0  # Will be fitted


@dataclass
class HandModel:
    """Complete hand model with all finger magnets."""
    fingers: Dict[str, FingerConfig]
    sensor_pos: np.ndarray = field(default_factory=lambda: np.array([0, 0, 0]))

    def get_magnet_positions(self, combo: str) -> Dict[str, np.ndarray]:
        """Get magnet positions for a given finger combination."""
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
        positions = {}
        for i, (name, char) in enumerate(zip(finger_names, combo)):
            finger = self.fingers[name]
            if char == 'e':
                positions[name] = finger.extended_pos.copy()
            else:
                positions[name] = finger.flexed_pos.copy()
        return positions


def create_default_hand_model() -> HandModel:
    """
    Create a hand model with CORRECT geometry per user clarification.

    Physical setup:
    - Sensor is on PALM of hand (not back)
    - Magnets are on MID-FINGER (middle phalanx, between knuckles)
    - Magnets are on INSIDE (palmar/curl side) of fingers

    Coordinate system:
    - Origin: sensor on palm (near center or wrist area)
    - X: toward fingertips (along hand length)
    - Y: across palm (positive toward pinky)
    - Z: perpendicular to palm, pointing UP (toward back of hand when palm up)

    Finger mechanics:
    - Extended: fingers flat, magnets on underside close to palm
    - Flexed: fingers curl UP, magnets rotate away from palm

    Key insight: The magnets on mid-finger palmar side are close to sensor
    when extended (~15-25mm above palm) and move further when flexed as
    the finger curls up and away.
    """
    # Mid-finger positions (middle phalanx, palmar side)
    # Extended: close to palm surface, facing sensor
    # Flexed: curled up and away from palm

    # Rough hand measurements (mm from palm sensor):
    # - Thumb: lateral, shorter reach
    # - Index to pinky: progressively more lateral (Y+)

    fingers = {
        'thumb': FingerConfig(
            name='thumb',
            # Thumb is special - lateral position, different curl axis
            extended_pos=np.array([25, -20, 15]),   # Close to palm when flat
            flexed_pos=np.array([20, -25, 25]),     # Curls inward and up
            orientation=np.array([0.2, 0.3, 0.93]), # Points up and slightly lateral
        ),
        'index': FingerConfig(
            name='index',
            extended_pos=np.array([45, -8, 12]),    # Mid-finger close to palm
            flexed_pos=np.array([25, -5, 35]),      # Curls up, moves toward palm center
            orientation=np.array([0.1, 0.05, 0.99]),
        ),
        'middle': FingerConfig(
            name='middle',
            extended_pos=np.array([50, 0, 12]),
            flexed_pos=np.array([28, 0, 38]),
            orientation=np.array([0, 0, 1]),
        ),
        'ring': FingerConfig(
            name='ring',
            extended_pos=np.array([45, 8, 12]),
            flexed_pos=np.array([25, 6, 35]),
            orientation=np.array([-0.1, 0.05, 0.99]),
        ),
        'pinky': FingerConfig(
            name='pinky',
            extended_pos=np.array([35, 18, 12]),
            flexed_pos=np.array([22, 14, 30]),
            orientation=np.array([-0.15, 0.1, 0.98]),
        ),
    }
    return HandModel(fingers=fingers)


def compute_magnetic_field(hand: HandModel, combo: str,
                          magnet_diameter: float = 6.0,
                          magnet_height: float = 3.0,
                          polarization: float = 1400) -> np.ndarray:
    """
    Compute total magnetic field at sensor from all finger magnets.

    Uses magpylib for accurate dipole field calculation.
    Returns field in μT (microtesla).

    Args:
        hand: Hand model with finger positions
        combo: 5-char string of 'e' (extended) or 'f' (flexed)
        magnet_diameter: Magnet diameter in mm (typical: 5-10mm)
        magnet_height: Magnet height in mm (typical: 2-5mm)
        polarization: Magnet polarization in mT (NdFeB: 1000-1400)
    """
    positions = hand.get_magnet_positions(combo)

    # Create magpy collection
    magnets = []
    for name, pos in positions.items():
        finger = hand.fingers[name]

        # Normalize orientation
        orientation = finger.orientation / (np.linalg.norm(finger.orientation) + 1e-9)

        # Create cylinder magnet with polarization along orientation
        pol_vector = polarization * orientation

        magnet = magpy.magnet.Cylinder(
            polarization=tuple(pol_vector),  # mT along orientation
            dimension=(magnet_diameter, magnet_height),
            position=pos,
        )

        magnets.append(magnet)

    # Compute total field at sensor position
    collection = magpy.Collection(*magnets)
    B = collection.getB(hand.sensor_pos)  # Returns in mT

    # Convert to μT
    return B * 1000


def compute_residual_field(hand: HandModel, combo: str, baseline_combo: str = 'eeeee') -> np.ndarray:
    """Compute residual field (relative to baseline/open palm)."""
    field = compute_magnetic_field(hand, combo)
    baseline = compute_magnetic_field(hand, baseline_combo)
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

    # Compute residual statistics per combo
    result = {}
    for combo, mags in combo_samples.items():
        mags = np.array(mags)
        residuals = mags - baseline
        result[combo] = (residuals.mean(axis=0), residuals.std(axis=0))

    return result


def fit_hand_model(observed: Dict[str, Tuple[np.ndarray, np.ndarray]],
                   max_iter: int = 100) -> HandModel:
    """
    Fit hand model parameters to match observed residuals.

    Optimizes finger positions and magnet orientations.
    """
    hand = create_default_hand_model()
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

    # Exclude baseline from fitting (it's zero by definition)
    fit_combos = [c for c in observed.keys() if c != 'eeeee']

    print(f"\nFitting model to {len(fit_combos)} observed combinations...")
    print(f"Combos: {', '.join(fit_combos)}")

    # Parameter vector: for each finger, we optimize:
    # - flexed position offset (3 values: dx, dy, dz)
    # - extended position offset (3 values)
    # Total: 5 fingers × 6 = 30 parameters

    def pack_params(hand: HandModel) -> np.ndarray:
        """Pack hand model into parameter vector."""
        params = []
        for name in finger_names:
            finger = hand.fingers[name]
            params.extend(finger.flexed_pos)
            params.extend(finger.extended_pos)
        return np.array(params)

    def unpack_params(params: np.ndarray, hand: HandModel):
        """Unpack parameter vector into hand model."""
        idx = 0
        for name in finger_names:
            finger = hand.fingers[name]
            finger.flexed_pos = params[idx:idx+3].copy()
            idx += 3
            finger.extended_pos = params[idx:idx+3].copy()
            idx += 3

    def objective(params: np.ndarray) -> float:
        """Compute fitting error."""
        unpack_params(params, hand)

        total_error = 0
        for combo in fit_combos:
            obs_mean, obs_std = observed[combo]
            sim_residual = compute_residual_field(hand, combo)

            # Weighted MSE (weight by inverse std)
            weights = 1.0 / (obs_std + 10)  # Add small value to avoid div by zero
            error = np.sum(weights * (sim_residual - obs_mean) ** 2)
            total_error += error

        return total_error

    # Initial parameters
    x0 = pack_params(hand)

    # Bounds: allow positions to vary by ±50mm from initial
    bounds = []
    for i in range(len(x0)):
        bounds.append((x0[i] - 50, x0[i] + 50))

    print(f"Optimizing {len(x0)} parameters...")

    # Use differential evolution for global optimization (single worker for pickling)
    result = differential_evolution(
        objective,
        bounds,
        maxiter=max_iter,
        seed=42,
        disp=True,
        workers=1,  # Single worker to avoid pickle issues
        polish=True
    )

    unpack_params(result.x, hand)
    print(f"\nOptimization complete. Final error: {result.fun:.2f}")

    return hand


def validate_model(hand: HandModel, observed: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    """Validate fitted model against observed data."""
    print("\n" + "=" * 70)
    print("MODEL VALIDATION")
    print("=" * 70)

    print(f"\n{'Combo':<10} {'Observed (μT)':<30} {'Simulated (μT)':<30} {'Error':<10}")
    print("-" * 80)

    total_error = 0
    for combo in sorted(observed.keys()):
        if combo == 'eeeee':
            continue

        obs_mean, obs_std = observed[combo]
        sim_residual = compute_residual_field(hand, combo)

        error = np.linalg.norm(sim_residual - obs_mean)
        rel_error = error / (np.linalg.norm(obs_mean) + 1e-6) * 100
        total_error += error

        obs_str = f"[{obs_mean[0]:+.0f}, {obs_mean[1]:+.0f}, {obs_mean[2]:+.0f}]"
        sim_str = f"[{sim_residual[0]:+.0f}, {sim_residual[1]:+.0f}, {sim_residual[2]:+.0f}]"

        print(f"{combo:<10} {obs_str:<30} {sim_str:<30} {rel_error:.1f}%")

    avg_error = total_error / (len(observed) - 1)
    print(f"\nAverage absolute error: {avg_error:.1f} μT")

    return avg_error


def generate_all_combos(hand: HandModel) -> Dict[str, np.ndarray]:
    """Generate simulated residuals for all 32 finger combinations."""
    all_combos = [f"{t}{i}{m}{r}{p}"
                  for t in 'ef' for i in 'ef' for m in 'ef' for r in 'ef' for p in 'ef']

    results = {}
    for combo in all_combos:
        results[combo] = compute_residual_field(hand, combo)

    return results


def visualize_hand_model(hand: HandModel):
    """Print hand model configuration."""
    print("\n" + "=" * 70)
    print("FITTED HAND MODEL")
    print("=" * 70)

    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

    print(f"\n{'Finger':<10} {'Extended Position (mm)':<30} {'Flexed Position (mm)':<30}")
    print("-" * 70)

    for name in finger_names:
        finger = hand.fingers[name]
        ext = finger.extended_pos
        flx = finger.flexed_pos
        ext_str = f"[{ext[0]:.1f}, {ext[1]:.1f}, {ext[2]:.1f}]"
        flx_str = f"[{flx[0]:.1f}, {flx[1]:.1f}, {flx[2]:.1f}]"
        print(f"{name:<10} {ext_str:<30} {flx_str:<30}")

    # Compute movement vectors
    print(f"\n{'Finger':<10} {'Movement Vector (mm)':<30} {'Distance (mm)':<15}")
    print("-" * 55)
    for name in finger_names:
        finger = hand.fingers[name]
        movement = finger.flexed_pos - finger.extended_pos
        distance = np.linalg.norm(movement)
        mov_str = f"[{movement[0]:+.1f}, {movement[1]:+.1f}, {movement[2]:+.1f}]"
        print(f"{name:<10} {mov_str:<30} {distance:.1f}")


def main():
    # Load observed data
    observed = load_observed_residuals()
    print(f"\nLoaded {len(observed)} observed combinations")

    print("\nObserved residuals:")
    for combo, (mean, std) in sorted(observed.items()):
        if combo == 'eeeee':
            continue
        print(f"  {combo}: [{mean[0]:+.0f}, {mean[1]:+.0f}, {mean[2]:+.0f}] ± [{std[0]:.0f}, {std[1]:.0f}, {std[2]:.0f}]")

    # Test default model first
    print("\n" + "=" * 70)
    print("TESTING DEFAULT MODEL (before fitting)")
    print("=" * 70)
    default_hand = create_default_hand_model()
    validate_model(default_hand, observed)

    # Fit model to observed data
    print("\n" + "=" * 70)
    print("FITTING MODEL TO OBSERVED DATA")
    print("=" * 70)
    fitted_hand = fit_hand_model(observed, max_iter=50)

    # Validate fitted model
    validate_model(fitted_hand, observed)
    visualize_hand_model(fitted_hand)

    # Generate predictions for all 32 combos
    print("\n" + "=" * 70)
    print("PREDICTIONS FOR ALL 32 COMBINATIONS")
    print("=" * 70)
    all_predictions = generate_all_combos(fitted_hand)

    print(f"\n{'Combo':<10} {'Simulated Residual (μT)':<30} {'Status':<15}")
    print("-" * 55)

    for combo in sorted(all_predictions.keys()):
        residual = all_predictions[combo]
        status = "OBSERVED" if combo in observed else "PREDICTED"
        res_str = f"[{residual[0]:+.0f}, {residual[1]:+.0f}, {residual[2]:+.0f}]"
        print(f"{combo:<10} {res_str:<30} {status:<15}")

    # Save results
    output = {
        'hand_model': {
            name: {
                'extended_pos': hand.fingers[name].extended_pos.tolist(),
                'flexed_pos': hand.fingers[name].flexed_pos.tolist(),
                'orientation': hand.fingers[name].orientation.tolist(),
            }
            for name, hand in [(n, fitted_hand) for n in ['thumb', 'index', 'middle', 'ring', 'pinky']]
        },
        'predictions': {
            combo: residual.tolist()
            for combo, residual in all_predictions.items()
        },
        'observed': {
            combo: {'mean': mean.tolist(), 'std': std.tolist()}
            for combo, (mean, std) in observed.items()
        }
    }

    # Fix the hand_model saving
    output['hand_model'] = {
        name: {
            'extended_pos': fitted_hand.fingers[name].extended_pos.tolist(),
            'flexed_pos': fitted_hand.fingers[name].flexed_pos.tolist(),
            'orientation': fitted_hand.fingers[name].orientation.tolist(),
        }
        for name in ['thumb', 'index', 'middle', 'ring', 'pinky']
    }

    output_path = Path(__file__).parent / 'physics_simulation_results.json'
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to {output_path}")


if __name__ == '__main__':
    main()
