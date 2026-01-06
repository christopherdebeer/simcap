"""
Physics-Based Magnetic Field Model for Finger State Classification

This module implements a magnetic dipole model to simulate the magnetic field
observed by a sensor on the palm from magnets attached to fingers.

Physics Background:
- Each finger has a small magnet attached after the first joint from the tip
- The sensor (magnetometer) is located on the palm
- When a finger flexes/curls, the magnet moves closer to the sensor
- The magnetic field follows the dipole equation:

  B(r) = (μ₀/4π) * (3(m·r̂)r̂ - m) / |r|³

  where:
  - μ₀/4π ≈ 10⁻⁷ T·m/A
  - m = dipole moment vector (A·m²)
  - r = position vector from dipole to observation point
  - r̂ = unit vector in direction of r

Author: Claude
Date: January 2026
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution
from scipy.spatial.transform import Rotation as R
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
import json
from pathlib import Path


# Physical constants
MU_0_OVER_4PI = 1e-7  # T·m/A (μ₀/4π)


@dataclass
class MagnetConfig:
    """Configuration for a single finger magnet."""
    # Position when finger is extended (relative to sensor, in meters)
    pos_extended: np.ndarray
    # Position when finger is flexed (relative to sensor, in meters)
    pos_flexed: np.ndarray
    # Dipole moment vector (A·m²) - direction and magnitude
    dipole_moment: np.ndarray

    def get_position(self, state: int) -> np.ndarray:
        """Get magnet position for given state (0=extended, 2=flexed)."""
        if state == 0:
            return self.pos_extended
        elif state == 2:
            return self.pos_flexed
        else:
            # Interpolate for intermediate states
            t = state / 2.0
            return (1 - t) * self.pos_extended + t * self.pos_flexed


@dataclass
class HandModel:
    """Complete hand model with 5 finger magnets and sensor."""
    # Magnets for each finger (thumb, index, middle, ring, pinky)
    magnets: Dict[str, MagnetConfig] = field(default_factory=dict)
    # Earth field in sensor frame (for reference, not used in optimization)
    earth_field: np.ndarray = field(default_factory=lambda: np.zeros(3))
    # Sensor noise standard deviation (μT)
    noise_std: float = 1.0

    FINGER_NAMES = ['thumb', 'index', 'middle', 'ring', 'pinky']

    def compute_field(self, finger_states: Dict[str, int],
                      include_earth: bool = False) -> np.ndarray:
        """
        Compute total magnetic field at sensor from all magnets.

        Args:
            finger_states: Dict mapping finger name to state (0=extended, 2=flexed)
            include_earth: Whether to include Earth's field

        Returns:
            Magnetic field vector (μT) at sensor location
        """
        total_field = np.zeros(3)

        for finger_name, state in finger_states.items():
            if finger_name in self.magnets:
                magnet = self.magnets[finger_name]
                pos = magnet.get_position(state)
                dipole = magnet.dipole_moment

                # Compute dipole field
                field = self._dipole_field(pos, dipole)
                total_field += field

        if include_earth:
            total_field += self.earth_field

        return total_field

    def _dipole_field(self, r: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        Compute magnetic field from a dipole.

        B(r) = (μ₀/4π) * (3(m·r̂)r̂ - m) / |r|³

        Args:
            r: Position vector from dipole to observation point (meters)
            m: Dipole moment vector (A·m²)

        Returns:
            Magnetic field in Tesla, converted to μT
        """
        r_mag = np.linalg.norm(r)
        if r_mag < 1e-10:
            return np.zeros(3)

        r_hat = r / r_mag

        # Dipole field formula
        m_dot_r = np.dot(m, r_hat)
        B = MU_0_OVER_4PI * (3 * m_dot_r * r_hat - m) / (r_mag ** 3)

        # Convert Tesla to μT
        return B * 1e6

    def predict_all_states(self) -> Dict[str, np.ndarray]:
        """
        Predict magnetic field for all 32 possible finger states.

        Returns:
            Dict mapping state code (e.g., '00000') to predicted field (μT)
        """
        predictions = {}

        for state_int in range(32):
            # Convert to binary string representing finger states
            state_bits = format(state_int, '05b')
            state_code = ''.join('2' if b == '1' else '0' for b in state_bits)

            # Build finger state dict
            finger_states = {
                name: int(state_code[i])
                for i, name in enumerate(self.FINGER_NAMES)
            }

            predictions[state_code] = self.compute_field(finger_states)

        return predictions


def create_initial_hand_model() -> HandModel:
    """
    Create an initial hand model with reasonable starting parameters.

    Coordinate system (right hand, palm facing down):
    - X: pointing right (towards pinky)
    - Y: pointing forward (towards fingertips)
    - Z: pointing up (away from palm)

    Sensor at origin (center of palm).
    """
    # Approximate finger positions when extended (in meters)
    # Fingers spread out in X, extended forward in Y
    finger_base_positions = {
        'thumb':  np.array([-0.04, 0.02, 0.01]),   # Thumb off to side
        'index':  np.array([-0.02, 0.08, 0.00]),   # Index finger
        'middle': np.array([0.00, 0.09, 0.00]),    # Middle (longest)
        'ring':   np.array([0.02, 0.08, 0.00]),    # Ring finger
        'pinky':  np.array([0.04, 0.06, 0.00]),    # Pinky (shortest)
    }

    # When flexed, magnets move closer to palm (less Y, more Z down)
    flex_offset = np.array([0.0, -0.04, -0.02])  # Move back and down

    # Initial dipole moments (pointing roughly along finger axis)
    # Typical small magnets: ~0.001 A·m² to 0.01 A·m²
    dipole_strength = 0.005  # A·m²

    magnets = {}
    for name, base_pos in finger_base_positions.items():
        # Dipole points roughly along finger (in Y direction for most fingers)
        if name == 'thumb':
            dipole_dir = np.array([0.5, 0.5, 0.0])
        else:
            dipole_dir = np.array([0.0, 1.0, 0.0])
        dipole_dir = dipole_dir / np.linalg.norm(dipole_dir)

        magnets[name] = MagnetConfig(
            pos_extended=base_pos.copy(),
            pos_flexed=base_pos + flex_offset,
            dipole_moment=dipole_strength * dipole_dir
        )

    return HandModel(magnets=magnets)


def pack_parameters(model: HandModel) -> np.ndarray:
    """
    Pack model parameters into a flat array for optimization.

    Parameters per finger (9):
    - pos_extended: 3 values (x, y, z)
    - pos_flexed: 3 values (x, y, z)
    - dipole_moment: 3 values (mx, my, mz)

    Total: 5 fingers × 9 parameters = 45 parameters
    """
    params = []
    for name in HandModel.FINGER_NAMES:
        magnet = model.magnets[name]
        params.extend(magnet.pos_extended)
        params.extend(magnet.pos_flexed)
        params.extend(magnet.dipole_moment)
    return np.array(params)


def unpack_parameters(params: np.ndarray) -> HandModel:
    """Unpack flat parameter array into a HandModel."""
    magnets = {}
    idx = 0
    for name in HandModel.FINGER_NAMES:
        pos_ext = params[idx:idx+3]
        pos_flex = params[idx+3:idx+6]
        dipole = params[idx+6:idx+9]
        idx += 9

        magnets[name] = MagnetConfig(
            pos_extended=pos_ext.copy(),
            pos_flexed=pos_flex.copy(),
            dipole_moment=dipole.copy()
        )

    return HandModel(magnets=magnets)


def load_observations(data_path: Path) -> Tuple[List[str], List[np.ndarray], List[np.ndarray]]:
    """
    Load labeled observations from session data.

    Returns:
        Tuple of (state_codes, mag_readings, orientations) where:
        - state_codes: List of finger state codes (e.g., '00000')
        - mag_readings: List of magnetic field vectors (μT)
        - orientations: List of quaternion orientations [x, y, z, w]
    """
    with open(data_path) as f:
        data = json.load(f)

    samples = data.get('samples', [])
    labels = data.get('labels', [])

    # Build mapping from sample index to finger state code
    FINGER_STATE_MAP = {'extended': '0', 'flexed': '2', 'unknown': None}
    FINGER_ORDER = ['thumb', 'index', 'middle', 'ring', 'pinky']

    index_to_code = {}
    for lbl in labels:
        # Handle both label formats: startIndex/endIndex and start_sample/end_sample
        start_idx = lbl.get('startIndex', lbl.get('start_sample', 0))
        end_idx = lbl.get('endIndex', lbl.get('end_sample', 0))

        # Handle both finger formats: top-level or nested in 'labels'
        fingers = lbl.get('fingers', {})
        if not fingers and 'labels' in lbl and isinstance(lbl['labels'], dict):
            fingers = lbl['labels'].get('fingers', {})

        # Build code from finger states
        codes = []
        for finger in FINGER_ORDER:
            state = fingers.get(finger, 'unknown')
            code = FINGER_STATE_MAP.get(state)
            if code is None:
                break
            codes.append(code)

        if len(codes) == 5:  # All fingers have valid states
            code = ''.join(codes)
            for i in range(start_idx, end_idx):
                index_to_code[i] = code

    state_codes = []
    mag_readings = []
    orientations = []

    for i, sample in enumerate(samples):
        if i not in index_to_code:
            continue

        # Extract magnetic field (iron-corrected if available)
        if 'iron_mx' in sample:
            mx = sample['iron_mx']
            my = sample['iron_my']
            mz = sample['iron_mz']
        elif 'mx_ut' in sample:
            mx = sample['mx_ut']
            my = sample['my_ut']
            mz = sample['mz_ut']
        else:
            mx = sample.get('mx', 0)
            my = sample.get('my', 0)
            mz = sample.get('mz', 0)

        # Skip samples with missing magnetometer data
        if mx == 0 and my == 0 and mz == 0:
            continue

        # Extract orientation if available
        quat = None
        if 'qx' in sample:
            quat = np.array([sample['qx'], sample['qy'], sample['qz'], sample['qw']])
        elif 'ctx' in sample and isinstance(sample['ctx'], dict):
            ctx = sample['ctx']
            if 'qx' in ctx:
                quat = np.array([ctx['qx'], ctx['qy'], ctx['qz'], ctx['qw']])

        state_codes.append(index_to_code[i])
        mag_readings.append(np.array([mx, my, mz]))
        orientations.append(quat)

    return state_codes, mag_readings, orientations


def compute_class_centroids(state_codes: List[str],
                           mag_readings: List[np.ndarray]) -> Dict[str, np.ndarray]:
    """Compute centroid (mean) of magnetic readings for each finger state."""
    from collections import defaultdict

    class_samples = defaultdict(list)
    for code, mag in zip(state_codes, mag_readings):
        class_samples[code].append(mag)

    centroids = {}
    for code, samples in class_samples.items():
        centroids[code] = np.mean(samples, axis=0)

    return centroids


def objective_function(params: np.ndarray,
                       target_centroids: Dict[str, np.ndarray],
                       regularization: float = 0.01) -> float:
    """
    Objective function for optimization.

    Minimizes the sum of squared errors between predicted and observed
    magnetic field centroids, plus regularization on parameter magnitudes.
    """
    model = unpack_parameters(params)
    predictions = model.predict_all_states()

    total_error = 0.0
    n_states = 0

    for state_code, target in target_centroids.items():
        if state_code in predictions:
            pred = predictions[state_code]
            error = np.sum((pred - target) ** 2)
            total_error += error
            n_states += 1

    # Regularization to prevent extreme parameter values
    reg_term = regularization * np.sum(params ** 2)

    return total_error / max(n_states, 1) + reg_term


def optimize_model(target_centroids: Dict[str, np.ndarray],
                   method: str = 'differential_evolution',
                   verbose: bool = True) -> Tuple[HandModel, float]:
    """
    Optimize hand model parameters to match observed centroids.

    Args:
        target_centroids: Dict mapping state codes to observed field centroids
        method: Optimization method ('differential_evolution' or 'minimize')
        verbose: Print progress

    Returns:
        Tuple of (optimized_model, final_error)
    """
    # Get initial model and parameters
    initial_model = create_initial_hand_model()
    initial_params = pack_parameters(initial_model)
    n_params = len(initial_params)

    if verbose:
        print(f"Optimizing {n_params} parameters...")
        print(f"Target states: {len(target_centroids)}")

    # Parameter bounds (reasonable physical limits)
    # Positions: -0.15m to 0.15m (15cm from sensor)
    # Dipole moments: -0.1 to 0.1 A·m²
    bounds = []
    for i in range(5):  # 5 fingers
        # Position extended (x, y, z)
        bounds.extend([(-0.15, 0.15), (-0.02, 0.15), (-0.05, 0.05)])
        # Position flexed (x, y, z)
        bounds.extend([(-0.15, 0.15), (-0.02, 0.10), (-0.08, 0.02)])
        # Dipole moment (mx, my, mz)
        bounds.extend([(-0.1, 0.1), (-0.1, 0.1), (-0.1, 0.1)])

    if method == 'differential_evolution':
        # Global optimization (slower but more robust)
        result = differential_evolution(
            objective_function,
            bounds=bounds,
            args=(target_centroids, 0.001),
            maxiter=500,
            tol=1e-6,
            seed=42,
            workers=-1,
            updating='deferred',
            disp=verbose
        )
    else:
        # Local optimization (faster but may find local minimum)
        result = minimize(
            objective_function,
            initial_params,
            args=(target_centroids, 0.001),
            method='L-BFGS-B',
            bounds=bounds,
            options={'maxiter': 1000, 'disp': verbose}
        )

    optimized_model = unpack_parameters(result.x)
    final_error = result.fun

    if verbose:
        print(f"\nOptimization complete!")
        print(f"Final MSE: {final_error:.4f} μT²")
        print(f"Final RMSE: {np.sqrt(final_error):.4f} μT")

    return optimized_model, final_error


def analyze_model(model: HandModel,
                  target_centroids: Dict[str, np.ndarray]) -> Dict:
    """
    Analyze the fitted model and compare predictions to observations.

    Returns:
        Dict with analysis results
    """
    predictions = model.predict_all_states()

    results = {
        'per_state_errors': {},
        'total_rmse': 0.0,
        'magnet_summary': {},
        'predictions': {},
        'targets': {}
    }

    errors = []
    for state_code, target in target_centroids.items():
        pred = predictions.get(state_code, np.zeros(3))
        error = np.linalg.norm(pred - target)
        errors.append(error)

        results['per_state_errors'][state_code] = error
        results['predictions'][state_code] = pred.tolist()
        results['targets'][state_code] = target.tolist()

    results['total_rmse'] = np.sqrt(np.mean(np.array(errors) ** 2))
    results['mean_error'] = np.mean(errors)
    results['max_error'] = np.max(errors)

    # Summarize magnet configurations
    for name in HandModel.FINGER_NAMES:
        magnet = model.magnets[name]

        # Distance from sensor
        dist_ext = np.linalg.norm(magnet.pos_extended)
        dist_flex = np.linalg.norm(magnet.pos_flexed)

        # Dipole strength
        dipole_mag = np.linalg.norm(magnet.dipole_moment)

        # Field contribution when flexed (closest to sensor)
        field_flex = model._dipole_field(magnet.pos_flexed, magnet.dipole_moment)
        field_ext = model._dipole_field(magnet.pos_extended, magnet.dipole_moment)

        results['magnet_summary'][name] = {
            'pos_extended_m': magnet.pos_extended.tolist(),
            'pos_flexed_m': magnet.pos_flexed.tolist(),
            'dipole_moment': magnet.dipole_moment.tolist(),
            'dist_extended_cm': dist_ext * 100,
            'dist_flexed_cm': dist_flex * 100,
            'dipole_magnitude': dipole_mag,
            'field_extended_uT': np.linalg.norm(field_ext),
            'field_flexed_uT': np.linalg.norm(field_flex),
            'field_ratio': np.linalg.norm(field_flex) / max(np.linalg.norm(field_ext), 1e-10)
        }

    return results


def print_analysis_report(results: Dict, target_centroids: Dict[str, np.ndarray]):
    """Print a detailed analysis report."""
    print("\n" + "=" * 70)
    print("PHYSICS MODEL ANALYSIS REPORT")
    print("=" * 70)

    print("\n## Model Fit Quality\n")
    print(f"Total RMSE:  {results['total_rmse']:.2f} μT")
    print(f"Mean Error:  {results['mean_error']:.2f} μT")
    print(f"Max Error:   {results['max_error']:.2f} μT")

    print("\n## Per-State Errors\n")
    print(f"{'State':<10} {'Error (μT)':<12} {'Quality'}")
    print("-" * 35)
    for state, error in sorted(results['per_state_errors'].items()):
        quality = "✓ Good" if error < 10 else ("~ Fair" if error < 20 else "✗ Poor")
        print(f"{state:<10} {error:>8.2f}     {quality}")

    print("\n## Magnet Configurations\n")
    for name, summary in results['magnet_summary'].items():
        print(f"### {name.capitalize()}")
        print(f"  Position (extended): [{summary['pos_extended_m'][0]*100:.1f}, "
              f"{summary['pos_extended_m'][1]*100:.1f}, "
              f"{summary['pos_extended_m'][2]*100:.1f}] cm")
        print(f"  Position (flexed):   [{summary['pos_flexed_m'][0]*100:.1f}, "
              f"{summary['pos_flexed_m'][1]*100:.1f}, "
              f"{summary['pos_flexed_m'][2]*100:.1f}] cm")
        print(f"  Distance: {summary['dist_extended_cm']:.1f} cm → {summary['dist_flexed_cm']:.1f} cm")
        print(f"  Dipole magnitude: {summary['dipole_magnitude']:.4f} A·m²")
        print(f"  Field: {summary['field_extended_uT']:.1f} μT → {summary['field_flexed_uT']:.1f} μT "
              f"(×{summary['field_ratio']:.1f})")
        print()

    print("\n## Predicted vs Observed Fields\n")
    print(f"{'State':<10} {'Observed (μT)':<30} {'Predicted (μT)':<30}")
    print("-" * 70)
    for state in sorted(target_centroids.keys()):
        obs = target_centroids[state]
        pred = np.array(results['predictions'][state])
        print(f"{state:<10} [{obs[0]:>6.1f}, {obs[1]:>6.1f}, {obs[2]:>6.1f}]   "
              f"[{pred[0]:>6.1f}, {pred[1]:>6.1f}, {pred[2]:>6.1f}]")


def run_physics_optimization():
    """Main function to run the physics model optimization."""
    print("=" * 70)
    print("PHYSICS-BASED MAGNETIC FIELD MODEL OPTIMIZATION")
    print("=" * 70)

    # Find labeled session data
    data_dir = Path("data/GAMBIT")
    if not data_dir.exists():
        data_dir = Path(".worktrees/data/GAMBIT")

    session_files = sorted(data_dir.glob("*.json"), key=lambda x: x.stat().st_size, reverse=True)

    # Find session with labels (skip manifest and small files)
    labeled_session = None
    FINGER_STATE_MAP = {'extended': '0', 'flexed': '2', 'unknown': None}
    FINGER_ORDER = ['thumb', 'index', 'middle', 'ring', 'pinky']

    for session_file in session_files:
        if session_file.name == 'manifest.json' or session_file.stat().st_size < 1000:
            continue
        try:
            with open(session_file) as f:
                data = json.load(f)

            # Check labels array for valid finger states
            labels = data.get('labels', [])
            valid_labels = 0
            for lbl in labels:
                # Handle both finger formats
                fingers = lbl.get('fingers', {})
                if not fingers and 'labels' in lbl and isinstance(lbl['labels'], dict):
                    fingers = lbl['labels'].get('fingers', {})
                codes = [FINGER_STATE_MAP.get(fingers.get(f, 'unknown'))
                         for f in FINGER_ORDER]
                if all(c is not None for c in codes):
                    # Handle both index formats
                    start = lbl.get('startIndex', lbl.get('start_sample', 0))
                    end = lbl.get('endIndex', lbl.get('end_sample', 0))
                    valid_labels += end - start

            if valid_labels > 100:
                labeled_session = session_file
                print(f"\nUsing session: {session_file.name}")
                print(f"Labeled samples: {valid_labels}")
                break
        except (json.JSONDecodeError, IOError) as e:
            print(f"Skipping {session_file.name}: {e}")
            continue

    if labeled_session is None:
        print("ERROR: No labeled session found!")
        return

    # Load observations
    print("\nLoading observations...")
    state_codes, mag_readings, orientations = load_observations(labeled_session)
    print(f"Total labeled samples: {len(state_codes)}")

    # Compute centroids
    print("\nComputing class centroids...")
    centroids = compute_class_centroids(state_codes, mag_readings)
    print(f"Unique states: {len(centroids)}")

    for state, centroid in sorted(centroids.items()):
        print(f"  {state}: [{centroid[0]:>7.1f}, {centroid[1]:>7.1f}, {centroid[2]:>7.1f}] μT")

    # Optimize model
    print("\n" + "-" * 70)
    print("OPTIMIZATION")
    print("-" * 70)

    optimized_model, final_error = optimize_model(
        centroids,
        method='differential_evolution',
        verbose=True
    )

    # Analyze results
    print("\n" + "-" * 70)
    print("ANALYSIS")
    print("-" * 70)

    results = analyze_model(optimized_model, centroids)
    print_analysis_report(results, centroids)

    # Save results
    output_path = Path("ml/physics_model_results.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return optimized_model, results


if __name__ == "__main__":
    run_physics_optimization()
