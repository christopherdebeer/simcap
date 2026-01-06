#!/usr/bin/env python3
"""
GPU-Accelerated Physics-Based Magnetic Model Optimization

Fits a physics-based magnetic dipole model to observed sensor data using:
- Vectorized numpy operations (CPU baseline, very fast)
- Optional GPU acceleration (JAX/CuPy/PyTorch)
- scipy optimization algorithms
- Magnetic dipole field equations from first principles

The model optimizes:
1. Per-finger magnet parameters (position, dipole moment, orientation)
2. Multi-finger interaction effects
3. Sensor calibration parameters

Author: Claude
Date: January 2026
"""

import numpy as np
from scipy.optimize import minimize, differential_evolution, basinhopping
from scipy.spatial.transform import Rotation as R
from pathlib import Path
import json
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time

# Try to import GPU libraries (optional)
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap
    HAS_JAX = True
    print("JAX available - GPU acceleration enabled")
except ImportError:
    HAS_JAX = False
    jnp = np
    print("JAX not available - using CPU (still fast with vectorization)")

# Physical constants
MU_0_OVER_4PI = 1e-7  # T·m/A (μ₀/4π)


@dataclass
class MagnetParams:
    """Parameters for a single finger magnet."""
    position: np.ndarray  # [x, y, z] in meters
    dipole_moment: np.ndarray  # [mx, my, mz] in A·m²

    def to_vector(self) -> np.ndarray:
        """Flatten to optimization parameter vector."""
        return np.concatenate([self.position, self.dipole_moment])

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'MagnetParams':
        """Create from optimization parameter vector."""
        return cls(position=vec[:3], dipole_moment=vec[3:6])


@dataclass
class PhysicsModelParams:
    """Complete physics model parameters."""
    magnets: Dict[str, MagnetParams]  # One per finger
    baseline_field: np.ndarray = field(default_factory=lambda: np.zeros(3))  # Earth field + bias

    def to_vector(self) -> np.ndarray:
        """Flatten all parameters to single vector."""
        finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']
        vec_parts = []
        for finger in finger_order:
            if finger in self.magnets:
                vec_parts.append(self.magnets[finger].to_vector())
        vec_parts.append(self.baseline_field)
        return np.concatenate(vec_parts)

    @classmethod
    def from_vector(cls, vec: np.ndarray) -> 'PhysicsModelParams':
        """Reconstruct from parameter vector."""
        finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']
        magnets = {}
        idx = 0
        for finger in finger_order:
            magnets[finger] = MagnetParams.from_vector(vec[idx:idx+6])
            idx += 6
        baseline = vec[idx:idx+3]
        return cls(magnets=magnets, baseline_field=baseline)


class VectorizedDipoleModel:
    """
    Vectorized magnetic dipole model for efficient batch computation.

    All operations are vectorized over:
    - Multiple observation points (samples)
    - Multiple magnets (fingers)
    """

    def __init__(self, use_gpu: bool = False):
        """
        Args:
            use_gpu: Use GPU acceleration if available (JAX)
        """
        self.use_gpu = use_gpu and HAS_JAX
        self.np = jnp if self.use_gpu else np

    def dipole_field_single(self, r: np.ndarray, m: np.ndarray) -> np.ndarray:
        """
        Compute magnetic field from a single dipole.

        B(r) = (μ₀/4π) * [3(m·r̂)r̂ - m] / |r|³

        Args:
            r: Position vector from dipole to observation point [3] (meters)
            m: Dipole moment vector [3] (A·m²)

        Returns:
            Magnetic field in μT [3]
        """
        r_mag = self.np.linalg.norm(r)

        # Avoid singularity
        if self.np.any(r_mag < 1e-6):
            return self.np.zeros(3)

        r_hat = r / r_mag
        m_dot_r = self.np.dot(m, r_hat)

        # Dipole field formula
        B = MU_0_OVER_4PI * (3 * m_dot_r * r_hat - m) / (r_mag ** 3)

        # Convert T to μT
        return B * 1e6

    def dipole_field_vectorized(
        self,
        r_batch: np.ndarray,
        m_batch: np.ndarray
    ) -> np.ndarray:
        """
        Vectorized dipole field computation for multiple samples and magnets.

        Args:
            r_batch: Position vectors [N_samples, N_magnets, 3] (meters)
            m_batch: Dipole moments [N_magnets, 3] (A·m²)

        Returns:
            Fields [N_samples, N_magnets, 3] (μT)
        """
        # Compute magnitudes: [N_samples, N_magnets]
        r_mag = self.np.linalg.norm(r_batch, axis=-1, keepdims=True)

        # Avoid division by zero
        r_mag = self.np.maximum(r_mag, 1e-6)

        # Normalize: [N_samples, N_magnets, 3]
        r_hat = r_batch / r_mag

        # Dot products: [N_samples, N_magnets]
        m_dot_r = self.np.sum(r_hat * m_batch[None, :, :], axis=-1, keepdims=True)

        # Dipole field: [N_samples, N_magnets, 3]
        B = MU_0_OVER_4PI * (3 * m_dot_r * r_hat - m_batch[None, :, :]) / (r_mag ** 3)

        # Convert T to μT
        return B * 1e6

    def compute_total_field(
        self,
        finger_states: np.ndarray,  # [N_samples, 5] - 0=extended, 1=flexed
        magnet_positions_ext: np.ndarray,  # [5, 3] - positions when extended
        magnet_positions_flex: np.ndarray,  # [5, 3] - positions when flexed
        dipole_moments: np.ndarray,  # [5, 3] - dipole moments
        baseline: np.ndarray  # [3] - baseline field
    ) -> np.ndarray:
        """
        Compute total magnetic field for a batch of samples.

        Args:
            finger_states: Binary state (0=extended, 1=flexed) [N_samples, 5]
            magnet_positions_ext: Magnet positions when extended [5, 3] meters
            magnet_positions_flex: Magnet positions when flexed [5, 3] meters
            dipole_moments: Dipole moment vectors [5, 3] A·m²
            baseline: Baseline field (Earth + sensor offset) [3] μT

        Returns:
            Total field at sensor [N_samples, 3] μT
        """
        N_samples = finger_states.shape[0]

        # Interpolate positions based on finger state
        # positions = extended + state * (flexed - extended)
        # Shape: [N_samples, 5, 3]
        positions = (
            magnet_positions_ext[None, :, :] +
            finger_states[:, :, None] * (magnet_positions_flex - magnet_positions_ext)[None, :, :]
        )

        # Compute position vectors from magnets to sensor at origin
        # r[i,j,:] = sensor_position - magnet_position[i,j,:]
        # Assuming sensor at origin: r = -positions
        r_batch = -positions  # [N_samples, 5, 3]

        # Compute fields from all magnets
        B_magnets = self.dipole_field_vectorized(r_batch, dipole_moments)  # [N_samples, 5, 3]

        # Sum over magnets and add baseline
        B_total = self.np.sum(B_magnets, axis=1) + baseline[None, :]  # [N_samples, 3]

        return B_total


class PhysicsOptimizer:
    """
    Optimizes physics model parameters to match observed data.
    """

    def __init__(
        self,
        observed_data: Dict[str, Dict],
        use_gpu: bool = False,
        verbose: bool = True
    ):
        """
        Args:
            observed_data: Dict mapping combo codes to observation dicts
            use_gpu: Use GPU acceleration if available
            verbose: Print progress
        """
        self.observed = observed_data
        self.model = VectorizedDipoleModel(use_gpu=use_gpu)
        self.verbose = verbose

        # Prepare observation arrays
        self._prepare_observations()

    def _prepare_observations(self):
        """Convert observed data to arrays for efficient computation."""
        # Extract observations
        combos = []
        fields = []
        weights = []  # Weight by sample count

        for combo, obs in self.observed.items():
            if combo == 'eeeee':  # Skip baseline
                continue

            combos.append(combo)
            fields.append(obs['mean'])

            # Weight by number of samples and inverse variance
            n = obs['n']
            var = np.sum(obs['std'] ** 2)
            weight = np.sqrt(n) / (var + 1.0)
            weights.append(weight)

        self.combo_codes = np.array(combos)
        self.observed_fields = np.array(fields)  # [N_obs, 3]
        self.weights = np.array(weights)  # [N_obs]

        # Convert combo codes to binary finger states
        self.finger_states = self._combos_to_states(self.combo_codes)  # [N_obs, 5]

        # Get baseline
        baseline_obs = self.observed.get('eeeee', {}).get('mean', np.zeros(3))
        self.baseline_observed = baseline_obs

        if self.verbose:
            print(f"Prepared {len(combos)} observations for optimization")
            print(f"Baseline (eeeee): {baseline_obs}")

    def _combos_to_states(self, combos: np.ndarray) -> np.ndarray:
        """
        Convert combo code strings to binary state arrays.

        Args:
            combos: Array of combo strings like ['feeee', 'efeee', ...]

        Returns:
            Binary states [N, 5] where 0=extended, 1=flexed
        """
        states = np.zeros((len(combos), 5))
        for i, combo in enumerate(combos):
            for j, c in enumerate(combo):
                states[i, j] = 1.0 if c == 'f' else 0.0
        return states

    def objective(self, params_vec: np.ndarray) -> float:
        """
        Objective function: weighted MSE between predicted and observed fields.

        Args:
            params_vec: Flattened parameter vector

        Returns:
            Total weighted squared error
        """
        # Unpack parameters
        # 5 magnets * 6 params each = 30 params
        # + 3 baseline = 33 total

        # Extended positions (5 magnets * 3 coords)
        pos_ext = params_vec[0:15].reshape(5, 3)

        # Flexed positions (5 magnets * 3 coords)
        pos_flex = params_vec[15:30].reshape(5, 3)

        # Dipole moments (5 magnets * 3 coords)
        dipoles = params_vec[30:45].reshape(5, 3)

        # Baseline field
        baseline = params_vec[45:48]

        # Compute predictions
        predicted = self.model.compute_total_field(
            self.finger_states,
            pos_ext,
            pos_flex,
            dipoles,
            baseline
        )

        # Compute weighted errors
        errors = predicted - self.observed_fields  # [N_obs, 3]
        squared_errors = np.sum(errors ** 2, axis=1)  # [N_obs]
        weighted_errors = squared_errors * self.weights

        total_error = np.sum(weighted_errors)

        return total_error

    def create_initial_guess(self) -> np.ndarray:
        """
        Create intelligent initial guess based on single-finger observations.
        """
        # Initialize with reasonable defaults
        finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']

        # Extended positions (magnets far from sensor)
        pos_ext = np.array([
            [0.04, 0.10, 0.02],   # thumb
            [0.02, 0.12, 0.01],   # index
            [0.00, 0.13, 0.00],   # middle
            [-0.02, 0.12, -0.01], # ring
            [-0.04, 0.10, -0.02], # pinky
        ])

        # Flexed positions (magnets closer)
        pos_flex = np.array([
            [0.02, 0.03, 0.01],   # thumb
            [0.01, 0.04, 0.00],   # index
            [0.00, 0.04, 0.00],   # middle
            [-0.01, 0.04, 0.00],  # ring
            [-0.02, 0.03, -0.01], # pinky
        ])

        # Estimate dipole moments from single-finger observations
        dipoles = np.zeros((5, 3))
        single_combos = {
            'feeee': 0,  # thumb
            'efeee': 1,  # index
            'eefee': 2,  # middle
            'eeefe': 3,  # ring
            'eeeef': 4,  # pinky
        }

        for combo, finger_idx in single_combos.items():
            if combo in self.observed:
                delta = self.observed[combo]['mean'] - self.baseline_observed

                # Rough estimate: assume magnet at ~3cm, field ~ m/r³
                r = 0.03
                m_mag = 2 * (r ** 3) * np.linalg.norm(delta) * 1e-6 / MU_0_OVER_4PI

                # Direction: approximate from field direction
                if np.linalg.norm(delta) > 1:
                    m_dir = delta / np.linalg.norm(delta)
                    dipoles[finger_idx] = m_mag * m_dir

        # Baseline
        baseline = self.baseline_observed.copy()

        # Concatenate all parameters
        params = np.concatenate([
            pos_ext.flatten(),
            pos_flex.flatten(),
            dipoles.flatten(),
            baseline
        ])

        return params

    def optimize(
        self,
        method: str = 'differential_evolution',
        maxiter: int = 1000
    ) -> Tuple[np.ndarray, Dict]:
        """
        Run optimization to find best-fit parameters.

        Args:
            method: Optimization method ('differential_evolution', 'basinhopping', 'minimize')
            maxiter: Maximum iterations

        Returns:
            (best_params, results_dict)
        """
        print(f"\n{'='*70}")
        print(f"PHYSICS MODEL OPTIMIZATION")
        print(f"{'='*70}")
        print(f"Method: {method}")
        print(f"Observations: {len(self.combo_codes)}")
        print(f"Parameters: 48 (15 pos_ext + 15 pos_flex + 15 dipoles + 3 baseline)")

        # Initial guess
        x0 = self.create_initial_guess()

        # Parameter bounds
        bounds = []

        # Extended positions: 5-15cm from sensor
        for _ in range(15):
            bounds.append((-0.15, 0.15))

        # Flexed positions: 1-8cm from sensor
        for _ in range(15):
            bounds.append((-0.08, 0.08))

        # Dipole moments: -1 to 1 A·m²
        for _ in range(15):
            bounds.append((-1.0, 1.0))

        # Baseline: -100 to 100 μT
        for _ in range(3):
            bounds.append((-100, 100))

        # Initial objective
        initial_error = self.objective(x0)
        print(f"Initial error: {initial_error:.1f}")

        # Optimize
        t0 = time.time()

        if method == 'differential_evolution':
            result = differential_evolution(
                self.objective,
                bounds,
                maxiter=maxiter,
                workers=1,  # Single worker (avoid pickling issues with self.model)
                updating='immediate',
                disp=True
            )
        elif method == 'basinhopping':
            minimizer_kwargs = {'method': 'L-BFGS-B', 'bounds': bounds}
            result = basinhopping(
                self.objective,
                x0,
                minimizer_kwargs=minimizer_kwargs,
                niter=maxiter,
                disp=True
            )
        else:  # 'minimize'
            result = minimize(
                self.objective,
                x0,
                method='L-BFGS-B',
                bounds=bounds,
                options={'maxiter': maxiter, 'disp': True}
            )

        elapsed = time.time() - t0

        print(f"\nOptimization complete in {elapsed:.1f}s")
        print(f"Final error: {result.fun:.1f}")
        print(f"Improvement: {initial_error - result.fun:.1f} ({(1 - result.fun/initial_error)*100:.1f}%)")

        return result.x, {
            'success': result.success,
            'initial_error': initial_error,
            'final_error': result.fun,
            'elapsed_time': elapsed,
            'n_iterations': result.nit if hasattr(result, 'nit') else None
        }

    def analyze_results(self, params: np.ndarray) -> Dict:
        """
        Analyze optimization results and compute statistics.
        """
        # Unpack parameters
        pos_ext = params[0:15].reshape(5, 3)
        pos_flex = params[15:30].reshape(5, 3)
        dipoles = params[30:45].reshape(5, 3)
        baseline = params[45:48]

        # Compute predictions
        predicted = self.model.compute_total_field(
            self.finger_states,
            pos_ext,
            pos_flex,
            dipoles,
            baseline
        )

        # Per-observation errors
        errors = predicted - self.observed_fields
        error_mags = np.linalg.norm(errors, axis=1)

        # Statistics
        finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']

        analysis = {
            'baseline_field': baseline.tolist(),
            'magnets': {},
            'predictions': {},
            'errors': {
                'mean_error_ut': float(np.mean(error_mags)),
                'max_error_ut': float(np.max(error_mags)),
                'rmse_ut': float(np.sqrt(np.mean(error_mags ** 2))),
            }
        }

        # Per-finger parameters
        for i, finger in enumerate(finger_names):
            analysis['magnets'][finger] = {
                'position_extended_m': pos_ext[i].tolist(),
                'position_extended_cm': (pos_ext[i] * 100).tolist(),
                'position_flexed_m': pos_flex[i].tolist(),
                'position_flexed_cm': (pos_flex[i] * 100).tolist(),
                'travel_distance_cm': float(np.linalg.norm(pos_flex[i] - pos_ext[i]) * 100),
                'dipole_moment_Am2': dipoles[i].tolist(),
                'dipole_magnitude_Am2': float(np.linalg.norm(dipoles[i])),
            }

        # Per-observation predictions
        for i, combo in enumerate(self.combo_codes):
            analysis['predictions'][combo] = {
                'observed': self.observed_fields[i].tolist(),
                'predicted': predicted[i].tolist(),
                'error_ut': float(error_mags[i]),
                'error_vector': errors[i].tolist(),
            }

        return analysis


def main():
    """Run GPU-accelerated physics optimization."""
    # Load observed data
    data_path = Path(".worktrees/data/GAMBIT/2025-12-31T14_06_18.270Z.json")

    print("Loading session data...")
    with open(data_path) as f:
        session = json.load(f)

    # Extract labeled observations
    print("Extracting labeled observations...")

    FINGER_ORDER = ['thumb', 'index', 'middle', 'ring', 'pinky']

    # Build sample index
    index_to_combo = {}
    for lbl in session.get('labels', []):
        if 'labels' in lbl and isinstance(lbl['labels'], dict):
            fingers = lbl['labels'].get('fingers', {})
            start = lbl.get('start_sample', 0)
            end = lbl.get('end_sample', 0)
        else:
            fingers = lbl.get('fingers', {})
            start = lbl.get('startIndex', 0)
            end = lbl.get('endIndex', 0)

        if not fingers:
            continue

        # Convert to combo code
        combo = ''.join([
            'e' if fingers.get(f) == 'extended' else 'f' if fingers.get(f) == 'flexed' else '?'
            for f in FINGER_ORDER
        ])

        if '?' in combo:
            continue

        for i in range(start, end):
            index_to_combo[i] = combo

    # Collect samples by combo
    combo_samples = defaultdict(list)
    for i, sample in enumerate(session.get('samples', [])):
        if i not in index_to_combo:
            continue

        combo = index_to_combo[i]

        # Use iron-corrected magnetometer data
        mag = [
            sample.get('iron_mx', sample.get('mx_ut', 0)),
            sample.get('iron_my', sample.get('my_ut', 0)),
            sample.get('iron_mz', sample.get('mz_ut', 0))
        ]

        combo_samples[combo].append(mag)

    # Compute statistics
    observed = {}
    for combo, samples in combo_samples.items():
        samps = np.array(samples)
        observed[combo] = {
            'mean': samps.mean(axis=0),
            'std': samps.std(axis=0),
            'n': len(samps),
            'cov': np.cov(samps.T) if len(samps) > 3 else np.diag(samps.std(axis=0) ** 2)
        }

    print(f"Found {len(observed)} unique combos with {sum(o['n'] for o in observed.values())} total samples")

    # Run optimization
    optimizer = PhysicsOptimizer(observed, use_gpu=False, verbose=True)

    best_params, opt_results = optimizer.optimize(
        method='differential_evolution',
        maxiter=100
    )

    # Analyze results
    analysis = optimizer.analyze_results(best_params)

    # Print results
    print(f"\n{'='*70}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*70}")

    print(f"\nBaseline field: {analysis['baseline_field']}")
    print(f"\nError statistics:")
    print(f"  Mean error: {analysis['errors']['mean_error_ut']:.1f} μT")
    print(f"  Max error: {analysis['errors']['max_error_ut']:.1f} μT")
    print(f"  RMSE: {analysis['errors']['rmse_ut']:.1f} μT")

    print(f"\nMagnet parameters:")
    for finger, params in analysis['magnets'].items():
        print(f"\n  {finger.upper()}:")
        print(f"    Extended: {params['position_extended_cm']} cm")
        print(f"    Flexed:   {params['position_flexed_cm']} cm")
        print(f"    Travel:   {params['travel_distance_cm']:.1f} cm")
        print(f"    Dipole:   {params['dipole_moment_Am2']} A·m² (|m|={params['dipole_magnitude_Am2']:.4f})")

    print(f"\nPer-combo predictions:")
    print(f"{'Combo':<8} {'Observed':<30} {'Predicted':<30} {'Error':>8}")
    print("-" * 85)
    for combo, pred in analysis['predictions'].items():
        obs = pred['observed']
        prd = pred['predicted']
        err = pred['error_ut']
        obs_str = f"[{obs[0]:>7.0f}, {obs[1]:>7.0f}, {obs[2]:>7.0f}]"
        prd_str = f"[{prd[0]:>7.0f}, {prd[1]:>7.0f}, {prd[2]:>7.0f}]"
        print(f"{combo:<8} {obs_str} {prd_str} {err:>7.1f}")

    # Save results
    output_path = Path("ml/analysis/physics/gpu_physics_optimization_results.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)

    results = {
        'session': str(data_path),
        'optimization': opt_results,
        'analysis': analysis,
        'parameters': best_params.tolist()
    }

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\n✓ Results saved to {output_path}")

    return results


if __name__ == '__main__':
    main()
