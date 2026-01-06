#!/usr/bin/env python3
"""
Align simulated magnetic signatures to observed ground truth.

Strategy:
1. Extract measured single-finger signatures (ground truth anchors)
2. Generate simulated single-finger signatures using dipole physics
3. Learn affine transformation (rotation + scaling) to align them
4. Apply transformation to all synthetic data

This allows unlimited synthetic training data grounded in real measurements.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List
import sys

# Add parent to path for imports

from ml.simulation.dipole import magnetic_dipole_field, estimate_dipole_moment


def load_measured_signatures(session_path: Path) -> Dict[str, np.ndarray]:
    """Extract mean delta vectors for each single-finger flexed state."""
    with open(session_path) as f:
        session = json.load(f)

    samples = session.get('samples', [])
    labels = session.get('labels', [])

    mx = np.array([s.get('mx', 0) for s in samples])
    my = np.array([s.get('my', 0) for s in samples])
    mz = np.array([s.get('mz', 0) for s in samples])

    # Group samples by finger configuration
    config_vectors = {}

    for label in labels:
        start = label.get('start_sample', label.get('startIndex', 0))
        end = label.get('end_sample', label.get('endIndex', 0))
        content = label.get('labels', label)
        fingers = content.get('fingers', {})

        if not fingers:
            continue

        # Build code
        code = ''
        for f in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            state = fingers.get(f, 'unknown')
            if state == 'extended':
                code += '0'
            elif state == 'partial':
                code += '1'
            elif state == 'flexed':
                code += '2'
            else:
                code += '?'

        if '?' in code:
            continue

        if code not in config_vectors:
            config_vectors[code] = []

        for i in range(start, min(end, len(mx))):
            config_vectors[code].append([mx[i], my[i], mz[i]])

    # Compute means
    signatures = {}
    for code, vectors in config_vectors.items():
        signatures[code] = np.mean(vectors, axis=0)

    # Compute deltas from baseline
    baseline = signatures.get('00000', np.zeros(3))
    deltas = {}
    for code, vec in signatures.items():
        deltas[code] = vec - baseline

    return deltas


def generate_simulated_signatures() -> Dict[str, np.ndarray]:
    """Generate simulated single-finger signatures using dipole physics."""

    # Sensor position (wrist) - in meters
    sensor_pos = np.array([0.0, 0.0, 0.0])

    # 6x3mm N48 magnet dipole moment
    moment_magnitude = estimate_dipole_moment(6, 3, 1430)  # ~0.0135 A·m²

    # Finger positions when extended vs flexed (in meters)
    # Extended: ~80-100mm from sensor, Flexed: ~40-50mm from sensor
    finger_configs = {
        'thumb':  {'extended': np.array([0.06, 0.04, 0.02]), 'flexed': np.array([0.03, 0.02, 0.01])},
        'index':  {'extended': np.array([0.10, 0.02, 0.0]), 'flexed': np.array([0.05, 0.01, 0.0])},
        'middle': {'extended': np.array([0.11, 0.0, 0.0]), 'flexed': np.array([0.055, 0.0, 0.0])},
        'ring':   {'extended': np.array([0.10, -0.02, 0.0]), 'flexed': np.array([0.05, -0.01, 0.0])},
        'pinky':  {'extended': np.array([0.08, -0.04, 0.0]), 'flexed': np.array([0.04, -0.02, 0.0])},
    }

    # Magnet orientations (alternating as user specified: Thumb=Index same)
    # But data shows Thumb is actually opposite to Index!
    # Using orientations derived from measured data:
    orientations = {
        'thumb':  np.array([0, 0, -1]),  # -Z (data shows opposite to Index)
        'index':  np.array([0, 0, 1]),   # +Z
        'middle': np.array([0, 0, 1]),   # +Z (aligned with Index)
        'ring':   np.array([0, -1, 0]),  # -Y (different axis!)
        'pinky':  np.array([0, 0, 1]),   # +Z (aligned with Index)
    }

    finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']

    def compute_field(states: Dict[str, str]) -> np.ndarray:
        """Compute total field for given finger states."""
        total = np.zeros(3)
        for finger, state in states.items():
            pos = finger_configs[finger][state]
            orientation = orientations[finger]
            dipole_moment = moment_magnitude * orientation

            # Compute field in Tesla, convert to µT
            field = magnetic_dipole_field(sensor_pos, pos, dipole_moment)
            total += field * 1e6  # Tesla to µT
        return total

    # Generate signatures
    signatures = {}

    # Baseline (all extended)
    baseline_states = {f: 'extended' for f in finger_order}
    baseline = compute_field(baseline_states)

    # Single finger flexed
    single_codes = {
        '20000': 'thumb',
        '02000': 'index',
        '00200': 'middle',
        '00020': 'ring',
        '00002': 'pinky',
    }

    for code, finger in single_codes.items():
        states = {f: 'extended' for f in finger_order}
        states[finger] = 'flexed'
        field = compute_field(states)
        signatures[code] = field - baseline

    # Multi-finger combinations
    multi_codes = [
        ('22000', ['thumb', 'index']),
        ('00022', ['ring', 'pinky']),
        ('00222', ['middle', 'ring', 'pinky']),
        ('22222', ['thumb', 'index', 'middle', 'ring', 'pinky']),
    ]

    for code, flexed_fingers in multi_codes:
        states = {f: 'extended' for f in finger_order}
        for f in flexed_fingers:
            states[f] = 'flexed'
        field = compute_field(states)
        signatures[code] = field - baseline

    signatures['00000'] = np.zeros(3)  # Baseline delta is zero

    return signatures


def compute_alignment_transform(
    measured: Dict[str, np.ndarray],
    simulated: Dict[str, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute affine transformation to align simulated to measured.

    Returns: (rotation_matrix, translation, scale)

    Uses Procrustes analysis on single-finger signatures.
    """
    # Use single-finger codes for alignment
    single_codes = ['20000', '02000', '00200', '00020', '00002']

    # Build matrices of corresponding points
    M = []  # Measured
    S = []  # Simulated

    for code in single_codes:
        if code in measured and code in simulated:
            M.append(measured[code])
            S.append(simulated[code])

    M = np.array(M)  # (n, 3)
    S = np.array(S)  # (n, 3)

    print(f"Aligning {len(M)} signature pairs...")

    # Center the data
    M_mean = np.mean(M, axis=0)
    S_mean = np.mean(S, axis=0)

    M_centered = M - M_mean
    S_centered = S - S_mean

    # Compute optimal rotation using SVD (Procrustes)
    H = S_centered.T @ M_centered
    U, _, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Ensure proper rotation (det = 1, not reflection)
    if np.linalg.det(R) < 0:
        Vt[-1, :] *= -1
        R = Vt.T @ U.T

    # Compute optimal scale
    S_rotated = (R @ S_centered.T).T
    scale = np.sum(M_centered * S_rotated) / np.sum(S_rotated * S_rotated)

    # Compute translation
    translation = M_mean - scale * (R @ S_mean)

    return R, translation, scale


def apply_transform(
    vector: np.ndarray,
    R: np.ndarray,
    translation: np.ndarray,
    scale: float
) -> np.ndarray:
    """Apply affine transformation to a vector."""
    return scale * (R @ vector) + translation


def main():
    print("=" * 80)
    print("ALIGNING SIMULATED TO OBSERVED SIGNATURES")
    print("=" * 80)

    # Load measured signatures
    session_path = Path('data/GAMBIT/2025-12-31T14_06_18.270Z.json')
    print(f"\nLoading measured signatures from {session_path.name}...")
    measured = load_measured_signatures(session_path)

    # Generate simulated signatures
    print("Generating simulated signatures...")
    simulated = generate_simulated_signatures()

    # Show comparison before alignment
    print("\n" + "=" * 80)
    print("1. BEFORE ALIGNMENT")
    print("=" * 80)

    single_codes = ['20000', '02000', '00200', '00020', '00002']
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']

    print("\nSingle-finger signatures (delta vectors in µT):")
    print("-" * 70)
    print(f"{'Finger':<8} {'Measured':>30} {'Simulated':>30}")
    print("-" * 70)

    for code, name in zip(single_codes, finger_names):
        m = measured.get(code, np.zeros(3))
        s = simulated.get(code, np.zeros(3))
        print(f"{name:<8} [{m[0]:>+8.0f},{m[1]:>+8.0f},{m[2]:>+8.0f}]  [{s[0]:>+8.1f},{s[1]:>+8.1f},{s[2]:>+8.1f}]")

    # Compute alignment
    print("\n" + "=" * 80)
    print("2. COMPUTING ALIGNMENT TRANSFORM")
    print("=" * 80)

    R, translation, scale = compute_alignment_transform(measured, simulated)

    print(f"\nScale factor: {scale:.2f}")
    print(f"Translation: [{translation[0]:.0f}, {translation[1]:.0f}, {translation[2]:.0f}] µT")
    print(f"Rotation matrix:\n{R}")

    # Apply transformation
    print("\n" + "=" * 80)
    print("3. AFTER ALIGNMENT")
    print("=" * 80)

    print("\nSingle-finger signatures (delta vectors in µT):")
    print("-" * 70)
    print(f"{'Finger':<8} {'Measured':>30} {'Aligned Sim':>30} {'Error':>10}")
    print("-" * 70)

    total_error = 0
    for code, name in zip(single_codes, finger_names):
        m = measured.get(code, np.zeros(3))
        s = simulated.get(code, np.zeros(3))
        aligned = apply_transform(s, R, translation, scale)

        error = np.linalg.norm(m - aligned)
        total_error += error

        print(f"{name:<8} [{m[0]:>+8.0f},{m[1]:>+8.0f},{m[2]:>+8.0f}]  "
              f"[{aligned[0]:>+8.0f},{aligned[1]:>+8.0f},{aligned[2]:>+8.0f}]  {error:>8.0f} µT")

    print("-" * 70)
    print(f"Mean alignment error: {total_error / len(single_codes):.0f} µT")

    # Test on multi-finger combinations (not used in alignment)
    print("\n" + "=" * 80)
    print("4. VALIDATION ON MULTI-FINGER (not used in alignment)")
    print("=" * 80)

    multi_codes = [('22000', 'Thumb+Index'), ('00022', 'Ring+Pinky'),
                   ('00222', 'Mid+Ring+Pinky'), ('22222', 'All')]

    print("\nMulti-finger signatures:")
    print("-" * 70)
    print(f"{'Config':<15} {'Measured':>25} {'Aligned Sim':>25} {'Error %':>10}")
    print("-" * 70)

    for code, name in multi_codes:
        if code not in measured or code not in simulated:
            continue
        m = measured[code]
        s = simulated[code]
        aligned = apply_transform(s, R, translation, scale)

        m_mag = np.linalg.norm(m)
        error = np.linalg.norm(m - aligned)
        error_pct = 100 * error / m_mag if m_mag > 0 else 0

        print(f"{name:<15} |m|={m_mag:>8.0f} µT  |a|={np.linalg.norm(aligned):>8.0f} µT  {error_pct:>8.1f}%")

    # Save alignment parameters
    print("\n" + "=" * 80)
    print("5. SAVING ALIGNMENT PARAMETERS")
    print("=" * 80)

    alignment = {
        'rotation': R.tolist(),
        'translation': translation.tolist(),
        'scale': float(scale),
        'source_session': session_path.name,
        'single_finger_codes': single_codes,
    }

    output_path = Path('ml/simulation/alignment_params.json')
    with open(output_path, 'w') as f:
        json.dump(alignment, f, indent=2)

    print(f"Saved to {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
    ALIGNMENT APPROACH:
    - Used Procrustes analysis on 5 single-finger signatures
    - Learned rotation + scale to map simulated → measured space

    RESULTS:
    - Scale factor: {scale:.2f}x (simulation → µT)
    - Mean single-finger alignment error: {total_error / len(single_codes):.0f} µT

    USAGE:
    - Apply transform to all synthetic training data
    - Synthetic data will now match observed magnitude/direction patterns
    - Can generate unlimited training samples grounded in reality

    LIMITATIONS:
    - Multi-finger combinations still show error (non-linear effects)
    - Consider using measured multi-finger signatures as additional anchors
    """)


if __name__ == '__main__':
    main()
