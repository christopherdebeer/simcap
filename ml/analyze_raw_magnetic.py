#!/usr/bin/env python3
"""
Raw Magnetic Field Analysis

Ignores stored calibration values and computes Earth field subtraction
directly from raw data per session. Tests if residual can be reliably
computed even with orientation changes.

Key approach:
1. Use raw magnetometer data (mx_ut, my_ut, mz_ut)
2. Estimate hard iron offset from session data
3. Estimate Earth field vector in world frame
4. Use orientation quaternion to rotate Earth field to sensor frame
5. Subtract to get residual - should be near zero without finger magnets
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional


def quaternion_to_rotation_matrix(q: Dict) -> np.ndarray:
    """Convert quaternion to 3x3 rotation matrix."""
    w, x, y, z = q['w'], q['x'], q['y'], q['z']

    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    return R


def load_session(json_path: Path) -> Tuple[Dict, List[Dict]]:
    """Load session and return metadata + samples."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        return {}, data
    elif isinstance(data, dict) and 'samples' in data:
        return data, data['samples']
    return {}, []


def estimate_hard_iron(samples: List[Dict]) -> np.ndarray:
    """
    Estimate hard iron offset using min/max method.
    Works best with data from varied orientations.
    """
    mx = np.array([s.get('mx_ut', 0) for s in samples])
    my = np.array([s.get('my_ut', 0) for s in samples])
    mz = np.array([s.get('mz_ut', 0) for s in samples])

    offset = np.array([
        (np.max(mx) + np.min(mx)) / 2,
        (np.max(my) + np.min(my)) / 2,
        (np.max(mz) + np.min(mz)) / 2
    ])

    return offset


def estimate_earth_field_world_frame(samples: List[Dict], hard_iron: np.ndarray) -> np.ndarray:
    """
    Estimate Earth field vector in world frame.

    Method: For each sample, rotate the hard-iron-corrected magnetometer
    reading from sensor frame to world frame, then average.
    This gives us the Earth field in world coordinates.
    """
    earth_vectors = []

    for s in samples:
        # Get magnetometer reading
        mag_sensor = np.array([
            s.get('mx_ut', 0) - hard_iron[0],
            s.get('my_ut', 0) - hard_iron[1],
            s.get('mz_ut', 0) - hard_iron[2]
        ])

        # Get orientation
        if 'orientation_w' not in s:
            continue

        q = {
            'w': s['orientation_w'],
            'x': s['orientation_x'],
            'y': s['orientation_y'],
            'z': s['orientation_z']
        }

        # Rotation matrix: sensor to world
        # R transforms world->sensor, so R.T transforms sensor->world
        R = quaternion_to_rotation_matrix(q)

        # Transform magnetometer reading to world frame
        mag_world = R.T @ mag_sensor
        earth_vectors.append(mag_world)

    if not earth_vectors:
        return np.zeros(3)

    # Average to get Earth field estimate
    earth_field = np.mean(earth_vectors, axis=0)
    return earth_field


def compute_residuals(samples: List[Dict], hard_iron: np.ndarray,
                      earth_field_world: np.ndarray) -> List[Dict]:
    """
    Compute residual for each sample by subtracting orientation-compensated Earth field.
    """
    results = []

    for i, s in enumerate(samples):
        # Get raw magnetometer (in µT)
        mag_raw = np.array([
            s.get('mx_ut', 0),
            s.get('my_ut', 0),
            s.get('mz_ut', 0)
        ])

        # Apply hard iron correction
        mag_corrected = mag_raw - hard_iron

        # Get orientation
        if 'orientation_w' not in s:
            continue

        q = {
            'w': s['orientation_w'],
            'x': s['orientation_x'],
            'y': s['orientation_y'],
            'z': s['orientation_z']
        }

        # Rotation matrix
        R = quaternion_to_rotation_matrix(q)

        # Rotate Earth field from world to sensor frame
        # R transforms world->sensor
        earth_sensor = R @ earth_field_world

        # Residual = measured - expected
        residual = mag_corrected - earth_sensor
        residual_mag = np.linalg.norm(residual)

        results.append({
            'index': i,
            'mag_raw': mag_raw,
            'mag_corrected': mag_corrected,
            'earth_sensor': earth_sensor,
            'residual': residual,
            'residual_magnitude': residual_mag,
            'euler_roll': s.get('euler_roll', 0),
            'euler_pitch': s.get('euler_pitch', 0),
            'euler_yaw': s.get('euler_yaw', 0)
        })

    return results


def analyze_session(json_path: Path) -> Optional[Dict]:
    """Analyze a single session using raw data only."""
    meta, samples = load_session(json_path)

    if len(samples) < 50:
        return None

    # Check for required fields
    if 'mx_ut' not in samples[0] or 'orientation_w' not in samples[0]:
        return None

    print(f"\n{'='*70}")
    print(f"Session: {json_path.name}")
    print(f"Samples: {len(samples)}")
    print(f"{'='*70}")

    # Step 1: Estimate hard iron from this session's data
    hard_iron = estimate_hard_iron(samples)
    print(f"\n1. HARD IRON ESTIMATE (from session data):")
    print(f"   Offset: [{hard_iron[0]:.2f}, {hard_iron[1]:.2f}, {hard_iron[2]:.2f}] µT")

    # Step 2: Estimate Earth field in world frame
    earth_world = estimate_earth_field_world_frame(samples, hard_iron)
    earth_mag = np.linalg.norm(earth_world)
    print(f"\n2. EARTH FIELD ESTIMATE (world frame):")
    print(f"   Vector: [{earth_world[0]:.2f}, {earth_world[1]:.2f}, {earth_world[2]:.2f}] µT")
    print(f"   Magnitude: {earth_mag:.2f} µT")

    if earth_mag < 15 or earth_mag > 80:
        print(f"   ⚠ Warning: Magnitude outside typical range (25-65 µT)")

    # Step 3: Compute residuals
    results = compute_residuals(samples, hard_iron, earth_world)

    if not results:
        print("   No valid samples with orientation data")
        return None

    residual_mags = np.array([r['residual_magnitude'] for r in results])

    print(f"\n3. RESIDUAL ANALYSIS (after Earth field subtraction):")
    print(f"   Mean:   {np.mean(residual_mags):.2f} µT")
    print(f"   Std:    {np.std(residual_mags):.2f} µT")
    print(f"   Median: {np.median(residual_mags):.2f} µT")
    print(f"   Min:    {np.min(residual_mags):.2f} µT")
    print(f"   Max:    {np.max(residual_mags):.2f} µT")

    # Check orientation variation
    rolls = [r['euler_roll'] for r in results]
    pitches = [r['euler_pitch'] for r in results]
    yaws = [r['euler_yaw'] for r in results]

    print(f"\n4. ORIENTATION COVERAGE:")
    print(f"   Roll:  [{np.min(rolls):.1f}° to {np.max(rolls):.1f}°], range={np.max(rolls)-np.min(rolls):.1f}°")
    print(f"   Pitch: [{np.min(pitches):.1f}° to {np.max(pitches):.1f}°], range={np.max(pitches)-np.min(pitches):.1f}°")
    print(f"   Yaw:   [{np.min(yaws):.1f}° to {np.max(yaws):.1f}°], range={np.max(yaws)-np.min(yaws):.1f}°")

    # Evaluate success
    mean_residual = np.mean(residual_mags)
    status = "✓ GOOD" if mean_residual < 5 else ("⚠ MARGINAL" if mean_residual < 15 else "✗ HIGH")

    print(f"\n5. STATUS: {status}")
    print(f"   Expected (no magnets): < 5 µT")
    print(f"   Actual mean: {mean_residual:.2f} µT")

    return {
        'filename': json_path.name,
        'num_samples': len(samples),
        'hard_iron': hard_iron.tolist(),
        'earth_field_world': earth_world.tolist(),
        'earth_field_magnitude': float(earth_mag),
        'residual_mean': float(np.mean(residual_mags)),
        'residual_std': float(np.std(residual_mags)),
        'residual_median': float(np.median(residual_mags)),
        'residual_min': float(np.min(residual_mags)),
        'residual_max': float(np.max(residual_mags)),
        'orientation_range': {
            'roll': float(np.max(rolls) - np.min(rolls)),
            'pitch': float(np.max(pitches) - np.min(pitches)),
            'yaw': float(np.max(yaws) - np.min(yaws))
        },
        'status': status
    }


def print_summary(results: List[Dict]):
    """Print summary across all sessions."""
    print("\n" + "=" * 70)
    print("SUMMARY ACROSS ALL SESSIONS")
    print("=" * 70)

    residual_means = [r['residual_mean'] for r in results]
    earth_mags = [r['earth_field_magnitude'] for r in results]

    print(f"\nSessions analyzed: {len(results)}")
    print(f"\nEarth field magnitude estimates:")
    print(f"  Mean: {np.mean(earth_mags):.2f} µT")
    print(f"  Std:  {np.std(earth_mags):.2f} µT")
    print(f"  Range: {np.min(earth_mags):.2f} - {np.max(earth_mags):.2f} µT")

    print(f"\nResidual magnitudes (session means):")
    print(f"  Mean: {np.mean(residual_means):.2f} µT")
    print(f"  Std:  {np.std(residual_means):.2f} µT")
    print(f"  Range: {np.min(residual_means):.2f} - {np.max(residual_means):.2f} µT")

    good = sum(1 for r in results if r['status'] == '✓ GOOD')
    marginal = sum(1 for r in results if r['status'] == '⚠ MARGINAL')
    high = sum(1 for r in results if r['status'] == '✗ HIGH')

    print(f"\nStatus breakdown:")
    print(f"  ✓ GOOD (<5 µT):     {good}/{len(results)}")
    print(f"  ⚠ MARGINAL (5-15):  {marginal}/{len(results)}")
    print(f"  ✗ HIGH (>15 µT):    {high}/{len(results)}")

    overall = "SUCCESS" if good == len(results) else ("PARTIAL" if good > 0 else "NEEDS WORK")
    print(f"\nOverall: {overall}")

    if np.mean(residual_means) > 5:
        print("\nPossible issues:")
        if np.std(earth_mags) > 10:
            print("  - Earth field estimates vary significantly between sessions")
            print("    → Consider using a fixed Earth field reference for your location")

        # Check if sessions with more orientation coverage have lower residuals
        orientation_ranges = [r['orientation_range']['roll'] + r['orientation_range']['pitch']
                             for r in results]
        if np.corrcoef(orientation_ranges, residual_means)[0,1] < -0.3:
            print("  - Sessions with more orientation movement have lower residuals")
            print("    → More movement helps estimate Earth field better")


def main():
    data_dir = Path('data/GAMBIT')

    sessions = sorted(data_dir.glob('*.json'))
    sessions = [s for s in sessions if not s.name.startswith('gambit') and 'manifest' not in s.name]

    print("=" * 70)
    print("RAW MAGNETIC FIELD ANALYSIS")
    print("Computing Earth field subtraction from raw data per session")
    print("(Ignoring stored calibration values)")
    print("=" * 70)

    results = []
    for session_path in sessions:
        result = analyze_session(session_path)
        if result:
            results.append(result)

    if results:
        print_summary(results)
    else:
        print("\nNo valid sessions found with required data")


if __name__ == '__main__':
    main()
