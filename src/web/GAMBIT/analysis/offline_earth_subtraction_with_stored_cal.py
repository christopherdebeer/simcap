#!/usr/bin/env python3
"""
Offline Earth Field Subtraction using STORED Calibration

Uses the calibration values from the calibration wizard (done WITHOUT magnets)
instead of estimating from session data (which is polluted by magnets).

From magnetometer-calibration-investigation.md:
  Hard Iron Offset: {x: 4.5, y: 520.5, z: -482} LSB
  Earth Field: {x: 18.11, y: 368.8, z: -284.52} LSB

LSB to µT conversion: ~0.09765625 µT/LSB (for ±4 gauss range)
"""

import json
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# ============================================================================
# STORED CALIBRATION VALUES (from calibration wizard, no magnets)
# ============================================================================

# Conversion factor: µT per LSB
LSB_TO_UT = 100 / 1024  # ~0.09765625

# Hard iron offset (in µT)
STORED_HARD_IRON = {
    'x': 4.5 * LSB_TO_UT,      # ~0.44 µT
    'y': 520.5 * LSB_TO_UT,    # ~50.8 µT
    'z': -482 * LSB_TO_UT,     # ~-47.1 µT
}

# Earth field in SENSOR frame at calibration reference orientation
# NOTE: This needs to be in WORLD frame for proper compensation
# We'll need to estimate world-frame Earth field differently
STORED_EARTH_SENSOR = {
    'x': 18.11 * LSB_TO_UT,    # ~1.8 µT
    'y': 368.8 * LSB_TO_UT,    # ~36.0 µT
    'z': -284.52 * LSB_TO_UT,  # ~-27.8 µT
}

# Typical Earth field magnitude for reference
EXPECTED_EARTH_MAGNITUDE = 50.0  # µT (typical value)


def mean(arr):
    return sum(arr) / len(arr) if arr else 0

def std(arr):
    if len(arr) < 2:
        return 0
    m = mean(arr)
    return math.sqrt(sum((x - m) ** 2 for x in arr) / len(arr))

def percentile(arr, p):
    if not arr:
        return 0
    sorted_arr = sorted(arr)
    k = (len(sorted_arr) - 1) * p / 100
    f, c = int(k), int(k) + 1
    if c >= len(sorted_arr):
        return sorted_arr[-1]
    return sorted_arr[f] * (c - k) + sorted_arr[min(c, len(sorted_arr)-1)] * (k - f)

def magnitude(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def matrix_mult(M, v):
    return [
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
    ]

def transpose(M):
    return [[M[j][i] for j in range(3)] for i in range(3)]

def quat_to_matrix(q):
    w, x, y, z = q['w'], q['x'], q['y'], q['z']
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n > 0:
        w, x, y, z = w/n, x/n, y/n, z/n
    return [
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ]


def estimate_earth_world_from_stationary(samples, hard_iron):
    """
    Estimate Earth field in world frame using only near-stationary samples.
    This reduces magnet influence since magnets vary more during movement.
    """
    earth_x, earth_y, earth_z = [], [], []

    for s in samples:
        if 'orientation_w' not in s:
            continue

        # Only use samples with low motion
        gyro_mag = magnitude(
            s.get('gx_dps', 0),
            s.get('gy_dps', 0),
            s.get('gz_dps', 0)
        )

        # Skip if rotating fast
        if gyro_mag > 10:  # degrees per second
            continue

        # Get iron-corrected magnetometer
        mag = [
            s.get('mx_ut', 0) - hard_iron['x'],
            s.get('my_ut', 0) - hard_iron['y'],
            s.get('mz_ut', 0) - hard_iron['z']
        ]

        q = {
            'w': s['orientation_w'],
            'x': s['orientation_x'],
            'y': s['orientation_y'],
            'z': s['orientation_z']
        }

        # Transform to world frame
        R = quat_to_matrix(q)
        R_T = transpose(R)
        mag_world = matrix_mult(R_T, mag)

        earth_x.append(mag_world[0])
        earth_y.append(mag_world[1])
        earth_z.append(mag_world[2])

    if not earth_x:
        return None

    return [mean(earth_x), mean(earth_y), mean(earth_z)]


def compute_residuals_with_stored_cal(samples):
    """
    Compute residuals using stored calibration values.
    """
    hard_iron = STORED_HARD_IRON

    # First, estimate Earth field in world frame from stationary samples
    earth_world = estimate_earth_world_from_stationary(samples, hard_iron)

    if earth_world is None:
        print("  Warning: Could not estimate Earth field from stationary samples")
        # Fall back to stored value (in sensor frame at reference)
        earth_world = [STORED_EARTH_SENSOR['x'], STORED_EARTH_SENSOR['y'], STORED_EARTH_SENSOR['z']]

    earth_mag = magnitude(earth_world[0], earth_world[1], earth_world[2])

    print(f"\n--- USING STORED HARD IRON ---")
    print(f"Hard Iron: [{hard_iron['x']:.1f}, {hard_iron['y']:.1f}, {hard_iron['z']:.1f}] µT")
    print(f"Earth Field (world): [{earth_world[0]:.1f}, {earth_world[1]:.1f}, {earth_world[2]:.1f}] µT")
    print(f"Earth Magnitude: {earth_mag:.1f} µT")

    if earth_mag < 20 or earth_mag > 80:
        print(f"  ⚠ Warning: Outside typical 25-65 µT range")
    else:
        print(f"  ✓ Within expected range")

    results = []
    for s in samples:
        if 'orientation_w' not in s:
            continue

        # Raw magnetometer
        raw = [s.get('mx_ut', 0), s.get('my_ut', 0), s.get('mz_ut', 0)]

        # Hard iron correction
        corrected = [
            raw[0] - hard_iron['x'],
            raw[1] - hard_iron['y'],
            raw[2] - hard_iron['z']
        ]

        q = {
            'w': s['orientation_w'],
            'x': s['orientation_x'],
            'y': s['orientation_y'],
            'z': s['orientation_z']
        }

        # Rotate Earth from world to sensor
        R = quat_to_matrix(q)
        earth_sensor = matrix_mult(R, earth_world)

        # Residual = corrected - earth (should be only finger magnets)
        residual = [
            corrected[0] - earth_sensor[0],
            corrected[1] - earth_sensor[1],
            corrected[2] - earth_sensor[2]
        ]

        results.append({
            'raw': raw,
            'corrected': corrected,
            'residual': residual,
            'residual_mag': magnitude(residual[0], residual[1], residual[2])
        })

    return results, earth_mag


def analyze_polarity(mx, my, mz):
    """Analyze polarity patterns."""
    def axis_signs(arr):
        pos = sum(1 for v in arr if v > 0)
        neg = sum(1 for v in arr if v < 0)
        trans = sum(1 for i in range(1, len(arr))
                   if (arr[i] > 0) != (arr[i-1] > 0) and arr[i] != 0 and arr[i-1] != 0)
        return {'pos_pct': 100*pos/len(arr), 'neg_pct': 100*neg/len(arr), 'transitions': trans}

    octants = defaultdict(int)
    for i in range(len(mx)):
        o = ('+' if mx[i] > 0 else '-') + ('+' if my[i] > 0 else '-') + ('+' if mz[i] > 0 else '-')
        octants[o] += 1

    dominant = max(octants.items(), key=lambda x: x[1])

    return {
        'mx': axis_signs(mx),
        'my': axis_signs(my),
        'mz': axis_signs(mz),
        'total_transitions': axis_signs(mx)['transitions'] + axis_signs(my)['transitions'] + axis_signs(mz)['transitions'],
        'octants': dict(octants),
        'unique_octants': len(octants),
        'dominant': dominant[0],
        'dominant_pct': 100 * dominant[1] / len(mx)
    }


def analyze_snr(magnitudes):
    baseline = percentile(magnitudes, 25)
    peak = percentile(magnitudes, 95)
    return {
        'mean': mean(magnitudes),
        'std': std(magnitudes),
        'baseline': baseline,
        'peak': peak,
        'snr': peak / baseline if baseline > 0 else 0,
        'range': peak - baseline
    }


def analyze_session(filepath):
    """Analyze session with stored calibration."""
    with open(filepath) as f:
        data = json.load(f)

    samples = data.get('samples', [])
    filename = Path(filepath).name

    print(f"\n{'='*70}")
    print(f"ANALYSIS: {filename}")
    print(f"{'='*70}")
    print(f"Samples: {len(samples)}")

    # RAW data
    raw_mx = [s.get('mx_ut', 0) for s in samples]
    raw_my = [s.get('my_ut', 0) for s in samples]
    raw_mz = [s.get('mz_ut', 0) for s in samples]
    raw_mags = [magnitude(raw_mx[i], raw_my[i], raw_mz[i]) for i in range(len(samples))]

    # CORRECTED with stored calibration
    results, earth_mag = compute_residuals_with_stored_cal(samples)

    if not results:
        print("No valid samples with orientation")
        return None

    res_mx = [r['residual'][0] for r in results]
    res_my = [r['residual'][1] for r in results]
    res_mz = [r['residual'][2] for r in results]
    res_mags = [r['residual_mag'] for r in results]

    # SNR comparison
    raw_snr = analyze_snr(raw_mags)
    res_snr = analyze_snr(res_mags)

    print(f"\n--- SNR COMPARISON ---")
    print(f"{'Metric':<20} {'RAW':>12} {'CORRECTED':>12} {'Change':>12}")
    print("-" * 58)
    print(f"{'Mean (µT)':<20} {raw_snr['mean']:>12.1f} {res_snr['mean']:>12.1f} {res_snr['mean']-raw_snr['mean']:>+12.1f}")
    print(f"{'Std (µT)':<20} {raw_snr['std']:>12.1f} {res_snr['std']:>12.1f} {res_snr['std']-raw_snr['std']:>+12.1f}")
    print(f"{'Baseline (25%)':<20} {raw_snr['baseline']:>12.1f} {res_snr['baseline']:>12.1f} {res_snr['baseline']-raw_snr['baseline']:>+12.1f}")
    print(f"{'Peak (95%)':<20} {raw_snr['peak']:>12.1f} {res_snr['peak']:>12.1f} {res_snr['peak']-raw_snr['peak']:>+12.1f}")
    print(f"{'SNR Ratio':<20} {raw_snr['snr']:>12.2f}x {res_snr['snr']:>12.2f}x {res_snr['snr']-raw_snr['snr']:>+12.2f}x")

    # Polarity comparison
    raw_pol = analyze_polarity(raw_mx, raw_my, raw_mz)
    res_pol = analyze_polarity(res_mx, res_my, res_mz)

    print(f"\n--- POLARITY COMPARISON ---")
    print(f"{'Metric':<25} {'RAW':>12} {'CORRECTED':>12} {'Change':>10}")
    print("-" * 61)
    print(f"{'Total Transitions':<25} {raw_pol['total_transitions']:>12} {res_pol['total_transitions']:>12} {res_pol['total_transitions']-raw_pol['total_transitions']:>+10}")
    print(f"{'Unique Octants':<25} {raw_pol['unique_octants']:>12} {res_pol['unique_octants']:>12} {res_pol['unique_octants']-raw_pol['unique_octants']:>+10}")
    print(f"{'Dominant Octant %':<25} {raw_pol['dominant_pct']:>11.1f}% {res_pol['dominant_pct']:>11.1f}% {res_pol['dominant_pct']-raw_pol['dominant_pct']:>+9.1f}%")
    print(f"{'Dominant Octant':<25} {raw_pol['dominant']:>12} {res_pol['dominant']:>12}")

    # Octant distribution
    print(f"\n--- OCTANT DISTRIBUTION ---")
    print(f"{'Octant':<10} {'RAW':>12} {'CORRECTED':>12}")
    print("-" * 36)
    all_oct = set(raw_pol['octants'].keys()) | set(res_pol['octants'].keys())
    for o in sorted(all_oct):
        raw_pct = 100 * raw_pol['octants'].get(o, 0) / len(raw_mx)
        res_pct = 100 * res_pol['octants'].get(o, 0) / len(res_mx)
        print(f"{o:<10} {raw_pct:>11.1f}% {res_pct:>11.1f}%")

    # Hypothesis evaluation
    snr_improved = res_snr['snr'] > raw_snr['snr']
    trans_improved = res_pol['total_transitions'] > raw_pol['total_transitions']
    oct_improved = res_pol['unique_octants'] > raw_pol['unique_octants']
    dom_reduced = res_pol['dominant_pct'] < raw_pol['dominant_pct']

    score = sum([snr_improved, trans_improved, oct_improved, dom_reduced])

    print(f"\n--- HYPOTHESIS EVALUATION ---")
    print(f"H1: SNR improved:           {'✓' if snr_improved else '✗'}")
    print(f"H2: More transitions:       {'✓' if trans_improved else '✗'}")
    print(f"H3: More octants:           {'✓' if oct_improved else '✗'}")
    print(f"H4: Less dominant:          {'✓' if dom_reduced else '✗'}")
    print(f"\nScore: {score}/4")

    return {
        'earth_mag': earth_mag,
        'raw_snr': raw_snr,
        'res_snr': res_snr,
        'raw_pol': raw_pol,
        'res_pol': res_pol,
        'score': score
    }


def main():
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    print("=" * 70)
    print("OFFLINE EARTH SUBTRACTION WITH STORED CALIBRATION")
    print("=" * 70)
    print("\nUsing calibration values from calibration wizard (no magnets)")
    print(f"Stored Hard Iron: [{STORED_HARD_IRON['x']:.1f}, {STORED_HARD_IRON['y']:.1f}, {STORED_HARD_IRON['z']:.1f}] µT")

    results = []
    for f in sorted(data_dir.glob('2025-12-15T22*.json')):
        if 'gambit' not in f.name.lower():
            r = analyze_session(str(f))
            if r:
                results.append(r)

    if len(results) > 1:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")

        avg_earth = mean([r['earth_mag'] for r in results])
        avg_raw_snr = mean([r['raw_snr']['snr'] for r in results])
        avg_res_snr = mean([r['res_snr']['snr'] for r in results])
        total_score = sum(r['score'] for r in results)

        print(f"\nAverage Earth Field Magnitude: {avg_earth:.1f} µT")
        print(f"Average SNR: {avg_raw_snr:.2f}x (raw) → {avg_res_snr:.2f}x (corrected)")
        print(f"Total Score: {total_score}/{4*len(results)}")


if __name__ == '__main__':
    main()
