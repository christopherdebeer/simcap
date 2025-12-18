#!/usr/bin/env python3
"""
Earth Field Estimation from RAW Session Data Only

No external calibration - estimate everything from the session itself.

Key insight: When we transform raw magnetometer readings to world frame and average:
- Earth field (constant in world) → stays constant → gives true Earth
- Hard iron (constant in sensor) → rotates with device → averages toward zero
- Finger magnets (sensor frame) → rotates with device → averages toward zero

This should work IF there's sufficient orientation variation.
"""

import json
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict


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
    s = sorted(arr)
    k = (len(s) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] * (c - k) + s[c] * (k - f) if f != c else s[f]

def mag3(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def mat_vec(M, v):
    return [sum(M[i][j] * v[j] for j in range(3)) for i in range(3)]

def transpose(M):
    return [[M[j][i] for j in range(3)] for i in range(3)]

def quat_to_mat(w, x, y, z):
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n > 0:
        w, x, y, z = w/n, x/n, y/n, z/n
    return [
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ]


def estimate_earth_from_raw(samples):
    """
    Estimate Earth field by transforming RAW readings to world frame and averaging.

    No hard iron correction - rely on rotation to average out sensor-frame biases.
    """
    world_x, world_y, world_z = [], [], []

    for s in samples:
        if 'orientation_w' not in s:
            continue

        # RAW magnetometer in µT (no corrections)
        mx = s.get('mx_ut', 0)
        my = s.get('my_ut', 0)
        mz = s.get('mz_ut', 0)

        # Transform to world frame using R.T (sensor→world)
        R = quat_to_mat(
            s['orientation_w'], s['orientation_x'],
            s['orientation_y'], s['orientation_z']
        )
        R_T = transpose(R)

        world = mat_vec(R_T, [mx, my, mz])
        world_x.append(world[0])
        world_y.append(world[1])
        world_z.append(world[2])

    if not world_x:
        return None

    # Average in world frame = Earth field (biases average out with rotation)
    return [mean(world_x), mean(world_y), mean(world_z)]


def compute_residuals(samples, earth_world):
    """
    Compute residuals by subtracting orientation-rotated Earth field from raw readings.
    """
    results = []

    for s in samples:
        if 'orientation_w' not in s:
            continue

        # RAW magnetometer
        raw = [s.get('mx_ut', 0), s.get('my_ut', 0), s.get('mz_ut', 0)]

        # Rotate Earth from world to sensor frame using R
        R = quat_to_mat(
            s['orientation_w'], s['orientation_x'],
            s['orientation_y'], s['orientation_z']
        )
        earth_sensor = mat_vec(R, earth_world)

        # Residual = raw - Earth (should be hard_iron + magnets)
        residual = [raw[i] - earth_sensor[i] for i in range(3)]

        results.append({
            'raw': raw,
            'earth_sensor': earth_sensor,
            'residual': residual,
            'residual_mag': mag3(*residual)
        })

    return results


def analyze_polarity(mx, my, mz):
    """Analyze sign patterns and octant distribution."""
    def axis_stats(arr):
        pos = sum(1 for v in arr if v > 0)
        trans = sum(1 for i in range(1, len(arr))
                   if (arr[i] > 0) != (arr[i-1] > 0) and arr[i] != 0 and arr[i-1] != 0)
        return {'pos_pct': 100*pos/len(arr) if arr else 0, 'transitions': trans}

    octants = defaultdict(int)
    for i in range(len(mx)):
        o = ('+' if mx[i] > 0 else '-') + ('+' if my[i] > 0 else '-') + ('+' if mz[i] > 0 else '-')
        octants[o] += 1

    dom = max(octants.items(), key=lambda x: x[1]) if octants else ('---', 0)
    total_trans = axis_stats(mx)['transitions'] + axis_stats(my)['transitions'] + axis_stats(mz)['transitions']

    return {
        'mx': axis_stats(mx),
        'my': axis_stats(my),
        'mz': axis_stats(mz),
        'total_transitions': total_trans,
        'octants': dict(octants),
        'unique_octants': len(octants),
        'dominant': dom[0],
        'dominant_pct': 100 * dom[1] / len(mx) if mx else 0
    }


def analyze_snr(mags):
    baseline = percentile(mags, 25)
    peak = percentile(mags, 95)
    return {
        'mean': mean(mags),
        'std': std(mags),
        'baseline': baseline,
        'peak': peak,
        'snr': peak / baseline if baseline > 0 else 0
    }


def analyze_session(filepath):
    with open(filepath) as f:
        data = json.load(f)

    samples = data.get('samples', [])
    name = Path(filepath).name

    print(f"\n{'='*70}")
    print(f"RAW-ONLY ANALYSIS: {name}")
    print(f"{'='*70}")
    print(f"Samples: {len(samples)}")

    # Check orientation variation
    rolls = [s.get('euler_roll', 0) for s in samples if 'euler_roll' in s]
    pitches = [s.get('euler_pitch', 0) for s in samples if 'euler_pitch' in s]
    yaws = [s.get('euler_yaw', 0) for s in samples if 'euler_yaw' in s]

    print(f"\nOrientation Ranges:")
    print(f"  Roll:  {max(rolls)-min(rolls):.0f}°")
    print(f"  Pitch: {max(pitches)-min(pitches):.0f}°")
    print(f"  Yaw:   {max(yaws)-min(yaws):.0f}°")

    # Estimate Earth field from raw data transformed to world frame
    earth_world = estimate_earth_from_raw(samples)
    if not earth_world:
        print("ERROR: No orientation data")
        return None

    earth_mag = mag3(*earth_world)

    print(f"\n--- EARTH FIELD ESTIMATE (from raw world-frame average) ---")
    print(f"Earth (world): [{earth_world[0]:.1f}, {earth_world[1]:.1f}, {earth_world[2]:.1f}] µT")
    print(f"Magnitude: {earth_mag:.1f} µT")

    if 25 <= earth_mag <= 65:
        print(f"  ✓ Within expected Earth field range (25-65 µT)")
    else:
        print(f"  ⚠ Outside expected range - biases may not have averaged out")

    # RAW data analysis
    raw_mx = [s.get('mx_ut', 0) for s in samples]
    raw_my = [s.get('my_ut', 0) for s in samples]
    raw_mz = [s.get('mz_ut', 0) for s in samples]
    raw_mags = [mag3(raw_mx[i], raw_my[i], raw_mz[i]) for i in range(len(samples))]

    # Compute residuals
    results = compute_residuals(samples, earth_world)
    res_mx = [r['residual'][0] for r in results]
    res_my = [r['residual'][1] for r in results]
    res_mz = [r['residual'][2] for r in results]
    res_mags = [r['residual_mag'] for r in results]

    # SNR comparison
    raw_snr = analyze_snr(raw_mags)
    res_snr = analyze_snr(res_mags)

    print(f"\n--- SNR COMPARISON ---")
    print(f"{'Metric':<20} {'RAW':>12} {'RESIDUAL':>12} {'Change':>12}")
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
    print(f"{'Metric':<25} {'RAW':>12} {'RESIDUAL':>12} {'Change':>10}")
    print("-" * 61)
    print(f"{'Total Transitions':<25} {raw_pol['total_transitions']:>12} {res_pol['total_transitions']:>12} {res_pol['total_transitions']-raw_pol['total_transitions']:>+10}")
    print(f"{'Unique Octants':<25} {raw_pol['unique_octants']:>12} {res_pol['unique_octants']:>12} {res_pol['unique_octants']-raw_pol['unique_octants']:>+10}")
    print(f"{'Dominant %':<25} {raw_pol['dominant_pct']:>11.1f}% {res_pol['dominant_pct']:>11.1f}% {res_pol['dominant_pct']-raw_pol['dominant_pct']:>+9.1f}%")
    print(f"{'Dominant Octant':<25} {raw_pol['dominant']:>12} {res_pol['dominant']:>12}")

    # Octant distribution
    print(f"\n--- OCTANT DISTRIBUTION ---")
    all_oct = set(raw_pol['octants'].keys()) | set(res_pol['octants'].keys())
    print(f"{'Octant':<10} {'RAW':>12} {'RESIDUAL':>12}")
    print("-" * 36)
    for o in sorted(all_oct):
        rp = 100 * raw_pol['octants'].get(o, 0) / len(raw_mx) if raw_mx else 0
        cp = 100 * res_pol['octants'].get(o, 0) / len(res_mx) if res_mx else 0
        print(f"{o:<10} {rp:>11.1f}% {cp:>11.1f}%")

    # Summary
    snr_up = res_snr['snr'] > raw_snr['snr']
    trans_up = res_pol['total_transitions'] > raw_pol['total_transitions']
    oct_up = res_pol['unique_octants'] > raw_pol['unique_octants']
    dom_down = res_pol['dominant_pct'] < raw_pol['dominant_pct']

    print(f"\n--- HYPOTHESIS RESULTS ---")
    print(f"H1: SNR improved:      {'✓' if snr_up else '✗'} ({raw_snr['snr']:.2f}x → {res_snr['snr']:.2f}x)")
    print(f"H2: More transitions:  {'✓' if trans_up else '✗'} ({raw_pol['total_transitions']} → {res_pol['total_transitions']})")
    print(f"H3: More octants:      {'✓' if oct_up else '✗'} ({raw_pol['unique_octants']} → {res_pol['unique_octants']})")
    print(f"H4: Less dominant:     {'✓' if dom_down else '✗'} ({raw_pol['dominant_pct']:.1f}% → {res_pol['dominant_pct']:.1f}%)")

    score = sum([snr_up, trans_up, oct_up, dom_down])
    print(f"\nScore: {score}/4")

    return {'earth_mag': earth_mag, 'score': score, 'snr_raw': raw_snr['snr'], 'snr_res': res_snr['snr']}


def main():
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    print("=" * 70)
    print("RAW-ONLY EARTH FIELD ESTIMATION")
    print("=" * 70)
    print("\nMethod: Transform raw readings to world frame, average.")
    print("Theory: Earth is constant in world; biases rotate and average out.")

    results = []
    for f in sorted(data_dir.glob('2025-12-15T22*.json')):
        if 'gambit' not in f.name.lower():
            r = analyze_session(str(f))
            if r:
                results.append(r)

    if results:
        print(f"\n{'='*70}")
        print("SUMMARY")
        print(f"{'='*70}")
        print(f"Sessions: {len(results)}")
        print(f"Avg Earth Magnitude: {mean([r['earth_mag'] for r in results]):.1f} µT")
        print(f"Avg SNR: {mean([r['snr_raw'] for r in results]):.2f}x (raw) → {mean([r['snr_res'] for r in results]):.2f}x (residual)")
        print(f"Total Score: {sum(r['score'] for r in results)}/{4*len(results)}")


if __name__ == '__main__':
    main()
