#!/usr/bin/env python3
"""
Offline Earth Field Subtraction Hypothesis Test

Tests whether proper orientation-compensated Earth field subtraction improves:
1. SNR (Signal-to-Noise Ratio)
2. Polarity pattern detection (alternating N/S finger magnets)
3. Octant distribution (should become more diverse)

Compares:
- RAW: Uncorrected magnetometer data
- OFFLINE: Recalculated with proper orientation-aware Earth subtraction

No external dependencies - uses only Python standard library.
"""

import json
import math
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from collections import defaultdict


def mean(arr: List[float]) -> float:
    """Calculate mean of list."""
    if not arr:
        return 0.0
    return sum(arr) / len(arr)


def std(arr: List[float]) -> float:
    """Calculate standard deviation."""
    if len(arr) < 2:
        return 0.0
    m = mean(arr)
    variance = sum((x - m) ** 2 for x in arr) / len(arr)
    return math.sqrt(variance)


def percentile(arr: List[float], p: float) -> float:
    """Calculate percentile (0-100)."""
    if not arr:
        return 0.0
    sorted_arr = sorted(arr)
    k = (len(sorted_arr) - 1) * p / 100
    f = int(math.floor(k))
    c = int(math.ceil(k))
    if f == c:
        return sorted_arr[int(k)]
    return sorted_arr[f] * (c - k) + sorted_arr[c] * (k - f)


def magnitude(x: float, y: float, z: float) -> float:
    """Calculate 3D vector magnitude."""
    return math.sqrt(x*x + y*y + z*z)


def matrix_multiply_3x3_vec(M: List[List[float]], v: List[float]) -> List[float]:
    """Multiply 3x3 matrix by 3-vector."""
    return [
        M[0][0]*v[0] + M[0][1]*v[1] + M[0][2]*v[2],
        M[1][0]*v[0] + M[1][1]*v[1] + M[1][2]*v[2],
        M[2][0]*v[0] + M[2][1]*v[1] + M[2][2]*v[2],
    ]


def transpose_3x3(M: List[List[float]]) -> List[List[float]]:
    """Transpose 3x3 matrix."""
    return [
        [M[0][0], M[1][0], M[2][0]],
        [M[0][1], M[1][1], M[2][1]],
        [M[0][2], M[1][2], M[2][2]],
    ]


def quaternion_to_rotation_matrix(q: Dict) -> List[List[float]]:
    """Convert quaternion to 3x3 rotation matrix."""
    w, x, y, z = q['w'], q['x'], q['y'], q['z']

    # Normalize
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n > 0:
        w, x, y, z = w/n, x/n, y/n, z/n

    return [
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ]


def load_session(filepath: str) -> Tuple[Dict, List[Dict]]:
    """Load GAMBIT session file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    metadata = {
        'version': data.get('version', 'unknown'),
        'timestamp': data.get('timestamp', ''),
    }
    samples = data.get('samples', [])
    return metadata, samples


def estimate_hard_iron(samples: List[Dict]) -> List[float]:
    """Estimate hard iron offset using min/max method."""
    mx = [s.get('mx_ut', 0) for s in samples]
    my = [s.get('my_ut', 0) for s in samples]
    mz = [s.get('mz_ut', 0) for s in samples]

    return [
        (max(mx) + min(mx)) / 2,
        (max(my) + min(my)) / 2,
        (max(mz) + min(mz)) / 2
    ]


def estimate_earth_field_world_frame(samples: List[Dict], hard_iron: List[float]) -> List[float]:
    """
    Estimate Earth field vector in world frame.

    For each sample, rotate the hard-iron-corrected magnetometer reading
    from sensor frame to world frame, then average.
    """
    earth_vectors_x = []
    earth_vectors_y = []
    earth_vectors_z = []

    for s in samples:
        if 'orientation_w' not in s:
            continue

        # Get magnetometer reading (iron-corrected)
        mag_sensor = [
            s.get('mx_ut', 0) - hard_iron[0],
            s.get('my_ut', 0) - hard_iron[1],
            s.get('mz_ut', 0) - hard_iron[2]
        ]

        q = {
            'w': s['orientation_w'],
            'x': s['orientation_x'],
            'y': s['orientation_y'],
            'z': s['orientation_z']
        }

        # R transforms world->sensor, so R.T transforms sensor->world
        R = quaternion_to_rotation_matrix(q)
        R_T = transpose_3x3(R)
        mag_world = matrix_multiply_3x3_vec(R_T, mag_sensor)

        earth_vectors_x.append(mag_world[0])
        earth_vectors_y.append(mag_world[1])
        earth_vectors_z.append(mag_world[2])

    if not earth_vectors_x:
        return [0.0, 0.0, 0.0]

    return [mean(earth_vectors_x), mean(earth_vectors_y), mean(earth_vectors_z)]


def compute_offline_residuals(samples: List[Dict], hard_iron: List[float],
                              earth_field_world: List[float]) -> List[Dict]:
    """
    Compute residual for each sample using proper orientation-compensated Earth subtraction.
    """
    results = []

    for i, s in enumerate(samples):
        if 'orientation_w' not in s:
            continue

        # Raw magnetometer (µT)
        mag_raw = [s.get('mx_ut', 0), s.get('my_ut', 0), s.get('mz_ut', 0)]

        # Hard iron correction
        mag_corrected = [
            mag_raw[0] - hard_iron[0],
            mag_raw[1] - hard_iron[1],
            mag_raw[2] - hard_iron[2]
        ]

        # Get orientation
        q = {
            'w': s['orientation_w'],
            'x': s['orientation_x'],
            'y': s['orientation_y'],
            'z': s['orientation_z']
        }

        # Rotate Earth field from world to sensor frame
        R = quaternion_to_rotation_matrix(q)
        earth_sensor = matrix_multiply_3x3_vec(R, earth_field_world)

        # Residual = measured - expected (finger magnet signal only)
        residual = [
            mag_corrected[0] - earth_sensor[0],
            mag_corrected[1] - earth_sensor[1],
            mag_corrected[2] - earth_sensor[2]
        ]

        results.append({
            'index': i,
            'raw': mag_raw,
            'corrected': mag_corrected,
            'earth_sensor': earth_sensor,
            'residual_x': residual[0],
            'residual_y': residual[1],
            'residual_z': residual[2],
            'residual_magnitude': magnitude(residual[0], residual[1], residual[2]),
        })

    return results


def analyze_polarity(mx: List[float], my: List[float], mz: List[float], name: str) -> Dict:
    """
    Analyze polarity patterns in 3D magnetic data.
    """
    # Sign analysis per axis
    def axis_signs(arr, axis_name):
        pos = sum(1 for v in arr if v > 0)
        neg = sum(1 for v in arr if v < 0)
        total = len(arr)

        # Count sign transitions
        transitions = 0
        for i in range(1, len(arr)):
            if (arr[i] > 0) != (arr[i-1] > 0) and arr[i] != 0 and arr[i-1] != 0:
                transitions += 1

        return {
            'positive_pct': 100 * pos / total if total > 0 else 0,
            'negative_pct': 100 * neg / total if total > 0 else 0,
            'transitions': transitions,
            'dominant': 'positive' if pos > neg else 'negative',
        }

    # Octant analysis (8 possible combinations of +/- for x,y,z)
    octant_counts = defaultdict(int)
    for i in range(len(mx)):
        octant = (
            ('+' if mx[i] > 0 else '-') +
            ('+' if my[i] > 0 else '-') +
            ('+' if mz[i] > 0 else '-')
        )
        octant_counts[octant] += 1

    # Find dominant octant
    if octant_counts:
        dominant_octant = max(octant_counts.items(), key=lambda x: x[1])
    else:
        dominant_octant = ('---', 0)

    mx_signs = axis_signs(mx, 'mx')
    my_signs = axis_signs(my, 'my')
    mz_signs = axis_signs(mz, 'mz')

    return {
        'name': name,
        'mx': mx_signs,
        'my': my_signs,
        'mz': mz_signs,
        'total_transitions': mx_signs['transitions'] + my_signs['transitions'] + mz_signs['transitions'],
        'octant_distribution': dict(octant_counts),
        'unique_octants': len(octant_counts),
        'dominant_octant': dominant_octant[0],
        'dominant_octant_pct': 100 * dominant_octant[1] / len(mx) if mx else 0,
    }


def analyze_snr(magnitudes: List[float], name: str) -> Dict:
    """
    Analyze Signal-to-Noise Ratio metrics.
    """
    if not magnitudes:
        return {'name': name, 'error': 'No data'}

    # Estimate baseline as 25th percentile
    baseline = percentile(magnitudes, 25)

    # Estimate peak signal as 95th percentile
    peak = percentile(magnitudes, 95)

    # SNR as ratio
    snr = peak / baseline if baseline > 0 else 0

    return {
        'name': name,
        'mean': mean(magnitudes),
        'std': std(magnitudes),
        'min': min(magnitudes),
        'max': max(magnitudes),
        'baseline_25pct': baseline,
        'peak_95pct': peak,
        'snr_ratio': snr,
        'dynamic_range': peak - baseline,
    }


def compare_before_after(filepath: str, verbose: bool = True) -> Dict:
    """
    Compare RAW vs OFFLINE-CORRECTED data for a session.

    Returns comparison metrics for SNR and polarity detection.
    """
    metadata, samples = load_session(filepath)

    if len(samples) < 50:
        return {'error': 'Insufficient samples'}

    filename = Path(filepath).name

    if verbose:
        print(f"\n{'='*80}")
        print(f"HYPOTHESIS TEST: {filename}")
        print(f"{'='*80}")
        print(f"Samples: {len(samples)}")

    # =========================================================================
    # STEP 1: Extract RAW data
    # =========================================================================
    raw_mx = [s.get('mx_ut', 0) for s in samples]
    raw_my = [s.get('my_ut', 0) for s in samples]
    raw_mz = [s.get('mz_ut', 0) for s in samples]
    raw_magnitudes = [magnitude(raw_mx[i], raw_my[i], raw_mz[i]) for i in range(len(samples))]

    # =========================================================================
    # STEP 2: Compute OFFLINE Earth field subtraction
    # =========================================================================
    hard_iron = estimate_hard_iron(samples)
    earth_world = estimate_earth_field_world_frame(samples, hard_iron)
    earth_mag = magnitude(earth_world[0], earth_world[1], earth_world[2])

    if verbose:
        print(f"\n--- OFFLINE CALIBRATION ---")
        print(f"Hard Iron Offset: [{hard_iron[0]:.1f}, {hard_iron[1]:.1f}, {hard_iron[2]:.1f}] µT")
        print(f"Earth Field (world): [{earth_world[0]:.1f}, {earth_world[1]:.1f}, {earth_world[2]:.1f}] µT")
        print(f"Earth Field Magnitude: {earth_mag:.1f} µT")

        if earth_mag < 20 or earth_mag > 80:
            print(f"  ⚠ Warning: Earth magnitude outside typical 25-65 µT range")

    # Compute offline residuals
    offline_results = compute_offline_residuals(samples, hard_iron, earth_world)

    if not offline_results:
        return {'error': 'No valid orientation data for offline correction'}

    offline_mx = [r['residual_x'] for r in offline_results]
    offline_my = [r['residual_y'] for r in offline_results]
    offline_mz = [r['residual_z'] for r in offline_results]
    offline_magnitudes = [r['residual_magnitude'] for r in offline_results]

    # =========================================================================
    # STEP 3: Compare SNR
    # =========================================================================
    raw_snr = analyze_snr(raw_magnitudes, 'RAW')
    offline_snr = analyze_snr(offline_magnitudes, 'OFFLINE')

    if verbose:
        print(f"\n--- SNR COMPARISON ---")
        print(f"{'Metric':<25} {'RAW':>12} {'OFFLINE':>12} {'Change':>12}")
        print("-" * 63)
        print(f"{'Mean Magnitude (µT)':<25} {raw_snr['mean']:>12.1f} {offline_snr['mean']:>12.1f} {offline_snr['mean'] - raw_snr['mean']:>+12.1f}")
        print(f"{'Std Dev (µT)':<25} {raw_snr['std']:>12.1f} {offline_snr['std']:>12.1f} {offline_snr['std'] - raw_snr['std']:>+12.1f}")
        print(f"{'Baseline (25th pct)':<25} {raw_snr['baseline_25pct']:>12.1f} {offline_snr['baseline_25pct']:>12.1f} {offline_snr['baseline_25pct'] - raw_snr['baseline_25pct']:>+12.1f}")
        print(f"{'Peak (95th pct)':<25} {raw_snr['peak_95pct']:>12.1f} {offline_snr['peak_95pct']:>12.1f} {offline_snr['peak_95pct'] - raw_snr['peak_95pct']:>+12.1f}")
        print(f"{'Dynamic Range':<25} {raw_snr['dynamic_range']:>12.1f} {offline_snr['dynamic_range']:>12.1f} {offline_snr['dynamic_range'] - raw_snr['dynamic_range']:>+12.1f}")
        print(f"{'SNR Ratio':<25} {raw_snr['snr_ratio']:>12.2f}x {offline_snr['snr_ratio']:>12.2f}x {offline_snr['snr_ratio'] - raw_snr['snr_ratio']:>+12.2f}x")

    # =========================================================================
    # STEP 4: Compare Polarity Detection
    # =========================================================================
    raw_polarity = analyze_polarity(raw_mx, raw_my, raw_mz, 'RAW')
    offline_polarity = analyze_polarity(offline_mx, offline_my, offline_mz, 'OFFLINE')

    if verbose:
        print(f"\n--- POLARITY COMPARISON ---")
        print(f"{'Metric':<25} {'RAW':>12} {'OFFLINE':>12} {'Change':>12}")
        print("-" * 63)
        print(f"{'Total Sign Transitions':<25} {raw_polarity['total_transitions']:>12} {offline_polarity['total_transitions']:>12} {offline_polarity['total_transitions'] - raw_polarity['total_transitions']:>+12}")
        print(f"{'Unique Octants (of 8)':<25} {raw_polarity['unique_octants']:>12} {offline_polarity['unique_octants']:>12} {offline_polarity['unique_octants'] - raw_polarity['unique_octants']:>+12}")
        print(f"{'Dominant Octant %':<25} {raw_polarity['dominant_octant_pct']:>11.1f}% {offline_polarity['dominant_octant_pct']:>11.1f}% {offline_polarity['dominant_octant_pct'] - raw_polarity['dominant_octant_pct']:>+11.1f}%")
        print(f"{'Dominant Octant':<25} {raw_polarity['dominant_octant']:>12} {offline_polarity['dominant_octant']:>12}")

    # Axis-specific transitions
    if verbose:
        print(f"\n--- AXIS TRANSITIONS ---")
        print(f"{'Axis':<10} {'RAW Trans':>12} {'OFFLINE Trans':>14} {'Change':>12}")
        print("-" * 50)
        for axis in ['mx', 'my', 'mz']:
            raw_t = raw_polarity[axis]['transitions']
            off_t = offline_polarity[axis]['transitions']
            print(f"{axis:<10} {raw_t:>12} {off_t:>14} {off_t - raw_t:>+12}")

    # Octant distribution
    if verbose:
        print(f"\n--- OCTANT DISTRIBUTION ---")
        print(f"{'Octant':<10} {'RAW %':>10} {'OFFLINE %':>12}")
        print("-" * 34)

        all_octants = set(raw_polarity['octant_distribution'].keys()) | set(offline_polarity['octant_distribution'].keys())
        for octant in sorted(all_octants):
            raw_count = raw_polarity['octant_distribution'].get(octant, 0)
            off_count = offline_polarity['octant_distribution'].get(octant, 0)
            raw_pct = 100 * raw_count / len(raw_mx) if raw_mx else 0
            off_pct = 100 * off_count / len(offline_mx) if offline_mx else 0
            print(f"{octant:<10} {raw_pct:>9.1f}% {off_pct:>11.1f}%")

    # =========================================================================
    # STEP 5: Hypothesis Evaluation
    # =========================================================================
    snr_improved = offline_snr['snr_ratio'] > raw_snr['snr_ratio']
    transitions_improved = offline_polarity['total_transitions'] > raw_polarity['total_transitions']
    octants_improved = offline_polarity['unique_octants'] > raw_polarity['unique_octants']
    dominance_reduced = offline_polarity['dominant_octant_pct'] < raw_polarity['dominant_octant_pct']

    score = sum([snr_improved, transitions_improved, octants_improved, dominance_reduced])

    if verbose:
        print(f"\n--- HYPOTHESIS EVALUATION ---")
        print(f"{'Hypothesis':<45} {'Result':>10}")
        print("-" * 57)
        print(f"{'H1: SNR improves with Earth subtraction':<45} {'✓ YES' if snr_improved else '✗ NO':>10}")
        print(f"{'H2: More polarity transitions detected':<45} {'✓ YES' if transitions_improved else '✗ NO':>10}")
        print(f"{'H3: More octants visited (diversity)':<45} {'✓ YES' if octants_improved else '✗ NO':>10}")
        print(f"{'H4: Dominant octant % decreases':<45} {'✓ YES' if dominance_reduced else '✗ NO':>10}")

        print(f"\nOverall: {score}/4 hypotheses validated")

        if score >= 3:
            print("→ Earth field subtraction SIGNIFICANTLY IMPROVES detection")
        elif score >= 2:
            print("→ Earth field subtraction provides MODERATE improvement")
        else:
            print("→ Earth field subtraction provides LIMITED improvement")
            print("  (Consider: magnet polarity configuration may be the issue)")

    return {
        'filename': filename,
        'samples': len(samples),
        'hard_iron': hard_iron,
        'earth_field_world': earth_world,
        'earth_magnitude': earth_mag,
        'raw_snr': raw_snr,
        'offline_snr': offline_snr,
        'raw_polarity': raw_polarity,
        'offline_polarity': offline_polarity,
        'hypotheses': {
            'snr_improved': snr_improved,
            'transitions_improved': transitions_improved,
            'octants_improved': octants_improved,
            'dominance_reduced': dominance_reduced,
            'score': score,
        }
    }


def main():
    """Run hypothesis test on today's sessions."""
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    # Find today's sessions after 22:00
    target_date = "2025-12-15"
    target_hour = 22

    json_files = list(data_dir.glob(f'{target_date}*.json'))

    # Filter by time
    filtered_files = []
    for f in json_files:
        name = f.name
        if 'gambit' in name.lower() or 'manifest' in name.lower():
            continue
        try:
            ts_str = name.replace('.json', '').replace('_', ':')
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            if ts.hour >= target_hour:
                filtered_files.append((f, ts))
        except ValueError:
            continue

    if not filtered_files:
        print(f"No sessions found for {target_date} after {target_hour}:00")
        return

    filtered_files.sort(key=lambda x: x[1])

    print("=" * 80)
    print("OFFLINE EARTH FIELD SUBTRACTION - HYPOTHESIS TEST")
    print("=" * 80)
    print(f"\nTesting whether proper orientation-compensated Earth field subtraction improves:")
    print("  H1: SNR (Signal-to-Noise Ratio)")
    print("  H2: Polarity transitions (alternating N/S detection)")
    print("  H3: Octant diversity (field direction variety)")
    print("  H4: Reduced dominance (less clustered readings)")

    all_results = []
    for filepath, _ in filtered_files:
        result = compare_before_after(str(filepath), verbose=True)
        if 'error' not in result:
            all_results.append(result)

    # Summary
    if len(all_results) > 1:
        print(f"\n{'='*80}")
        print("CROSS-SESSION SUMMARY")
        print(f"{'='*80}")

        total_score = sum(r['hypotheses']['score'] for r in all_results)
        max_score = 4 * len(all_results)

        print(f"\nSessions analyzed: {len(all_results)}")
        print(f"Total hypothesis validations: {total_score}/{max_score}")

        # Average improvements
        avg_snr_raw = mean([r['raw_snr']['snr_ratio'] for r in all_results])
        avg_snr_off = mean([r['offline_snr']['snr_ratio'] for r in all_results])
        avg_trans_raw = mean([r['raw_polarity']['total_transitions'] for r in all_results])
        avg_trans_off = mean([r['offline_polarity']['total_transitions'] for r in all_results])

        print(f"\nAverage SNR: {avg_snr_raw:.2f}x (raw) → {avg_snr_off:.2f}x (offline)")
        print(f"Average Transitions: {avg_trans_raw:.0f} (raw) → {avg_trans_off:.0f} (offline)")

    print(f"\n{'='*80}")
    print("ANALYSIS COMPLETE")
    print(f"{'='*80}")

    return all_results


if __name__ == '__main__':
    main()
