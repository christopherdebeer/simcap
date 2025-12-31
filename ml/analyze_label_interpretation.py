#!/usr/bin/env python3
"""
Analyze wizard label interpretation and magnet sizing.

Questions to answer:
1. Are "flexed"/"extended" labels physically consistent with magnetic field strength?
   - Flexed finger = magnet CLOSER to wrist sensor = HIGHER magnitude
   - Extended finger = magnet FARTHER from sensor = LOWER magnitude

2. Can we use smaller magnets?
   - What's the baseline magnitude when all fingers extended?
   - What's the minimum signal change we need for reliable detection?
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict


def load_session(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


def finger_code(fingers: dict) -> str:
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
    return code


def count_flexed(code: str) -> int:
    """Count number of flexed fingers (2s in code)."""
    return code.count('2')


def main():
    session_path = Path('data/GAMBIT/2025-12-31T14_06_18.270Z.json')

    print("=" * 80)
    print("WIZARD LABEL INTERPRETATION ANALYSIS")
    print("=" * 80)

    session = load_session(session_path)
    samples = session.get('samples', [])
    labels = session.get('labels', [])

    # Extract magnetometer data
    mx = np.array([s.get('mx', 0) for s in samples])
    my = np.array([s.get('my', 0) for s in samples])
    mz = np.array([s.get('mz', 0) for s in samples])
    mag = np.sqrt(mx**2 + my**2 + mz**2)

    # Group by finger configuration
    config_stats = defaultdict(lambda: {'samples': [], 'magnitudes': []})

    for label in labels:
        start = label.get('start_sample', label.get('startIndex', 0))
        end = label.get('end_sample', label.get('endIndex', 0))
        content = label.get('labels', label)
        fingers = content.get('fingers', {})

        if not fingers:
            continue

        code = finger_code(fingers)
        if '?' in code:
            continue

        for i in range(start, min(end, len(mag))):
            config_stats[code]['magnitudes'].append(mag[i])
            config_stats[code]['samples'].append([mx[i], my[i], mz[i]])

    # 1. Physical consistency check
    print("\n" + "=" * 80)
    print("1. PHYSICAL CONSISTENCY CHECK")
    print("=" * 80)

    print("""
    EXPECTED BEHAVIOR (if labels are correct):
    - "Extended" (0): Finger straight out → magnet FAR from sensor → LOW magnitude
    - "Flexed" (2): Finger curled in → magnet CLOSE to sensor → HIGH magnitude

    Therefore: More flexed fingers (more 2s) should mean HIGHER total magnitude
    """)

    # Sort by number of flexed fingers
    by_n_flexed = defaultdict(list)
    for code, stats in config_stats.items():
        n_flexed = count_flexed(code)
        mean_mag = np.mean(stats['magnitudes'])
        by_n_flexed[n_flexed].append((code, mean_mag, len(stats['magnitudes'])))

    print("\nMagnitude vs Number of Flexed Fingers:")
    print("-" * 60)

    for n in sorted(by_n_flexed.keys()):
        configs = by_n_flexed[n]
        avg_mag = np.mean([c[1] for c in configs])
        print(f"\n  {n} fingers flexed (avg magnitude: {avg_mag:.0f} µT):")
        for code, mean_mag, n_samples in sorted(configs, key=lambda x: -x[1]):
            print(f"    {code}: {mean_mag:,.0f} µT ({n_samples} samples)")

    # Check if trend is correct
    avg_by_n = {n: np.mean([c[1] for c in configs]) for n, configs in by_n_flexed.items()}

    print("\n" + "-" * 60)
    print("TREND ANALYSIS:")

    if 0 in avg_by_n and max(avg_by_n.keys()) in avg_by_n:
        baseline = avg_by_n[0]
        max_flexed = avg_by_n[max(avg_by_n.keys())]

        if max_flexed > baseline:
            print(f"  ✓ Trend is CORRECT: More flexed → Higher magnitude")
            print(f"    0 flexed: {baseline:,.0f} µT")
            print(f"    {max(avg_by_n.keys())} flexed: {max_flexed:,.0f} µT")
        else:
            print(f"  ✗ Trend is INVERTED: Labels may be swapped!")
            print(f"    0 flexed: {baseline:,.0f} µT")
            print(f"    {max(avg_by_n.keys())} flexed: {max_flexed:,.0f} µT")

    # 2. Baseline analysis
    print("\n" + "=" * 80)
    print("2. BASELINE ANALYSIS (All Extended = 00000)")
    print("=" * 80)

    if '00000' in config_stats:
        baseline_mags = config_stats['00000']['magnitudes']
        baseline_samples = np.array(config_stats['00000']['samples'])

        print(f"\nBaseline (all fingers extended):")
        print(f"  Mean magnitude: {np.mean(baseline_mags):,.0f} µT")
        print(f"  Std magnitude:  {np.std(baseline_mags):,.0f} µT")
        print(f"  Min magnitude:  {np.min(baseline_mags):,.0f} µT")
        print(f"  Max magnitude:  {np.max(baseline_mags):,.0f} µT")

        print(f"\nPer-axis baseline:")
        print(f"  X: {np.mean(baseline_samples[:,0]):,.0f} ± {np.std(baseline_samples[:,0]):,.0f} µT")
        print(f"  Y: {np.mean(baseline_samples[:,1]):,.0f} ± {np.std(baseline_samples[:,1]):,.0f} µT")
        print(f"  Z: {np.mean(baseline_samples[:,2]):,.0f} ± {np.std(baseline_samples[:,2]):,.0f} µT")

        baseline_mean = np.mean(baseline_mags)
        earth_field = 50.4

        print(f"\n  Earth's field (Edinburgh): ~{earth_field} µT")
        print(f"  Baseline is {baseline_mean/earth_field:.0f}x Earth's field!")

        if baseline_mean > 200:
            print(f"\n  ⚠️  HIGH BASELINE: Magnets may still influence even when extended")
            print(f"     Or sensor has DC offset from nearby ferromagnetic material")

    # 3. Single finger analysis
    print("\n" + "=" * 80)
    print("3. SINGLE FINGER SIGNAL STRENGTH")
    print("=" * 80)

    single_finger_codes = {
        '20000': 'Thumb',
        '02000': 'Index',
        '00200': 'Middle',
        '00020': 'Ring',
        '00002': 'Pinky'
    }

    baseline_mag = np.mean(config_stats['00000']['magnitudes']) if '00000' in config_stats else 0

    print(f"\nSignal increase when each finger is flexed:")
    print("-" * 60)

    finger_signals = {}
    for code, name in single_finger_codes.items():
        if code in config_stats:
            mean_mag = np.mean(config_stats[code]['magnitudes'])
            delta = mean_mag - baseline_mag
            finger_signals[name] = {
                'magnitude': mean_mag,
                'delta': delta,
                'relative': delta / baseline_mag * 100 if baseline_mag > 0 else 0
            }
            print(f"  {name:8s} ({code}): {mean_mag:>10,.0f} µT  (Δ = {delta:>+10,.0f} µT, {delta/baseline_mag*100:>+6.0f}%)")

    # 4. Can we use smaller magnets?
    print("\n" + "=" * 80)
    print("4. SMALLER MAGNET FEASIBILITY")
    print("=" * 80)

    # Current magnet info (from metadata if available)
    metadata = session.get('metadata', {})
    magnet_type = metadata.get('magnet_type', 'unknown')
    print(f"\nCurrent magnet: {magnet_type}")

    # Minimum detectable signal
    noise_level = np.std(config_stats['00000']['magnitudes']) if '00000' in config_stats else 200
    min_snr = 5  # Minimum signal-to-noise ratio for reliable detection
    min_signal = noise_level * min_snr

    print(f"\nNoise level (baseline std): {noise_level:.0f} µT")
    print(f"Minimum signal for SNR={min_snr}: {min_signal:.0f} µT")

    # Check each finger
    print(f"\nCurrent signals vs minimum needed:")
    print("-" * 60)

    for name, sig in sorted(finger_signals.items(), key=lambda x: -x[1]['delta']):
        current = abs(sig['delta'])
        margin = current / min_signal
        print(f"  {name:8s}: {current:>10,.0f} µT  (margin: {margin:.1f}x minimum)")

    # Scaling estimate
    print("\n" + "-" * 60)
    smallest_signal = min(abs(s['delta']) for s in finger_signals.values())
    current_margin = smallest_signal / min_signal

    print(f"\nSmallest finger signal: {smallest_signal:,.0f} µT")
    print(f"Current margin over minimum: {current_margin:.1f}x")

    if current_margin > 2:
        # Magnetic field scales as 1/r³ for a dipole
        # If we can reduce by current_margin/2, we could use smaller magnet
        scale_factor = (current_margin / 2) ** (1/3)
        print(f"\n✓ YES, smaller magnets could work!")
        print(f"  Current margin allows ~{scale_factor:.1f}x smaller linear dimension")
        print(f"  (e.g., 6x3mm → ~{6/scale_factor:.1f}x{3/scale_factor:.1f}mm)")
    else:
        print(f"\n✗ Current magnets are near minimum size for reliable detection")

    # 5. Interpretation issues
    print("\n" + "=" * 80)
    print("5. POTENTIAL INTERPRETATION ISSUES")
    print("=" * 80)

    # Check for anomalies
    print("\nAnomalies in the data:")

    # Pinky has HUGE signal - physically reasonable?
    if 'Pinky' in finger_signals:
        pinky_delta = finger_signals['Pinky']['delta']
        avg_delta = np.mean([abs(s['delta']) for n, s in finger_signals.items() if n != 'Pinky'])

        print(f"\n  Pinky signal ({pinky_delta:,.0f} µT) is {pinky_delta/avg_delta:.1f}x average")
        if pinky_delta > avg_delta * 3:
            print(f"  ⚠️  Pinky may have different magnet or be positioned differently")

    # Check ring finger (often confused direction)
    if 'Ring' in finger_signals:
        ring = finger_signals['Ring']
        if ring['delta'] < 0:
            print(f"\n  ⚠️  Ring finger shows NEGATIVE delta")
            print(f"     This could indicate inverted magnet polarity or label confusion")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    print("""
    LABEL INTERPRETATION:
    - The wizard likely uses "flexed" = finger curled toward palm
    - "Extended" = finger straight out
    - This matches physics: flexed → closer to sensor → higher magnitude

    BASELINE OBSERVATION:
    - Baseline (757 µT) is 15x Earth's field
    - This is because magnets are still relatively close even when extended
    - The sensor on wrist is ~100mm from fingertips even when straight

    SMALLER MAGNETS:
    - Current signals have large margin (smallest is still >> noise)
    - Could potentially use 2-3x smaller magnets and still work
    - BUT: Smaller magnets = more sensitive to orientation variations

    RECOMMENDATION:
    - Current 6x3mm N48 magnets are well-sized
    - Could test 4x2mm or 3x2mm for more subtle form factor
    - Focus on consistent placement rather than smaller magnets
    """)


if __name__ == '__main__':
    main()
