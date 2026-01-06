#!/usr/bin/env python3
"""
Diagnose Live Calibration Confidence Issues

This script analyzes why live calibration confidence stays low (~30%) even with
full device motion. It checks magnetometer readings against expected values.

Root Cause Identified (2025-12-15):
The code was using the LIS3MDL conversion factor (0.014616 µT/LSB) but the device
is a Puck.js v2.1a with MMC5603NJ magnetometer which has a different sensitivity:
- LIS3MDL (Puck.js v2): 6842 LSB/Gauss → 0.014616 µT/LSB
- MMC5603NJ (Puck.js v2.1a): 1024 LSB/Gauss → 0.09765625 µT/LSB

The MMC5603NJ has ~6.7x LOWER sensitivity, so raw LSB values are ~6.7x smaller
for the same magnetic field. With the wrong conversion factor, 485 LSB was
incorrectly converted to 7 µT instead of the correct 47 µT.

FIX: Updated sensor_units.py to use MMC5603NJ conversion factor as default.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from ml.sensor_units import MAG_SPEC, mag_lsb_to_microtesla


def analyze_session(session_path: Path) -> dict:
    """Analyze a single session for magnetometer health."""
    with open(session_path) as f:
        data = json.load(f)
    
    samples = data.get('samples', data) if isinstance(data, dict) else data
    
    # Get samples with magnetometer data
    valid_samples = [s for s in samples if 'mx' in s]
    if not valid_samples:
        return None
    
    # Calculate raw magnitude
    mx_raw = np.array([s['mx'] for s in valid_samples])
    my_raw = np.array([s['my'] for s in valid_samples])
    mz_raw = np.array([s['mz'] for s in valid_samples])
    raw_mag = np.sqrt(mx_raw**2 + my_raw**2 + mz_raw**2)
    
    # Calculate µT magnitude if available
    ut_mag = 0
    if 'mx_ut' in valid_samples[0]:
        mx_ut = np.array([s['mx_ut'] for s in valid_samples])
        my_ut = np.array([s['my_ut'] for s in valid_samples])
        mz_ut = np.array([s['mz_ut'] for s in valid_samples])
        ut_mag = np.sqrt(mx_ut**2 + my_ut**2 + mz_ut**2).mean()
    
    return {
        'filename': session_path.name,
        'timestamp': session_path.stem,
        'sample_count': len(valid_samples),
        'raw_mag_mean': raw_mag.mean(),
        'raw_mag_std': raw_mag.std(),
        'ut_mag_mean': ut_mag,
        'expected_raw_for_earth': int(45 / MAG_SPEC['conversion_factor']),  # ~461 for MMC5603NJ
        'expected_ut': 45,
        'raw_ratio': raw_mag.mean() / int(45 / MAG_SPEC['conversion_factor']),
        'ut_ratio': ut_mag / 45 if ut_mag > 0 else 0
    }


def simulate_confidence(earth_field_magnitude: float, sample_count: int = 500) -> dict:
    """Simulate the JavaScript confidence calculation."""
    # Hard iron confidence (assume good sphericity and coverage)
    sphericity = 0.8
    coverage = 0.75
    sample_factor_hi = min(1, sample_count / 500)
    hard_iron_confidence = sphericity * coverage * sample_factor_hi
    
    # Earth field confidence
    sample_factor_ef = min(1, sample_count / 200)
    
    # Magnitude sanity check (from incremental-calibration.js)
    mag = earth_field_magnitude
    if 25 <= mag <= 65:
        magnitude_sanity = 1.0
    elif 15 <= mag < 25:
        magnitude_sanity = (mag - 15) / 10
    elif 65 < mag <= 80:
        magnitude_sanity = 1 - (mag - 65) / 15
    else:
        magnitude_sanity = 0.0
    
    stability = 0.7  # Assume moderate stability
    earth_field_confidence = sample_factor_ef * max(0.3, magnitude_sanity) * max(0.5, stability)
    
    overall = min(hard_iron_confidence, earth_field_confidence)
    
    return {
        'hard_iron': hard_iron_confidence,
        'earth_field': earth_field_confidence,
        'magnitude_sanity': magnitude_sanity,
        'overall': overall
    }


def main():
    data_dir = Path('data/GAMBIT')
    sessions = sorted(data_dir.glob('*.json'))
    sessions = [s for s in sessions if not s.name.startswith('gambit') and 'manifest' not in s.name]
    
    print("=" * 80)
    print("LIVE CALIBRATION CONFIDENCE DIAGNOSTIC")
    print("=" * 80)
    print()
    print("This script diagnoses why live calibration confidence stays low (~30%)")
    print("even with full device motion.")
    print()
    
    # Analyze all sessions
    results = []
    for session_path in sessions:
        result = analyze_session(session_path)
        if result:
            results.append(result)
    
    if not results:
        print("No valid sessions found!")
        return
    
    # Find the transition point
    print("MAGNETOMETER READINGS OVER TIME:")
    print("-" * 80)
    print(f"{'Timestamp':<25} | {'Raw (LSB)':<12} | {'µT':<8} | {'% of Expected':<15} | Status")
    print("-" * 80)
    
    transition_found = False
    for r in results:
        pct = r['raw_ratio'] * 100
        if pct >= 80:
            status = "✓ GOOD"
        elif pct >= 50:
            status = "⚠ LOW"
        else:
            status = "✗ CRITICAL"
            if not transition_found and r['raw_mag_mean'] < 1000:
                transition_found = True
                print("-" * 80)
                print(">>> TRANSITION POINT - Magnetometer readings dropped significantly <<<")
                print("-" * 80)
        
        print(f"{r['timestamp'][:25]:<25} | {r['raw_mag_mean']:>10.1f} | {r['ut_mag_mean']:>6.1f} | {pct:>13.1f}% | {status}")
    
    print("-" * 80)
    print()
    
    # Analyze recent sessions
    recent = results[-5:]
    avg_ut = np.mean([r['ut_mag_mean'] for r in recent])
    
    print("RECENT SESSION ANALYSIS (last 5 sessions):")
    print(f"  Average magnetometer magnitude: {avg_ut:.1f} µT")
    print(f"  Expected for Earth field: 25-65 µT (typically ~45 µT)")
    print()
    
    # Simulate confidence
    conf = simulate_confidence(avg_ut)
    print("SIMULATED CONFIDENCE CALCULATION:")
    print(f"  Earth field magnitude: {avg_ut:.1f} µT")
    print(f"  Magnitude sanity check: {conf['magnitude_sanity']:.2f}")
    print(f"  Hard iron confidence: {conf['hard_iron']:.2f} ({conf['hard_iron']*100:.0f}%)")
    print(f"  Earth field confidence: {conf['earth_field']:.2f} ({conf['earth_field']*100:.0f}%)")
    print(f"  Overall confidence: {conf['overall']:.2f} ({conf['overall']*100:.0f}%)")
    print()
    
    # Root cause analysis
    print("=" * 80)
    print("ROOT CAUSE ANALYSIS")
    print("=" * 80)
    print()
    
    if avg_ut < 15:
        print("✗ CRITICAL: Magnetometer readings are significantly below expected values!")
        print()
        print("  The raw magnetometer magnitude is ~{:.0f}% of expected.".format(avg_ut / 45 * 100))
        print("  This causes the 'magnitudeSanity' check in the confidence calculation")
        print("  to fail, capping confidence at ~21%.")
        print()
        print("POSSIBLE CAUSES:")
        print("  1. Device is in a magnetically shielded location")
        print("  2. Magnetometer sensor issue (hardware failure)")
        print("  3. Firmware configuration changed (wrong sensitivity setting)")
        print("  4. Strong magnetic interference canceling Earth's field")
        print()
        print("RECOMMENDED ACTIONS:")
        print("  1. Move device to a different location (away from metal/electronics)")
        print("  2. Check if magnetometer readings increase in open outdoor area")
        print("  3. Verify firmware magnetometer configuration")
        print("  4. If issue persists, check hardware (sensor may be damaged)")
    elif avg_ut < 25:
        print("⚠ WARNING: Magnetometer readings are below optimal range.")
        print()
        print("  Consider moving to a location with less magnetic shielding.")
    else:
        print("✓ Magnetometer readings are within expected range.")
        print()
        print("  If confidence is still low, check:")
        print("  - Orientation coverage (rotate device in all directions)")
        print("  - Sample count (need 200+ samples for good confidence)")
    
    print()
    print("=" * 80)


if __name__ == '__main__':
    main()
