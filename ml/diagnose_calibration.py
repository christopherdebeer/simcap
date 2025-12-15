#!/usr/bin/env python3
"""
Diagnose Calibration Pipeline Issues

Traces through exactly what happens at each stage of the calibration
to identify where the high residual values come from.
"""

import json
import numpy as np
from pathlib import Path
from sensor_units import MAG_SPEC


def load_calibration(cal_path: Path) -> dict:
    """Load calibration file."""
    with open(cal_path, 'r') as f:
        return json.load(f)


def load_session_sample(json_path: Path, sample_idx: int = 100) -> dict:
    """Load a single sample from a session."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    if isinstance(data, list):
        samples = data
    else:
        samples = data.get('samples', [])

    if sample_idx < len(samples):
        return samples[sample_idx]
    return samples[0] if samples else {}


def diagnose_single_sample(sample: dict, cal: dict):
    """Trace through calibration for a single sample."""

    print("=" * 80)
    print("CALIBRATION PIPELINE DIAGNOSIS")
    print("=" * 80)

    # Step 0: Raw values
    print("\n1. RAW SENSOR VALUES (LSB):")
    mx_raw = sample.get('mx', 0)
    my_raw = sample.get('my', 0)
    mz_raw = sample.get('mz', 0)
    raw_mag = np.sqrt(mx_raw**2 + my_raw**2 + mz_raw**2)
    print(f"   mx={mx_raw}, my={my_raw}, mz={mz_raw}")
    print(f"   magnitude = {raw_mag:.1f} LSB")

    # Step 1: Convert to µT
    print("\n2. CONVERTED TO µT (from file):")
    mx_ut = sample.get('mx_ut', 0)
    my_ut = sample.get('my_ut', 0)
    mz_ut = sample.get('mz_ut', 0)
    ut_mag = np.sqrt(mx_ut**2 + my_ut**2 + mz_ut**2)
    print(f"   mx_ut={mx_ut:.3f}, my_ut={my_ut:.3f}, mz_ut={mz_ut:.3f}")
    print(f"   magnitude = {ut_mag:.2f} µT")

    # Verify conversion factor
    if mx_raw != 0:
        conv_factor = mx_ut / mx_raw
        print(f"   Implied conversion factor: {conv_factor:.6f}")
        print(f"   (Expected {MAG_SPEC['sensor']} factor: {MAG_SPEC['conversion_factor']:.6f})")

    # Step 2: Hard iron correction
    print("\n3. HARD IRON CORRECTION:")
    hi = cal.get('hardIronOffset', {})
    print(f"   Hard iron offset: x={hi.get('x', 0):.3f}, y={hi.get('y', 0):.3f}, z={hi.get('z', 0):.3f} µT")

    mx_hi = mx_ut - hi.get('x', 0)
    my_hi = my_ut - hi.get('y', 0)
    mz_hi = mz_ut - hi.get('z', 0)
    hi_mag = np.sqrt(mx_hi**2 + my_hi**2 + mz_hi**2)
    print(f"   After hard iron: x={mx_hi:.3f}, y={my_hi:.3f}, z={mz_hi:.3f}")
    print(f"   magnitude = {hi_mag:.2f} µT")

    # Step 3: Soft iron correction
    print("\n4. SOFT IRON CORRECTION:")
    si_flat = cal.get('softIronMatrix', [1, 0, 0, 0, 1, 0, 0, 0, 1])
    si = np.array(si_flat).reshape(3, 3)
    print(f"   Soft iron matrix (diagonal): [{si[0,0]:.3f}, {si[1,1]:.3f}, {si[2,2]:.3f}]")

    vec_hi = np.array([mx_hi, my_hi, mz_hi])
    vec_si = si @ vec_hi
    si_mag = np.linalg.norm(vec_si)
    print(f"   After soft iron: x={vec_si[0]:.3f}, y={vec_si[1]:.3f}, z={vec_si[2]:.3f}")
    print(f"   magnitude = {si_mag:.2f} µT")

    # Compare with stored calibrated values
    print("\n5. COMPARISON WITH STORED VALUES:")
    cal_mx = sample.get('calibrated_mx', None)
    cal_my = sample.get('calibrated_my', None)
    cal_mz = sample.get('calibrated_mz', None)

    if cal_mx is not None:
        cal_mag = np.sqrt(cal_mx**2 + cal_my**2 + cal_mz**2)
        print(f"   Stored calibrated: x={cal_mx:.3f}, y={cal_my:.3f}, z={cal_mz:.3f}")
        print(f"   Stored magnitude = {cal_mag:.2f}")
        print(f"\n   *** Expected (computed): {si_mag:.2f} µT")
        print(f"   *** Stored (in file):    {cal_mag:.2f}")

        if abs(si_mag - cal_mag) > 1:
            print(f"\n   ⚠ MISMATCH! Difference: {abs(si_mag - cal_mag):.2f}")

            # Check if stored value matches raw LSB
            if abs(cal_mag - raw_mag) < 200:
                print(f"   → Stored value is close to RAW LSB value ({raw_mag:.1f})")
                print(f"   → This suggests calibration was applied to RAW values, not µT!")

            # Check if there's a scaling factor
            if si_mag > 0:
                ratio = cal_mag / si_mag
                print(f"   → Ratio: stored/computed = {ratio:.2f}")
    else:
        print("   No calibrated values stored in this sample")

    # Step 4: Earth field subtraction
    print("\n6. EARTH FIELD SUBTRACTION:")
    ef = cal.get('earthField', {})
    ef_mag = cal.get('earthFieldMagnitude', 0)
    print(f"   Earth field: x={ef.get('x', 0):.3f}, y={ef.get('y', 0):.3f}, z={ef.get('z', 0):.3f} µT")
    print(f"   Earth field magnitude: {ef_mag:.3f} µT")
    print(f"   (Typical Earth field: 25-65 µT depending on location)")

    if ef_mag < 20:
        print(f"\n   ⚠ WARNING: Earth field magnitude ({ef_mag:.1f}) is unusually low!")

        # Estimate actual Earth field from raw data
        print(f"\n   Comparing to raw magnetometer magnitude:")
        print(f"   Raw magnitude (µT): {ut_mag:.2f}")
        if ut_mag > 20 and ut_mag < 70:
            print(f"   → Raw magnitude is reasonable for Earth field")
            print(f"   → Calibration earth field value appears incorrect!")

    # Check orientation data
    print("\n7. ORIENTATION DATA:")
    ow = sample.get('orientation_w', None)
    if ow is not None:
        ox = sample.get('orientation_x', 0)
        oy = sample.get('orientation_y', 0)
        oz = sample.get('orientation_z', 0)
        print(f"   Quaternion: w={ow:.4f}, x={ox:.4f}, y={oy:.4f}, z={oz:.4f}")

        euler_r = sample.get('euler_roll', 0)
        euler_p = sample.get('euler_pitch', 0)
        euler_y = sample.get('euler_yaw', 0)
        print(f"   Euler: roll={euler_r:.1f}°, pitch={euler_p:.1f}°, yaw={euler_y:.1f}°")
    else:
        print("   No orientation data in sample")

    # Final residual analysis
    print("\n8. RESIDUAL ANALYSIS:")
    fused_mx = sample.get('fused_mx', None)
    residual_mag = sample.get('residual_magnitude', None)

    if fused_mx is not None:
        fused_my = sample.get('fused_my', 0)
        fused_mz = sample.get('fused_mz', 0)
        print(f"   Fused (stored): x={fused_mx:.3f}, y={fused_my:.3f}, z={fused_mz:.3f}")

    if residual_mag is not None:
        print(f"   Residual magnitude (stored): {residual_mag:.2f}")
        print(f"\n   Expected (no finger magnets): < 5 µT")
        print(f"   Status: {'✓ OK' if residual_mag < 5 else '✗ TOO HIGH'}")

    # Summary
    print("\n" + "=" * 80)
    print("DIAGNOSIS SUMMARY")
    print("=" * 80)

    issues = []

    if ef_mag < 20:
        issues.append("Earth field magnitude in calibration is too low (should be 25-65 µT)")

    if cal_mx is not None:
        cal_mag = np.sqrt(cal_mx**2 + cal_my**2 + cal_mz**2)
        if abs(si_mag - cal_mag) > 10:
            issues.append("Stored calibrated values don't match expected computation")

    if residual_mag is not None and residual_mag > 50:
        issues.append(f"Residual magnitude ({residual_mag:.0f}) is too high (expected <5 µT)")

    if issues:
        print("\nIssues found:")
        for i, issue in enumerate(issues, 1):
            print(f"  {i}. {issue}")
    else:
        print("\nNo obvious issues detected.")

    return {
        'raw_mag': raw_mag,
        'ut_mag': ut_mag,
        'computed_cal_mag': si_mag,
        'stored_cal_mag': cal_mag if cal_mx else None,
        'residual_mag': residual_mag
    }


def main():
    data_dir = Path('data/GAMBIT')
    cal_path = data_dir / 'gambit_calibration.json'

    if not cal_path.exists():
        print(f"Error: Calibration file not found at {cal_path}")
        return

    cal = load_calibration(cal_path)

    # Find a session with calibrated data
    sessions = sorted(data_dir.glob('*.json'))
    sessions = [s for s in sessions if not s.name.startswith('gambit') and 'manifest' not in s.name]

    for session_path in sessions[-3:]:  # Check last 3 sessions
        print(f"\n\n{'#' * 80}")
        print(f"# SESSION: {session_path.name}")
        print(f"{'#' * 80}")

        sample = load_session_sample(session_path, sample_idx=100)
        if sample:
            diagnose_single_sample(sample, cal)
        else:
            print("No samples in this session")


if __name__ == '__main__':
    main()
