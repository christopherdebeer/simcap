#!/usr/bin/env python3
"""
Deep investigation of calibration issues discovered in 2025-12-12 session data.
"""

import json
import numpy as np
from pathlib import Path

def load_session(filepath):
    """Load a session file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_calibration_parameters(sessions_data):
    """Compare calibration parameters across sessions."""
    print("=" * 80)
    print("CALIBRATION PARAMETERS COMPARISON")
    print("=" * 80)
    print()

    for name, session in sessions_data.items():
        cal = session.get('metadata', {}).get('calibration')
        if not cal:
            print(f"{name}: No calibration metadata")
            continue

        print(f"\n{name}:")
        print(f"  Hard Iron Offset: {cal['hardIronOffset']}")

        # Calculate hard iron offset magnitude
        offset = cal['hardIronOffset']
        offset_mag = np.sqrt(offset['x']**2 + offset['y']**2 + offset['z']**2)
        print(f"  Hard Iron Offset Magnitude: {offset_mag:.2f} µT")

        print(f"  Earth Field: {cal['earthField']}")
        print(f"  Earth Field Magnitude: {cal['earthFieldMagnitude']:.2f} µT")

        # Calculate actual earth field from components
        ef = cal['earthField']
        ef_calculated = np.sqrt(ef['x']**2 + ef['y']**2 + ef['z']**2)
        print(f"  Earth Field (calculated): {ef_calculated:.2f} µT")

        # Check if this looks reasonable
        if cal['earthFieldMagnitude'] > 100:
            print(f"  ⚠️  WARNING: Earth field magnitude is too high!")
            print(f"      Expected: 25-65 µT (typical Earth's magnetic field)")
            print(f"      Got: {cal['earthFieldMagnitude']:.2f} µT")
            print(f"      This suggests calibration captured environmental distortions, not just Earth's field")

def analyze_earth_field_calibration_data(sessions_data):
    """Analyze the raw data during Earth field calibration steps."""
    print("\n" + "=" * 80)
    print("EARTH FIELD CALIBRATION DATA ANALYSIS")
    print("=" * 80)
    print()

    for name, session in sessions_data.items():
        labels = session.get('labels', [])
        samples = session.get('samples', [])

        for label in labels:
            if label.get('metadata', {}).get('calibration_step') == 'EARTH_FIELD':
                start = label['start_sample']
                end = label['end_sample']

                print(f"\n{name} - Segment [{start}:{end}]:")
                print(f"  Calibration Step: EARTH_FIELD")
                print(f"  Quality: {label['metadata'].get('quality', 'N/A')}")

                # Extract magnetometer data for this segment
                seg_samples = samples[start:end]
                # Use converted µT values if available, otherwise use raw LSB
                if 'mx_ut' in seg_samples[0]:
                    mx = np.array([s['mx_ut'] for s in seg_samples])
                    my = np.array([s['my_ut'] for s in seg_samples])
                    mz = np.array([s['mz_ut'] for s in seg_samples])
                else:
                    mx = np.array([s['mx'] for s in seg_samples])
                    my = np.array([s['my'] for s in seg_samples])
                    mz = np.array([s['mz'] for s in seg_samples])

                # Calculate statistics
                print(f"  Raw Magnetometer Statistics:")
                print(f"    MX: mean={np.mean(mx):.2f}, std={np.std(mx):.2f} µT")
                print(f"    MY: mean={np.mean(my):.2f}, std={np.std(my):.2f} µT")
                print(f"    MZ: mean={np.mean(mz):.2f}, std={np.std(mz):.2f} µT")

                mag = np.sqrt(mx**2 + my**2 + mz**2)
                print(f"    Magnitude: mean={np.mean(mag):.2f}, std={np.std(mag):.2f} µT")

                # Check for stability (should be very stable during Earth field calibration)
                if np.std(mag) > 20:
                    print(f"  ⚠️  WARNING: High variance in magnitude (std={np.std(mag):.2f} µT)")
                    print(f"      Device may have been moving during calibration")

                # Check for reasonableness
                mean_mag = np.mean(mag)
                if mean_mag > 200:
                    print(f"  ⚠️  WARNING: Magnitude too high ({mean_mag:.2f} µT)")
                    print(f"      Expected: ~50 µT (Earth's field only)")
                    print(f"      Actual: {mean_mag:.2f} µT")
                    print(f"      Excess: ~{mean_mag - 50:.2f} µT")
                    print(f"      This indicates strong environmental magnetic distortion")

def analyze_hard_iron_calibration_data(sessions_data):
    """Analyze the raw data during Hard iron calibration steps."""
    print("\n" + "=" * 80)
    print("HARD IRON CALIBRATION DATA ANALYSIS")
    print("=" * 80)
    print()

    for name, session in sessions_data.items():
        labels = session.get('labels', [])
        samples = session.get('samples', [])

        for label in labels:
            if label.get('metadata', {}).get('calibration_step') == 'HARD_IRON':
                start = label['start_sample']
                end = label['end_sample']

                print(f"\n{name} - Segment [{start}:{end}]:")
                print(f"  Calibration Step: HARD_IRON")
                print(f"  Quality: {label['metadata'].get('quality', 'N/A')}")

                # Extract magnetometer data for this segment
                seg_samples = samples[start:end]
                # Use converted µT values if available, otherwise use raw LSB
                if 'mx_ut' in seg_samples[0]:
                    mx = np.array([s['mx_ut'] for s in seg_samples])
                    my = np.array([s['my_ut'] for s in seg_samples])
                    mz = np.array([s['mz_ut'] for s in seg_samples])
                else:
                    mx = np.array([s['mx'] for s in seg_samples])
                    my = np.array([s['my'] for s in seg_samples])
                    mz = np.array([s['mz'] for s in seg_samples])

                # For hard iron calibration, we expect the device to be rotated
                # Check the range of values
                print(f"  Raw Magnetometer Range:")
                print(f"    MX: [{np.min(mx):.2f}, {np.max(mx):.2f}] µT (range: {np.max(mx)-np.min(mx):.2f})")
                print(f"    MY: [{np.min(my):.2f}, {np.max(my):.2f}] µT (range: {np.max(my)-np.min(my):.2f})")
                print(f"    MZ: [{np.min(mz):.2f}, {np.max(mz):.2f}] µT (range: {np.max(mz)-np.min(mz):.2f})")

                # Calculate expected hard iron offset (center of min/max)
                offset_x = (np.max(mx) + np.min(mx)) / 2
                offset_y = (np.max(my) + np.min(my)) / 2
                offset_z = (np.max(mz) + np.min(mz)) / 2

                print(f"  Calculated Hard Iron Offset (from data):")
                print(f"    X: {offset_x:.2f} µT")
                print(f"    Y: {offset_y:.2f} µT")
                print(f"    Z: {offset_z:.2f} µT")

                # Compare with metadata
                cal = session.get('metadata', {}).get('calibration', {})
                if cal:
                    meta_offset = cal.get('hardIronOffset', {})
                    print(f"  Metadata Hard Iron Offset:")
                    print(f"    X: {meta_offset.get('x', 'N/A')} µT")
                    print(f"    Y: {meta_offset.get('y', 'N/A')} µT")
                    print(f"    Z: {meta_offset.get('z', 'N/A')} µT")

                    # Check for discrepancy
                    if meta_offset:
                        diff_x = abs(offset_x - meta_offset['x'])
                        diff_y = abs(offset_y - meta_offset['y'])
                        diff_z = abs(offset_z - meta_offset['z'])
                        print(f"  Discrepancy:")
                        print(f"    X: {diff_x:.2f} µT")
                        print(f"    Y: {diff_y:.2f} µT")
                        print(f"    Z: {diff_z:.2f} µT")

                        if max(diff_x, diff_y, diff_z) > 50:
                            print(f"  ⚠️  WARNING: Large discrepancy between calculated and metadata offsets")

def check_session_1_mystery(session1_data):
    """Investigate why session 1 has calibrated fields but others don't."""
    print("\n" + "=" * 80)
    print("SESSION 1 MYSTERY: Why does it have calibrated fields?")
    print("=" * 80)
    print()

    # Check if there are additional fields that might indicate real-time processing
    samples = session1_data.get('samples', [])
    if not samples:
        print("No samples found")
        return

    first_sample = samples[0]

    print("Fields present in Session 1 but not in others:")
    session1_fields = set(first_sample.keys())
    print(f"  Total fields: {len(session1_fields)}")

    # Key fields to check
    calibration_related = ['calibrated_mx', 'calibrated_my', 'calibrated_mz',
                          'fused_mx', 'fused_my', 'fused_mz',
                          'filtered_mx', 'filtered_my', 'filtered_mz',
                          'orientation_w', 'orientation_x', 'orientation_y', 'orientation_z',
                          'euler_roll', 'euler_pitch', 'euler_yaw',
                          'residual_magnitude', 'isMoving', 'gyroBiasCalibrated']

    print("\nCalibration-related fields present:")
    for field in calibration_related:
        if field in session1_fields:
            print(f"  ✓ {field}")

    # Sample some values to see if they're meaningful
    print("\nSample values from first data point:")
    print(f"  Raw: mx={first_sample['mx']}, my={first_sample['my']}, mz={first_sample['mz']}")
    print(f"  Calibrated: mx={first_sample.get('calibrated_mx')}, my={first_sample.get('calibrated_my')}, mz={first_sample.get('calibrated_mz')}")
    print(f"  Fused: mx={first_sample.get('fused_mx')}, my={first_sample.get('fused_my')}, mz={first_sample.get('fused_mz')}")
    print(f"  Filtered: mx={first_sample.get('filtered_mx')}, my={first_sample.get('filtered_my')}, mz={first_sample.get('filtered_mz')}")

    # Check if fused values are reasonable (should be near zero without magnets)
    fused_mx = np.array([s.get('fused_mx', 0) for s in samples])
    fused_my = np.array([s.get('fused_my', 0) for s in samples])
    fused_mz = np.array([s.get('fused_mz', 0) for s in samples])
    fused_mag = np.sqrt(fused_mx**2 + fused_my**2 + fused_mz**2)

    print(f"\nFused field statistics (should be near zero without magnets):")
    print(f"  Mean magnitude: {np.mean(fused_mag):.2f} µT")
    print(f"  Std magnitude: {np.std(fused_mag):.2f} µT")
    print(f"  Max magnitude: {np.max(fused_mag):.2f} µT")

    if np.mean(fused_mag) < 20:
        print(f"  ✓ Fused field looks reasonable (low residual)")
    else:
        print(f"  ⚠️  Fused field is high - calibration may not be working correctly")

def main():
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    # Load all sessions
    session_files = sorted(data_dir.glob('2025-12-12T*.json'))
    sessions_data = {}

    for session_file in session_files:
        name = session_file.stem
        sessions_data[name] = load_session(session_file)

    print("\n" + "=" * 80)
    print("GAMBIT CALIBRATION INVESTIGATION")
    print("Deep dive into calibration issues")
    print("=" * 80)

    # Run analyses
    analyze_calibration_parameters(sessions_data)
    analyze_earth_field_calibration_data(sessions_data)
    analyze_hard_iron_calibration_data(sessions_data)

    # Special investigation of session 1
    if '2025-12-12T11_14_50.144Z' in sessions_data:
        check_session_1_mystery(sessions_data['2025-12-12T11_14_50.144Z'])

    print("\n" + "=" * 80)
    print("KEY FINDINGS SUMMARY")
    print("=" * 80)
    print()
    print("1. INCONSISTENT CALIBRATION APPLICATION:")
    print("   - Session 1 has all calibrated fields (calibrated, fused, orientation)")
    print("   - Sessions 2-4 are missing all calibrated fields")
    print("   - This confirms previous finding: real-time calibration not consistently applied")
    print()
    print("2. INCORRECT EARTH FIELD CALIBRATION:")
    print("   - Earth field magnitudes are 1200-1600 µT (way too high!)")
    print("   - Expected: 25-65 µT for Earth's magnetic field")
    print("   - Issue: Calibration is capturing environmental distortions + Earth field")
    print("   - This explains poor SNR and high noise in previous analysis")
    print()
    print("3. VARYING HARD IRON OFFSETS:")
    print("   - Hard iron offset changes dramatically between sessions")
    print("   - Session 1: {x: -51.5, y: 504, z: -436}")
    print("   - Session 4: {x: 408.5, y: -402, z: -885}")
    print("   - Suggests: Device moved, or calibration environment changed")
    print()
    print("4. RECOMMENDATIONS:")
    print("   a. Fix real-time calibration pipeline to consistently apply corrections")
    print("   b. Recalibrate in magnetically clean environment:")
    print("      - Remove all ferromagnetic materials (watches, rings, metal furniture)")
    print("      - Stay away from electronics, motors, power cables")
    print("      - Verify Earth field magnitude is 25-65 µT after calibration")
    print("   c. Add validation checks to reject bad calibrations")
    print("   d. Consider implementing gyro-only orientation (no accel feedback during motion)")
    print()

if __name__ == '__main__':
    main()
