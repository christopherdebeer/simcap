#!/usr/bin/env python3
"""
Analyze GAMBIT calibration session data collected 2025-12-12.
This script investigates calibration quality and data characteristics.
"""

import json
import numpy as np
from pathlib import Path
import sys

def load_session(filepath):
    """Load a session file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def analyze_fields(session):
    """Check what fields are present in the session data."""
    if not session['samples']:
        return set()

    first_sample = session['samples'][0]
    return set(first_sample.keys())

def analyze_magnetometer_data(session):
    """Analyze raw magnetometer data statistics."""
    samples = session['samples']

    mx = np.array([s['mx'] for s in samples])
    my = np.array([s['my'] for s in samples])
    mz = np.array([s['mz'] for s in samples])

    magnitude = np.sqrt(mx**2 + my**2 + mz**2)

    return {
        'mx': {'mean': np.mean(mx), 'std': np.std(mx), 'min': np.min(mx), 'max': np.max(mx)},
        'my': {'mean': np.mean(my), 'std': np.std(my), 'min': np.min(my), 'max': np.max(my)},
        'mz': {'mean': np.mean(mz), 'std': np.std(mz), 'min': np.min(mz), 'max': np.max(mz)},
        'magnitude': {'mean': np.mean(magnitude), 'std': np.std(magnitude), 'min': np.min(magnitude), 'max': np.max(magnitude)},
        'sample_count': len(samples)
    }

def analyze_calibration_metadata(session):
    """Extract calibration metadata if present."""
    metadata = session.get('metadata', {})
    calibration = metadata.get('calibration', None)

    if calibration:
        return {
            'hard_iron_offset': calibration.get('hardIronOffset'),
            'earth_field': calibration.get('earthField'),
            'earth_field_magnitude': calibration.get('earthFieldMagnitude'),
            'hard_iron_calibrated': calibration.get('hardIronCalibrated'),
            'soft_iron_calibrated': calibration.get('softIronCalibrated'),
            'earth_field_calibrated': calibration.get('earthFieldCalibrated')
        }
    return None

def analyze_label_segments(session):
    """Analyze label segments and calibration steps."""
    labels = session.get('labels', [])

    segments = []
    for label in labels:
        segment_info = {
            'start': label['start_sample'],
            'end': label['end_sample'],
            'count': label['end_sample'] - label['start_sample'],
            'calibration_step': label.get('metadata', {}).get('calibration_step'),
            'quality': label.get('metadata', {}).get('quality'),
            'result_summary': label.get('metadata', {}).get('result_summary')
        }
        segments.append(segment_info)

    return segments

def check_for_calibration_application(fields):
    """Check if calibration has been applied to the data."""
    expected_calibrated_fields = ['calibrated_mx', 'calibrated_my', 'calibrated_mz']
    expected_fused_fields = ['fused_mx', 'fused_my', 'fused_mz']
    expected_orientation_fields = ['orientation_w', 'orientation_x', 'orientation_y', 'orientation_z']

    has_calibrated = all(f in fields for f in expected_calibrated_fields)
    has_fused = all(f in fields for f in expected_fused_fields)
    has_orientation = all(f in fields for f in expected_orientation_fields)

    return {
        'has_calibrated_fields': has_calibrated,
        'has_fused_fields': has_fused,
        'has_orientation_fields': has_orientation,
        'missing_calibrated': [f for f in expected_calibrated_fields if f not in fields],
        'missing_fused': [f for f in expected_fused_fields if f not in fields],
        'missing_orientation': [f for f in expected_orientation_fields if f not in fields]
    }

def main():
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    # Find all 2025-12-12 session files
    session_files = sorted(data_dir.glob('2025-12-12T*.json'))

    if not session_files:
        print("No session files found!")
        return

    print("=" * 80)
    print("GAMBIT CALIBRATION SESSION ANALYSIS - 2025-12-12")
    print("=" * 80)
    print()

    for session_file in session_files:
        print(f"\n{'=' * 80}")
        print(f"Session: {session_file.name}")
        print(f"{'=' * 80}\n")

        session = load_session(session_file)

        # 1. Field Analysis
        print("1. FIELD ANALYSIS")
        print("-" * 80)
        fields = analyze_fields(session)
        print(f"Available fields: {sorted(fields)}")
        print()

        calibration_check = check_for_calibration_application(fields)
        print("Calibration Application Status:")
        print(f"  ✓ Has calibrated fields (calibrated_mx/my/mz): {calibration_check['has_calibrated_fields']}")
        print(f"  ✓ Has fused fields (fused_mx/my/mz): {calibration_check['has_fused_fields']}")
        print(f"  ✓ Has orientation fields (orientation_w/x/y/z): {calibration_check['has_orientation_fields']}")

        if not calibration_check['has_calibrated_fields']:
            print(f"  ⚠ Missing: {calibration_check['missing_calibrated']}")
        if not calibration_check['has_fused_fields']:
            print(f"  ⚠ Missing: {calibration_check['missing_fused']}")
        if not calibration_check['has_orientation_fields']:
            print(f"  ⚠ Missing: {calibration_check['missing_orientation']}")
        print()

        # 2. Magnetometer Statistics
        print("2. RAW MAGNETOMETER STATISTICS")
        print("-" * 80)
        mag_stats = analyze_magnetometer_data(session)
        print(f"Sample count: {mag_stats['sample_count']}")
        print()
        for axis in ['mx', 'my', 'mz']:
            stats = mag_stats[axis]
            print(f"{axis.upper()}:")
            print(f"  Mean: {stats['mean']:8.2f} µT")
            print(f"  Std:  {stats['std']:8.2f} µT")
            print(f"  Range: [{stats['min']:8.2f}, {stats['max']:8.2f}] µT")
        print()
        mag_mag = mag_stats['magnitude']
        print(f"MAGNITUDE:")
        print(f"  Mean: {mag_mag['mean']:8.2f} µT")
        print(f"  Std:  {mag_mag['std']:8.2f} µT")
        print(f"  Range: [{mag_mag['min']:8.2f}, {mag_mag['max']:8.2f}] µT")
        print()

        # 3. Calibration Metadata
        print("3. CALIBRATION METADATA")
        print("-" * 80)
        cal_meta = analyze_calibration_metadata(session)
        if cal_meta:
            print("Calibration data embedded in metadata:")
            print(f"  Hard Iron Offset: {cal_meta['hard_iron_offset']}")
            print(f"  Earth Field: {cal_meta['earth_field']}")
            print(f"  Earth Field Magnitude: {cal_meta['earth_field_magnitude']:.2f} µT")
            print()
            print("Calibration Status Flags:")
            print(f"  Hard Iron Calibrated: {cal_meta['hard_iron_calibrated']}")
            print(f"  Soft Iron Calibrated: {cal_meta['soft_iron_calibrated']}")
            print(f"  Earth Field Calibrated: {cal_meta['earth_field_calibrated']}")
        else:
            print("No calibration metadata found in session.")
        print()

        # 4. Label Segments
        print("4. LABEL SEGMENTS")
        print("-" * 80)
        segments = analyze_label_segments(session)
        for i, seg in enumerate(segments):
            print(f"Segment {i+1}:")
            print(f"  Samples: {seg['start']} - {seg['end']} (count: {seg['count']})")
            print(f"  Calibration Step: {seg['calibration_step']}")
            if seg['quality'] is not None:
                print(f"  Quality: {seg['quality']:.3f}")
            if seg['result_summary']:
                print(f"  Result Summary: {seg['result_summary']}")
            print()

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
