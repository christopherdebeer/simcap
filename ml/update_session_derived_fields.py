#!/usr/bin/env python3
"""
Update Session Derived Fields with Corrected Earth Field Algorithm

This script updates all session JSON files with correctly computed derived fields:
- calibrated_mx/my/mz (iron correction only)
- fused_mx/my/mz (iron + correct earth field subtraction)
- filtered_mx/my/mz (simple low-pass filter on fused)

The key fix:
- OLD (buggy): rotatedEarth = R @ earthField
- NEW (correct): rotatedEarth = R.T @ earthField (world→sensor transform)
"""

import json
import numpy as np
from pathlib import Path
import shutil
from datetime import datetime

DATA_DIR = Path('/home/user/simcap/data/GAMBIT')
CALIBRATION_FILE = DATA_DIR / 'gambit_calibration.json'


def quaternion_to_rotation_matrix(w, x, y, z):
    """Convert quaternion to 3x3 rotation matrix (sensor → world)."""
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])


def apply_iron_correction(mx, my, mz, calibration):
    """Apply hard/soft iron calibration."""
    hi = calibration['hardIronOffset']
    si = np.array(calibration['softIronMatrix']).reshape(3, 3)
    corrected = np.array([mx - hi['x'], my - hi['y'], mz - hi['z']])
    return si @ corrected


def correct_earth_subtraction(corrected_mag, qw, qx, qy, qz, earth_field_world):
    """
    CORRECT algorithm: R.T @ earthField (world→sensor).
    Earth field is in world frame, we transform to current sensor frame.
    """
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    ef = np.array([earth_field_world['x'], earth_field_world['y'], earth_field_world['z']])
    rotated_earth = R.T @ ef
    return corrected_mag - rotated_earth


def estimate_world_frame_earth_field(samples, calibration):
    """
    Estimate earth field in world frame from samples with diverse orientations.
    Uses the heading-informed approach: average B_world = R @ B_sensor across orientations.
    """
    world_estimates = []

    for s in samples:
        if s.get('mx') is None or s.get('orientation_w') is None:
            continue

        # Iron-corrected sensor reading
        b_sensor = apply_iron_correction(
            s['mx'], s['my'], s['mz'], calibration
        )

        # Rotation matrix from quaternion
        R = quaternion_to_rotation_matrix(
            s['orientation_w'], s['orientation_x'],
            s['orientation_y'], s['orientation_z']
        )

        # Transform to world frame: B_world = R @ B_sensor
        b_world = R @ b_sensor
        world_estimates.append(b_world)

    if len(world_estimates) < 10:
        return None

    world_estimates = np.array(world_estimates)
    mean_world = np.mean(world_estimates, axis=0)

    return {
        'x': float(mean_world[0]),
        'y': float(mean_world[1]),
        'z': float(mean_world[2])
    }


class SimpleKalmanFilter:
    """Simple 1D Kalman filter for smoothing."""
    def __init__(self, process_noise=0.1, measurement_noise=1.0):
        self.Q = process_noise
        self.R = measurement_noise
        self.x = 0.0
        self.P = 1.0
        self.initialized = False

    def update(self, z):
        if not self.initialized:
            self.x = z
            self.initialized = True
            return self.x

        # Predict
        P_pred = self.P + self.Q

        # Update
        K = P_pred / (P_pred + self.R)
        self.x = self.x + K * (z - self.x)
        self.P = (1 - K) * P_pred

        return self.x


def load_calibration():
    """Load the checked-in calibration file."""
    with open(CALIBRATION_FILE, 'r') as f:
        return json.load(f)


def update_session(session_path, calibration, backup=True):
    """
    Update a session file with corrected derived fields.
    Returns statistics about the update.
    """
    # Load session
    with open(session_path, 'r') as f:
        data = json.load(f)

    # Handle different formats
    if isinstance(data, dict) and 'samples' in data:
        samples = data['samples']
        metadata = data.get('metadata', {})
        is_wrapped = True
    elif isinstance(data, list):
        samples = data
        metadata = {}
        is_wrapped = False
    else:
        return None

    if not samples:
        return None

    # Check if session has zeroed calibration - use checked-in calibration
    session_cal = metadata.get('calibration', {})
    if not session_cal.get('hardIronCalibrated', False):
        use_cal = calibration
        cal_source = 'checked-in'
    else:
        use_cal = session_cal
        cal_source = 'embedded'

    # Check if we have orientation data
    has_orientation = any(s.get('orientation_w') is not None for s in samples)
    if not has_orientation:
        print(f"  No orientation data - skipping")
        return None

    # Estimate world-frame earth field from the session's orientation diversity
    earth_field_world = estimate_world_frame_earth_field(samples, use_cal)
    if earth_field_world is None:
        print(f"  Could not estimate world-frame earth field - skipping")
        return None

    print(f"  Using {cal_source} calibration")
    print(f"  Estimated world-frame earth field: [{earth_field_world['x']:.1f}, {earth_field_world['y']:.1f}, {earth_field_world['z']:.1f}]")

    # Create Kalman filters for each axis
    kf_x = SimpleKalmanFilter(process_noise=0.1, measurement_noise=1.0)
    kf_y = SimpleKalmanFilter(process_noise=0.1, measurement_noise=1.0)
    kf_z = SimpleKalmanFilter(process_noise=0.1, measurement_noise=1.0)

    # Update each sample
    updated_count = 0
    for s in samples:
        if s.get('mx') is None:
            continue

        # Iron correction (calibrated_*)
        iron_corrected = apply_iron_correction(s['mx'], s['my'], s['mz'], use_cal)
        s['calibrated_mx'] = float(iron_corrected[0])
        s['calibrated_my'] = float(iron_corrected[1])
        s['calibrated_mz'] = float(iron_corrected[2])

        # Earth field subtraction (fused_*)
        if s.get('orientation_w') is not None:
            qw, qx, qy, qz = s['orientation_w'], s['orientation_x'], s['orientation_y'], s['orientation_z']
            fused = correct_earth_subtraction(iron_corrected, qw, qx, qy, qz, earth_field_world)
            s['fused_mx'] = float(fused[0])
            s['fused_my'] = float(fused[1])
            s['fused_mz'] = float(fused[2])

            # Kalman filtered (filtered_*)
            s['filtered_mx'] = float(kf_x.update(fused[0]))
            s['filtered_my'] = float(kf_y.update(fused[1]))
            s['filtered_mz'] = float(kf_z.update(fused[2]))
        else:
            # No orientation - use iron-corrected only
            s['fused_mx'] = s['calibrated_mx']
            s['fused_my'] = s['calibrated_my']
            s['fused_mz'] = s['calibrated_mz']
            s['filtered_mx'] = float(kf_x.update(s['calibrated_mx']))
            s['filtered_my'] = float(kf_y.update(s['calibrated_my']))
            s['filtered_mz'] = float(kf_z.update(s['calibrated_mz']))

        updated_count += 1

    # Backup original file
    if backup:
        backup_path = session_path.with_suffix('.json.bak')
        if not backup_path.exists():
            shutil.copy(session_path, backup_path)
            print(f"  Backed up to {backup_path.name}")

    # Save updated session
    if is_wrapped:
        # Store the world-frame earth field in metadata for reference
        if 'calibration' not in metadata:
            metadata['calibration'] = {}
        metadata['calibration']['earthFieldWorld'] = earth_field_world
        metadata['calibration']['reprocessedAt'] = datetime.now().isoformat()
        data['samples'] = samples
        data['metadata'] = metadata
    else:
        data = samples

    with open(session_path, 'w') as f:
        json.dump(data, f)

    return {
        'updated_samples': updated_count,
        'earth_field_world': earth_field_world,
        'cal_source': cal_source
    }


def main():
    """Main entry point."""
    print("=" * 70)
    print("UPDATING SESSION DERIVED FIELDS WITH CORRECTED ALGORITHM")
    print("=" * 70)

    # Load calibration
    calibration = load_calibration()
    print(f"\nLoaded calibration from {CALIBRATION_FILE}")

    # Find all session files
    session_files = sorted(DATA_DIR.glob('2025-*.json'))
    print(f"Found {len(session_files)} session files\n")

    # Process each session
    results = []
    for session_path in session_files:
        print(f"Processing: {session_path.name}")

        result = update_session(session_path, calibration, backup=True)
        if result:
            print(f"  Updated {result['updated_samples']} samples")
            results.append({
                'session': session_path.name,
                **result
            })
        print()

    # Summary
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"\nUpdated {len(results)} sessions:")
    for r in results:
        print(f"  {r['session']}: {r['updated_samples']} samples ({r['cal_source']} calibration)")

    print("\nSession files have been updated with corrected derived fields.")
    print("Original files backed up with .bak extension.")
    print("\nYou can now run visualize.py to regenerate all graphics.")


if __name__ == '__main__':
    main()
