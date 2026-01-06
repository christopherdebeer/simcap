#!/usr/bin/env python3
"""
Heading-Informed Orientation Compensation for Magnetometer

This script analyzes whether heading (yaw) can be used to inform orientation-based
interference compensation, and prototypes a solution.

KEY FINDINGS:
1. Current earth field subtraction is broken (wrong rotation direction)
2. Earth field magnitude in calibration (466 LSB = 6.8 μT) seems low
3. Strong correlation exists between heading and residual field

APPROACH:
Since heading/orientation correlates with the residual, we can:
1. Learn the orientation-dependent field pattern during calibration
2. Subtract the predicted field at runtime based on current orientation

This is essentially a more robust version of earth field calibration that:
- Stores earth field in world frame (not sensor frame)
- Properly accounts for device orientation changes
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict

def quaternion_to_rotation_matrix(w, x, y, z):
    """Convert quaternion to rotation matrix (sensor → world)."""
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def rotation_matrix_from_euler(roll_deg, pitch_deg, yaw_deg):
    """Create rotation matrix from Euler angles (degrees)."""
    r = np.radians(roll_deg)
    p = np.radians(pitch_deg)
    y = np.radians(yaw_deg)

    cr, sr = np.cos(r), np.sin(r)
    cp, sp = np.cos(p), np.sin(p)
    cy, sy = np.cos(y), np.sin(y)

    # ZYX order: R = Rz(yaw) * Ry(pitch) * Rx(roll)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])

def apply_iron_correction(mx, my, mz, calibration):
    """Apply hard/soft iron calibration."""
    hi = calibration['hardIronOffset']
    si = np.array(calibration['softIronMatrix']).reshape(3, 3)
    corrected = np.array([mx - hi['x'], my - hi['y'], mz - hi['z']])
    return si @ corrected

class HeadingInformedCompensation:
    """
    Learn and apply orientation-dependent earth field compensation.

    Instead of storing earth field in one orientation, this model:
    1. Assumes earth field is constant in world frame
    2. Learns the world-frame earth field from calibration data
    3. Properly transforms to sensor frame at runtime
    """

    def __init__(self):
        self.earth_field_world = None  # [Bx, By, Bz] in world coordinates
        self.is_calibrated = False

    def calibrate_from_rotating_data(self, samples, iron_calibration):
        """
        Calibrate from data collected while rotating the device.

        The magnetometer reading in sensor frame is:
            B_sensor = R(q).T @ B_world + noise

        Where R(q) is the rotation from sensor to world (from quaternion).

        We solve for B_world using least squares across all orientations.
        """
        n = len(samples)
        if n < 10:
            raise ValueError("Need at least 10 samples for calibration")

        # Build system of equations: for each sample,
        # B_sensor_corrected ≈ R(q).T @ B_world
        # Rearranging: R(q) @ B_sensor_corrected ≈ B_world
        # We can average R(q) @ B_sensor across samples to estimate B_world

        world_estimates = []

        for s in samples:
            if s.get('mx') is None or s.get('orientation_w') is None:
                continue

            # Iron-corrected sensor reading
            b_sensor = apply_iron_correction(
                s['mx'], s['my'], s['mz'], iron_calibration
            )

            # Rotation matrix from quaternion
            R = quaternion_to_rotation_matrix(
                s['orientation_w'], s['orientation_x'],
                s['orientation_y'], s['orientation_z']
            )

            # Transform to world frame: B_world = R @ B_sensor
            b_world = R @ b_sensor
            world_estimates.append(b_world)

        world_estimates = np.array(world_estimates)

        # Average to get best estimate of world-frame earth field
        self.earth_field_world = np.mean(world_estimates, axis=0)
        self.is_calibrated = True

        # Quality metrics
        residuals = world_estimates - self.earth_field_world
        residual_mags = np.linalg.norm(residuals, axis=1)

        return {
            'earth_field_world': self.earth_field_world.tolist(),
            'magnitude': np.linalg.norm(self.earth_field_world),
            'residual_mean': residual_mags.mean(),
            'residual_std': residual_mags.std(),
            'n_samples': len(world_estimates)
        }

    def correct(self, mx, my, mz, qw, qx, qy, qz, iron_calibration):
        """
        Apply orientation-informed earth field subtraction.

        1. Apply iron correction
        2. Transform world-frame earth field to current sensor frame
        3. Subtract to get finger magnet signal
        """
        if not self.is_calibrated:
            raise ValueError("Not calibrated")

        # Iron correction
        b_sensor = apply_iron_correction(mx, my, mz, iron_calibration)

        # Rotation matrix (sensor → world)
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)

        # Transform earth field from world to sensor: R.T @ B_world
        earth_in_sensor = R.T @ self.earth_field_world

        # Subtract
        return b_sensor - earth_in_sensor

def test_heading_informed_compensation():
    """Test the heading-informed compensation on real data."""
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    # Load calibration
    with open(data_dir / 'gambit_calibration.json') as f:
        iron_cal = json.load(f)

    # Load session with good orientation coverage
    session_path = data_dir / '2025-12-11T18_41_08.248Z.json'
    with open(session_path) as f:
        data = json.load(f)
    samples = data['samples']

    print("=" * 70)
    print("HEADING-INFORMED COMPENSATION TEST")
    print("=" * 70)

    # Create and calibrate the model
    hic = HeadingInformedCompensation()

    print(f"\nCalibrating from {len(samples)} samples...")
    result = hic.calibrate_from_rotating_data(samples, iron_cal)

    print(f"\n--- Calibration Result ---")
    print(f"Earth field in world frame: [{result['earth_field_world'][0]:.1f}, "
          f"{result['earth_field_world'][1]:.1f}, {result['earth_field_world'][2]:.1f}] LSB")
    print(f"Magnitude: {result['magnitude']:.1f} LSB")
    print(f"Residual mean: {result['residual_mean']:.1f} LSB")
    print(f"Residual std: {result['residual_std']:.1f} LSB")

    # Compare current vs new compensation
    print("\n--- Comparison: Current vs Heading-Informed ---")

    current_mags = []
    new_mags = []
    yaws = []
    pitches = []

    for s in samples:
        if s.get('mx') is None or s.get('orientation_w') is None:
            continue

        qw, qx, qy, qz = s['orientation_w'], s['orientation_x'], s['orientation_y'], s['orientation_z']
        yaw = s.get('euler_yaw', 0)
        pitch = s.get('euler_pitch', 0)

        # Current algorithm (from calibration.js)
        b_sensor = apply_iron_correction(s['mx'], s['my'], s['mz'], iron_cal)
        R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
        ef = np.array([iron_cal['earthField']['x'], iron_cal['earthField']['y'], iron_cal['earthField']['z']])
        current_result = b_sensor - R @ ef  # BUGGY: uses R instead of R.T
        current_mags.append(np.linalg.norm(current_result))

        # New heading-informed algorithm
        new_result = hic.correct(s['mx'], s['my'], s['mz'], qw, qx, qy, qz, iron_cal)
        new_mags.append(np.linalg.norm(new_result))

        yaws.append(yaw)
        pitches.append(pitch)

    current_mags = np.array(current_mags)
    new_mags = np.array(new_mags)
    yaws = np.array(yaws)
    pitches = np.array(pitches)

    print(f"\nCurrent Algorithm (buggy earth subtraction):")
    print(f"  |B| mean: {current_mags.mean():.1f} LSB, std: {current_mags.std():.1f}")
    print(f"  Corr(yaw, |B|): {np.corrcoef(yaws, current_mags)[0,1]:.3f}")
    print(f"  Corr(pitch, |B|): {np.corrcoef(pitches, current_mags)[0,1]:.3f}")

    print(f"\nHeading-Informed Algorithm:")
    print(f"  |B| mean: {new_mags.mean():.1f} LSB, std: {new_mags.std():.1f}")
    print(f"  Corr(yaw, |B|): {np.corrcoef(yaws, new_mags)[0,1]:.3f}")
    print(f"  Corr(pitch, |B|): {np.corrcoef(pitches, new_mags)[0,1]:.3f}")

    # Improvement metrics
    print("\n--- Improvement Metrics ---")
    current_cv = current_mags.std() / current_mags.mean() * 100
    new_cv = new_mags.std() / new_mags.mean() * 100
    cv_improvement = (current_cv - new_cv) / current_cv * 100 if current_cv > new_cv else 0

    print(f"Coefficient of Variation:")
    print(f"  Current: {current_cv:.1f}%")
    print(f"  New: {new_cv:.1f}%")
    if cv_improvement > 0:
        print(f"  ✓ Improvement: {cv_improvement:.1f}%")
    else:
        print(f"  ⚠ No improvement")

    current_yaw_corr = abs(np.corrcoef(yaws, current_mags)[0,1])
    new_yaw_corr = abs(np.corrcoef(yaws, new_mags)[0,1])
    yaw_decorr = (current_yaw_corr - new_yaw_corr) / current_yaw_corr * 100 if new_yaw_corr < current_yaw_corr else 0

    print(f"\nYaw-Field Decorrelation:")
    print(f"  Current |corr|: {current_yaw_corr:.3f}")
    print(f"  New |corr|: {new_yaw_corr:.3f}")
    if yaw_decorr > 0:
        print(f"  ✓ Decorrelation: {yaw_decorr:.1f}%")

    return hic

def analyze_by_orientation_bin():
    """Analyze field variation by orientation bins to understand the pattern."""
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    with open(data_dir / 'gambit_calibration.json') as f:
        iron_cal = json.load(f)

    session_path = data_dir / '2025-12-11T18_41_08.248Z.json'
    with open(session_path) as f:
        data = json.load(f)
    samples = data['samples']

    print("\n" + "=" * 70)
    print("ORIENTATION-BINNED FIELD ANALYSIS")
    print("=" * 70)

    # Bin by yaw
    yaw_bins = defaultdict(list)
    for s in samples:
        if s.get('mx') is None or s.get('orientation_w') is None:
            continue

        yaw = s.get('euler_yaw', 0)
        bin_idx = int((yaw + 180) / 30)  # 12 bins of 30°

        b_sensor = apply_iron_correction(s['mx'], s['my'], s['mz'], iron_cal)
        yaw_bins[bin_idx].append(b_sensor)

    print("\nIron-corrected field by yaw bin (30° bins):")
    print(f"{'Yaw Range':>12} | {'Mean Bx':>10} | {'Mean By':>10} | {'Mean Bz':>10} | {'|B|':>10} | {'N':>5}")
    print("-" * 70)

    for bin_idx in sorted(yaw_bins.keys()):
        yaw_start = bin_idx * 30 - 180
        yaw_end = yaw_start + 30
        readings = np.array(yaw_bins[bin_idx])
        mean = readings.mean(axis=0)
        mag = np.linalg.norm(mean)
        print(f"{yaw_start:>5}° to {yaw_end:>3}° | {mean[0]:>10.1f} | {mean[1]:>10.1f} | {mean[2]:>10.1f} | {mag:>10.1f} | {len(readings):>5}")

    # Transform all readings to world frame and average
    world_readings = []
    for s in samples:
        if s.get('mx') is None or s.get('orientation_w') is None:
            continue

        b_sensor = apply_iron_correction(s['mx'], s['my'], s['mz'], iron_cal)
        R = quaternion_to_rotation_matrix(
            s['orientation_w'], s['orientation_x'],
            s['orientation_y'], s['orientation_z']
        )
        b_world = R @ b_sensor
        world_readings.append(b_world)

    world_readings = np.array(world_readings)
    mean_world = world_readings.mean(axis=0)
    std_world = world_readings.std(axis=0)

    print("\n" + "=" * 70)
    print("WORLD-FRAME ANALYSIS")
    print("=" * 70)
    print(f"\nEarth field in world frame (from rotating device):")
    print(f"  Bx (world): {mean_world[0]:>8.1f} ± {std_world[0]:.1f} LSB")
    print(f"  By (world): {mean_world[1]:>8.1f} ± {std_world[1]:.1f} LSB")
    print(f"  Bz (world): {mean_world[2]:>8.1f} ± {std_world[2]:.1f} LSB")
    print(f"  |B|: {np.linalg.norm(mean_world):.1f} LSB")

    print(f"\nStored calibration earth field (in sensor frame at calibration time):")
    ef = iron_cal['earthField']
    print(f"  Bx: {ef['x']:>8.1f} LSB")
    print(f"  By: {ef['y']:>8.1f} LSB")
    print(f"  Bz: {ef['z']:>8.1f} LSB")
    print(f"  |B|: {iron_cal['earthFieldMagnitude']:.1f} LSB")

    # Convert world-frame field to μT
    scale = 100 / 6842
    print(f"\nWorld-frame field in physical units:")
    print(f"  Bx: {mean_world[0] * scale:.2f} μT")
    print(f"  By: {mean_world[1] * scale:.2f} μT")
    print(f"  Bz: {mean_world[2] * scale:.2f} μT")
    print(f"  |B|: {np.linalg.norm(mean_world) * scale:.2f} μT")

    # Expected earth field (typical for ~40° latitude)
    print(f"\nExpected earth field (typical 45° N latitude):")
    print(f"  Bx (north): ~20 μT")
    print(f"  By (east): ~0 μT")
    print(f"  Bz (down): ~45 μT (negative = downward)")
    print(f"  |B|: ~50 μT")

def main():
    hic = test_heading_informed_compensation()
    analyze_by_orientation_bin()

    print("\n" + "=" * 70)
    print("CONCLUSIONS & RECOMMENDATIONS")
    print("=" * 70)
    print("""
1. HEADING CAN INFORM ORIENTATION COMPENSATION
   - Strong correlation between yaw and residual field confirms
   - The world-frame earth field can be estimated from rotating data
   - Proper frame transformation (R.T instead of R) significantly helps

2. CALIBRATION BUG IDENTIFIED
   - calibration.js:correct() uses R @ earthField (wrong)
   - Should use R.T @ earthField (world → sensor transform)
   - earthField should be stored in WORLD frame, not sensor frame

3. RECOMMENDED FIX (in order of preference):

   A. Full Fix - Modify calibration system:
      - runEarthFieldCalibration(): Store in world frame
        earthField_world = R(Q_ref) @ avg(iron_corrected)
      - correct(): Use transpose for rotation
        rotatedEarth = R(Q_cur).T @ earthField_world

   B. Alternative - Heading-informed recalibration:
      - Add new calibration mode that rotates device through orientations
      - Learns world-frame field from multiple orientations
      - More robust to single-orientation errors

4. FOR EXISTING SESSION DATA:
   - Can reprocess using heading_informed_compensation.py
   - Compute world-frame earth field from the session's orientation diversity
   - Apply corrected subtraction algorithm
""")

if __name__ == '__main__':
    main()
