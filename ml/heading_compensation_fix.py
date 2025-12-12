#!/usr/bin/env python3
"""
Analysis of Earth Field Subtraction Bug and Correct Algorithm

PROBLEM IDENTIFIED:
The current calibration.js:correct() does:
    rotatedEarth = Q_cur.toRotationMatrix() * earthField

This is WRONG because:
- earthField is stored in sensor frame at calibration reference orientation (Q_ref)
- Multiplying by Q_cur's rotation matrix gives nonsense

CORRECT ALGORITHM:
Earth's magnetic field is constant in world frame. When device rotates,
we need to express that constant world-frame field in the current sensor frame.

If earthField was captured at reference orientation Q_ref (stored in sensor frame):
    earthField_world = R(Q_ref) * earthField_sensor  // Transform to world

At runtime with current orientation Q_cur:
    earthField_in_sensor = R(Q_cur).T * earthField_world  // World to sensor

Combined:
    earthField_in_sensor = R(Q_cur).T * R(Q_ref) * earthField_sensor

If we assume Q_ref = identity (device flat, aligned with world):
    earthField_in_sensor = R(Q_cur).T * earthField_sensor
    = R(Q_cur^-1) * earthField_sensor
"""

import json
import numpy as np
from pathlib import Path

def quaternion_to_rotation_matrix(w, x, y, z):
    """Convert quaternion to 3x3 rotation matrix (transforms FROM sensor TO world)."""
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def apply_iron_correction(mx, my, mz, calibration):
    """Apply hard/soft iron calibration."""
    hi = calibration['hardIronOffset']
    si = np.array(calibration['softIronMatrix']).reshape(3, 3)

    # Hard iron
    corrected = np.array([mx - hi['x'], my - hi['y'], mz - hi['z']])
    # Soft iron
    return si @ corrected

def current_earth_subtraction(corrected_mag, qw, qx, qy, qz, earth_field):
    """
    CURRENT (BUGGY) algorithm from calibration.js:
    rotatedEarth = R(Q_cur) * earthField
    result = corrected - rotatedEarth
    """
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    ef = np.array([earth_field['x'], earth_field['y'], earth_field['z']])
    rotated_earth = R @ ef  # This is wrong!
    return corrected_mag - rotated_earth

def correct_earth_subtraction(corrected_mag, qw, qx, qy, qz, earth_field_world):
    """
    CORRECT algorithm:
    Earth field is constant in world frame.
    To subtract from sensor reading, rotate world→sensor (inverse rotation).
    rotatedEarth = R(Q_cur).T * earthField_world
    result = corrected - rotatedEarth
    """
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)
    ef = np.array([earth_field_world['x'], earth_field_world['y'], earth_field_world['z']])
    # R.T rotates from world to sensor
    rotated_earth = R.T @ ef
    return corrected_mag - rotated_earth

def load_session(filepath):
    """Load session data."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data.get('samples', data), data.get('metadata', {}).get('calibration', {})

def analyze_earth_field_frame():
    """
    Determine the correct earth field frame by analyzing the calibration procedure.

    During Earth field calibration (runEarthFieldCalibration):
    - Device is held still in reference orientation
    - Raw readings are iron-corrected
    - Average is stored as earthField

    This means earthField is in SENSOR FRAME at the reference orientation.

    If reference orientation was Q_ref, to get earthField in world frame:
        earthField_world = R(Q_ref) * earthField_sensor

    The problem: Q_ref is NOT stored during calibration!

    SOLUTION: Store earthField in WORLD frame during calibration:
        earthField_world = R(Q_ref) * avg(iron_corrected_readings)
    """
    print("=" * 60)
    print("EARTH FIELD FRAME ANALYSIS")
    print("=" * 60)

    print("""
CURRENT BUG:
- earthField stored in sensor frame at unknown reference orientation Q_ref
- Runtime correction uses: rotatedEarth = R(Q_cur) * earthField
- This produces nonsense because earthField is in sensor, not world frame

FIX OPTIONS:

1. Store earth field in WORLD frame during calibration:
   - During calibration at Q_ref: earthField_world = R(Q_ref) * earthField_sensor
   - At runtime: rotatedEarth = R(Q_cur).T * earthField_world

2. Store reference quaternion Q_ref during calibration:
   - At runtime: rotatedEarth = R(Q_cur).T * R(Q_ref) * earthField_sensor

Option 1 is cleaner - modify runEarthFieldCalibration() to store in world frame.
""")

def test_algorithms():
    """Test current vs corrected algorithm on real data."""
    data_dir = Path('/home/user/simcap/data/GAMBIT')
    calibration_path = data_dir / 'gambit_calibration.json'
    session_path = data_dir / '2025-12-11T16_16_16.613Z.json'

    # Load calibration
    with open(calibration_path) as f:
        calibration = json.load(f)

    # Load session with embedded calibration
    samples, session_cal = load_session(session_path)

    print("\n" + "=" * 60)
    print("ALGORITHM COMPARISON TEST")
    print("=" * 60)

    print(f"\nCalibration Earth Field: {calibration['earthField']}")
    print(f"Magnitude: {calibration['earthFieldMagnitude']:.1f} LSB")

    # Analyze a subset of samples
    n_samples = min(100, len(samples))

    # Results storage
    current_mags = []
    fixed_mags = []
    raw_mags = []
    yaws = []
    pitches = []

    for i in range(n_samples):
        s = samples[i]
        if s.get('mx') is None or s.get('orientation_w') is None:
            continue

        # Raw magnetometer
        mx, my, mz = s['mx'], s['my'], s['mz']
        raw_mag = np.sqrt(mx**2 + my**2 + mz**2)

        # Iron correction
        corrected = apply_iron_correction(mx, my, mz, calibration)

        # Orientation
        qw = s['orientation_w']
        qx = s['orientation_x']
        qy = s['orientation_y']
        qz = s['orientation_z']
        yaw = s.get('euler_yaw', 0)
        pitch = s.get('euler_pitch', 0)

        # Current (buggy) algorithm
        current_result = current_earth_subtraction(corrected, qw, qx, qy, qz, calibration['earthField'])
        current_mag = np.linalg.norm(current_result)

        # Corrected algorithm - assuming earthField is already in world frame
        # (In practice, we'd need to convert it during calibration)
        fixed_result = correct_earth_subtraction(corrected, qw, qx, qy, qz, calibration['earthField'])
        fixed_mag = np.linalg.norm(fixed_result)

        current_mags.append(current_mag)
        fixed_mags.append(fixed_mag)
        raw_mags.append(raw_mag)
        yaws.append(yaw)
        pitches.append(pitch)

    current_mags = np.array(current_mags)
    fixed_mags = np.array(fixed_mags)
    raw_mags = np.array(raw_mags)
    yaws = np.array(yaws)
    pitches = np.array(pitches)

    print(f"\nAnalyzed {len(current_mags)} samples")

    print("\n--- Raw Magnetometer ---")
    print(f"|B| mean: {raw_mags.mean():.1f}, std: {raw_mags.std():.1f}")
    print(f"Corr(yaw, |B|): {np.corrcoef(yaws, raw_mags)[0,1]:.3f}")

    print("\n--- Current Algorithm (BUGGY) ---")
    print(f"|B| mean: {current_mags.mean():.1f}, std: {current_mags.std():.1f}")
    print(f"Corr(yaw, |B|): {np.corrcoef(yaws, current_mags)[0,1]:.3f}")
    print(f"Corr(pitch, |B|): {np.corrcoef(pitches, current_mags)[0,1]:.3f}")

    print("\n--- Fixed Algorithm (R.T instead of R) ---")
    print(f"|B| mean: {fixed_mags.mean():.1f}, std: {fixed_mags.std():.1f}")
    print(f"Corr(yaw, |B|): {np.corrcoef(yaws, fixed_mags)[0,1]:.3f}")
    print(f"Corr(pitch, |B|): {np.corrcoef(pitches, fixed_mags)[0,1]:.3f}")

    # Stability improvement
    print("\n--- Stability Analysis ---")
    current_cv = current_mags.std() / current_mags.mean() * 100
    fixed_cv = fixed_mags.std() / fixed_mags.mean() * 100
    print(f"Current CV: {current_cv:.1f}%")
    print(f"Fixed CV: {fixed_cv:.1f}%")
    if fixed_cv < current_cv:
        improvement = (current_cv - fixed_cv) / current_cv * 100
        print(f"✓ Stability improved by {improvement:.1f}%")
    else:
        print("⚠ Fixed algorithm not better - may need earth field in world frame")

    # The key insight
    print("\n" + "=" * 60)
    print("KEY INSIGHT")
    print("=" * 60)
    print("""
The 'fixed' algorithm using R.T may not fully solve the problem because
the earthField was stored in SENSOR frame during calibration, not WORLD frame.

COMPLETE FIX requires modifying calibration to store earth field in world frame:

1. In runEarthFieldCalibration(), after computing the average field:

   // Get current orientation during calibration
   const Q_ref = imuFusion.getQuaternion();
   const R_ref = Q_ref.toRotationMatrix();

   // Transform sensor-frame earth field to world frame
   const earthFieldWorld = R_ref.multiply(earthFieldSensor);
   this.earthField = earthFieldWorld;  // Store in world frame!

2. In correct(), use R.T to transform back to sensor frame:

   const rotatedEarth = rotMatrix.transpose().multiply(this.earthField);
""")

def analyze_reference_orientation():
    """
    Try to infer what reference orientation was used during calibration
    by analyzing the stored earth field components.

    Earth's field in world frame (typical northern hemisphere):
    - Points north with downward inclination (~60° from horizontal)
    - Bx ≈ 20 μT (north), By ≈ 0 (east), Bz ≈ -45 μT (down)

    In LSB units (with scale ~0.0146 μT/LSB for LIS3MDL at ±4G):
    - Bx ≈ 1370 LSB, Bz ≈ -3100 LSB
    """
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    with open(data_dir / 'gambit_calibration.json') as f:
        cal = json.load(f)

    ef = cal['earthField']
    print("\n" + "=" * 60)
    print("REFERENCE ORIENTATION INFERENCE")
    print("=" * 60)

    print(f"\nStored Earth Field (sensor frame at calibration):")
    print(f"  x: {ef['x']:.1f} LSB")
    print(f"  y: {ef['y']:.1f} LSB")
    print(f"  z: {ef['z']:.1f} LSB")
    print(f"  magnitude: {cal['earthFieldMagnitude']:.1f} LSB")

    # Convert to μT
    scale = 100 / 6842  # From calibration.js
    print(f"\nIn μT (scale factor {scale:.6f}):")
    print(f"  x: {ef['x'] * scale:.2f} μT")
    print(f"  y: {ef['y'] * scale:.2f} μT")
    print(f"  z: {ef['z'] * scale:.2f} μT")

    # Typical earth field magnitude is 25-65 μT
    mag_ut = cal['earthFieldMagnitude'] * scale
    print(f"  magnitude: {mag_ut:.2f} μT")

    if 25 < mag_ut < 65:
        print("  ✓ Magnitude is in typical Earth field range")
    else:
        print(f"  ⚠ Magnitude {mag_ut:.1f} μT is outside typical 25-65 μT range")

    # Compute inclination (angle from horizontal)
    horizontal = np.sqrt(ef['x']**2 + ef['y']**2)
    inclination = np.degrees(np.arctan2(-ef['z'], horizontal))
    print(f"\nInclination: {inclination:.1f}° (typical: 50-70° in northern hemisphere)")

    # The large Y component suggests device wasn't aligned with magnetic north
    declination = np.degrees(np.arctan2(ef['y'], ef['x']))
    print(f"Declination from sensor X: {declination:.1f}°")

    print("""
INTERPRETATION:
The large Y component (368.8) suggests either:
1. Device wasn't aligned with magnetic north during calibration, OR
2. There's local magnetic anomaly

For accurate compensation, the calibration should:
1. Record the IMU orientation (Q_ref) at calibration time
2. Transform earthField to world frame using Q_ref
3. Store earthField in world frame for proper runtime subtraction
""")

def main():
    analyze_earth_field_frame()
    test_algorithms()
    analyze_reference_orientation()

    print("\n" + "=" * 60)
    print("RECOMMENDED FIX")
    print("=" * 60)
    print("""
1. MODIFY runEarthFieldCalibration() to store earth field in WORLD frame:
   - Capture orientation quaternion Q_ref during calibration
   - Transform: earthField_world = R(Q_ref) * earthField_sensor
   - Store earthField_world

2. MODIFY correct() to use transpose for world→sensor transform:
   - rotatedEarth = R(Q_cur).T * earthField_world

3. ALTERNATIVE: Use heading-informed compensation
   - Since yaw (heading) strongly correlates with residual (-0.619),
   - Could use yaw to predict and subtract orientation-dependent component
   - This would work even without fixing the fundamental frame bug
""")

if __name__ == '__main__':
    main()
