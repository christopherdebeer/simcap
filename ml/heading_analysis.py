#!/usr/bin/env python3
"""
Analyze the relationship between heading/orientation and magnetic field readings.
Goal: Determine if we can use heading to inform orientation-based interference compensation.
"""

import json
import numpy as np
import sys
from pathlib import Path

def load_session(filepath):
    """Load session data, applying checked-in calibration if session has zeroed calibration."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    samples = data.get('samples', data)  # Handle v1 vs v2 format
    if isinstance(samples, dict):
        samples = [samples]

    metadata = data.get('metadata', {})
    calibration = metadata.get('calibration', {})

    return samples, calibration, metadata

def load_calibration(calibration_path):
    """Load the checked-in calibration file."""
    with open(calibration_path, 'r') as f:
        return json.load(f)

def quaternion_to_rotation_matrix(w, x, y, z):
    """Convert quaternion to 3x3 rotation matrix."""
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w, 2*x*z + 2*y*w],
        [2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w, 2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y]
    ])

def apply_calibration(mx, my, mz, calibration):
    """Apply hard/soft iron calibration to raw magnetometer data."""
    hi = calibration.get('hardIronOffset', {'x': 0, 'y': 0, 'z': 0})
    si = calibration.get('softIronMatrix', [1,0,0, 0,1,0, 0,0,1])

    # Hard iron correction
    mx_corr = mx - hi['x']
    my_corr = my - hi['y']
    mz_corr = mz - hi['z']

    # Soft iron correction (3x3 matrix)
    si_mat = np.array(si).reshape(3, 3)
    corrected = si_mat @ np.array([mx_corr, my_corr, mz_corr])

    return corrected[0], corrected[1], corrected[2]

def subtract_earth_field_by_orientation(mx, my, mz, qw, qx, qy, qz, earth_field):
    """
    Subtract earth field rotated by current device orientation.
    This is the key operation - rotate earth field to sensor frame then subtract.
    """
    ef = np.array([earth_field['x'], earth_field['y'], earth_field['z']])

    # Rotation matrix from device orientation quaternion
    R = quaternion_to_rotation_matrix(qw, qx, qy, qz)

    # Rotate earth field from reference frame to current sensor frame
    # R.T rotates from world to sensor frame
    ef_rotated = R.T @ ef

    # Subtract rotated earth field
    return mx - ef_rotated[0], my - ef_rotated[1], mz - ef_rotated[2]

def analyze_session(session_path, calibration_path=None):
    """Analyze a single session for heading vs magnetic field correlation."""
    samples, session_cal, metadata = load_session(session_path)

    # Check if we need to use checked-in calibration
    use_checked_in_cal = not session_cal.get('hardIronCalibrated', False)

    if use_checked_in_cal and calibration_path:
        print(f"Session has zeroed calibration - using checked-in calibration")
        calibration = load_calibration(calibration_path)
    else:
        print(f"Session has embedded calibration")
        calibration = session_cal

    print(f"Calibration: HI={calibration.get('hardIronOffset')}")
    print(f"             EF={calibration.get('earthField')}")

    earth_field = calibration.get('earthField', {'x': 0, 'y': 0, 'z': 0})

    # Extract data arrays
    n = len(samples)
    print(f"\nAnalyzing {n} samples...")

    # Raw and processed magnetic fields
    raw_mx, raw_my, raw_mz = [], [], []
    cal_mx, cal_my, cal_mz = [], [], []  # After hard/soft iron
    fused_mx, fused_my, fused_mz = [], [], []  # After earth field subtraction

    # Orientation
    yaw, pitch, roll = [], [], []
    qw, qx, qy, qz = [], [], [], []

    # Existing filtered values (from session)
    existing_filtered_mx, existing_filtered_my, existing_filtered_mz = [], [], []

    for s in samples:
        # Raw magnetometer
        raw_mx.append(s.get('mx', 0))
        raw_my.append(s.get('my', 0))
        raw_mz.append(s.get('mz', 0))

        # Orientation
        yaw.append(s.get('euler_yaw', 0))
        pitch.append(s.get('euler_pitch', 0))
        roll.append(s.get('euler_roll', 0))

        qw.append(s.get('orientation_w', 1))
        qx.append(s.get('orientation_x', 0))
        qy.append(s.get('orientation_y', 0))
        qz.append(s.get('orientation_z', 0))

        # Existing filtered (what the system produced)
        existing_filtered_mx.append(s.get('filtered_mx', s.get('mx', 0)))
        existing_filtered_my.append(s.get('filtered_my', s.get('my', 0)))
        existing_filtered_mz.append(s.get('filtered_mz', s.get('mz', 0)))

        # Apply calibration
        cmx, cmy, cmz = apply_calibration(s.get('mx', 0), s.get('my', 0), s.get('mz', 0), calibration)
        cal_mx.append(cmx)
        cal_my.append(cmy)
        cal_mz.append(cmz)

    # Convert to arrays
    raw_mx, raw_my, raw_mz = np.array(raw_mx), np.array(raw_my), np.array(raw_mz)
    cal_mx, cal_my, cal_mz = np.array(cal_mx), np.array(cal_my), np.array(cal_mz)
    yaw, pitch, roll = np.array(yaw), np.array(pitch), np.array(roll)
    qw, qx, qy, qz = np.array(qw), np.array(qx), np.array(qy), np.array(qz)
    existing_filtered_mx = np.array(existing_filtered_mx)
    existing_filtered_my = np.array(existing_filtered_my)
    existing_filtered_mz = np.array(existing_filtered_mz)

    # Now compute fused (orientation-compensated) values
    fused_mx, fused_my, fused_mz = [], [], []
    for i in range(n):
        fx, fy, fz = subtract_earth_field_by_orientation(
            cal_mx[i], cal_my[i], cal_mz[i],
            qw[i], qx[i], qy[i], qz[i],
            earth_field
        )
        fused_mx.append(fx)
        fused_my.append(fy)
        fused_mz.append(fz)

    fused_mx, fused_my, fused_mz = np.array(fused_mx), np.array(fused_my), np.array(fused_mz)

    # Compute field magnitudes
    raw_mag = np.sqrt(raw_mx**2 + raw_my**2 + raw_mz**2)
    cal_mag = np.sqrt(cal_mx**2 + cal_my**2 + cal_mz**2)
    fused_mag = np.sqrt(fused_mx**2 + fused_my**2 + fused_mz**2)
    existing_filtered_mag = np.sqrt(existing_filtered_mx**2 + existing_filtered_my**2 + existing_filtered_mz**2)

    print("\n" + "="*60)
    print("STATISTICS")
    print("="*60)

    print("\nOrientation Range:")
    print(f"  Yaw:   {yaw.min():.1f}° to {yaw.max():.1f}° (range: {yaw.max()-yaw.min():.1f}°)")
    print(f"  Pitch: {pitch.min():.1f}° to {pitch.max():.1f}° (range: {pitch.max()-pitch.min():.1f}°)")
    print(f"  Roll:  {roll.min():.1f}° to {roll.max():.1f}° (range: {roll.max()-roll.min():.1f}°)")

    print("\nRaw Magnetometer Field:")
    print(f"  mx: mean={raw_mx.mean():.1f}, std={raw_mx.std():.1f}, range=[{raw_mx.min():.0f}, {raw_mx.max():.0f}]")
    print(f"  my: mean={raw_my.mean():.1f}, std={raw_my.std():.1f}, range=[{raw_my.min():.0f}, {raw_my.max():.0f}]")
    print(f"  mz: mean={raw_mz.mean():.1f}, std={raw_mz.std():.1f}, range=[{raw_mz.min():.0f}, {raw_mz.max():.0f}]")
    print(f"  |B|: mean={raw_mag.mean():.1f}, std={raw_mag.std():.1f}")

    print("\nCalibrated (Hard/Soft Iron Corrected):")
    print(f"  mx: mean={cal_mx.mean():.1f}, std={cal_mx.std():.1f}")
    print(f"  my: mean={cal_my.mean():.1f}, std={cal_my.std():.1f}")
    print(f"  mz: mean={cal_mz.mean():.1f}, std={cal_mz.std():.1f}")
    print(f"  |B|: mean={cal_mag.mean():.1f}, std={cal_mag.std():.1f}")

    print("\nFused (After Earth Field Subtraction by Orientation):")
    print(f"  mx: mean={fused_mx.mean():.1f}, std={fused_mx.std():.1f}")
    print(f"  my: mean={fused_my.mean():.1f}, std={fused_my.std():.1f}")
    print(f"  mz: mean={fused_mz.mean():.1f}, std={fused_mz.std():.1f}")
    print(f"  |B|: mean={fused_mag.mean():.1f}, std={fused_mag.std():.1f}")

    print("\nExisting Session Filtered Values:")
    print(f"  mx: mean={existing_filtered_mx.mean():.1f}, std={existing_filtered_mx.std():.1f}")
    print(f"  my: mean={existing_filtered_my.mean():.1f}, std={existing_filtered_my.std():.1f}")
    print(f"  mz: mean={existing_filtered_mz.mean():.1f}, std={existing_filtered_mz.std():.1f}")
    print(f"  |B|: mean={existing_filtered_mag.mean():.1f}, std={existing_filtered_mag.std():.1f}")

    # Correlation analysis
    print("\n" + "="*60)
    print("CORRELATION: YAW (HEADING) vs MAGNETIC FIELD")
    print("="*60)

    print("\nRaw field vs Yaw:")
    print(f"  corr(yaw, raw_mx) = {np.corrcoef(yaw, raw_mx)[0,1]:.3f}")
    print(f"  corr(yaw, raw_my) = {np.corrcoef(yaw, raw_my)[0,1]:.3f}")
    print(f"  corr(yaw, raw_mz) = {np.corrcoef(yaw, raw_mz)[0,1]:.3f}")
    print(f"  corr(yaw, raw_|B|) = {np.corrcoef(yaw, raw_mag)[0,1]:.3f}")

    print("\nCalibrated field vs Yaw:")
    print(f"  corr(yaw, cal_mx) = {np.corrcoef(yaw, cal_mx)[0,1]:.3f}")
    print(f"  corr(yaw, cal_my) = {np.corrcoef(yaw, cal_my)[0,1]:.3f}")
    print(f"  corr(yaw, cal_mz) = {np.corrcoef(yaw, cal_mz)[0,1]:.3f}")
    print(f"  corr(yaw, cal_|B|) = {np.corrcoef(yaw, cal_mag)[0,1]:.3f}")

    print("\nFused field vs Yaw (should be decorrelated if orientation compensation works):")
    print(f"  corr(yaw, fused_mx) = {np.corrcoef(yaw, fused_mx)[0,1]:.3f}")
    print(f"  corr(yaw, fused_my) = {np.corrcoef(yaw, fused_my)[0,1]:.3f}")
    print(f"  corr(yaw, fused_mz) = {np.corrcoef(yaw, fused_mz)[0,1]:.3f}")
    print(f"  corr(yaw, fused_|B|) = {np.corrcoef(yaw, fused_mag)[0,1]:.3f}")

    # Correlation with pitch and roll too
    print("\n" + "="*60)
    print("CORRELATION: FULL ORIENTATION vs FUSED FIELD")
    print("="*60)
    print("\nFused field vs Pitch:")
    print(f"  corr(pitch, fused_mx) = {np.corrcoef(pitch, fused_mx)[0,1]:.3f}")
    print(f"  corr(pitch, fused_my) = {np.corrcoef(pitch, fused_my)[0,1]:.3f}")
    print(f"  corr(pitch, fused_mz) = {np.corrcoef(pitch, fused_mz)[0,1]:.3f}")

    print("\nFused field vs Roll:")
    print(f"  corr(roll, fused_mx) = {np.corrcoef(roll, fused_mx)[0,1]:.3f}")
    print(f"  corr(roll, fused_my) = {np.corrcoef(roll, fused_my)[0,1]:.3f}")
    print(f"  corr(roll, fused_mz) = {np.corrcoef(roll, fused_mz)[0,1]:.3f}")

    # Check if there's systematic bias that could be a second iron source
    print("\n" + "="*60)
    print("ENVIRONMENTAL INTERFERENCE ANALYSIS")
    print("="*60)

    # If fused_mag varies significantly while device isn't moving (no finger magnets),
    # there's environmental interference not captured by calibration

    # Segment by yaw bins to see if there's a systematic pattern
    yaw_bins = np.linspace(yaw.min(), yaw.max(), 13)  # 12 bins of 30° each (roughly)

    print("\nFused field magnitude by yaw bin (12 bins):")
    print("(If |B| varies with yaw, environmental field isn't fully compensated)")
    for i in range(len(yaw_bins)-1):
        mask = (yaw >= yaw_bins[i]) & (yaw < yaw_bins[i+1])
        if mask.sum() > 0:
            bin_center = (yaw_bins[i] + yaw_bins[i+1]) / 2
            mean_mag = fused_mag[mask].mean()
            std_mag = fused_mag[mask].std()
            n_samples = mask.sum()
            print(f"  Yaw {bin_center:6.1f}°: |B|={mean_mag:6.1f} ± {std_mag:5.1f} (n={n_samples})")

    # Same for pitch bins
    print("\nFused field magnitude by pitch bin:")
    pitch_bins = np.linspace(pitch.min(), pitch.max(), 7)  # 6 bins
    for i in range(len(pitch_bins)-1):
        mask = (pitch >= pitch_bins[i]) & (pitch < pitch_bins[i+1])
        if mask.sum() > 0:
            bin_center = (pitch_bins[i] + pitch_bins[i+1]) / 2
            mean_mag = fused_mag[mask].mean()
            std_mag = fused_mag[mask].std()
            n_samples = mask.sum()
            print(f"  Pitch {bin_center:6.1f}°: |B|={mean_mag:6.1f} ± {std_mag:5.1f} (n={n_samples})")

    # Residual analysis - what's left after orientation compensation?
    print("\n" + "="*60)
    print("RESIDUAL ANALYSIS")
    print("="*60)

    # Ideal: fused field should be near zero without finger magnets
    expected_magnitude = calibration.get('earthFieldMagnitude', 50)
    print(f"\nExpected earth field magnitude: {expected_magnitude:.1f} LSB")
    print(f"Fused field magnitude: mean={fused_mag.mean():.1f}, std={fused_mag.std():.1f}")

    # If no finger magnets, fused should be ~0
    # Residual tells us about environmental interference
    residual_ratio = fused_mag.mean() / expected_magnitude if expected_magnitude > 0 else 0
    print(f"\nResidual ratio (fused/expected): {residual_ratio:.2%}")

    if residual_ratio > 0.2:
        print("⚠️  High residual suggests environmental interference OR orientation drift")
    elif residual_ratio > 0.05:
        print("⚡ Moderate residual - some uncompensated interference")
    else:
        print("✓ Low residual - orientation compensation is working well")

    # Return data for further analysis
    return {
        'yaw': yaw, 'pitch': pitch, 'roll': roll,
        'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
        'raw_mx': raw_mx, 'raw_my': raw_my, 'raw_mz': raw_mz,
        'cal_mx': cal_mx, 'cal_my': cal_my, 'cal_mz': cal_mz,
        'fused_mx': fused_mx, 'fused_my': fused_my, 'fused_mz': fused_mz,
        'fused_mag': fused_mag,
        'calibration': calibration
    }


def main():
    data_dir = Path('/home/user/simcap/data/GAMBIT')
    calibration_path = data_dir / 'gambit_calibration.json'

    # Find all session files
    session_files = sorted(data_dir.glob('2025-*.json'))

    print("="*60)
    print("HEADING-ORIENTATION ANALYSIS FOR INTERFERENCE COMPENSATION")
    print("="*60)

    for session_file in session_files:
        print(f"\n{'='*60}")
        print(f"SESSION: {session_file.name}")
        print("="*60)
        try:
            data = analyze_session(session_file, calibration_path)
        except Exception as e:
            print(f"Error analyzing session: {e}")
            import traceback
            traceback.print_exc()

if __name__ == '__main__':
    main()
