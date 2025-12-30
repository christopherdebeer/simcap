#!/usr/bin/env python3
"""
Deep Analysis: Magnetometer Calibration Session 2025-12-30T22_46_28.771Z

Investigating:
1. Why calibration auto-completes at 100% too quickly
2. Why Earth residual is ~67µT instead of near-zero (no finger magnets present)
3. H/V ratio inversion (measured 1.28 vs expected 0.33)
4. Why orientation-aware calibration also fails
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from datetime import datetime

# Edinburgh geomagnetic reference
EARTH_H = 16.0  # Horizontal component (µT)
EARTH_V = 47.8  # Vertical component (µT)
EARTH_MAG = np.sqrt(EARTH_H**2 + EARTH_V**2)  # ~50.4 µT
EARTH_WORLD = np.array([EARTH_H, 0, EARTH_V])  # [North, East, Down] in NED

print("=" * 80)
print("MAGNETOMETER CALIBRATION DEEP ANALYSIS")
print("Session: 2025-12-30T22_46_28.771Z")
print("=" * 80)
print(f"\nExpected Earth field (Edinburgh):")
print(f"  Horizontal (H): {EARTH_H} µT")
print(f"  Vertical (V): {EARTH_V} µT")
print(f"  Total magnitude: {EARTH_MAG:.1f} µT")
print(f"  H/V ratio: {EARTH_H/EARTH_V:.3f}")


def load_session(filepath):
    """Load session data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_arrays(samples):
    """Extract numpy arrays from samples."""
    n = len(samples)
    result = {
        # Raw magnetometer (µT)
        'mx': np.array([s.get('mx_ut', s.get('mx', 0)) for s in samples]),
        'my': np.array([s.get('my_ut', s.get('my', 0)) for s in samples]),
        'mz': np.array([s.get('mz_ut', s.get('mz', 0)) for s in samples]),
        # Accelerometer (g)
        'ax': np.array([s.get('ax_g', s.get('ax', 0)) for s in samples]),
        'ay': np.array([s.get('ay_g', s.get('ay', 0)) for s in samples]),
        'az': np.array([s.get('az_g', s.get('az', 0)) for s in samples]),
        # Gyroscope (deg/s or rad/s)
        'gx': np.array([s.get('gx_dps', s.get('gx', 0)) for s in samples]),
        'gy': np.array([s.get('gy_dps', s.get('gy', 0)) for s in samples]),
        'gz': np.array([s.get('gz_dps', s.get('gz', 0)) for s in samples]),
        # Orientation quaternion
        'qw': np.array([s.get('orientation_w', 1) for s in samples]),
        'qx': np.array([s.get('orientation_x', 0) for s in samples]),
        'qy': np.array([s.get('orientation_y', 0) for s in samples]),
        'qz': np.array([s.get('orientation_z', 0) for s in samples]),
        # Euler angles
        'yaw': np.array([s.get('euler_yaw', 0) for s in samples]),
        'pitch': np.array([s.get('euler_pitch', 0) for s in samples]),
        'roll': np.array([s.get('euler_roll', 0) for s in samples]),
        # Motion state
        'is_moving': np.array([s.get('isMoving', True) for s in samples]),
        # Timestamps
        'timestamp': np.array([s.get('timestamp', i*20) for i, s in enumerate(samples)]),
        'n': n
    }
    return result


def analyze_raw_magnetometer(arrays):
    """Analyze raw magnetometer data characteristics."""
    mx, my, mz = arrays['mx'], arrays['my'], arrays['mz']
    n = arrays['n']

    raw_mag = np.sqrt(mx**2 + my**2 + mz**2)

    print("\n" + "=" * 80)
    print("1. RAW MAGNETOMETER ANALYSIS")
    print("=" * 80)

    print(f"\nTotal samples: {n}")
    print(f"Duration: ~{n/50:.1f} seconds (assuming 50Hz)")

    print(f"\n--- Raw Magnetometer Statistics ---")
    print(f"X: min={mx.min():.1f}, max={mx.max():.1f}, range={mx.max()-mx.min():.1f} µT")
    print(f"Y: min={my.min():.1f}, max={my.max():.1f}, range={my.max()-my.min():.1f} µT")
    print(f"Z: min={mz.min():.1f}, max={mz.max():.1f}, range={mz.max()-mz.min():.1f} µT")
    print(f"\nMagnitude: mean={raw_mag.mean():.1f}, std={raw_mag.std():.1f}, range=[{raw_mag.min():.1f}, {raw_mag.max():.1f}] µT")

    # Check for expected range (~2*EARTH_MAG for full rotation)
    expected_range = 2 * EARTH_MAG
    print(f"\n--- Range Analysis ---")
    print(f"Expected range for full rotation: ~{expected_range:.0f} µT (2 × {EARTH_MAG:.0f})")
    x_coverage = (mx.max() - mx.min()) / expected_range * 100
    y_coverage = (my.max() - my.min()) / expected_range * 100
    z_coverage = (mz.max() - mz.min()) / expected_range * 100
    print(f"X coverage: {x_coverage:.0f}%")
    print(f"Y coverage: {y_coverage:.0f}%")
    print(f"Z coverage: {z_coverage:.0f}%")

    # Check if ranges are suspiciously large (indicating sensor issues or strong interference)
    if any([mx.max()-mx.min() > 200, my.max()-my.min() > 200, mz.max()-mz.min() > 200]):
        print("\n⚠️  WARNING: Range exceeds 200µT - possible sensor issue or strong magnetic interference!")

    return raw_mag


def analyze_calibration_progress(arrays):
    """Analyze why calibration completes at 100% so quickly."""
    mx, my, mz = arrays['mx'], arrays['my'], arrays['mz']
    n = arrays['n']

    print("\n" + "=" * 80)
    print("2. CALIBRATION PROGRESS ANALYSIS")
    print("=" * 80)

    # Simulate min-max calibration progress
    print("\n--- Simulating Min-Max Calibration Progress ---")

    # The calibration typically measures coverage of each axis
    # Progress = min(x_coverage, y_coverage, z_coverage) / target_coverage

    expected_range = 2 * EARTH_MAG  # ~100µT
    min_acceptable_range = 0.5 * expected_range  # 50µT minimum for "good" calibration

    # Track when each axis reaches sufficient coverage
    window_sizes = [10, 25, 50, 100, 200]

    for window in window_sizes:
        if window >= n:
            continue
        x_range = mx[:window].max() - mx[:window].min()
        y_range = my[:window].max() - my[:window].min()
        z_range = mz[:window].max() - mz[:window].min()

        # Progress based on coverage
        x_prog = min(100, x_range / min_acceptable_range * 100)
        y_prog = min(100, y_range / min_acceptable_range * 100)
        z_prog = min(100, z_range / min_acceptable_range * 100)
        overall = min(x_prog, y_prog, z_prog)

        print(f"After {window} samples ({window/50:.1f}s): X={x_prog:.0f}% Y={y_prog:.0f}% Z={z_prog:.0f}% → Overall={overall:.0f}%")
        print(f"  Ranges: X={x_range:.0f}µT Y={y_range:.0f}µT Z={z_range:.0f}µT")

    # Find when 100% is reached
    for i in range(10, n):
        x_range = mx[:i].max() - mx[:i].min()
        y_range = my[:i].max() - my[:i].min()
        z_range = mz[:i].max() - mz[:i].min()

        if x_range >= min_acceptable_range and y_range >= min_acceptable_range and z_range >= min_acceptable_range:
            print(f"\n100% reached at sample {i} ({i/50:.1f}s)")
            print(f"  Ranges at completion: X={x_range:.0f}µT Y={y_range:.0f}µT Z={z_range:.0f}µT")
            break

    # Issue: Large ranges mean quick completion but poor calibration
    final_x = mx.max() - mx.min()
    final_y = my.max() - my.min()
    final_z = mz.max() - mz.min()
    print(f"\nFinal ranges: X={final_x:.0f}µT Y={final_y:.0f}µT Z={final_z:.0f}µT")

    if max(final_x, final_y, final_z) > 150:
        print("\n⚠️  ISSUE IDENTIFIED: Ranges are too large!")
        print("   Expected ~100µT for pure Earth field rotation")
        print("   Large ranges suggest:")
        print("   - Strong hard iron offset (device magnetic bias)")
        print("   - External magnetic interference")
        print("   - Sensor saturation or non-linearity")


def compute_iron_calibration(arrays):
    """Compute and analyze hard/soft iron calibration."""
    mx, my, mz = arrays['mx'], arrays['my'], arrays['mz']

    print("\n" + "=" * 80)
    print("3. HARD/SOFT IRON CALIBRATION ANALYSIS")
    print("=" * 80)

    # Min-max hard iron
    hard_iron = np.array([
        (mx.max() + mx.min()) / 2,
        (my.max() + my.min()) / 2,
        (mz.max() + mz.min()) / 2
    ])

    ranges = np.array([
        mx.max() - mx.min(),
        my.max() - my.min(),
        mz.max() - mz.min()
    ])

    # Soft iron scale to normalize to expected magnitude
    expected_range = 2 * EARTH_MAG
    soft_iron_scale = expected_range / ranges

    print(f"\n--- Min-Max Calibration Results ---")
    print(f"Hard iron offset: [{hard_iron[0]:.1f}, {hard_iron[1]:.1f}, {hard_iron[2]:.1f}] µT")
    print(f"Hard iron magnitude: |{np.linalg.norm(hard_iron):.1f}| µT")
    print(f"\nRaw ranges: [{ranges[0]:.1f}, {ranges[1]:.1f}, {ranges[2]:.1f}] µT")
    print(f"Soft iron scale: [{soft_iron_scale[0]:.3f}, {soft_iron_scale[1]:.3f}, {soft_iron_scale[2]:.3f}]")

    # Check sphericity
    avg_range = np.mean(ranges)
    sphericity = np.min(ranges) / np.max(ranges)
    print(f"\nSphericity: {sphericity:.2f} (1.0 = perfect sphere)")

    axis_deviation = (ranges - avg_range) / avg_range * 100
    print(f"Axis deviation from mean: X={axis_deviation[0]:+.1f}%, Y={axis_deviation[1]:+.1f}%, Z={axis_deviation[2]:+.1f}%")

    # Apply calibration
    corrected = np.array([
        (mx - hard_iron[0]) * soft_iron_scale[0],
        (my - hard_iron[1]) * soft_iron_scale[1],
        (mz - hard_iron[2]) * soft_iron_scale[2]
    ])

    corr_mag = np.sqrt(corrected[0]**2 + corrected[1]**2 + corrected[2]**2)

    print(f"\n--- Corrected Magnetometer ---")
    print(f"Magnitude: mean={corr_mag.mean():.1f}, std={corr_mag.std():.1f} µT")
    print(f"Expected: {EARTH_MAG:.1f} µT")
    print(f"Error: {abs(corr_mag.mean() - EARTH_MAG) / EARTH_MAG * 100:.1f}%")

    return hard_iron, soft_iron_scale, corrected


def analyze_earth_residual(arrays, corrected):
    """Analyze Earth field residual."""
    qw, qx, qy, qz = arrays['qw'], arrays['qx'], arrays['qy'], arrays['qz']
    n = arrays['n']

    print("\n" + "=" * 80)
    print("4. EARTH FIELD RESIDUAL ANALYSIS")
    print("=" * 80)

    def quat_to_rotation(qw, qx, qy, qz):
        """Convert quaternion to rotation matrix."""
        norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
        qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm

        return np.array([
            [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
            [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
            [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
        ])

    residuals = []
    earth_devices = []
    dot_products = []

    for i in range(n):
        # Rotate Earth field from world to device frame
        R = quat_to_rotation(qw[i], qx[i], qy[i], qz[i])
        earth_device = R @ EARTH_WORLD
        earth_devices.append(earth_device)

        # Corrected magnetometer reading
        mag_vec = np.array([corrected[0][i], corrected[1][i], corrected[2][i]])

        # Residual
        residual = mag_vec - earth_device
        residuals.append(residual)

        # Dot product (alignment quality)
        mag_norm = np.linalg.norm(mag_vec)
        earth_norm = np.linalg.norm(earth_device)
        if mag_norm > 0 and earth_norm > 0:
            dot = np.dot(mag_vec, earth_device) / (mag_norm * earth_norm)
            dot_products.append(dot)

    residuals = np.array(residuals)
    earth_devices = np.array(earth_devices)
    residual_mag = np.linalg.norm(residuals, axis=1)

    print(f"\n--- Residual Statistics ---")
    print(f"Mean residual: {residual_mag.mean():.1f} µT")
    print(f"Std residual: {residual_mag.std():.1f} µT")
    print(f"Range: [{residual_mag.min():.1f}, {residual_mag.max():.1f}] µT")
    print(f"\nTarget (no magnets): <10 µT")

    if len(dot_products) > 0:
        dot_products = np.array(dot_products)
        print(f"\n--- Alignment Quality ---")
        print(f"Mean dot product: {dot_products.mean():.3f} (1.0 = perfect alignment)")
        print(f"Samples with good alignment (dot > 0.9): {(dot_products > 0.9).sum()}/{len(dot_products)} ({(dot_products > 0.9).mean()*100:.1f}%)")

    return residual_mag, earth_devices


def analyze_hv_ratio(arrays, corrected):
    """Analyze H/V component ratio - key indicator of calibration quality."""
    ax, ay, az = arrays['ax'], arrays['ay'], arrays['az']
    n = arrays['n']

    print("\n" + "=" * 80)
    print("5. H/V COMPONENT ANALYSIS (KEY DIAGNOSTIC)")
    print("=" * 80)

    def accel_to_roll_pitch(ax, ay, az):
        """Get roll and pitch from accelerometer."""
        a_norm = np.sqrt(ax**2 + ay**2 + az**2)
        if a_norm < 0.1:
            return 0, 0
        ax, ay, az = ax/a_norm, ay/a_norm, az/a_norm
        roll = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        return roll, pitch

    def tilt_compensate(mx, my, mz, roll, pitch):
        """Tilt-compensate magnetometer."""
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)

        mx_h = mx * cos_p + my * sin_r * sin_p + mz * cos_r * sin_p
        my_h = my * cos_r - mz * sin_r
        mz_h = -mx * sin_p + my * cos_r * sin_p + mz * cos_r * cos_p

        return mx_h, my_h, mz_h

    h_components = []
    v_components = []

    for i in range(n):
        roll, pitch = accel_to_roll_pitch(ax[i], ay[i], az[i])
        mx_h, my_h, mz_h = tilt_compensate(
            corrected[0][i], corrected[1][i], corrected[2][i],
            roll, pitch
        )
        h_components.append(np.sqrt(mx_h**2 + my_h**2))
        v_components.append(mz_h)

    h_components = np.array(h_components)
    v_components = np.array(v_components)

    print(f"\n--- Tilt-Compensated Components ---")
    print(f"Horizontal (H): mean={h_components.mean():.1f} µT (expected {EARTH_H:.1f})")
    print(f"Vertical (V):   mean={v_components.mean():.1f} µT (expected {EARTH_V:.1f})")

    # Avoid division by zero
    valid_mask = np.abs(v_components) > 1
    if valid_mask.sum() > 0:
        hv_ratio = np.abs(h_components[valid_mask] / v_components[valid_mask])
        print(f"\nH/V ratio: mean={hv_ratio.mean():.2f} (expected {EARTH_H/EARTH_V:.2f})")

        if hv_ratio.mean() > 0.8:
            print("\n⚠️  H/V RATIO INVERTED!")
            print("   This indicates the soft iron calibration is distorting field direction.")
            print("   The diagonal soft iron matrix cannot correct cross-axis coupling.")

    return h_components, v_components


def analyze_orientation_aware_calibration(arrays):
    """Attempt orientation-aware calibration and analyze why it might fail."""
    mx, my, mz = arrays['mx'], arrays['my'], arrays['mz']
    ax, ay, az = arrays['ax'], arrays['ay'], arrays['az']
    n = arrays['n']

    print("\n" + "=" * 80)
    print("6. ORIENTATION-AWARE CALIBRATION ATTEMPT")
    print("=" * 80)

    def accel_to_roll_pitch(ax, ay, az):
        a_norm = np.sqrt(ax**2 + ay**2 + az**2)
        if a_norm < 0.1:
            return 0, 0
        ax, ay, az = ax/a_norm, ay/a_norm, az/a_norm
        roll = np.arctan2(ay, az)
        pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
        return roll, pitch

    def euler_to_rotation(roll, pitch, yaw):
        cy, sy = np.cos(yaw), np.sin(yaw)
        cp, sp = np.cos(pitch), np.sin(pitch)
        cr, sr = np.cos(roll), np.sin(roll)
        return np.array([
            [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
            [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
            [-sp, cp*sr, cp*cr]
        ])

    def tilt_compensate(mx, my, mz, roll, pitch):
        cos_r, sin_r = np.cos(roll), np.sin(roll)
        cos_p, sin_p = np.cos(pitch), np.sin(pitch)
        mx_h = mx * cos_p + my * sin_r * sin_p + mz * cos_r * sin_p
        my_h = my * cos_r - mz * sin_r
        return mx_h, my_h

    # Use subset for optimization
    step = max(1, n // 200)
    indices = list(range(0, n, step))[:200]

    mx_sub = mx[indices]
    my_sub = my[indices]
    mz_sub = mz[indices]
    ax_sub = ax[indices]
    ay_sub = ay[indices]
    az_sub = az[indices]

    print(f"Using {len(indices)} samples for optimization")

    def residual_func(params):
        offset = params[:3]
        S = params[3:12].reshape(3, 3)

        raw = np.array([mx_sub, my_sub, mz_sub])
        centered = raw - offset.reshape(3, 1)
        corrected = S @ centered

        residuals = []
        for i in range(len(indices)):
            roll, pitch = accel_to_roll_pitch(ax_sub[i], ay_sub[i], az_sub[i])
            cmx, cmy, cmz = corrected[0][i], corrected[1][i], corrected[2][i]

            mx_h, my_h = tilt_compensate(cmx, cmy, cmz, roll, pitch)
            yaw = np.arctan2(-my_h, mx_h)

            R = euler_to_rotation(roll, pitch, yaw)
            earth_device = R.T @ EARTH_WORLD
            mag_corr = np.array([cmx, cmy, cmz])
            diff = mag_corr - earth_device
            residuals.extend(diff.tolist())

        return np.array(residuals)

    # Initial guess from min-max
    offset_init = np.array([(mx.max()+mx.min())/2, (my.max()+my.min())/2, (mz.max()+mz.min())/2])
    S_init = np.eye(3).flatten()
    x0 = np.concatenate([offset_init, S_init])

    print("\nRunning optimization...")

    try:
        result = least_squares(residual_func, x0, method='lm', max_nfev=10000, verbose=0)

        offset = result.x[:3]
        S = result.x[3:12].reshape(3, 3)

        print(f"\n--- Optimization Results ---")
        print(f"Hard iron: [{offset[0]:.2f}, {offset[1]:.2f}, {offset[2]:.2f}] µT")
        print(f"\nSoft iron matrix:")
        for row in S:
            print(f"  [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}]")

        # Apply and evaluate
        raw = np.array([mx, my, mz])
        centered = raw - offset.reshape(3, 1)
        corrected = S @ centered

        corr_mag = np.sqrt(corrected[0]**2 + corrected[1]**2 + corrected[2]**2)
        print(f"\nCorrected magnitude: {corr_mag.mean():.1f} ± {corr_mag.std():.1f} µT (expected {EARTH_MAG:.1f})")
        print(f"Magnitude error: {abs(corr_mag.mean() - EARTH_MAG) / EARTH_MAG * 100:.1f}%")

        # Final residual
        final_residual = np.sqrt(np.mean(result.fun**2))
        print(f"\nFinal RMS residual: {final_residual:.1f} µT")

        if final_residual > 20:
            print("\n⚠️  HIGH RESIDUAL - Calibration failed to converge!")
            print("   Possible causes:")
            print("   1. Magnetic interference in environment")
            print("   2. Sensor hardware issues")
            print("   3. Accelerometer-magnetometer misalignment")
            print("   4. Non-linear sensor distortion")

        return offset, S, corrected, final_residual

    except Exception as e:
        print(f"\n❌ Optimization failed: {e}")
        return None, None, None, None


def analyze_sensor_data_quality(arrays):
    """Check for sensor data quality issues."""
    mx, my, mz = arrays['mx'], arrays['my'], arrays['mz']
    ax, ay, az = arrays['ax'], arrays['ay'], arrays['az']
    gx, gy, gz = arrays['gx'], arrays['gy'], arrays['gz']

    print("\n" + "=" * 80)
    print("7. SENSOR DATA QUALITY CHECK")
    print("=" * 80)

    # Accelerometer sanity check
    accel_mag = np.sqrt(ax**2 + ay**2 + az**2)
    print(f"\n--- Accelerometer ---")
    print(f"Magnitude: mean={accel_mag.mean():.3f}g, std={accel_mag.std():.3f}g")
    print(f"Expected: ~1.0g when stationary")

    if abs(accel_mag.mean() - 1.0) > 0.1:
        print("⚠️  Accelerometer magnitude off - check units or calibration")

    # Gyroscope check
    gyro_mag = np.sqrt(gx**2 + gy**2 + gz**2)
    print(f"\n--- Gyroscope ---")
    print(f"Magnitude: mean={gyro_mag.mean():.1f}, std={gyro_mag.std():.1f}")

    # Check for magnetometer saturation
    print(f"\n--- Magnetometer Saturation Check ---")
    raw_mag = np.sqrt(mx**2 + my**2 + mz**2)

    # Typical magnetometer range is ±4900µT or ±1600µT
    if raw_mag.max() > 1500:
        print(f"⚠️  Max magnitude {raw_mag.max():.0f}µT - possible saturation!")
    else:
        print(f"Max magnitude: {raw_mag.max():.0f}µT - within typical range")

    # Check for constant/stuck values
    print(f"\n--- Value Variation Check ---")
    for name, data in [('mx', mx), ('my', my), ('mz', mz)]:
        unique = len(np.unique(np.round(data, 1)))
        if unique < 10:
            print(f"⚠️  {name}: Only {unique} unique values - sensor may be stuck!")
        else:
            print(f"{name}: {unique} unique values - OK")


def generate_diagnostic_plots(arrays, corrected, residual_mag):
    """Generate diagnostic visualization."""
    mx, my, mz = arrays['mx'], arrays['my'], arrays['mz']
    n = arrays['n']
    ts = arrays['timestamp']

    # Normalize timestamp to seconds from start
    ts_sec = (ts - ts[0]) / 1000.0

    fig, axes = plt.subplots(3, 2, figsize=(14, 10))
    fig.suptitle('Magnetometer Calibration Diagnostic - Session 2025-12-30', fontsize=14)

    # Raw magnetometer
    ax1 = axes[0, 0]
    ax1.plot(ts_sec, mx, label='X', alpha=0.7)
    ax1.plot(ts_sec, my, label='Y', alpha=0.7)
    ax1.plot(ts_sec, mz, label='Z', alpha=0.7)
    ax1.set_ylabel('µT')
    ax1.set_title('Raw Magnetometer')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Raw magnitude
    ax2 = axes[0, 1]
    raw_mag = np.sqrt(mx**2 + my**2 + mz**2)
    ax2.plot(ts_sec, raw_mag, 'b-', alpha=0.7)
    ax2.axhline(EARTH_MAG, color='g', linestyle='--', label=f'Expected {EARTH_MAG:.0f}µT')
    ax2.set_ylabel('µT')
    ax2.set_title('Raw Magnitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Corrected magnetometer
    ax3 = axes[1, 0]
    ax3.plot(ts_sec, corrected[0], label='X', alpha=0.7)
    ax3.plot(ts_sec, corrected[1], label='Y', alpha=0.7)
    ax3.plot(ts_sec, corrected[2], label='Z', alpha=0.7)
    ax3.set_ylabel('µT')
    ax3.set_title('Corrected Magnetometer')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Corrected magnitude
    ax4 = axes[1, 1]
    corr_mag = np.sqrt(corrected[0]**2 + corrected[1]**2 + corrected[2]**2)
    ax4.plot(ts_sec, corr_mag, 'b-', alpha=0.7)
    ax4.axhline(EARTH_MAG, color='g', linestyle='--', label=f'Expected {EARTH_MAG:.0f}µT')
    ax4.set_ylabel('µT')
    ax4.set_title('Corrected Magnitude')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    # Residual
    ax5 = axes[2, 0]
    ax5.plot(ts_sec, residual_mag, 'r-', alpha=0.7)
    ax5.axhline(10, color='g', linestyle='--', label='Target <10µT')
    ax5.axhline(60, color='orange', linestyle='--', label='Trust threshold 60µT')
    ax5.set_xlabel('Time (s)')
    ax5.set_ylabel('µT')
    ax5.set_title('Earth Residual (should be ~0 without magnets)')
    ax5.legend()
    ax5.grid(True, alpha=0.3)

    # 3D scatter of raw mag
    ax6 = axes[2, 1]
    scatter = ax6.scatter(mx[::10], my[::10], c=mz[::10], cmap='coolwarm', alpha=0.5, s=10)
    ax6.set_xlabel('X (µT)')
    ax6.set_ylabel('Y (µT)')
    ax6.set_title('Raw Mag XY (color=Z)')
    plt.colorbar(scatter, ax=ax6, label='Z (µT)')
    ax6.axis('equal')
    ax6.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path('ml/mag_calibration_diagnostic.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nDiagnostic plot saved to: {output_path}")
    plt.close()

    return output_path


def main():
    # Load session data
    session_path = Path('data/GAMBIT/2025-12-30T22_46_28.771Z.json')

    if not session_path.exists():
        print(f"Session file not found: {session_path}")
        return

    print(f"\nLoading: {session_path}")
    data = load_session(session_path)

    # Check session metadata
    if 'metadata' in data:
        print(f"\n--- Session Metadata ---")
        for key, value in data['metadata'].items():
            print(f"  {key}: {value}")

    samples = data.get('samples', [])
    if not samples:
        print("No samples found in session!")
        return

    arrays = extract_arrays(samples)
    print(f"\nExtracted {arrays['n']} samples")

    # Run analyses
    raw_mag = analyze_raw_magnetometer(arrays)
    analyze_calibration_progress(arrays)
    hard_iron, soft_iron, corrected = compute_iron_calibration(arrays)
    residual_mag, earth_devices = analyze_earth_residual(arrays, corrected)
    h_comp, v_comp = analyze_hv_ratio(arrays, corrected)

    # Try orientation-aware calibration
    offset_oa, S_oa, corrected_oa, res_oa = analyze_orientation_aware_calibration(arrays)

    analyze_sensor_data_quality(arrays)

    # Generate plots
    plot_path = generate_diagnostic_plots(arrays, corrected, residual_mag)

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY & RECOMMENDATIONS")
    print("=" * 80)

    print("\n--- Key Findings ---")
    print(f"1. Calibration completion: Too fast due to large magnetometer ranges")
    print(f"2. Earth residual: {residual_mag.mean():.0f}µT (target: <10µT)")
    print(f"3. H/V ratio: Likely inverted (indicates direction distortion)")

    if residual_mag.mean() > 40:
        print("\n--- Root Cause Analysis ---")
        print("The high residual despite calibration suggests:")
        print("1. Diagonal soft iron correction is insufficient")
        print("2. The sensor has significant cross-axis coupling")
        print("3. Full 3x3 soft iron matrix is required")
        print("4. May need factory calibration data or manual alignment")

    print("\n--- Recommendations ---")
    print("1. Verify sensor is not near magnetic interference sources")
    print("2. Consider implementing full ellipsoid fit with orientation constraint")
    print("3. Add quality gate: don't trust calibration if H/V ratio > 0.8")
    print("4. Investigate if sensor axes are correctly oriented")


if __name__ == '__main__':
    main()
