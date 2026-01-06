#!/usr/bin/env python3
"""
Comprehensive analysis of the Y-axis fix session.
Validates the magnetometer axis alignment and calibration quality.

Session: 2025-12-19T15_38_28.988Z (after Y-axis negation fix)
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

# Edinburgh geomagnetic reference
H_EXP = 16.0  # Horizontal component (µT)
V_EXP = 47.8  # Vertical component (µT)
EXPECTED_MAG = np.sqrt(H_EXP**2 + V_EXP**2)

def load_session(filepath):
    """Load session data from JSON file."""
    with open(filepath) as f:
        data = json.load(f)
    return data['samples']

def extract_arrays(samples):
    """Extract numpy arrays from samples."""
    return {
        'mx': np.array([s.get('mx_ut', 0) for s in samples]),
        'my': np.array([s.get('my_ut', 0) for s in samples]),
        'mz': np.array([s.get('mz_ut', 0) for s in samples]),
        'ax': np.array([s.get('ax_g', 0) for s in samples]),
        'ay': np.array([s.get('ay_g', 0) for s in samples]),
        'az': np.array([s.get('az_g', 0) for s in samples]),
        'iron_mx': np.array([s.get('iron_mx', 0) for s in samples]),
        'iron_my': np.array([s.get('iron_my', 0) for s in samples]),
        'iron_mz': np.array([s.get('iron_mz', 0) for s in samples]),
        'qw': np.array([s.get('orientation_w', 1) for s in samples]),
        'qx': np.array([s.get('orientation_x', 0) for s in samples]),
        'qy': np.array([s.get('orientation_y', 0) for s in samples]),
        'qz': np.array([s.get('orientation_z', 0) for s in samples]),
        'euler_yaw': np.array([s.get('euler_yaw', 0) for s in samples]),
        'euler_pitch': np.array([s.get('euler_pitch', 0) for s in samples]),
        'euler_roll': np.array([s.get('euler_roll', 0) for s in samples]),
        'is_moving': np.array([s.get('isMoving', True) for s in samples]),
    }

def analyze_axis_alignment(data):
    """Analyze magnetometer-accelerometer axis alignment."""
    print("\n" + "="*60)
    print("AXIS ALIGNMENT VERIFICATION")
    print("="*60)
    
    # Correlations (should all be positive after Y-axis fix)
    corr_ax_mx = np.corrcoef(data['ax'], data['mx'])[0,1]
    corr_ay_my = np.corrcoef(data['ay'], data['my'])[0,1]
    corr_az_mz = np.corrcoef(data['az'], data['mz'])[0,1]
    
    print("\nPrimary axis correlations (should be POSITIVE):")
    print(f"  ax vs mx: {corr_ax_mx:+.3f} {'✓' if corr_ax_mx > 0 else '✗'}")
    print(f"  ay vs my: {corr_ay_my:+.3f} {'✓' if corr_ay_my > 0 else '✗'}")
    print(f"  az vs mz: {corr_az_mz:+.3f} {'✓' if corr_az_mz > 0 else '✗'}")
    
    # Cross-correlations (should be near zero)
    print("\nCross-correlations (should be ~0):")
    print(f"  ax vs my: {np.corrcoef(data['ax'], data['my'])[0,1]:+.3f}")
    print(f"  ax vs mz: {np.corrcoef(data['ax'], data['mz'])[0,1]:+.3f}")
    print(f"  ay vs mx: {np.corrcoef(data['ay'], data['mx'])[0,1]:+.3f}")
    print(f"  ay vs mz: {np.corrcoef(data['ay'], data['mz'])[0,1]:+.3f}")
    print(f"  az vs mx: {np.corrcoef(data['az'], data['mx'])[0,1]:+.3f}")
    print(f"  az vs my: {np.corrcoef(data['az'], data['my'])[0,1]:+.3f}")
    
    return all([corr_ax_mx > 0, corr_ay_my > 0, corr_az_mz > 0])

def analyze_orientation_specific(data):
    """Analyze magnetometer readings at specific orientations."""
    print("\n" + "="*60)
    print("ORIENTATION-SPECIFIC ANALYSIS")
    print("="*60)
    
    mx, my, mz = data['mx'], data['my'], data['mz']
    ax, ay, az = data['ax'], data['ay'], data['az']
    
    # Level (Z up)
    level_mask = (np.abs(az - 1.0) < 0.2) & (np.abs(ax) < 0.2) & (np.abs(ay) < 0.2)
    n_level = np.sum(level_mask)
    print(f"\nLEVEL (Z up): {n_level} samples")
    if n_level > 0:
        H_level = np.sqrt(np.mean(mx[level_mask])**2 + np.mean(my[level_mask])**2)
        V_level = np.mean(mz[level_mask])
        print(f"  Raw mag: [{np.mean(mx[level_mask]):.1f}, {np.mean(my[level_mask]):.1f}, {np.mean(mz[level_mask]):.1f}]")
        print(f"  H (XY): {H_level:.1f} µT (expected {H_EXP})")
        print(f"  V (Z):  {V_level:.1f} µT (expected {V_EXP})")
    
    # Y up
    yup_mask = (np.abs(ay - 1.0) < 0.2) & (np.abs(ax) < 0.2) & (np.abs(az) < 0.2)
    n_yup = np.sum(yup_mask)
    print(f"\nY UP: {n_yup} samples")
    if n_yup > 0:
        print(f"  Raw mag: [{np.mean(mx[yup_mask]):.1f}, {np.mean(my[yup_mask]):.1f}, {np.mean(mz[yup_mask]):.1f}]")
        print(f"  V (Y): {np.mean(my[yup_mask]):.1f} µT (expected +{V_EXP})")
    
    # X up
    xup_mask = (np.abs(ax - 1.0) < 0.2) & (np.abs(ay) < 0.2) & (np.abs(az) < 0.2)
    n_xup = np.sum(xup_mask)
    print(f"\nX UP: {n_xup} samples")
    if n_xup > 0:
        print(f"  Raw mag: [{np.mean(mx[xup_mask]):.1f}, {np.mean(my[xup_mask]):.1f}, {np.mean(mz[xup_mask]):.1f}]")
        print(f"  V (X): {np.mean(mx[xup_mask]):.1f} µT (expected +{V_EXP})")

def analyze_calibration(data):
    """Analyze hard and soft iron calibration."""
    print("\n" + "="*60)
    print("CALIBRATION ANALYSIS")
    print("="*60)
    
    mx, my, mz = data['mx'], data['my'], data['mz']
    
    # Min-max hard iron
    offset = np.array([
        (mx.max() + mx.min()) / 2,
        (my.max() + my.min()) / 2,
        (mz.max() + mz.min()) / 2
    ])
    ranges = np.array([
        mx.max() - mx.min(),
        my.max() - my.min(),
        mz.max() - mz.min()
    ])
    
    print(f"\nMin-max hard iron offset: [{offset[0]:.1f}, {offset[1]:.1f}, {offset[2]:.1f}] µT")
    print(f"Ranges: [{ranges[0]:.1f}, {ranges[1]:.1f}, {ranges[2]:.1f}] µT")
    
    # Sphericity
    sphericity = np.min(ranges) / np.max(ranges)
    quality = "good" if sphericity > 0.7 else "fair" if sphericity > 0.5 else "poor"
    print(f"Sphericity: {sphericity:.2f} ({quality})")
    
    # Soft iron scale
    expected_range = 2 * EXPECTED_MAG
    soft_scale = expected_range / ranges
    print(f"Soft iron scale: [{soft_scale[0]:.3f}, {soft_scale[1]:.3f}, {soft_scale[2]:.3f}]")
    
    return offset, soft_scale

def compute_hv_tilt_compensated(mx, my, mz, ax, ay, az):
    """Compute H/V using accelerometer-based tilt compensation."""
    a_mag = np.sqrt(ax**2 + ay**2 + az**2)
    if a_mag < 0.1:
        return None, None
    
    ax_n, ay_n, az_n = ax/a_mag, ay/a_mag, az/a_mag
    
    roll = np.arctan2(ay_n, az_n)
    pitch = np.arctan2(-ax_n, np.sqrt(ay_n**2 + az_n**2))
    
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)
    
    # Tilt-compensate
    mx_h = mx * cp + my * sr * sp + mz * cr * sp
    my_h = my * cr - mz * sr
    mz_h = -mx * sp + my * sr * cp + mz * cr * cp
    
    H = np.sqrt(mx_h**2 + my_h**2)
    V = mz_h
    
    return H, V

def analyze_hv_ratio(data, offset, soft_scale):
    """Analyze H/V ratio with different calibration methods."""
    print("\n" + "="*60)
    print("H/V RATIO ANALYSIS")
    print("="*60)
    
    mx, my, mz = data['mx'], data['my'], data['mz']
    ax, ay, az = data['ax'], data['ay'], data['az']
    
    # Apply calibration
    mx_c = (mx - offset[0]) * soft_scale[0]
    my_c = (my - offset[1]) * soft_scale[1]
    mz_c = (mz - offset[2]) * soft_scale[2]
    
    # Compute H/V for all samples
    H_all, V_all = [], []
    for i in range(len(mx)):
        H, V = compute_hv_tilt_compensated(mx_c[i], my_c[i], mz_c[i], ax[i], ay[i], az[i])
        if H is not None:
            H_all.append(H)
            V_all.append(V)
    
    H_all = np.array(H_all)
    V_all = np.array(V_all)
    
    print(f"\nAll samples ({len(H_all)}):")
    print(f"  H: mean={np.mean(H_all):.1f}, std={np.std(H_all):.1f} (expected {H_EXP})")
    print(f"  V: mean={np.mean(V_all):.1f}, std={np.std(V_all):.1f} (expected {V_EXP})")
    hv_ratio = np.mean(H_all) / abs(np.mean(V_all))
    expected_hv = H_EXP / V_EXP
    print(f"  H/V ratio: {hv_ratio:.3f} (expected {expected_hv:.3f})")
    
    # By motion state
    still_mask = ~data['is_moving']
    if np.sum(still_mask) > 0:
        # Recompute for still samples
        H_still, V_still = [], []
        still_indices = np.where(still_mask)[0]
        for i in still_indices:
            H, V = compute_hv_tilt_compensated(mx_c[i], my_c[i], mz_c[i], ax[i], ay[i], az[i])
            if H is not None:
                H_still.append(H)
                V_still.append(V)
        
        H_still = np.array(H_still)
        V_still = np.array(V_still)
        
        print(f"\nWhen STILL ({len(H_still)} samples):")
        print(f"  H: mean={np.mean(H_still):.1f}, std={np.std(H_still):.1f}")
        print(f"  V: mean={np.mean(V_still):.1f}, std={np.std(V_still):.1f}")
        print(f"  H/V ratio: {np.mean(H_still)/abs(np.mean(V_still)):.3f}")
    
    return H_all, V_all

def analyze_earth_residual(data, offset, soft_scale):
    """Analyze Earth field residual."""
    print("\n" + "="*60)
    print("EARTH RESIDUAL ANALYSIS")
    print("="*60)
    
    mx, my, mz = data['mx'], data['my'], data['mz']
    ax, ay, az = data['ax'], data['ay'], data['az']
    qw, qx, qy, qz = data['qw'], data['qx'], data['qy'], data['qz']
    
    # Apply calibration
    mx_c = (mx - offset[0]) * soft_scale[0]
    my_c = (my - offset[1]) * soft_scale[1]
    mz_c = (mz - offset[2]) * soft_scale[2]
    
    # Compute residuals using quaternion orientation
    def quat_to_rotation_matrix(w, x, y, z):
        return np.array([
            [1 - 2*(y**2 + z**2), 2*(x*y - w*z), 2*(x*z + w*y)],
            [2*(x*y + w*z), 1 - 2*(x**2 + z**2), 2*(y*z - w*x)],
            [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x**2 + y**2)]
        ])
    
    residual_mags = []
    for i in range(len(mx)):
        # Earth field in NED frame
        earth_ned = np.array([H_EXP, 0, V_EXP])
        
        # Rotate to body frame
        R = quat_to_rotation_matrix(qw[i], qx[i], qy[i], qz[i])
        earth_body = R.T @ earth_ned
        
        # Measured field
        measured = np.array([mx_c[i], my_c[i], mz_c[i]])
        
        # Residual
        residual = measured - earth_body
        residual_mags.append(np.linalg.norm(residual))
    
    residual_mags = np.array(residual_mags)
    
    print(f"\nAll samples:")
    print(f"  Residual: mean={np.mean(residual_mags):.1f}, std={np.std(residual_mags):.1f}")
    print(f"  Range: [{np.min(residual_mags):.1f}, {np.max(residual_mags):.1f}]")
    
    # By motion state
    still_mask = ~data['is_moving']
    moving_mask = data['is_moving']
    
    if np.sum(still_mask) > 0:
        print(f"\nWhen STILL ({np.sum(still_mask)} samples):")
        print(f"  Residual: mean={np.mean(residual_mags[still_mask]):.1f}, std={np.std(residual_mags[still_mask]):.1f}")
    
    if np.sum(moving_mask) > 0:
        print(f"\nWhen MOVING ({np.sum(moving_mask)} samples):")
        print(f"  Residual: mean={np.mean(residual_mags[moving_mask]):.1f}, std={np.std(residual_mags[moving_mask]):.1f}")
    
    return residual_mags

def main():
    """Main analysis function."""
    session_path = Path("data/GAMBIT/2025-12-19T15_38_28.988Z.json")
    
    print("="*60)
    print("Y-AXIS FIX SESSION ANALYSIS")
    print(f"Session: {session_path.name}")
    print("="*60)
    
    # Load data
    samples = load_session(session_path)
    print(f"\nTotal samples: {len(samples)}")
    
    data = extract_arrays(samples)
    
    # Run analyses
    axis_ok = analyze_axis_alignment(data)
    analyze_orientation_specific(data)
    offset, soft_scale = analyze_calibration(data)
    H_all, V_all = analyze_hv_ratio(data, offset, soft_scale)
    residual_mags = analyze_earth_residual(data, offset, soft_scale)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    print(f"\n✓ Axis alignment: {'CORRECT' if axis_ok else 'INCORRECT'}")
    
    hv_ratio = np.mean(H_all) / abs(np.mean(V_all))
    expected_hv = H_EXP / V_EXP
    hv_ok = abs(hv_ratio - expected_hv) < 0.3
    print(f"{'✓' if hv_ok else '⚠'} H/V ratio: {hv_ratio:.3f} (expected {expected_hv:.3f})")
    
    still_mask = ~data['is_moving']
    still_residual = np.mean(residual_mags[still_mask]) if np.sum(still_mask) > 0 else np.mean(residual_mags)
    residual_ok = still_residual < 30
    print(f"{'✓' if residual_ok else '⚠'} Earth residual (still): {still_residual:.1f} µT (target <30)")
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()
