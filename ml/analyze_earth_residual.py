#!/usr/bin/env python3
"""
Earth Field Residual Analysis

Reproduces the magnetometer calibration and Earth field subtraction
to understand why residuals are high and how to minimize them.

Goal: Get Earth-subtracted residual to near zero when no finger magnets present.
"""

import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
from scipy.spatial.transform import Rotation

# Edinburgh geomagnetic reference
GEOMAG_REF = {
    'horizontal': 16.0,  # µT
    'vertical': 47.8,    # µT
    'declination': -1.5  # degrees
}
EXPECTED_MAG = np.sqrt(GEOMAG_REF['horizontal']**2 + GEOMAG_REF['vertical']**2)
print(f"Expected Earth field magnitude: {EXPECTED_MAG:.1f} µT")

# Earth field in world frame (NED: North-East-Down)
# Horizontal component points North, Vertical component points Down
EARTH_FIELD_WORLD = np.array([
    GEOMAG_REF['horizontal'],  # North
    0,                          # East
    GEOMAG_REF['vertical']      # Down
])
print(f"Earth field world (NED): {EARTH_FIELD_WORLD}")


def load_session(filepath):
    """Load session data from JSON file."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def extract_arrays(samples):
    """Extract numpy arrays from samples."""
    n = len(samples)
    
    # Raw magnetometer (already in µT from session)
    mx = np.array([s.get('mx_ut', 0) for s in samples])
    my = np.array([s.get('my_ut', 0) for s in samples])
    mz = np.array([s.get('mz_ut', 0) for s in samples])
    
    # Orientation quaternion
    qw = np.array([s.get('orientation_w', 1) for s in samples])
    qx = np.array([s.get('orientation_x', 0) for s in samples])
    qy = np.array([s.get('orientation_y', 0) for s in samples])
    qz = np.array([s.get('orientation_z', 0) for s in samples])
    
    # Motion state
    is_moving = np.array([s.get('isMoving', True) for s in samples])
    
    # Euler angles
    yaw = np.array([s.get('euler_yaw', 0) for s in samples])
    
    return {
        'mx': mx, 'my': my, 'mz': mz,
        'qw': qw, 'qx': qx, 'qy': qy, 'qz': qz,
        'is_moving': is_moving,
        'yaw': yaw,
        'n': n
    }


def compute_hard_iron_minmax(mx, my, mz):
    """Compute hard iron offset using min-max method."""
    offset = np.array([
        (np.max(mx) + np.min(mx)) / 2,
        (np.max(my) + np.min(my)) / 2,
        (np.max(mz) + np.min(mz)) / 2
    ])
    ranges = np.array([
        np.max(mx) - np.min(mx),
        np.max(my) - np.min(my),
        np.max(mz) - np.min(mz)
    ])
    return offset, ranges


def compute_soft_iron_scale(ranges, expected_range=None):
    """Compute soft iron scale factors."""
    if expected_range is None:
        expected_range = 2 * EXPECTED_MAG
    
    scale = expected_range / ranges
    return scale


def apply_iron_correction(mx, my, mz, hard_iron, soft_iron_scale):
    """Apply hard and soft iron correction."""
    corrected = np.array([
        (mx - hard_iron[0]) * soft_iron_scale[0],
        (my - hard_iron[1]) * soft_iron_scale[1],
        (mz - hard_iron[2]) * soft_iron_scale[2]
    ])
    return corrected


def quat_to_rotation_matrix(qw, qx, qy, qz):
    """Convert quaternion to rotation matrix."""
    # Normalize
    norm = np.sqrt(qw**2 + qx**2 + qy**2 + qz**2)
    qw, qx, qy, qz = qw/norm, qx/norm, qy/norm, qz/norm
    
    # Rotation matrix
    R = np.array([
        [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
        [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
        [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
    ])
    return R


def rotate_earth_to_device(earth_world, qw, qx, qy, qz):
    """Rotate Earth field from world frame to device frame."""
    R = quat_to_rotation_matrix(qw, qx, qy, qz)
    # R transforms from device to world, so R.T transforms from world to device
    earth_device = R @ earth_world
    return earth_device


def compute_residual(mag_corrected, earth_device):
    """Compute residual = corrected_mag - expected_earth."""
    residual = mag_corrected - earth_device
    return residual


def analyze_session(filepath):
    """Full analysis of a session."""
    print(f"\n{'='*60}")
    print(f"Analyzing: {filepath}")
    print('='*60)
    
    data = load_session(filepath)
    samples = data['samples']
    arrays = extract_arrays(samples)
    
    mx, my, mz = arrays['mx'], arrays['my'], arrays['mz']
    qw, qx, qy, qz = arrays['qw'], arrays['qx'], arrays['qy'], arrays['qz']
    is_moving = arrays['is_moving']
    yaw = arrays['yaw']
    n = arrays['n']
    
    print(f"\nTotal samples: {n}")
    print(f"STILL samples: {np.sum(~is_moving)}")
    
    # Raw magnetometer statistics
    raw_mag = np.sqrt(mx**2 + my**2 + mz**2)
    print(f"\n--- Raw Magnetometer ---")
    print(f"Magnitude: mean={np.mean(raw_mag):.1f}, std={np.std(raw_mag):.1f}, range=[{np.min(raw_mag):.1f}, {np.max(raw_mag):.1f}] µT")
    
    # Compute hard iron offset
    hard_iron, ranges = compute_hard_iron_minmax(mx, my, mz)
    print(f"\n--- Hard Iron (min-max) ---")
    print(f"Offset: [{hard_iron[0]:.1f}, {hard_iron[1]:.1f}, {hard_iron[2]:.1f}] µT")
    print(f"Ranges: [{ranges[0]:.1f}, {ranges[1]:.1f}, {ranges[2]:.1f}] µT")
    
    # Compute soft iron scale
    soft_iron = compute_soft_iron_scale(ranges)
    print(f"\n--- Soft Iron Scale ---")
    print(f"Scale: [{soft_iron[0]:.3f}, {soft_iron[1]:.3f}, {soft_iron[2]:.3f}]")
    
    # Apply iron correction
    corrected = apply_iron_correction(mx, my, mz, hard_iron, soft_iron)
    corr_mag = np.sqrt(corrected[0]**2 + corrected[1]**2 + corrected[2]**2)
    print(f"\n--- Iron-Corrected Magnetometer ---")
    print(f"Magnitude: mean={np.mean(corr_mag):.1f}, std={np.std(corr_mag):.1f}, range=[{np.min(corr_mag):.1f}, {np.max(corr_mag):.1f}] µT")
    print(f"Expected: {EXPECTED_MAG:.1f} µT")
    print(f"Error: {(np.mean(corr_mag) - EXPECTED_MAG) / EXPECTED_MAG * 100:.1f}%")
    
    # Compute Earth field residual for each sample
    residuals = []
    earth_devices = []
    for i in range(n):
        earth_device = rotate_earth_to_device(EARTH_FIELD_WORLD, qw[i], qx[i], qy[i], qz[i])
        earth_devices.append(earth_device)
        
        mag_vec = np.array([corrected[0][i], corrected[1][i], corrected[2][i]])
        residual = compute_residual(mag_vec, earth_device)
        residuals.append(residual)
    
    residuals = np.array(residuals)
    earth_devices = np.array(earth_devices)
    residual_mag = np.sqrt(residuals[:, 0]**2 + residuals[:, 1]**2 + residuals[:, 2]**2)
    
    print(f"\n--- Earth Field Residual ---")
    print(f"Magnitude: mean={np.mean(residual_mag):.1f}, std={np.std(residual_mag):.1f}, range=[{np.min(residual_mag):.1f}, {np.max(residual_mag):.1f}] µT")
    
    # Analyze STILL periods
    still_mask = ~is_moving
    if np.sum(still_mask) > 0:
        print(f"\n--- STILL Period Analysis ---")
        print(f"STILL samples: {np.sum(still_mask)}")
        print(f"Corrected mag (STILL): mean={np.mean(corr_mag[still_mask]):.1f} µT")
        print(f"Residual mag (STILL): mean={np.mean(residual_mag[still_mask]):.1f} µT")
        print(f"Yaw (STILL): start={yaw[still_mask][0]:.1f}°, end={yaw[still_mask][-1]:.1f}°")
        
        # Yaw drift
        still_yaw = yaw[still_mask]
        yaw_unwrapped = np.unwrap(np.radians(still_yaw))
        yaw_change = np.degrees(yaw_unwrapped[-1] - yaw_unwrapped[0])
        duration = len(still_yaw) / 50.0
        print(f"Yaw change: {yaw_change:.1f}° over {duration:.1f}s = {yaw_change/duration:.1f}°/s")
    
    # Optimization: Find better hard iron offset using scipy
    print(f"\n{'='*60}")
    print("OPTIMIZATION: Finding optimal hard iron offset")
    print('='*60)
    
    from scipy.optimize import minimize
    
    def objective(offset_delta):
        """Objective function: mean residual magnitude."""
        test_offset = hard_iron + offset_delta
        test_corrected = apply_iron_correction(mx, my, mz, test_offset, soft_iron)
        
        # Vectorized residual computation
        mag_vecs = np.array([test_corrected[0], test_corrected[1], test_corrected[2]]).T
        residuals = mag_vecs - earth_devices
        residual_mags = np.sqrt(np.sum(residuals**2, axis=1))
        return np.mean(residual_mags)
    
    # Optimize using Nelder-Mead
    result = minimize(objective, [0, 0, 0], method='Nelder-Mead', 
                      options={'maxiter': 500, 'xatol': 0.1, 'fatol': 0.1})
    
    best_offset = hard_iron + result.x
    best_residual = result.fun
    
    print(f"\nOriginal offset: [{hard_iron[0]:.1f}, {hard_iron[1]:.1f}, {hard_iron[2]:.1f}] µT")
    print(f"Optimal offset:  [{best_offset[0]:.1f}, {best_offset[1]:.1f}, {best_offset[2]:.1f}] µT")
    print(f"Offset change:   [{best_offset[0]-hard_iron[0]:.1f}, {best_offset[1]-hard_iron[1]:.1f}, {best_offset[2]-hard_iron[2]:.1f}] µT")
    print(f"\nOriginal mean residual: {np.mean(residual_mag):.1f} µT")
    print(f"Optimal mean residual:  {best_residual:.1f} µT")
    print(f"Improvement: {(np.mean(residual_mag) - best_residual):.1f} µT ({(np.mean(residual_mag) - best_residual)/np.mean(residual_mag)*100:.0f}%)")
    
    # Apply optimal offset and recompute
    opt_corrected = apply_iron_correction(mx, my, mz, best_offset, soft_iron)
    opt_corr_mag = np.sqrt(opt_corrected[0]**2 + opt_corrected[1]**2 + opt_corrected[2]**2)
    
    opt_residuals = []
    for i in range(n):
        mag_vec = np.array([opt_corrected[0][i], opt_corrected[1][i], opt_corrected[2][i]])
        residual = compute_residual(mag_vec, earth_devices[i])
        opt_residuals.append(residual)
    opt_residuals = np.array(opt_residuals)
    opt_residual_mag = np.sqrt(opt_residuals[:, 0]**2 + opt_residuals[:, 1]**2 + opt_residuals[:, 2]**2)
    
    print(f"\n--- With Optimal Offset ---")
    print(f"Corrected mag: mean={np.mean(opt_corr_mag):.1f} µT (expected {EXPECTED_MAG:.1f})")
    print(f"Residual mag: mean={np.mean(opt_residual_mag):.1f} µT")
    
    if np.sum(still_mask) > 0:
        print(f"Residual mag (STILL): mean={np.mean(opt_residual_mag[still_mask]):.1f} µT")
    
    return {
        'hard_iron': hard_iron,
        'soft_iron': soft_iron,
        'best_offset': best_offset,
        'original_residual': np.mean(residual_mag),
        'optimal_residual': best_residual,
        'corr_mag': corr_mag,
        'residual_mag': residual_mag,
        'opt_residual_mag': opt_residual_mag,
        'yaw': yaw,
        'is_moving': is_moving
    }


def main():
    # Find most recent session (exclude manifest.json)
    data_dir = Path('data/GAMBIT')
    sessions = sorted([f for f in data_dir.glob('*.json') if f.name != 'manifest.json'])
    
    if not sessions:
        print("No sessions found!")
        return
    
    print(f"Found {len(sessions)} sessions")
    print(f"Latest: {sessions[-1].name}")
    
    # Analyze most recent session
    latest = sessions[-1]
    results = analyze_session(latest)
    
    # Plot results
    fig, axes = plt.subplots(3, 1, figsize=(12, 10))
    
    # Plot 1: Corrected magnitude
    ax1 = axes[0]
    ax1.plot(results['corr_mag'], label='Iron-corrected', alpha=0.7)
    ax1.axhline(EXPECTED_MAG, color='g', linestyle='--', label=f'Expected ({EXPECTED_MAG:.1f} µT)')
    ax1.fill_between(range(len(results['is_moving'])), 0, 100, 
                     where=~results['is_moving'], alpha=0.2, color='blue', label='STILL')
    ax1.set_ylabel('Magnitude (µT)')
    ax1.set_title('Iron-Corrected Magnetometer Magnitude')
    ax1.legend()
    ax1.set_ylim(0, 100)
    
    # Plot 2: Residual magnitude
    ax2 = axes[1]
    ax2.plot(results['residual_mag'], label='Original offset', alpha=0.7)
    ax2.plot(results['opt_residual_mag'], label='Optimal offset', alpha=0.7)
    ax2.axhline(0, color='g', linestyle='--', label='Target (0 µT)')
    ax2.fill_between(range(len(results['is_moving'])), 0, 150, 
                     where=~results['is_moving'], alpha=0.2, color='blue', label='STILL')
    ax2.set_ylabel('Residual (µT)')
    ax2.set_title('Earth Field Residual (should be ~0 without finger magnets)')
    ax2.legend()
    ax2.set_ylim(0, 150)
    
    # Plot 3: Yaw
    ax3 = axes[2]
    ax3.plot(results['yaw'], label='Yaw', alpha=0.7)
    ax3.fill_between(range(len(results['is_moving'])), -180, 180, 
                     where=~results['is_moving'], alpha=0.2, color='blue', label='STILL')
    ax3.set_ylabel('Yaw (°)')
    ax3.set_xlabel('Sample')
    ax3.set_title('Yaw Angle')
    ax3.legend()
    
    plt.tight_layout()
    plt.savefig('ml/earth_residual_analysis.png', dpi=150)
    print(f"\nPlot saved to: ml/earth_residual_analysis.png")
    plt.show()


if __name__ == '__main__':
    main()
