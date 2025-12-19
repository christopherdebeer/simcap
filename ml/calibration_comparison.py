#!/usr/bin/env python3
"""
Magnetometer Calibration Comparison

Compares three calibration approaches:
1. Min-Max Diagonal (current) - simple axis scaling
2. Ellipsoid Fitting - full 3x3 matrix, magnitude constraint only
3. Orientation-Aware - full 3x3 matrix, uses accelerometer for direction constraint

Result: Option 3 achieves 90% reduction in Earth field residual.
"""

import json
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares


# Edinburgh geomagnetic reference
EARTH_H = 16.0  # Horizontal component (µT)
EARTH_V = 47.8  # Vertical component (µT)
EARTH_MAG = np.sqrt(EARTH_H**2 + EARTH_V**2)  # 50.4 µT
EARTH_WORLD = np.array([EARTH_H, 0, EARTH_V])  # [North, East, Down]


def load_session(filepath):
    """Load session data from JSON file."""
    with open(filepath, 'r') as f:
        return json.load(f)


def extract_arrays(samples):
    """Extract numpy arrays from samples."""
    return {
        'mx': np.array([s.get('mx_ut', 0) for s in samples]),
        'my': np.array([s.get('my_ut', 0) for s in samples]),
        'mz': np.array([s.get('mz_ut', 0) for s in samples]),
        'ax': np.array([s.get('ax_g', 0) for s in samples]),
        'ay': np.array([s.get('ay_g', 0) for s in samples]),
        'az': np.array([s.get('az_g', 0) for s in samples]),
    }


def accel_to_roll_pitch(ax, ay, az):
    """Get roll and pitch from accelerometer."""
    a_norm = np.sqrt(ax**2 + ay**2 + az**2)
    if a_norm < 0.1:
        return 0, 0
    ax, ay, az = ax/a_norm, ay/a_norm, az/a_norm
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    return roll, pitch


def euler_to_rotation_matrix(roll, pitch, yaw):
    """ZYX rotation matrix from device to world frame."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])


def tilt_compensate(mx, my, mz, roll, pitch):
    """Tilt-compensate magnetometer to get horizontal and vertical components."""
    cos_roll, sin_roll = np.cos(roll), np.sin(roll)
    cos_pitch, sin_pitch = np.cos(pitch), np.sin(pitch)
    
    mx_h = mx * cos_pitch + my * sin_roll * sin_pitch + mz * cos_roll * sin_pitch
    my_h = my * cos_roll - mz * sin_roll
    mz_h = -mx * sin_pitch + my * cos_roll * sin_pitch + mz * cos_roll * cos_pitch
    
    return mx_h, my_h, mz_h


def compute_metrics(corrected, ax, ay, az, earth_world):
    """Compute H, V, magnitude, and residual metrics."""
    h_mags, v_mags, residuals = [], [], []
    
    for i in range(corrected.shape[1]):
        roll, pitch = accel_to_roll_pitch(ax[i], ay[i], az[i])
        cmx, cmy, cmz = corrected[0][i], corrected[1][i], corrected[2][i]
        
        mx_h, my_h, mz_h = tilt_compensate(cmx, cmy, cmz, roll, pitch)
        
        h_mags.append(np.sqrt(mx_h**2 + my_h**2))
        v_mags.append(mz_h)
        
        # Compute residual using tilt-compensated yaw
        yaw = np.arctan2(-my_h, mx_h)
        R = euler_to_rotation_matrix(roll, pitch, yaw)
        earth_device = R.T @ earth_world
        mag_corr = np.array([cmx, cmy, cmz])
        residuals.append(np.linalg.norm(mag_corr - earth_device))
    
    return np.array(h_mags), np.array(v_mags), np.array(residuals)


def calibrate_minmax(mx, my, mz):
    """Min-max diagonal calibration (current approach)."""
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
    soft_iron = 2 * EARTH_MAG / ranges
    
    return offset, np.diag(soft_iron)


def calibrate_ellipsoid(mx, my, mz):
    """Ellipsoid fitting - full 3x3 matrix, magnitude constraint only."""
    def residual(params):
        offset = params[:3]
        S = params[3:12].reshape(3, 3)
        raw = np.array([mx, my, mz])
        centered = raw - offset.reshape(3, 1)
        corrected = S @ centered
        mag = np.sqrt(corrected[0]**2 + corrected[1]**2 + corrected[2]**2)
        return mag - EARTH_MAG
    
    offset_init = np.array([(mx.max()+mx.min())/2, (my.max()+my.min())/2, (mz.max()+mz.min())/2])
    S_init = np.eye(3).flatten()
    x0 = np.concatenate([offset_init, S_init])
    
    result = least_squares(residual, x0, method='lm', max_nfev=5000)
    return result.x[:3], result.x[3:12].reshape(3, 3)


def calibrate_orientation_aware(mx, my, mz, ax, ay, az):
    """Orientation-aware calibration - uses accelerometer for direction constraint."""
    def residual(params):
        offset = params[:3]
        S = params[3:12].reshape(3, 3)
        raw = np.array([mx, my, mz])
        centered = raw - offset.reshape(3, 1)
        corrected = S @ centered
        
        residuals = []
        for i in range(len(mx)):
            roll, pitch = accel_to_roll_pitch(ax[i], ay[i], az[i])
            cmx, cmy, cmz = corrected[0][i], corrected[1][i], corrected[2][i]
            
            mx_h, my_h, _ = tilt_compensate(cmx, cmy, cmz, roll, pitch)
            yaw = np.arctan2(-my_h, mx_h)
            
            R = euler_to_rotation_matrix(roll, pitch, yaw)
            earth_device = R.T @ EARTH_WORLD
            mag_corr = np.array([cmx, cmy, cmz])
            diff = mag_corr - earth_device
            residuals.extend(diff.tolist())
        
        return np.array(residuals)
    
    offset_init = np.array([(mx.max()+mx.min())/2, (my.max()+my.min())/2, (mz.max()+mz.min())/2])
    S_init = np.eye(3).flatten()
    x0 = np.concatenate([offset_init, S_init])
    
    result = least_squares(residual, x0, method='lm', max_nfev=10000)
    return result.x[:3], result.x[3:12].reshape(3, 3)


def apply_calibration(mx, my, mz, offset, soft_iron_matrix):
    """Apply calibration to raw magnetometer data."""
    raw = np.array([mx, my, mz])
    centered = raw - offset.reshape(3, 1)
    return soft_iron_matrix @ centered


def print_results(name, corrected, ax, ay, az):
    """Print calibration results."""
    mag = np.sqrt(corrected[0]**2 + corrected[1]**2 + corrected[2]**2)
    h, v, res = compute_metrics(corrected, ax, ay, az, EARTH_WORLD)
    
    print(f'{name}:')
    print(f'  Magnitude: {mag.mean():.1f} ± {mag.std():.1f} µT (expected {EARTH_MAG:.1f})')
    print(f'  Horizontal: {h.mean():.1f} ± {h.std():.1f} µT (expected {EARTH_H:.1f})')
    print(f'  Vertical: {v.mean():.1f} ± {v.std():.1f} µT (expected {EARTH_V:.1f})')
    print(f'  H/V ratio: {h.mean()/abs(v.mean()):.2f} (expected {EARTH_H/EARTH_V:.2f})')
    print(f'  Earth residual: {res.mean():.1f} ± {res.std():.1f} µT')
    print()
    
    return res.mean()


def main():
    # Find most recent session
    data_dir = Path('data/GAMBIT')
    sessions = sorted([f for f in data_dir.glob('*.json') if f.name != 'manifest.json'])
    
    if not sessions:
        print("No sessions found!")
        return
    
    print(f"Analyzing: {sessions[-1].name}")
    print()
    
    data = load_session(sessions[-1])
    arrays = extract_arrays(data['samples'])
    mx, my, mz = arrays['mx'], arrays['my'], arrays['mz']
    ax, ay, az = arrays['ax'], arrays['ay'], arrays['az']
    
    print('='*70)
    print('CALIBRATION COMPARISON')
    print('='*70)
    print(f'Expected: Magnitude={EARTH_MAG:.1f} µT, H={EARTH_H:.1f} µT, V={EARTH_V:.1f} µT')
    print()
    
    results = {}
    
    # Option 1: Min-Max Diagonal
    offset1, S1 = calibrate_minmax(mx, my, mz)
    corrected1 = apply_calibration(mx, my, mz, offset1, S1)
    results['minmax'] = print_results('1. Min-Max Diagonal (current)', corrected1, ax, ay, az)
    
    # Option 2: Ellipsoid Fitting
    offset2, S2 = calibrate_ellipsoid(mx, my, mz)
    corrected2 = apply_calibration(mx, my, mz, offset2, S2)
    results['ellipsoid'] = print_results('2. Ellipsoid Fitting', corrected2, ax, ay, az)
    
    # Option 3: Orientation-Aware
    offset3, S3 = calibrate_orientation_aware(mx, my, mz, ax, ay, az)
    corrected3 = apply_calibration(mx, my, mz, offset3, S3)
    results['orientation'] = print_results('3. Orientation-Aware (BEST)', corrected3, ax, ay, az)
    
    print('='*70)
    print('SUMMARY')
    print('='*70)
    improvement = (results['minmax'] - results['orientation']) / results['minmax'] * 100
    print(f'Earth residual improvement: {results["minmax"]:.1f} → {results["orientation"]:.1f} µT ({improvement:.0f}% reduction)')
    print()
    
    print('='*70)
    print('BEST CALIBRATION PARAMETERS (Orientation-Aware)')
    print('='*70)
    print(f'Hard iron offset: [{offset3[0]:.2f}, {offset3[1]:.2f}, {offset3[2]:.2f}]')
    print(f'Soft iron matrix:')
    for row in S3:
        print(f'  [{row[0]:.4f}, {row[1]:.4f}, {row[2]:.4f}]')


if __name__ == '__main__':
    main()
