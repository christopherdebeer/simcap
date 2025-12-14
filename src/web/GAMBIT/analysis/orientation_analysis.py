#!/usr/bin/env python3
"""
Orientation Calibration Analysis Script

Analyzes raw sensor data and AHRS outputs to diagnose axis mapping issues.
Uses only standard library (no numpy required).
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import math

# Euler angle extraction orders to test
EULER_ORDERS = ['ZYX', 'YXZ', 'XYZ', 'XZY', 'YZX', 'ZXY']


def mean(arr: List[float]) -> float:
    """Calculate mean of list."""
    if not arr:
        return 0
    return sum(arr) / len(arr)


def std(arr: List[float]) -> float:
    """Calculate standard deviation of list."""
    if len(arr) < 2:
        return 0
    m = mean(arr)
    variance = sum((x - m) ** 2 for x in arr) / len(arr)
    return math.sqrt(variance)


def quaternion_to_euler_zyx(q: Dict[str, float]) -> Dict[str, float]:
    """
    Convert quaternion to Euler angles using ZYX (aerospace) convention.
    This is what the MadgwickAHRS filter uses.
    """
    w, x, y, z = q['w'], q['x'], q['y'], q['z']

    # Normalize
    n = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n

    # Roll (X)
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    # Pitch (Y)
    sinp = 2 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (Z)
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return {
        'roll': math.degrees(roll),
        'pitch': math.degrees(pitch),
        'yaw': math.degrees(yaw)
    }


def quaternion_to_euler_yxz(q: Dict[str, float]) -> Dict[str, float]:
    """
    Convert quaternion to Euler using YXZ (Three.js default) convention.
    """
    w, x, y, z = q['w'], q['x'], q['y'], q['z']

    n = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n

    # Pitch (X) - computed first in YXZ order
    sinp = 2 * (w * x - y * z)
    if abs(sinp) >= 1:
        pitch = math.copysign(math.pi / 2, sinp)
    else:
        pitch = math.asin(sinp)

    # Yaw (Y)
    siny = 2 * (w * y + x * z)
    cosy = 1 - 2 * (x * x + y * y)
    yaw = math.atan2(siny, cosy)

    # Roll (Z)
    sinr = 2 * (w * z + x * y)
    cosr = 1 - 2 * (x * x + z * z)
    roll = math.atan2(sinr, cosr)

    return {
        'roll': math.degrees(roll),
        'pitch': math.degrees(pitch),
        'yaw': math.degrees(yaw)
    }


def quaternion_to_euler_xyz(q: Dict[str, float]) -> Dict[str, float]:
    """
    Convert quaternion to Euler using XYZ convention.
    """
    w, x, y, z = q['w'], q['x'], q['y'], q['z']

    n = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n

    # Roll (X)
    sinr = 2 * (w * x - y * z)
    if abs(sinr) >= 1:
        roll = math.copysign(math.pi / 2, sinr)
    else:
        roll = math.asin(sinr)

    # Pitch (Y)
    sinp = 2 * (w * y + x * z)
    cosp = 1 - 2 * (x * x + y * y)
    pitch = math.atan2(sinp, cosp)

    # Yaw (Z)
    siny = 2 * (w * z - x * y)
    cosy = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny, cosy)

    return {
        'roll': math.degrees(roll),
        'pitch': math.degrees(pitch),
        'yaw': math.degrees(yaw)
    }


def accel_to_tilt(ax: float, ay: float, az: float) -> Dict[str, float]:
    """
    Calculate tilt angles directly from accelerometer (gravity vector).
    This gives us ground truth roll and pitch (ignoring yaw).
    """
    # Normalize
    mag = math.sqrt(ax*ax + ay*ay + az*az)
    if mag < 0.01:
        return {'roll': 0, 'pitch': 0}

    ax, ay, az = ax/mag, ay/mag, az/mag

    # Roll: rotation around X axis (tilt left/right)
    roll = math.atan2(ay, az)

    # Pitch: rotation around Y axis (tilt forward/back)
    pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))

    return {
        'roll': math.degrees(roll),
        'pitch': math.degrees(pitch)
    }


def analyze_session(filepath: str) -> Dict:
    """
    Analyze a session JSON file for orientation issues.
    """
    with open(filepath, 'r') as f:
        data = json.load(f)

    samples = data.get('samples', [])
    if not samples:
        return {'error': 'No samples found'}

    results = {
        'file': filepath,
        'sample_count': len(samples),
        'analysis': []
    }

    # Analyze samples with quaternion data
    for i, sample in enumerate(samples):
        if 'orientation_w' not in sample:
            continue

        q = {
            'w': sample['orientation_w'],
            'x': sample['orientation_x'],
            'y': sample['orientation_y'],
            'z': sample['orientation_z']
        }

        # AHRS reported Euler angles
        ahrs_euler = {
            'roll': sample.get('euler_roll', 0),
            'pitch': sample.get('euler_pitch', 0),
            'yaw': sample.get('euler_yaw', 0)
        }

        # Calculate Euler angles using different conventions
        euler_zyx = quaternion_to_euler_zyx(q)
        euler_yxz = quaternion_to_euler_yxz(q)
        euler_xyz = quaternion_to_euler_xyz(q)

        # Calculate tilt from accelerometer (ground truth)
        accel_tilt = accel_to_tilt(
            sample.get('ax_g', sample.get('ax', 0) / 8192),
            sample.get('ay_g', sample.get('ay', 0) / 8192),
            sample.get('az_g', sample.get('az', 0) / 8192)
        )

        results['analysis'].append({
            'index': i,
            'quaternion': q,
            'ahrs_euler': ahrs_euler,
            'euler_zyx': euler_zyx,
            'euler_yxz': euler_yxz,
            'euler_xyz': euler_xyz,
            'accel_tilt': accel_tilt,
            'isMoving': sample.get('isMoving', False)
        })

    return results


def find_stationary_segments(analysis: List[Dict], max_samples: int = 10) -> List[Dict]:
    """
    Find segments where the device is stationary (not moving).
    Returns representative samples from stationary periods.
    """
    stationary = []
    for sample in analysis:
        if not sample.get('isMoving', True):
            stationary.append(sample)
            if len(stationary) >= max_samples:
                break
    return stationary


def compare_euler_with_accel(sample: Dict) -> Dict:
    """
    Compare AHRS Euler angles with accelerometer-derived tilt.
    This helps identify which Euler axis maps to which physical axis.
    """
    ahrs = sample['ahrs_euler']
    accel = sample['accel_tilt']

    # Try different mappings to see which one matches accelerometer best
    mappings = {
        'roll_to_roll': abs(ahrs['roll'] - accel['roll']),
        'roll_to_pitch': abs(ahrs['roll'] - accel['pitch']),
        'pitch_to_roll': abs(ahrs['pitch'] - accel['roll']),
        'pitch_to_pitch': abs(ahrs['pitch'] - accel['pitch']),
        'neg_roll_to_roll': abs(-ahrs['roll'] - accel['roll']),
        'neg_roll_to_pitch': abs(-ahrs['roll'] - accel['pitch']),
        'neg_pitch_to_roll': abs(-ahrs['pitch'] - accel['roll']),
        'neg_pitch_to_pitch': abs(-ahrs['pitch'] - accel['pitch']),
    }

    # Find best mapping
    best = min(mappings.items(), key=lambda x: x[1])

    return {
        'mappings': mappings,
        'best_mapping': best[0],
        'best_error': best[1],
        'accel_roll': accel['roll'],
        'accel_pitch': accel['pitch'],
        'ahrs_roll': ahrs['roll'],
        'ahrs_pitch': ahrs['pitch']
    }


def detect_axis_swap(samples: List[Dict]) -> Dict:
    """
    Analyze multiple samples to detect if axes are swapped or inverted.
    """
    if len(samples) < 2:
        return {'error': 'Need at least 2 samples'}

    roll_deltas = []
    pitch_deltas = []
    accel_roll_deltas = []
    accel_pitch_deltas = []

    for i in range(1, len(samples)):
        prev = samples[i-1]
        curr = samples[i]

        roll_deltas.append(curr['ahrs_euler']['roll'] - prev['ahrs_euler']['roll'])
        pitch_deltas.append(curr['ahrs_euler']['pitch'] - prev['ahrs_euler']['pitch'])
        accel_roll_deltas.append(curr['accel_tilt']['roll'] - prev['accel_tilt']['roll'])
        accel_pitch_deltas.append(curr['accel_tilt']['pitch'] - prev['accel_tilt']['pitch'])

    # Calculate correlation between AHRS deltas and accel deltas
    def correlation(a, b):
        if len(a) < 2 or len(b) < 2:
            return 0
        a_mean = mean(a)
        b_mean = mean(b)
        num = sum((ai - a_mean) * (bi - b_mean) for ai, bi in zip(a, b))
        denom_a = sum((ai - a_mean)**2 for ai in a)
        denom_b = sum((bi - b_mean)**2 for bi in b)
        denom = math.sqrt(denom_a * denom_b)
        if denom < 1e-10:
            return 0
        return num / denom

    return {
        'roll_to_accel_roll': correlation(roll_deltas, accel_roll_deltas),
        'roll_to_accel_pitch': correlation(roll_deltas, accel_pitch_deltas),
        'pitch_to_accel_roll': correlation(pitch_deltas, accel_roll_deltas),
        'pitch_to_accel_pitch': correlation(pitch_deltas, accel_pitch_deltas),
        'interpretation': {
            'roll_to_accel_roll > 0.8': 'Roll maps correctly to physical roll',
            'roll_to_accel_roll < -0.8': 'Roll is INVERTED',
            'roll_to_accel_pitch > 0.8': 'Roll is SWAPPED with pitch',
            'pitch_to_accel_pitch > 0.8': 'Pitch maps correctly to physical pitch',
            'pitch_to_accel_pitch < -0.8': 'Pitch is INVERTED',
            'pitch_to_accel_roll > 0.8': 'Pitch is SWAPPED with roll',
        }
    }


def main():
    """Main analysis routine."""
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    # Find most recent session file
    json_files = list(data_dir.glob('2025-*.json'))
    if not json_files:
        print("No session files found")
        return

    # Sort by modification time
    json_files.sort(key=lambda x: x.stat().st_mtime, reverse=True)

    print("=" * 80)
    print("ORIENTATION CALIBRATION ANALYSIS")
    print("=" * 80)

    # Analyze most recent files
    for filepath in json_files[:3]:
        print(f"\n### Analyzing: {filepath.name}")
        print("-" * 60)

        results = analyze_session(str(filepath))

        if 'error' in results:
            print(f"Error: {results['error']}")
            continue

        print(f"Sample count: {results['sample_count']}")

        analysis = results['analysis']
        if not analysis:
            print("No orientation data found")
            continue

        # Get stationary samples
        stationary = find_stationary_segments(analysis, 20)
        print(f"Stationary samples: {len(stationary)}")

        if stationary:
            # Analyze first stationary sample in detail
            sample = stationary[0]
            print(f"\n--- Sample {sample['index']} (stationary) ---")
            print(f"Quaternion: w={sample['quaternion']['w']:.4f}, "
                  f"x={sample['quaternion']['x']:.4f}, "
                  f"y={sample['quaternion']['y']:.4f}, "
                  f"z={sample['quaternion']['z']:.4f}")

            print(f"\nAHRS Euler (ZYX): roll={sample['ahrs_euler']['roll']:.1f}°, "
                  f"pitch={sample['ahrs_euler']['pitch']:.1f}°, "
                  f"yaw={sample['ahrs_euler']['yaw']:.1f}°")

            print(f"Euler ZYX recalc: roll={sample['euler_zyx']['roll']:.1f}°, "
                  f"pitch={sample['euler_zyx']['pitch']:.1f}°, "
                  f"yaw={sample['euler_zyx']['yaw']:.1f}°")

            print(f"Euler YXZ:        roll={sample['euler_yxz']['roll']:.1f}°, "
                  f"pitch={sample['euler_yxz']['pitch']:.1f}°, "
                  f"yaw={sample['euler_yxz']['yaw']:.1f}°")

            print(f"Euler XYZ:        roll={sample['euler_xyz']['roll']:.1f}°, "
                  f"pitch={sample['euler_xyz']['pitch']:.1f}°, "
                  f"yaw={sample['euler_xyz']['yaw']:.1f}°")

            print(f"\nAccelerometer tilt: roll={sample['accel_tilt']['roll']:.1f}°, "
                  f"pitch={sample['accel_tilt']['pitch']:.1f}°")

            # Compare with accelerometer
            comparison = compare_euler_with_accel(sample)
            print(f"\nBest axis mapping: {comparison['best_mapping']} "
                  f"(error: {comparison['best_error']:.1f}°)")

            # Detect axis swap over multiple samples
            if len(stationary) >= 3:
                swap_analysis = detect_axis_swap(stationary)
                print("\n--- Axis Correlation Analysis ---")
                print(f"AHRS roll  <-> accel roll:  {swap_analysis['roll_to_accel_roll']:+.2f}")
                print(f"AHRS roll  <-> accel pitch: {swap_analysis['roll_to_accel_pitch']:+.2f}")
                print(f"AHRS pitch <-> accel roll:  {swap_analysis['pitch_to_accel_roll']:+.2f}")
                print(f"AHRS pitch <-> accel pitch: {swap_analysis['pitch_to_accel_pitch']:+.2f}")

                # Interpret correlations
                print("\n--- Interpretation ---")
                for key, corr in [
                    ('roll_to_accel_roll', swap_analysis['roll_to_accel_roll']),
                    ('roll_to_accel_pitch', swap_analysis['roll_to_accel_pitch']),
                    ('pitch_to_accel_roll', swap_analysis['pitch_to_accel_roll']),
                    ('pitch_to_accel_pitch', swap_analysis['pitch_to_accel_pitch'])
                ]:
                    if abs(corr) > 0.7:
                        sign = "+" if corr > 0 else "-"
                        print(f"  {key}: {sign}{abs(corr):.2f} -> SIGNIFICANT")

    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
