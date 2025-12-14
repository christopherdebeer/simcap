#!/usr/bin/env python3
"""
Axis Coupling Analysis Script

Specifically analyzes orientation data to detect:
1. Axis swapping (roll/pitch/yaw confusion)
2. Sign inversions
3. Euler order mismatches
"""

import json
from pathlib import Path
import math
from typing import Dict, List


def quaternion_to_euler_zyx(w, x, y, z) -> Dict[str, float]:
    """ZYX (aerospace) convention - what MadgwickAHRS uses."""
    n = math.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/n, x/n, y/n, z/n

    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = math.atan2(sinr_cosp, cosr_cosp)

    sinp = 2 * (w * y - z * x)
    pitch = math.asin(max(-1, min(1, sinp)))

    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)

    return {'roll': math.degrees(roll), 'pitch': math.degrees(pitch), 'yaw': math.degrees(yaw)}


def accel_to_tilt(ax, ay, az) -> Dict[str, float]:
    """Get tilt angles from accelerometer (ground truth for roll/pitch)."""
    mag = math.sqrt(ax*ax + ay*ay + az*az)
    if mag < 0.01:
        return {'roll': 0, 'pitch': 0}
    ax, ay, az = ax/mag, ay/mag, az/mag
    roll = math.atan2(ay, az)
    pitch = math.atan2(-ax, math.sqrt(ay*ay + az*az))
    return {'roll': math.degrees(roll), 'pitch': math.degrees(pitch)}


def analyze_pose_transitions(filepath: str) -> None:
    """Analyze transitions between poses to detect axis issues."""
    print(f"\n{'='*80}")
    print(f"ANALYZING: {Path(filepath).name}")
    print(f"{'='*80}")

    with open(filepath, 'r') as f:
        data = json.load(f)

    samples = data.get('samples', [])
    if not samples:
        print("No samples found")
        return

    # Find samples with orientation data and group by approximate pose
    poses = []
    current_pose = None
    pose_samples = []

    for s in samples:
        if 'orientation_w' not in s:
            continue

        euler_roll = s.get('euler_roll', 0)
        euler_pitch = s.get('euler_pitch', 0)
        euler_yaw = s.get('euler_yaw', 0)

        # Classify into rough pose categories based on pitch
        if -20 < euler_pitch < 20 and -30 < euler_roll < 30:
            pose = "FLAT"
        elif euler_pitch < -30:
            pose = "PITCH_NEG"  # Pitched forward
        elif euler_pitch > 30:
            pose = "PITCH_POS"  # Pitched back
        elif euler_roll < -30:
            pose = "ROLL_NEG"   # Rolled left
        elif euler_roll > 30:
            pose = "ROLL_POS"   # Rolled right
        else:
            pose = "OTHER"

        # Detect pose changes
        if pose != current_pose:
            if pose_samples:
                # Average the pose samples
                avg_roll = sum(p['euler_roll'] for p in pose_samples) / len(pose_samples)
                avg_pitch = sum(p['euler_pitch'] for p in pose_samples) / len(pose_samples)
                avg_yaw = sum(p['euler_yaw'] for p in pose_samples) / len(pose_samples)
                avg_accel_roll = sum(p['accel_roll'] for p in pose_samples) / len(pose_samples)
                avg_accel_pitch = sum(p['accel_pitch'] for p in pose_samples) / len(pose_samples)

                poses.append({
                    'name': current_pose,
                    'count': len(pose_samples),
                    'euler_roll': avg_roll,
                    'euler_pitch': avg_pitch,
                    'euler_yaw': avg_yaw,
                    'accel_roll': avg_accel_roll,
                    'accel_pitch': avg_accel_pitch
                })

            current_pose = pose
            pose_samples = []

        # Add sample
        ax_g = s.get('ax_g', s.get('ax', 0) / 8192)
        ay_g = s.get('ay_g', s.get('ay', 0) / 8192)
        az_g = s.get('az_g', s.get('az', 0) / 8192)
        accel = accel_to_tilt(ax_g, ay_g, az_g)

        pose_samples.append({
            'euler_roll': euler_roll,
            'euler_pitch': euler_pitch,
            'euler_yaw': euler_yaw,
            'accel_roll': accel['roll'],
            'accel_pitch': accel['pitch']
        })

    # Handle last pose
    if pose_samples:
        avg_roll = sum(p['euler_roll'] for p in pose_samples) / len(pose_samples)
        avg_pitch = sum(p['euler_pitch'] for p in pose_samples) / len(pose_samples)
        avg_yaw = sum(p['euler_yaw'] for p in pose_samples) / len(pose_samples)
        avg_accel_roll = sum(p['accel_roll'] for p in pose_samples) / len(pose_samples)
        avg_accel_pitch = sum(p['accel_pitch'] for p in pose_samples) / len(pose_samples)
        poses.append({
            'name': current_pose,
            'count': len(pose_samples),
            'euler_roll': avg_roll,
            'euler_pitch': avg_pitch,
            'euler_yaw': avg_yaw,
            'accel_roll': avg_accel_roll,
            'accel_pitch': avg_accel_pitch
        })

    if not poses:
        print("No pose segments found")
        return

    print(f"\nFound {len(poses)} pose segments:")
    print("-" * 60)
    print(f"{'Pose':<12} {'Count':>6} | {'AHRS Roll':>10} {'AHRS Pitch':>11} {'AHRS Yaw':>9} | {'Accel Roll':>11} {'Accel Pitch':>12}")
    print("-" * 60)

    for p in poses:
        print(f"{p['name']:<12} {p['count']:>6} | "
              f"{p['euler_roll']:>10.1f} {p['euler_pitch']:>11.1f} {p['euler_yaw']:>9.1f} | "
              f"{p['accel_roll']:>11.1f} {p['accel_pitch']:>12.1f}")

    # Analyze for axis issues
    print("\n" + "=" * 60)
    print("AXIS MAPPING ANALYSIS")
    print("=" * 60)

    # Find FLAT pose as baseline
    flat_poses = [p for p in poses if p['name'] == 'FLAT']
    if flat_poses:
        baseline = flat_poses[0]
        print(f"\nBaseline (FLAT): AHRS r={baseline['euler_roll']:.1f}° p={baseline['euler_pitch']:.1f}°")
        print(f"                 Accel r={baseline['accel_roll']:.1f}° p={baseline['accel_pitch']:.1f}°")

        # Analyze other poses relative to baseline
        for p in poses:
            if p['name'] == 'FLAT':
                continue

            delta_ahrs_roll = p['euler_roll'] - baseline['euler_roll']
            delta_ahrs_pitch = p['euler_pitch'] - baseline['euler_pitch']
            delta_ahrs_yaw = p['euler_yaw'] - baseline['euler_yaw']
            delta_accel_roll = p['accel_roll'] - baseline['accel_roll']
            delta_accel_pitch = p['accel_pitch'] - baseline['accel_pitch']

            print(f"\n{p['name']}:")
            print(f"  AHRS deltas:  Δroll={delta_ahrs_roll:+.1f}° Δpitch={delta_ahrs_pitch:+.1f}° Δyaw={delta_ahrs_yaw:+.1f}°")
            print(f"  Accel deltas: Δroll={delta_accel_roll:+.1f}° Δpitch={delta_accel_pitch:+.1f}°")

            # Check for coupling
            expected_axis = None
            if 'PITCH' in p['name']:
                expected_axis = 'pitch'
            elif 'ROLL' in p['name']:
                expected_axis = 'roll'

            if expected_axis:
                # Check if the expected axis moved
                ahrs_deltas = {'roll': abs(delta_ahrs_roll), 'pitch': abs(delta_ahrs_pitch), 'yaw': abs(delta_ahrs_yaw)}
                accel_deltas = {'roll': abs(delta_accel_roll), 'pitch': abs(delta_accel_pitch)}

                # Which axis moved most?
                max_ahrs_axis = max(ahrs_deltas, key=ahrs_deltas.get)
                max_ahrs_delta = ahrs_deltas[max_ahrs_axis]

                # Check other axes
                other_axes_moved = []
                for axis, delta in ahrs_deltas.items():
                    if axis != expected_axis and delta > 15:
                        other_axes_moved.append(f"{axis}={delta:.1f}°")

                print(f"  Expected: {expected_axis} to change")
                print(f"  Largest AHRS change: {max_ahrs_axis} ({max_ahrs_delta:.1f}°)")

                if max_ahrs_axis != expected_axis:
                    print(f"  ⚠️  AXIS MISMATCH: Expected {expected_axis}, got {max_ahrs_axis}")

                if other_axes_moved:
                    print(f"  ⚠️  COUPLING: Other axes also moved: {', '.join(other_axes_moved)}")

    # Check for inversions
    print("\n" + "=" * 60)
    print("INVERSION CHECK")
    print("=" * 60)

    for p in poses:
        if p['name'] == 'FLAT':
            continue

        roll_match = (p['euler_roll'] > 0) == (p['accel_roll'] > 0)
        pitch_match = (p['euler_pitch'] > 0) == (p['accel_pitch'] > 0)

        if not roll_match and abs(p['accel_roll']) > 20:
            print(f"  ⚠️  {p['name']}: Roll sign mismatch (AHRS={p['euler_roll']:.1f}°, Accel={p['accel_roll']:.1f}°)")

        if not pitch_match and abs(p['accel_pitch']) > 20:
            print(f"  ⚠️  {p['name']}: Pitch sign mismatch (AHRS={p['euler_pitch']:.1f}°, Accel={p['accel_pitch']:.1f}°)")


def main():
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    # Find files with orientation data (most recent first)
    json_files = sorted(data_dir.glob('2025-12-13T21*.json'), reverse=True)

    for filepath in json_files[:5]:
        analyze_pose_transitions(str(filepath))


if __name__ == '__main__':
    main()
