#!/usr/bin/env python3
"""
Check orientation variation in sessions to understand Earth field estimation quality.
"""

import json
import math
from pathlib import Path
from datetime import datetime


def mean(arr):
    return sum(arr) / len(arr) if arr else 0

def std(arr):
    if len(arr) < 2:
        return 0
    m = mean(arr)
    return math.sqrt(sum((x - m) ** 2 for x in arr) / len(arr))


def analyze_orientation_variation(filepath):
    """Check how much orientation varied during session."""
    with open(filepath) as f:
        data = json.load(f)

    samples = data.get('samples', [])

    rolls = [s.get('euler_roll', 0) for s in samples if 'euler_roll' in s]
    pitches = [s.get('euler_pitch', 0) for s in samples if 'euler_pitch' in s]
    yaws = [s.get('euler_yaw', 0) for s in samples if 'euler_yaw' in s]

    print(f"\n{'='*60}")
    print(f"Orientation Variation: {Path(filepath).name}")
    print(f"{'='*60}")
    print(f"Samples: {len(samples)}")

    print(f"\n{'Axis':<10} {'Min':>10} {'Max':>10} {'Range':>10} {'Std':>10}")
    print("-" * 52)

    for name, arr in [('Roll', rolls), ('Pitch', pitches), ('Yaw', yaws)]:
        if arr:
            print(f"{name:<10} {min(arr):>10.1f} {max(arr):>10.1f} {max(arr)-min(arr):>10.1f} {std(arr):>10.1f}")

    # Check if we have enough variation for Earth field averaging
    roll_range = max(rolls) - min(rolls) if rolls else 0
    pitch_range = max(pitches) - min(pitches) if pitches else 0
    yaw_range = max(yaws) - min(yaws) if yaws else 0

    print(f"\n--- ORIENTATION COVERAGE ASSESSMENT ---")

    # For good Earth field estimation via averaging, we need rotation in multiple axes
    has_roll = roll_range > 30
    has_pitch = pitch_range > 30
    has_yaw = yaw_range > 30

    axes_covered = sum([has_roll, has_pitch, has_yaw])

    print(f"Roll coverage (>30°):  {'✓' if has_roll else '✗'} ({roll_range:.0f}°)")
    print(f"Pitch coverage (>30°): {'✓' if has_pitch else '✗'} ({pitch_range:.0f}°)")
    print(f"Yaw coverage (>30°):   {'✓' if has_yaw else '✗'} ({yaw_range:.0f}°)")

    if axes_covered >= 2:
        print(f"\n→ GOOD: {axes_covered}/3 axes have significant rotation")
        print("  Finger magnets should average out in world frame")
    else:
        print(f"\n→ POOR: Only {axes_covered}/3 axes have significant rotation")
        print("  Finger magnet field may NOT average out properly!")
        print("  This explains inflated 'Earth field' estimate")


def main():
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    for f in sorted(data_dir.glob('2025-12-15T22*.json')):
        if 'gambit' not in f.name.lower():
            analyze_orientation_variation(str(f))


if __name__ == '__main__':
    main()
