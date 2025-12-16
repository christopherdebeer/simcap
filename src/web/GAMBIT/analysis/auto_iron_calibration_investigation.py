#!/usr/bin/env python3
"""
Automatic Hard Iron Calibration Investigation

Goal: Can we estimate hard iron offset automatically from streaming data,
even when finger magnets are present?

Key Insight from Earth Field Investigation:
- Earth field (constant in world) → averages to true value in world frame
- Hard iron (constant in sensor) → rotates with device → averages toward zero in world frame
- Finger magnets (sensor frame) → rotates with device → averages toward zero in world frame

NEW INSIGHT for Hard Iron:
After Earth subtraction, the residual in sensor frame is:
    residual = raw - Earth_in_sensor
    residual ≈ hard_iron + finger_magnets

If finger positions vary over time, their contribution should average toward zero,
leaving the hard iron offset.

Approaches to Test:
1. Sensor-frame residual averaging (after Earth subtraction)
2. Gyroscope-based stationary detection (hard iron dominates when still)
3. World-frame Earth + sensor-frame residual decomposition
"""

import json
import math
from pathlib import Path
from collections import defaultdict
from typing import List, Dict, Optional, Tuple


def mean(arr):
    return sum(arr) / len(arr) if arr else 0

def std(arr):
    if len(arr) < 2:
        return 0
    m = mean(arr)
    return math.sqrt(sum((x - m) ** 2 for x in arr) / len(arr))

def percentile(arr, p):
    if not arr:
        return 0
    s = sorted(arr)
    k = (len(s) - 1) * p / 100
    f, c = int(k), min(int(k) + 1, len(s) - 1)
    return s[f] * (c - k) + s[c] * (k - f) if f != c else s[f]

def mag3(x, y, z):
    return math.sqrt(x*x + y*y + z*z)

def mat_vec(M, v):
    return [sum(M[i][j] * v[j] for j in range(3)) for i in range(3)]

def transpose(M):
    return [[M[j][i] for j in range(3)] for i in range(3)]

def quat_to_mat(w, x, y, z):
    n = math.sqrt(w*w + x*x + y*y + z*z)
    if n > 0:
        w, x, y, z = w/n, x/n, y/n, z/n
    return [
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ]


class AutoIronCalibrator:
    """
    Estimates hard iron offset automatically from streaming data.

    Two-phase approach:
    1. First estimate Earth field using world-frame averaging
    2. Then estimate hard iron from sensor-frame residual averaging
    """

    def __init__(self, earth_window=200, iron_window=500):
        # Earth estimation (world frame)
        self.earth_window = earth_window
        self.world_samples = []
        self.earth_world = [0, 0, 0]

        # Hard iron estimation (sensor frame)
        self.iron_window = iron_window
        self.residual_samples = []  # Sensor-frame residuals
        self.hard_iron_estimate = [0, 0, 0]

        # Statistics
        self.total_samples = 0

    def update(self, mx_ut, my_ut, mz_ut, qw, qx, qy, qz) -> Dict:
        """
        Process a sample and update both Earth and hard iron estimates.

        Returns dict with current estimates and metrics.
        """
        self.total_samples += 1

        # === Phase 1: Earth Field Estimation ===
        # Transform raw to world frame
        R = quat_to_mat(qw, qx, qy, qz)
        R_T = transpose(R)
        world = mat_vec(R_T, [mx_ut, my_ut, mz_ut])

        self.world_samples.append(world)
        if len(self.world_samples) > self.earth_window:
            self.world_samples.pop(0)

        # Update Earth estimate from world-frame average
        if len(self.world_samples) >= 50:
            self.earth_world = [
                mean([s[0] for s in self.world_samples]),
                mean([s[1] for s in self.world_samples]),
                mean([s[2] for s in self.world_samples])
            ]

        # === Phase 2: Hard Iron Estimation ===
        # Compute residual in sensor frame after Earth subtraction
        earth_sensor = mat_vec(R, self.earth_world)
        residual = [
            mx_ut - earth_sensor[0],
            my_ut - earth_sensor[1],
            mz_ut - earth_sensor[2]
        ]

        self.residual_samples.append(residual)
        if len(self.residual_samples) > self.iron_window:
            self.residual_samples.pop(0)

        # Update hard iron estimate from sensor-frame residual average
        # Key insight: finger magnets vary with finger position, averaging toward zero
        #              hard iron is constant, so average converges to hard iron
        if len(self.residual_samples) >= 100:
            self.hard_iron_estimate = [
                mean([r[0] for r in self.residual_samples]),
                mean([r[1] for r in self.residual_samples]),
                mean([r[2] for r in self.residual_samples])
            ]

        # Compute final residual (after both corrections)
        final_residual = [
            residual[0] - self.hard_iron_estimate[0],
            residual[1] - self.hard_iron_estimate[1],
            residual[2] - self.hard_iron_estimate[2]
        ]

        return {
            'earth_world': self.earth_world.copy(),
            'earth_mag': mag3(*self.earth_world),
            'hard_iron': self.hard_iron_estimate.copy(),
            'hard_iron_mag': mag3(*self.hard_iron_estimate),
            'residual': final_residual,
            'residual_mag': mag3(*final_residual),
            'n_samples': self.total_samples
        }


class TwoPhaseCalibrator:
    """
    Alternative approach: Two distinct phases.

    Phase 1: Collect data, estimate Earth field only
    Phase 2: After Earth stabilizes, estimate hard iron from residuals
    """

    def __init__(self, earth_window=200, min_earth_samples=100):
        self.earth_window = earth_window
        self.min_earth_samples = min_earth_samples

        self.world_samples = []
        self.earth_world = [0, 0, 0]
        self.earth_stable = False

        # Hard iron tracking
        self.residual_history = []
        self.hard_iron = [0, 0, 0]

        self.total_samples = 0
        self.earth_magnitude_history = []

    def update(self, mx_ut, my_ut, mz_ut, qw, qx, qy, qz) -> Dict:
        self.total_samples += 1

        # Transform to world frame
        R = quat_to_mat(qw, qx, qy, qz)
        R_T = transpose(R)
        world = mat_vec(R_T, [mx_ut, my_ut, mz_ut])

        self.world_samples.append(world)
        if len(self.world_samples) > self.earth_window:
            self.world_samples.pop(0)

        # Update Earth estimate
        if len(self.world_samples) >= self.min_earth_samples:
            self.earth_world = [
                mean([s[0] for s in self.world_samples]),
                mean([s[1] for s in self.world_samples]),
                mean([s[2] for s in self.world_samples])
            ]

            earth_mag = mag3(*self.earth_world)
            self.earth_magnitude_history.append(earth_mag)

            # Check if Earth estimate has stabilized (less than 5% change over 50 samples)
            if len(self.earth_magnitude_history) > 50:
                recent = self.earth_magnitude_history[-50:]
                old = self.earth_magnitude_history[-100:-50] if len(self.earth_magnitude_history) >= 100 else recent
                change = abs(mean(recent) - mean(old)) / mean(old) if mean(old) > 0 else 1
                if change < 0.05:
                    self.earth_stable = True

        # Compute sensor-frame residual
        earth_sensor = mat_vec(R, self.earth_world)
        residual = [
            mx_ut - earth_sensor[0],
            my_ut - earth_sensor[1],
            mz_ut - earth_sensor[2]
        ]

        # Only update hard iron estimate after Earth is stable
        if self.earth_stable:
            self.residual_history.append(residual)
            if len(self.residual_history) > 500:
                self.residual_history.pop(0)

            if len(self.residual_history) >= 50:
                self.hard_iron = [
                    mean([r[0] for r in self.residual_history]),
                    mean([r[1] for r in self.residual_history]),
                    mean([r[2] for r in self.residual_history])
                ]

        # Final residual
        final_residual = [
            residual[0] - self.hard_iron[0],
            residual[1] - self.hard_iron[1],
            residual[2] - self.hard_iron[2]
        ]

        return {
            'phase': 'iron' if self.earth_stable else 'earth',
            'earth_world': self.earth_world.copy(),
            'earth_mag': mag3(*self.earth_world),
            'earth_stable': self.earth_stable,
            'hard_iron': self.hard_iron.copy(),
            'hard_iron_mag': mag3(*self.hard_iron),
            'residual': final_residual,
            'residual_mag': mag3(*final_residual),
            'n_samples': self.total_samples
        }


def analyze_session(filepath: str):
    """Analyze a session with automatic iron calibration."""
    with open(filepath) as f:
        data = json.load(f)

    samples = data.get('samples', [])
    name = Path(filepath).name

    print(f"\n{'='*80}")
    print(f"AUTO IRON CALIBRATION: {name}")
    print(f"{'='*80}")
    print(f"Total samples: {len(samples)}")

    # Check for existing calibration data to compare
    stored_cal = None
    cal_path = Path('/home/user/simcap/data/GAMBIT/gambit_calibration.json')
    try:
        if cal_path.exists():
            with open(cal_path) as f:
                content = f.read()
                # Check if it's LFS pointer or actual JSON
                if content.startswith('version https://git-lfs'):
                    print(f"\nStored Calibration: (Git LFS pointer - not available)")
                else:
                    stored_cal = json.loads(content)
                    if 'hardIronOffset' in stored_cal:
                        hi = stored_cal['hardIronOffset']
                        print(f"\nStored Hard Iron: [{hi['x']:.1f}, {hi['y']:.1f}, {hi['z']:.1f}] µT")
                        print(f"Stored Hard Iron Magnitude: {mag3(hi['x'], hi['y'], hi['z']):.1f} µT")
    except Exception as e:
        print(f"\nStored Calibration: (not available - {e})")

    # Run both calibration approaches
    auto_cal = AutoIronCalibrator(earth_window=200, iron_window=500)
    two_phase = TwoPhaseCalibrator(earth_window=200)

    # Track results over time
    checkpoints = [50, 100, 200, 300, 500, 1000, 2000]
    auto_results = []
    phase_results = []

    raw_mags = []
    auto_residual_mags = []
    phase_residual_mags = []

    for i, s in enumerate(samples):
        if 'orientation_w' not in s:
            continue

        mx = s.get('mx_ut', 0)
        my = s.get('my_ut', 0)
        mz = s.get('mz_ut', 0)
        qw, qx, qy, qz = s['orientation_w'], s['orientation_x'], s['orientation_y'], s['orientation_z']

        auto_r = auto_cal.update(mx, my, mz, qw, qx, qy, qz)
        phase_r = two_phase.update(mx, my, mz, qw, qx, qy, qz)

        raw_mags.append(mag3(mx, my, mz))
        auto_residual_mags.append(auto_r['residual_mag'])
        phase_residual_mags.append(phase_r['residual_mag'])

        n = len(raw_mags)
        if n in checkpoints or n == len(samples):
            auto_results.append(auto_r.copy())
            phase_results.append(phase_r.copy())

    # Display results
    print(f"\n--- SIMULTANEOUS CALIBRATION APPROACH ---")
    print(f"{'N':>6} {'Earth|':>8} {'HardIron':^25} {'Residual':>10}")
    print(f"{'':>6} {'(µT)':>8} {'X':>8} {'Y':>8} {'Z':>8} {'|':>5} {'(µT)':>10}")
    print("-" * 70)

    for r in auto_results:
        hi = r['hard_iron']
        print(f"{r['n_samples']:>6} {r['earth_mag']:>7.0f} "
              f"{hi[0]:>8.1f} {hi[1]:>8.1f} {hi[2]:>8.1f} "
              f"{r['hard_iron_mag']:>5.1f} {r['residual_mag']:>10.1f}")

    print(f"\n--- TWO-PHASE CALIBRATION APPROACH ---")
    print(f"{'N':>6} {'Phase':>6} {'Earth|':>8} {'HardIron':^25} {'Residual':>10}")
    print(f"{'':>6} {'':>6} {'(µT)':>8} {'X':>8} {'Y':>8} {'Z':>8} {'|':>5} {'(µT)':>10}")
    print("-" * 76)

    for r in phase_results:
        hi = r['hard_iron']
        print(f"{r['n_samples']:>6} {r['phase']:>6} {r['earth_mag']:>7.0f} "
              f"{hi[0]:>8.1f} {hi[1]:>8.1f} {hi[2]:>8.1f} "
              f"{r['hard_iron_mag']:>5.1f} {r['residual_mag']:>10.1f}")

    # Compare final results
    print(f"\n--- COMPARISON ---")
    final_auto = auto_results[-1] if auto_results else None
    final_phase = phase_results[-1] if phase_results else None

    if final_auto and final_phase:
        print(f"\nFinal Hard Iron Estimates:")
        print(f"  Simultaneous: [{final_auto['hard_iron'][0]:.1f}, {final_auto['hard_iron'][1]:.1f}, {final_auto['hard_iron'][2]:.1f}] |{final_auto['hard_iron_mag']:.1f}| µT")
        print(f"  Two-Phase:    [{final_phase['hard_iron'][0]:.1f}, {final_phase['hard_iron'][1]:.1f}, {final_phase['hard_iron'][2]:.1f}] |{final_phase['hard_iron_mag']:.1f}| µT")

        if stored_cal and 'hardIronOffset' in stored_cal:
            hi = stored_cal['hardIronOffset']
            stored_vec = [hi['x'], hi['y'], hi['z']]
            print(f"  Stored (ref): [{hi['x']:.1f}, {hi['y']:.1f}, {hi['z']:.1f}] |{mag3(*stored_vec):.1f}| µT")

            # Calculate error
            auto_err = mag3(
                final_auto['hard_iron'][0] - stored_vec[0],
                final_auto['hard_iron'][1] - stored_vec[1],
                final_auto['hard_iron'][2] - stored_vec[2]
            )
            phase_err = mag3(
                final_phase['hard_iron'][0] - stored_vec[0],
                final_phase['hard_iron'][1] - stored_vec[1],
                final_phase['hard_iron'][2] - stored_vec[2]
            )

            print(f"\nError vs Stored Calibration:")
            print(f"  Simultaneous: {auto_err:.1f} µT")
            print(f"  Two-Phase:    {phase_err:.1f} µT")

    # SNR Analysis
    print(f"\n--- SNR COMPARISON ---")

    def calc_snr(mags):
        baseline = percentile(mags, 25)
        peak = percentile(mags, 95)
        return peak / baseline if baseline > 0 else 0

    raw_snr = calc_snr(raw_mags)
    auto_snr = calc_snr(auto_residual_mags)
    phase_snr = calc_snr(phase_residual_mags)

    print(f"RAW SNR:         {raw_snr:.2f}x")
    print(f"Simultaneous:    {auto_snr:.2f}x (Δ={auto_snr - raw_snr:+.2f}x)")
    print(f"Two-Phase:       {phase_snr:.2f}x (Δ={phase_snr - raw_snr:+.2f}x)")

    # Conclusion
    print(f"\n--- CONCLUSION ---")
    if auto_snr > raw_snr or phase_snr > raw_snr:
        print("✓ Auto iron calibration IMPROVES SNR")
        best = "Simultaneous" if auto_snr > phase_snr else "Two-Phase"
        print(f"  Best approach: {best}")
    else:
        print("✗ Auto iron calibration does not improve SNR")
        print("  Possible reasons:")
        print("  - Finger magnets are too strong/variable")
        print("  - Insufficient rotation coverage")
        print("  - Hard iron is already small compared to magnets")

    return {
        'raw_snr': raw_snr,
        'auto_snr': auto_snr,
        'phase_snr': phase_snr,
        'auto_iron': final_auto['hard_iron'] if final_auto else None,
        'phase_iron': final_phase['hard_iron'] if final_phase else None
    }


def analyze_without_magnets(filepath: str):
    """
    Analyze a session WITHOUT finger magnets (if available).

    This tests if our method can recover the true hard iron
    when magnets aren't confusing things.
    """
    with open(filepath) as f:
        data = json.load(f)

    samples = data.get('samples', [])
    name = Path(filepath).name

    # Check if this is likely a magnet-free session
    # (we'd need metadata or session naming convention)

    print(f"\n{'='*80}")
    print(f"MAGNET-FREE CALIBRATION TEST: {name}")
    print(f"{'='*80}")

    # Run the two-phase calibrator
    cal = TwoPhaseCalibrator(earth_window=200)

    for s in samples:
        if 'orientation_w' not in s:
            continue

        mx = s.get('mx_ut', 0)
        my = s.get('my_ut', 0)
        mz = s.get('mz_ut', 0)
        qw, qx, qy, qz = s['orientation_w'], s['orientation_x'], s['orientation_y'], s['orientation_z']

        cal.update(mx, my, mz, qw, qx, qy, qz)

    print(f"Final Earth: [{cal.earth_world[0]:.1f}, {cal.earth_world[1]:.1f}, {cal.earth_world[2]:.1f}] |{mag3(*cal.earth_world):.1f}| µT")
    print(f"Final Hard Iron: [{cal.hard_iron[0]:.1f}, {cal.hard_iron[1]:.1f}, {cal.hard_iron[2]:.1f}] |{mag3(*cal.hard_iron):.1f}| µT")

    return cal.hard_iron


def investigate_hard_iron_stability():
    """
    Investigate whether hard iron estimate stabilizes over time.

    Key question: Does averaging residuals converge to a stable value?
    """
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    print("=" * 80)
    print("HARD IRON STABILITY INVESTIGATION")
    print("=" * 80)
    print("\nQuestion: Does sensor-frame residual averaging converge to hard iron?")

    # Find sessions with finger magnets (post-22:00 sessions from investigation)
    magnet_sessions = list(sorted(data_dir.glob('2025-12-15T22*.json')))

    if not magnet_sessions:
        print("No magnet sessions found")
        return

    for session_file in magnet_sessions[:3]:  # Analyze up to 3 sessions
        if 'gambit' not in session_file.name.lower():
            analyze_session(str(session_file))


def main():
    print("=" * 80)
    print("AUTOMATIC IRON CALIBRATION INVESTIGATION")
    print("=" * 80)
    print("""
Goal: Estimate hard iron offset automatically from streaming data,
      even when finger magnets are present.

Approach:
1. Earth field estimation (already validated - world-frame averaging)
2. Hard iron estimation from sensor-frame residual averaging
   - After Earth subtraction: residual ≈ hard_iron + magnets
   - If magnet positions vary: averaging should converge to hard_iron

Testing with existing session data...
""")

    investigate_hard_iron_stability()

    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    print("""
Based on this investigation:

1. If auto iron calibration IMPROVES SNR:
   - Implement in UnifiedMagCalibration class
   - Use two-phase approach (Earth first, then iron)
   - Track residual stability as quality metric

2. If auto iron calibration has MINIMAL effect:
   - Hard iron may be small compared to finger magnets
   - The Earth-only approach may be sufficient
   - Keep manual iron calibration as optional enhancement

3. If auto iron calibration DEGRADES SNR:
   - Magnet signals are too dominant
   - Cannot separate hard iron from magnet patterns
   - Require manual calibration (without magnets) for best results
""")


if __name__ == '__main__':
    main()
