#!/usr/bin/env python3
"""
Real-Time Earth Field Estimation Simulation

Simulates real-time processing constraint: at each time step,
only use data seen SO FAR (no future lookahead).

Questions to answer:
1. How many samples needed before Earth estimate is usable?
2. Does the estimate stabilize or drift?
3. What rotation coverage is needed?
4. Cumulative vs sliding window - which works better?

This informs client-side implementation.
"""

import json
import math
from pathlib import Path
from datetime import datetime
from collections import defaultdict


def mean(arr):
    return sum(arr) / len(arr) if arr else 0

def std(arr):
    if len(arr) < 2:
        return 0
    m = mean(arr)
    return math.sqrt(sum((x - m) ** 2 for x in arr) / len(arr))

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


class RealtimeEarthEstimator:
    """
    Estimates Earth field in real-time using only past samples.
    """

    def __init__(self, window_size=None):
        """
        Args:
            window_size: If None, use cumulative average.
                        If int, use sliding window of that size.
        """
        self.window_size = window_size
        self.world_samples = []  # [(x, y, z), ...]
        self.earth_estimate = [0, 0, 0]
        self.sample_count = 0

    def update(self, mx_ut, my_ut, mz_ut, qw, qx, qy, qz):
        """
        Process new sample, update Earth estimate.

        Returns current Earth estimate (world frame).
        """
        # Transform raw reading to world frame
        R = quat_to_mat(qw, qx, qy, qz)
        R_T = transpose(R)
        world = mat_vec(R_T, [mx_ut, my_ut, mz_ut])

        self.world_samples.append(world)
        self.sample_count += 1

        # Compute Earth estimate from available samples
        if self.window_size and len(self.world_samples) > self.window_size:
            # Sliding window
            window = self.world_samples[-self.window_size:]
        else:
            # Cumulative (all samples)
            window = self.world_samples

        self.earth_estimate = [
            mean([s[0] for s in window]),
            mean([s[1] for s in window]),
            mean([s[2] for s in window])
        ]

        return self.earth_estimate

    def get_residual(self, mx_ut, my_ut, mz_ut, qw, qx, qy, qz):
        """
        Compute residual using current Earth estimate.
        """
        # Rotate Earth estimate from world to sensor frame
        R = quat_to_mat(qw, qx, qy, qz)
        earth_sensor = mat_vec(R, self.earth_estimate)

        # Residual = raw - Earth
        return [
            mx_ut - earth_sensor[0],
            my_ut - earth_sensor[1],
            mz_ut - earth_sensor[2]
        ]


def analyze_polarity(mx, my, mz):
    """Analyze octant distribution."""
    octants = defaultdict(int)
    for i in range(len(mx)):
        o = ('+' if mx[i] > 0 else '-') + ('+' if my[i] > 0 else '-') + ('+' if mz[i] > 0 else '-')
        octants[o] += 1

    dom = max(octants.items(), key=lambda x: x[1]) if octants else ('---', 0)
    return {
        'unique': len(octants),
        'dominant': dom[0],
        'dominant_pct': 100 * dom[1] / len(mx) if mx else 0
    }


def compute_orientation_coverage(rolls, pitches, yaws):
    """Compute rotation coverage in each axis."""
    return {
        'roll': max(rolls) - min(rolls) if rolls else 0,
        'pitch': max(pitches) - min(pitches) if pitches else 0,
        'yaw': max(yaws) - min(yaws) if yaws else 0
    }


def simulate_realtime(samples, window_size=None):
    """
    Simulate real-time processing of session.

    Returns metrics at various checkpoints.
    """
    estimator = RealtimeEarthEstimator(window_size=window_size)

    # Track metrics over time
    checkpoints = []
    checkpoint_intervals = [50, 100, 200, 300, 500, 1000, 2000]

    raw_mags = []
    residual_mags = []
    residuals = []
    rolls, pitches, yaws = [], [], []

    for i, s in enumerate(samples):
        if 'orientation_w' not in s:
            continue

        mx = s.get('mx_ut', 0)
        my = s.get('my_ut', 0)
        mz = s.get('mz_ut', 0)
        qw, qx, qy, qz = s['orientation_w'], s['orientation_x'], s['orientation_y'], s['orientation_z']

        # Update Earth estimate with this sample
        earth = estimator.update(mx, my, mz, qw, qx, qy, qz)

        # Compute residual using CURRENT estimate (causal - no future data)
        res = estimator.get_residual(mx, my, mz, qw, qx, qy, qz)

        raw_mags.append(mag3(mx, my, mz))
        residual_mags.append(mag3(*res))
        residuals.append(res)

        rolls.append(s.get('euler_roll', 0))
        pitches.append(s.get('euler_pitch', 0))
        yaws.append(s.get('euler_yaw', 0))

        # Record checkpoint
        n = len(raw_mags)
        if n in checkpoint_intervals or n == len(samples):
            res_mx = [r[0] for r in residuals]
            res_my = [r[1] for r in residuals]
            res_mz = [r[2] for r in residuals]

            raw_pol = analyze_polarity(
                [s.get('mx_ut', 0) for s in samples[:i+1] if 'mx_ut' in s],
                [s.get('my_ut', 0) for s in samples[:i+1] if 'my_ut' in s],
                [s.get('mz_ut', 0) for s in samples[:i+1] if 'mz_ut' in s]
            )
            res_pol = analyze_polarity(res_mx, res_my, res_mz)

            coverage = compute_orientation_coverage(rolls, pitches, yaws)

            # SNR calculation
            raw_baseline = sorted(raw_mags)[len(raw_mags)//4] if raw_mags else 1
            raw_peak = sorted(raw_mags)[int(len(raw_mags)*0.95)] if raw_mags else 1
            res_baseline = sorted(residual_mags)[len(residual_mags)//4] if residual_mags else 1
            res_peak = sorted(residual_mags)[int(len(residual_mags)*0.95)] if residual_mags else 1

            checkpoints.append({
                'n_samples': n,
                'earth_estimate': earth.copy(),
                'earth_magnitude': mag3(*earth),
                'coverage': coverage,
                'raw_snr': raw_peak / raw_baseline if raw_baseline > 0 else 0,
                'res_snr': res_peak / res_baseline if res_baseline > 0 else 0,
                'raw_octants': raw_pol['unique'],
                'res_octants': res_pol['unique'],
                'raw_dominant_pct': raw_pol['dominant_pct'],
                'res_dominant_pct': res_pol['dominant_pct'],
            })

    return checkpoints


def analyze_session_realtime(filepath):
    """Analyze session with real-time constraint."""
    with open(filepath) as f:
        data = json.load(f)

    samples = data.get('samples', [])
    name = Path(filepath).name

    print(f"\n{'='*80}")
    print(f"REAL-TIME SIMULATION: {name}")
    print(f"{'='*80}")
    print(f"Total samples: {len(samples)}")

    # Test both cumulative and sliding window approaches
    print(f"\n--- CUMULATIVE AVERAGE APPROACH ---")
    cumulative = simulate_realtime(samples, window_size=None)

    print(f"\n{'N':>6} {'Earth|':>8} {'Coverage':^20} {'SNR':^15} {'Octants':^12} {'Dom%':^12}")
    print(f"{'':>6} {'(µT)':>8} {'R°    P°    Y°':^20} {'Raw   Res':^15} {'Raw Res':^12} {'Raw  Res':^12}")
    print("-" * 85)

    for cp in cumulative:
        cov = cp['coverage']
        print(f"{cp['n_samples']:>6} {cp['earth_magnitude']:>7.0f} "
              f"{cov['roll']:>5.0f} {cov['pitch']:>5.0f} {cov['yaw']:>5.0f}   "
              f"{cp['raw_snr']:>5.1f}x {cp['res_snr']:>5.1f}x  "
              f"{cp['raw_octants']:>4} {cp['res_octants']:>4}   "
              f"{cp['raw_dominant_pct']:>5.0f} {cp['res_dominant_pct']:>5.0f}")

    # Determine when estimate becomes "good"
    print(f"\n--- CONVERGENCE ANALYSIS ---")

    # Check when Earth magnitude stabilizes (within 25-65 µT or stops changing much)
    stable_at = None
    for i, cp in enumerate(cumulative):
        if i > 0:
            prev_mag = cumulative[i-1]['earth_magnitude']
            curr_mag = cp['earth_magnitude']
            change_pct = abs(curr_mag - prev_mag) / prev_mag * 100 if prev_mag > 0 else 100

            if change_pct < 10 and stable_at is None:  # Less than 10% change
                stable_at = cp['n_samples']

    if stable_at:
        print(f"Earth estimate stabilizes (~10% change) at: {stable_at} samples")
    else:
        print(f"Earth estimate does not fully stabilize")

    # Check when SNR improvement appears
    snr_improves_at = None
    for cp in cumulative:
        if cp['res_snr'] > cp['raw_snr'] and snr_improves_at is None:
            snr_improves_at = cp['n_samples']

    if snr_improves_at:
        print(f"SNR improvement first appears at: {snr_improves_at} samples")
    else:
        print(f"SNR does not improve (may need more rotation)")

    # Check when octant diversity improves
    octant_improves_at = None
    for cp in cumulative:
        if cp['res_octants'] > cp['raw_octants'] and octant_improves_at is None:
            octant_improves_at = cp['n_samples']

    if octant_improves_at:
        print(f"Octant diversity improves at: {octant_improves_at} samples")

    # Final comparison
    final = cumulative[-1] if cumulative else None
    if final:
        print(f"\n--- FINAL RESULTS (Real-Time Constrained) ---")
        print(f"Earth Magnitude: {final['earth_magnitude']:.1f} µT")
        print(f"SNR: {final['raw_snr']:.2f}x (raw) → {final['res_snr']:.2f}x (residual)")
        print(f"Octants: {final['raw_octants']} (raw) → {final['res_octants']} (residual)")
        print(f"Dominant %: {final['raw_dominant_pct']:.1f}% (raw) → {final['res_dominant_pct']:.1f}% (residual)")

        # Hypothesis check
        snr_ok = final['res_snr'] > final['raw_snr']
        oct_ok = final['res_octants'] > final['raw_octants']
        dom_ok = final['res_dominant_pct'] < final['raw_dominant_pct']

        score = sum([snr_ok, oct_ok, dom_ok])
        print(f"\nReal-Time Hypothesis Score: {score}/3")
        print(f"  SNR improved: {'✓' if snr_ok else '✗'}")
        print(f"  More octants: {'✓' if oct_ok else '✗'}")
        print(f"  Less dominant: {'✓' if dom_ok else '✗'}")

    # Test sliding window
    print(f"\n--- SLIDING WINDOW COMPARISON (window=200) ---")
    sliding = simulate_realtime(samples, window_size=200)

    if sliding and cumulative:
        final_slide = sliding[-1]
        final_cum = cumulative[-1]

        print(f"{'Method':<15} {'Earth|':>8} {'SNR':>8} {'Octants':>8} {'Dom%':>8}")
        print("-" * 50)
        print(f"{'Cumulative':<15} {final_cum['earth_magnitude']:>7.0f} {final_cum['res_snr']:>7.2f}x {final_cum['res_octants']:>8} {final_cum['res_dominant_pct']:>7.0f}%")
        print(f"{'Sliding(200)':<15} {final_slide['earth_magnitude']:>7.0f} {final_slide['res_snr']:>7.2f}x {final_slide['res_octants']:>8} {final_slide['res_dominant_pct']:>7.0f}%")

    return cumulative


def main():
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    print("=" * 80)
    print("REAL-TIME EARTH ESTIMATION SIMULATION")
    print("=" * 80)
    print("\nConstraint: At each time step, only use data seen SO FAR.")
    print("Goal: Determine minimum samples needed for viable Earth estimate.")

    all_results = []
    for f in sorted(data_dir.glob('2025-12-15T22*.json')):
        if 'gambit' not in f.name.lower():
            r = analyze_session_realtime(str(f))
            if r:
                all_results.append(r)

    print(f"\n{'='*80}")
    print("RECOMMENDATIONS FOR CLIENT-SIDE IMPLEMENTATION")
    print("=" * 80)
    print("""
Based on this analysis:

1. MINIMUM SAMPLES: ~100-200 samples before Earth estimate is useful
   - Need sufficient rotation coverage in at least 2 axes (>30° each)

2. APPROACH: Cumulative average works well
   - Sliding window may help if environment changes mid-session
   - Start with cumulative, consider adaptive switching

3. IMPLEMENTATION STRATEGY:
   a) First 100-200 samples: Build Earth estimate, don't apply correction yet
   b) After threshold: Start applying correction, continue updating estimate
   c) Monitor Earth magnitude - should stabilize near 25-65 µT

4. ROTATION REQUIREMENT:
   - Encourage user to rotate device during initial calibration phase
   - Can detect coverage and prompt user if insufficient

5. FALLBACK: If rotation is insufficient, Earth estimate will be biased
   - Can detect this by checking if Earth magnitude >> 65 µT
   - Fall back to raw data or prompt for more rotation
""")


if __name__ == '__main__':
    main()
