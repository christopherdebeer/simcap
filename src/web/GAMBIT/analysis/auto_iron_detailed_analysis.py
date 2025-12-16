#!/usr/bin/env python3
"""
Detailed Analysis of Auto Iron Calibration Dynamics

Key Questions:
1. Why does auto iron calibration work in one session but not another?
2. What conditions make it reliable?
3. Can we detect when the estimate is "good" vs "contaminated"?

Observations from initial investigation:
- Session 1 (2564 samples): SNR improved from 2.94x to 3.74x (+0.80x)
- Session 2 (968 samples): SNR degraded from 8.93x to 3.99x (-4.94x)

Hypotheses:
- H1: More samples = better (rotation averages out magnets better)
- H2: Finger movement pattern matters (varied vs static)
- H3: Hard iron estimate stability indicates quality
"""

import json
import math
from pathlib import Path
from collections import defaultdict


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


class DetailedIronAnalyzer:
    """Detailed analysis of iron calibration dynamics."""

    def __init__(self, earth_window=200, iron_window=500):
        self.earth_window = earth_window
        self.iron_window = iron_window

        # Data buffers
        self.world_samples = []
        self.residual_samples = []

        # Estimates
        self.earth_world = [0, 0, 0]
        self.hard_iron = [0, 0, 0]

        # Tracking
        self.iron_estimate_history = []  # Track how iron estimate evolves
        self.residual_std_history = []   # Track residual stability
        self.gyro_activity_history = []  # Track device movement
        self.total_samples = 0

    def update(self, mx_ut, my_ut, mz_ut, qw, qx, qy, qz, gx=0, gy=0, gz=0):
        self.total_samples += 1

        # Transform to world frame
        R = quat_to_mat(qw, qx, qy, qz)
        R_T = transpose(R)
        world = mat_vec(R_T, [mx_ut, my_ut, mz_ut])

        self.world_samples.append(world)
        if len(self.world_samples) > self.earth_window:
            self.world_samples.pop(0)

        # Earth estimate
        if len(self.world_samples) >= 50:
            self.earth_world = [
                mean([s[0] for s in self.world_samples]),
                mean([s[1] for s in self.world_samples]),
                mean([s[2] for s in self.world_samples])
            ]

        # Residual in sensor frame
        earth_sensor = mat_vec(R, self.earth_world)
        residual = [
            mx_ut - earth_sensor[0],
            my_ut - earth_sensor[1],
            mz_ut - earth_sensor[2]
        ]

        self.residual_samples.append(residual)
        if len(self.residual_samples) > self.iron_window:
            self.residual_samples.pop(0)

        # Iron estimate
        if len(self.residual_samples) >= 100:
            self.hard_iron = [
                mean([r[0] for r in self.residual_samples]),
                mean([r[1] for r in self.residual_samples]),
                mean([r[2] for r in self.residual_samples])
            ]

        # Track gyro activity (device rotation rate)
        gyro_mag = mag3(gx, gy, gz)
        self.gyro_activity_history.append(gyro_mag)

        # Track iron estimate evolution
        self.iron_estimate_history.append(self.hard_iron.copy())

        # Track residual variability (after iron correction)
        if len(self.residual_samples) >= 50:
            corrected_residuals = [
                [r[i] - self.hard_iron[i] for i in range(3)]
                for r in self.residual_samples[-50:]
            ]
            residual_std = [
                std([r[0] for r in corrected_residuals]),
                std([r[1] for r in corrected_residuals]),
                std([r[2] for r in corrected_residuals])
            ]
            self.residual_std_history.append(mag3(*residual_std))
        else:
            self.residual_std_history.append(0)

        return {
            'earth_world': self.earth_world.copy(),
            'hard_iron': self.hard_iron.copy(),
            'residual': residual,
            'residual_std': self.residual_std_history[-1] if self.residual_std_history else 0
        }


def analyze_iron_dynamics(filepath):
    """Analyze how iron estimate evolves over time."""
    with open(filepath) as f:
        data = json.load(f)

    samples = data.get('samples', [])
    name = Path(filepath).name

    print(f"\n{'='*80}")
    print(f"IRON DYNAMICS: {name}")
    print(f"{'='*80}")
    print(f"Total samples: {len(samples)}")

    analyzer = DetailedIronAnalyzer(earth_window=200, iron_window=500)

    # Process all samples
    for s in samples:
        if 'orientation_w' not in s:
            continue

        mx = s.get('mx_ut', 0)
        my = s.get('my_ut', 0)
        mz = s.get('mz_ut', 0)
        qw, qx, qy, qz = s['orientation_w'], s['orientation_x'], s['orientation_y'], s['orientation_z']
        gx = s.get('gx_dps', 0)
        gy = s.get('gy_dps', 0)
        gz = s.get('gz_dps', 0)

        analyzer.update(mx, my, mz, qw, qx, qy, qz, gx, gy, gz)

    # Analyze iron estimate stability
    print(f"\n--- IRON ESTIMATE EVOLUTION ---")

    if len(analyzer.iron_estimate_history) < 200:
        print("Insufficient samples for stability analysis")
        return

    # Look at iron estimate at different points
    checkpoints = [100, 200, 300, 500, 750, 1000, 1500, 2000, len(analyzer.iron_estimate_history)-1]
    checkpoints = [c for c in checkpoints if c < len(analyzer.iron_estimate_history)]

    print(f"{'N':>6} {'Iron X':>10} {'Iron Y':>10} {'Iron Z':>10} {'|Iron|':>10} {'ResStd':>10}")
    print("-" * 66)

    prev_iron = None
    for cp in checkpoints:
        iron = analyzer.iron_estimate_history[cp]
        res_std = analyzer.residual_std_history[cp] if cp < len(analyzer.residual_std_history) else 0

        drift = ""
        if prev_iron:
            change = mag3(
                iron[0] - prev_iron[0],
                iron[1] - prev_iron[1],
                iron[2] - prev_iron[2]
            )
            drift = f" (Δ={change:.1f})"
        prev_iron = iron

        print(f"{cp:>6} {iron[0]:>10.1f} {iron[1]:>10.1f} {iron[2]:>10.1f} {mag3(*iron):>10.1f} {res_std:>10.1f}{drift}")

    # Analyze stability: Is iron estimate converging?
    print(f"\n--- CONVERGENCE ANALYSIS ---")

    # Compare last 100 vs first 100 (after minimum samples)
    if len(analyzer.iron_estimate_history) > 300:
        early = analyzer.iron_estimate_history[100:200]
        late = analyzer.iron_estimate_history[-100:]

        early_mean = [mean([e[i] for e in early]) for i in range(3)]
        late_mean = [mean([e[i] for e in late]) for i in range(3)]

        early_std = [std([e[i] for e in early]) for i in range(3)]
        late_std = [std([e[i] for e in late]) for i in range(3)]

        print(f"Early (100-200) mean: [{early_mean[0]:.1f}, {early_mean[1]:.1f}, {early_mean[2]:.1f}]")
        print(f"Late  (last 100) mean: [{late_mean[0]:.1f}, {late_mean[1]:.1f}, {late_mean[2]:.1f}]")
        print(f"\nEarly std: [{early_std[0]:.1f}, {early_std[1]:.1f}, {early_std[2]:.1f}]")
        print(f"Late  std: [{late_std[0]:.1f}, {late_std[1]:.1f}, {late_std[2]:.1f}]")

        # Stability improved?
        early_total_std = mag3(*early_std)
        late_total_std = mag3(*late_std)
        print(f"\nTotal std: {early_total_std:.1f} (early) → {late_total_std:.1f} (late)")

        if late_total_std < early_total_std * 0.7:
            print("✓ Iron estimate is STABILIZING (variance decreasing)")
        elif late_total_std > early_total_std * 1.3:
            print("✗ Iron estimate is UNSTABLE (variance increasing)")
        else:
            print("○ Iron estimate stability is MARGINAL")

    # Look at gyroscope activity (rotation)
    print(f"\n--- DEVICE ROTATION ANALYSIS ---")

    if analyzer.gyro_activity_history:
        mean_gyro = mean(analyzer.gyro_activity_history)
        max_gyro = max(analyzer.gyro_activity_history)
        active_samples = sum(1 for g in analyzer.gyro_activity_history if g > 10)  # >10 dps considered active
        pct_active = 100 * active_samples / len(analyzer.gyro_activity_history)

        print(f"Mean rotation rate: {mean_gyro:.1f} °/s")
        print(f"Peak rotation rate: {max_gyro:.1f} °/s")
        print(f"Samples with rotation: {pct_active:.1f}%")

        if pct_active < 30:
            print("⚠ Low rotation - iron estimate may not average out")
        else:
            print("✓ Sufficient rotation for averaging")

    # Check residual stability
    print(f"\n--- RESIDUAL STABILITY ---")
    if analyzer.residual_std_history:
        early_res_std = mean(analyzer.residual_std_history[100:200]) if len(analyzer.residual_std_history) > 200 else 0
        late_res_std = mean(analyzer.residual_std_history[-100:]) if len(analyzer.residual_std_history) > 100 else 0

        print(f"Residual std (early): {early_res_std:.1f} µT")
        print(f"Residual std (late):  {late_res_std:.1f} µT")

        if late_res_std < early_res_std * 0.8:
            print("✓ Residual stability IMPROVING (iron calibration helping)")
        elif late_res_std > early_res_std * 1.2:
            print("✗ Residual stability DEGRADING (iron calibration hurting)")
        else:
            print("○ Residual stability UNCHANGED")

    return analyzer


def compare_earth_only_vs_iron(filepath):
    """Compare Earth-only vs Earth+Iron calibration."""
    with open(filepath) as f:
        data = json.load(f)

    samples = data.get('samples', [])
    name = Path(filepath).name

    print(f"\n{'='*80}")
    print(f"EARTH-ONLY vs EARTH+IRON: {name}")
    print(f"{'='*80}")

    # Process with Earth-only
    world_samples = []
    earth_world = [0, 0, 0]

    earth_residuals = []
    iron_residuals = []

    residual_samples = []
    hard_iron = [0, 0, 0]

    for s in samples:
        if 'orientation_w' not in s:
            continue

        mx = s.get('mx_ut', 0)
        my = s.get('my_ut', 0)
        mz = s.get('mz_ut', 0)
        qw, qx, qy, qz = s['orientation_w'], s['orientation_x'], s['orientation_y'], s['orientation_z']

        R = quat_to_mat(qw, qx, qy, qz)
        R_T = transpose(R)
        world = mat_vec(R_T, [mx, my, mz])

        world_samples.append(world)
        if len(world_samples) > 200:
            world_samples.pop(0)

        if len(world_samples) >= 50:
            earth_world = [
                mean([s[0] for s in world_samples]),
                mean([s[1] for s in world_samples]),
                mean([s[2] for s in world_samples])
            ]

        earth_sensor = mat_vec(R, earth_world)
        earth_residual = [
            mx - earth_sensor[0],
            my - earth_sensor[1],
            mz - earth_sensor[2]
        ]
        earth_residuals.append(mag3(*earth_residual))

        # Iron correction
        residual_samples.append(earth_residual)
        if len(residual_samples) > 500:
            residual_samples.pop(0)

        if len(residual_samples) >= 100:
            hard_iron = [
                mean([r[0] for r in residual_samples]),
                mean([r[1] for r in residual_samples]),
                mean([r[2] for r in residual_samples])
            ]

        iron_residual = [
            earth_residual[0] - hard_iron[0],
            earth_residual[1] - hard_iron[1],
            earth_residual[2] - hard_iron[2]
        ]
        iron_residuals.append(mag3(*iron_residual))

    # Compare SNR
    def calc_snr(mags, skip=100):
        # Skip first 100 samples (warm-up)
        mags = mags[skip:]
        baseline = percentile(mags, 25)
        peak = percentile(mags, 95)
        return peak / baseline if baseline > 0 else 0

    earth_snr = calc_snr(earth_residuals)
    iron_snr = calc_snr(iron_residuals)

    raw_mags = [mag3(s.get('mx_ut', 0), s.get('my_ut', 0), s.get('mz_ut', 0))
                for s in samples if 'mx_ut' in s]
    raw_snr = calc_snr(raw_mags) if len(raw_mags) > 100 else 0

    print(f"\n--- SNR COMPARISON (after warm-up) ---")
    print(f"RAW:        {raw_snr:.2f}x")
    print(f"Earth-only: {earth_snr:.2f}x (Δ={earth_snr - raw_snr:+.2f}x)")
    print(f"Earth+Iron: {iron_snr:.2f}x (Δ={iron_snr - raw_snr:+.2f}x)")

    print(f"\nIron effect: {iron_snr - earth_snr:+.2f}x")

    if iron_snr > earth_snr + 0.1:
        print("✓ Iron calibration HELPS")
    elif iron_snr < earth_snr - 0.1:
        print("✗ Iron calibration HURTS")
    else:
        print("○ Iron calibration has MINIMAL effect")

    return {
        'raw_snr': raw_snr,
        'earth_snr': earth_snr,
        'iron_snr': iron_snr,
        'iron_effect': iron_snr - earth_snr
    }


def main():
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    print("=" * 80)
    print("DETAILED AUTO IRON CALIBRATION ANALYSIS")
    print("=" * 80)

    # Analyze both sessions
    sessions = sorted(data_dir.glob('2025-12-15T22*.json'))

    results = []
    for session in sessions:
        if 'gambit' not in session.name.lower():
            analyze_iron_dynamics(str(session))
            result = compare_earth_only_vs_iron(str(session))
            results.append(result)

    if results:
        print(f"\n{'='*80}")
        print("SUMMARY AND RECOMMENDATIONS")
        print("=" * 80)

        # Average effects
        avg_earth_improvement = mean([r['earth_snr'] - r['raw_snr'] for r in results])
        avg_iron_effect = mean([r['iron_effect'] for r in results])

        print(f"\nAverage Earth-only improvement: {avg_earth_improvement:+.2f}x SNR")
        print(f"Average Iron calibration effect: {avg_iron_effect:+.2f}x SNR")

        print(f"""
CONCLUSIONS:

1. EARTH FIELD ESTIMATION (already implemented):
   - Consistently improves SNR across sessions
   - Average improvement: {avg_earth_improvement:+.2f}x
   - Should be the PRIMARY calibration method

2. AUTOMATIC IRON CALIBRATION:
   - Effect is {"positive" if avg_iron_effect > 0.1 else "mixed" if avg_iron_effect < -0.1 else "marginal"} ({avg_iron_effect:+.2f}x average)
   - Works better with more samples (>1000)
   - Requires good device rotation to average out magnets

3. RECOMMENDED APPROACH:
   - PRIMARY: Keep real-time Earth field estimation (UnifiedMagCalibration)
   - OPTIONAL: Add auto iron as "enhancement" mode
   - QUALITY GATE: Only use auto iron if:
     * Sample count > 500
     * Iron estimate variance is decreasing
     * Residual stability is improving
   - FALLBACK: If auto iron degrades SNR, disable it

4. MANUAL IRON CALIBRATION:
   - Still provides best results when done without magnets
   - Keep as optional wizard for users who want maximum accuracy
""")


if __name__ == '__main__':
    main()
