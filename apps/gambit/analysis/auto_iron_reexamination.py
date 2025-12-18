#!/usr/bin/env python3
"""
Re-examination of Auto Iron Calibration

Hypothesis: The "auto iron" estimate is actually capturing:
  hard_iron + mean_finger_magnet_field

This could be USEFUL if:
1. We want to detect finger MOVEMENT (deviation from mean position)
2. Fingers are in a consistent "rest" position during calibration

This would HURT if:
1. We want to detect absolute finger position
2. Fingers move a lot during the averaging window

Key question: What does subtracting the mean residual actually do?
- It centers the signal around zero
- Positive values = fingers closer than average
- Negative values = fingers farther than average

This is valid for GESTURE detection (relative movement)
but loses ABSOLUTE position information.
"""

import json
import math
from pathlib import Path


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


def analyze_finger_variation(filepath):
    """Analyze how much finger position varies during session."""
    with open(filepath) as f:
        data = json.load(f)

    samples = data.get('samples', [])
    name = Path(filepath).name

    print(f"\n{'='*80}")
    print(f"FINGER VARIATION ANALYSIS: {name}")
    print(f"{'='*80}")

    # Compute Earth-subtracted residuals
    world_samples = []
    earth_world = [0, 0, 0]
    residuals = []

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
        residual = [
            mx - earth_sensor[0],
            my - earth_sensor[1],
            mz - earth_sensor[2]
        ]
        residuals.append(residual)

    # Skip warmup
    residuals = residuals[100:]

    if not residuals:
        print("Insufficient data")
        return

    # Analyze residual distribution
    rx = [r[0] for r in residuals]
    ry = [r[1] for r in residuals]
    rz = [r[2] for r in residuals]
    rmag = [mag3(*r) for r in residuals]

    print(f"\nResidual Statistics (Earth-subtracted, after warmup):")
    print(f"  Samples: {len(residuals)}")
    print(f"\n  {'Axis':<6} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Range':>10}")
    print(f"  {'-'*56}")
    print(f"  {'X':<6} {mean(rx):>10.1f} {std(rx):>10.1f} {min(rx):>10.1f} {max(rx):>10.1f} {max(rx)-min(rx):>10.1f}")
    print(f"  {'Y':<6} {mean(ry):>10.1f} {std(ry):>10.1f} {min(ry):>10.1f} {max(ry):>10.1f} {max(ry)-min(ry):>10.1f}")
    print(f"  {'Z':<6} {mean(rz):>10.1f} {std(rz):>10.1f} {min(rz):>10.1f} {max(rz):>10.1f} {max(rz)-min(rz):>10.1f}")
    print(f"  {'|Mag|':<6} {mean(rmag):>10.1f} {std(rmag):>10.1f} {min(rmag):>10.1f} {max(rmag):>10.1f} {max(rmag)-min(rmag):>10.1f}")

    # The mean residual is what "auto iron" estimates
    mean_residual = [mean(rx), mean(ry), mean(rz)]
    print(f"\n  Mean residual (what auto iron captures): [{mean_residual[0]:.1f}, {mean_residual[1]:.1f}, {mean_residual[2]:.1f}] |{mag3(*mean_residual):.1f}| µT")

    # After subtracting mean (what auto iron does)
    centered_rx = [r - mean_residual[0] for r in rx]
    centered_ry = [r - mean_residual[1] for r in ry]
    centered_rz = [r - mean_residual[2] for r in rz]
    centered_rmag = [mag3(centered_rx[i], centered_ry[i], centered_rz[i]) for i in range(len(residuals))]

    print(f"\n  After centering (subtracting mean):")
    print(f"  {'Axis':<6} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10} {'Range':>10}")
    print(f"  {'-'*56}")
    print(f"  {'X':<6} {mean(centered_rx):>10.1f} {std(centered_rx):>10.1f} {min(centered_rx):>10.1f} {max(centered_rx):>10.1f} {max(centered_rx)-min(centered_rx):>10.1f}")
    print(f"  {'Y':<6} {mean(centered_ry):>10.1f} {std(centered_ry):>10.1f} {min(centered_ry):>10.1f} {max(centered_ry):>10.1f} {max(centered_ry)-min(centered_ry):>10.1f}")
    print(f"  {'Z':<6} {mean(centered_rz):>10.1f} {std(centered_rz):>10.1f} {min(centered_rz):>10.1f} {max(centered_rz):>10.1f} {max(centered_rz)-min(centered_rz):>10.1f}")
    print(f"  {'|Mag|':<6} {mean(centered_rmag):>10.1f} {std(centered_rmag):>10.1f} {min(centered_rmag):>10.1f} {max(centered_rmag):>10.1f} {max(centered_rmag)-min(centered_rmag):>10.1f}")

    # SNR analysis
    def calc_snr(mags):
        baseline = percentile(mags, 25)
        peak = percentile(mags, 95)
        return peak / baseline if baseline > 0 else 0

    earth_snr = calc_snr(rmag)
    centered_snr = calc_snr(centered_rmag)

    print(f"\n  SNR Comparison:")
    print(f"    Earth-only:   {earth_snr:.2f}x")
    print(f"    Centered:     {centered_snr:.2f}x")
    print(f"    Difference:   {centered_snr - earth_snr:+.2f}x")

    # Key insight: coefficient of variation (relative spread)
    cv_earth = std(rmag) / mean(rmag) if mean(rmag) > 0 else 0
    cv_centered = std(centered_rmag) / mean(centered_rmag) if mean(centered_rmag) > 0 else 0

    print(f"\n  Coefficient of Variation (std/mean):")
    print(f"    Earth-only:   {cv_earth:.2f}")
    print(f"    Centered:     {cv_centered:.2f}")

    # The problem: when we center, we reduce the mean but not the std
    # This INCREASES the coefficient of variation
    # But SNR (peak/baseline) depends on the RATIO, not CV

    print(f"\n--- INTERPRETATION ---")
    if mean(rmag) > std(rmag) * 2:
        print(f"  Mean ({mean(rmag):.1f}) >> Std ({std(rmag):.1f})")
        print(f"  → Signal has large DC offset relative to variation")
        print(f"  → Centering helps by removing the offset")
    else:
        print(f"  Mean ({mean(rmag):.1f}) ≈ Std ({std(rmag):.1f})")
        print(f"  → Signal variation is significant relative to offset")
        print(f"  → Centering may hurt by reducing meaningful variation")

    return {
        'earth_snr': earth_snr,
        'centered_snr': centered_snr,
        'mean_residual': mean_residual,
        'std_residual': [std(rx), std(ry), std(rz)]
    }


def main():
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    print("=" * 80)
    print("RE-EXAMINATION: What Does Auto Iron Actually Capture?")
    print("=" * 80)
    print("""
Hard Iron Definition:
  - Constant magnetic offset from permanently magnetized device components
  - Device-specific, environment-independent
  - Typically 10-50 µT for consumer electronics

What "Auto Iron" Estimate Actually Captures:
  mean(sensor_residual) = hard_iron + mean(finger_magnets)

If finger magnets have significant mean field:
  - Auto iron estimate is dominated by finger magnets, not hard iron
  - Subtracting it removes the MEAN finger position signal
  - This "centers" the data around the average finger configuration

Question: Is centering useful?
  - YES for gesture detection (relative movement from mean)
  - NO for absolute finger position detection
  - Depends on whether the "mean position" is a meaningful reference
""")

    sessions = sorted(data_dir.glob('2025-12-15T22*.json'))

    results = []
    for session in sessions:
        if 'gambit' not in session.name.lower():
            r = analyze_finger_variation(str(session))
            if r:
                results.append(r)

    if len(results) >= 2:
        print(f"\n{'='*80}")
        print("SESSION COMPARISON")
        print("=" * 80)

        print(f"\n{'Session':<40} {'Earth SNR':>12} {'Centered SNR':>14} {'Effect':>10}")
        print("-" * 78)
        for i, r in enumerate(results):
            print(f"Session {i+1:<35} {r['earth_snr']:>12.2f}x {r['centered_snr']:>14.2f}x {r['centered_snr']-r['earth_snr']:>+10.2f}x")

        print(f"\n--- KEY INSIGHT ---")
        print(f"""
The session with HIGHER Earth-only SNR (more finger variation) suffers MORE
from centering. This is because:

1. High SNR = large signal range relative to baseline
2. The "mean residual" falls somewhere in the MIDDLE of this range
3. Subtracting the mean shifts the signal, changing the baseline
4. If baseline was already low, shifting it UP hurts SNR

The session with LOWER Earth-only SNR (less finger variation) is less affected
because there's less signal range to disturb.

This suggests "auto iron" is NOT measuring true hard iron, but rather
capturing the mean finger magnet configuration.
""")


if __name__ == '__main__':
    main()
