#!/usr/bin/env python3
"""
Cross-Session Baseline Test

Hypothesis: If Session 1 had relatively static fingers (low variation),
its mean residual could serve as a "rest baseline" for Session 2.

Naming options for this baseline:
- "Rest Baseline" - assumes fingers at rest position
- "Reference Offset" - neutral term for the combined offset
- "Magnet Baseline" - acknowledges it includes finger magnets
- "Sensor Baseline" - generic sensor-frame offset
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


def compute_earth_residuals(samples):
    """Compute Earth-subtracted residuals for a session."""
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

    return residuals[100:]  # Skip warmup


def calc_snr(mags):
    baseline = percentile(mags, 25)
    peak = percentile(mags, 95)
    return peak / baseline if baseline > 0 else 0


def main():
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    # Load both sessions
    session1_path = data_dir / '2025-12-15T22:40:44.984Z.json'  # 2564 samples, lower variation
    session2_path = data_dir / '2025-12-15T22_35_15.567Z.json'  # 968 samples, higher variation

    with open(session1_path) as f:
        session1 = json.load(f)
    with open(session2_path) as f:
        session2 = json.load(f)

    print("=" * 80)
    print("CROSS-SESSION BASELINE TEST")
    print("=" * 80)

    # Compute residuals for both sessions
    residuals1 = compute_earth_residuals(session1['samples'])
    residuals2 = compute_earth_residuals(session2['samples'])

    print(f"\nSession 1 (baseline source): {len(residuals1)} samples")
    print(f"Session 2 (test target):     {len(residuals2)} samples")

    # Session 1 statistics (potential baseline)
    r1_mean = [
        mean([r[0] for r in residuals1]),
        mean([r[1] for r in residuals1]),
        mean([r[2] for r in residuals1])
    ]
    r1_std = [
        std([r[0] for r in residuals1]),
        std([r[1] for r in residuals1]),
        std([r[2] for r in residuals1])
    ]

    print(f"\n--- SESSION 1 (Baseline Candidate) ---")
    print(f"Mean residual: [{r1_mean[0]:.1f}, {r1_mean[1]:.1f}, {r1_mean[2]:.1f}] |{mag3(*r1_mean):.1f}| µT")
    print(f"Std residual:  [{r1_std[0]:.1f}, {r1_std[1]:.1f}, {r1_std[2]:.1f}] |{mag3(*r1_std):.1f}| µT")
    print(f"Coefficient of Variation: {mag3(*r1_std) / mag3(*r1_mean):.2f}")

    # Session 2 statistics
    r2_mean = [
        mean([r[0] for r in residuals2]),
        mean([r[1] for r in residuals2]),
        mean([r[2] for r in residuals2])
    ]
    r2_std = [
        std([r[0] for r in residuals2]),
        std([r[1] for r in residuals2]),
        std([r[2] for r in residuals2])
    ]

    print(f"\n--- SESSION 2 (Test Target) ---")
    print(f"Mean residual: [{r2_mean[0]:.1f}, {r2_mean[1]:.1f}, {r2_mean[2]:.1f}] |{mag3(*r2_mean):.1f}| µT")
    print(f"Std residual:  [{r2_std[0]:.1f}, {r2_std[1]:.1f}, {r2_std[2]:.1f}] |{mag3(*r2_std):.1f}| µT")
    print(f"Coefficient of Variation: {mag3(*r2_std) / mag3(*r2_mean):.2f}")

    # Compare baselines
    baseline_diff = [r2_mean[i] - r1_mean[i] for i in range(3)]
    print(f"\n--- BASELINE COMPARISON ---")
    print(f"Difference (S2 - S1): [{baseline_diff[0]:.1f}, {baseline_diff[1]:.1f}, {baseline_diff[2]:.1f}] |{mag3(*baseline_diff):.1f}| µT")

    # Now test: Apply Session 1's baseline to Session 2
    print(f"\n{'='*80}")
    print("APPLYING SESSION 1 BASELINE TO SESSION 2")
    print("=" * 80)

    # Compute different corrections for Session 2
    r2_mags_raw = [mag3(*r) for r in residuals2]

    # Using Session 2's own mean (self-centering)
    r2_self_centered = [[r[i] - r2_mean[i] for i in range(3)] for r in residuals2]
    r2_mags_self = [mag3(*r) for r in r2_self_centered]

    # Using Session 1's mean as baseline
    r2_cross_centered = [[r[i] - r1_mean[i] for i in range(3)] for r in residuals2]
    r2_mags_cross = [mag3(*r) for r in r2_cross_centered]

    # Calculate SNRs
    snr_raw = calc_snr(r2_mags_raw)
    snr_self = calc_snr(r2_mags_self)
    snr_cross = calc_snr(r2_mags_cross)

    print(f"\nSession 2 SNR Results:")
    print(f"  {'Method':<35} {'SNR':>10} {'vs Raw':>12}")
    print(f"  {'-'*57}")
    print(f"  {'Earth-only (no baseline):':<35} {snr_raw:>10.2f}x {'-':>12}")
    print(f"  {'Self-centered (own mean):':<35} {snr_self:>10.2f}x {snr_self - snr_raw:>+12.2f}x")
    print(f"  {'Cross-session (S1 baseline):':<35} {snr_cross:>10.2f}x {snr_cross - snr_raw:>+12.2f}x")

    # Detailed analysis of what cross-session baseline does
    print(f"\n--- DETAILED ANALYSIS ---")

    # Statistics after each correction
    print(f"\n  After self-centering:")
    print(f"    Mean magnitude: {mean(r2_mags_self):.1f} µT")
    print(f"    25th percentile (baseline): {percentile(r2_mags_self, 25):.1f} µT")
    print(f"    95th percentile (peak): {percentile(r2_mags_self, 95):.1f} µT")

    print(f"\n  After cross-session baseline:")
    print(f"    Mean magnitude: {mean(r2_mags_cross):.1f} µT")
    print(f"    25th percentile (baseline): {percentile(r2_mags_cross, 25):.1f} µT")
    print(f"    95th percentile (peak): {percentile(r2_mags_cross, 95):.1f} µT")

    # The key insight: where does the baseline shift take us?
    print(f"\n--- INTERPRETATION ---")

    if snr_cross > snr_raw:
        print(f"✓ Cross-session baseline IMPROVES SNR by {snr_cross - snr_raw:+.2f}x")
        print(f"  This suggests Session 1's mean represents a useful reference point")
    elif snr_cross > snr_self:
        print(f"○ Cross-session baseline is BETTER than self-centering")
        print(f"  ({snr_cross:.2f}x vs {snr_self:.2f}x)")
        print(f"  But still worse than no baseline ({snr_raw:.2f}x)")
    else:
        print(f"✗ Cross-session baseline does NOT help")
        print(f"  Session 1's mean differs too much from Session 2's finger positions")

    # Check if the baselines are similar enough
    baseline_similarity = 1 - (mag3(*baseline_diff) / max(mag3(*r1_mean), mag3(*r2_mean)))
    print(f"\n  Baseline similarity: {baseline_similarity*100:.1f}%")

    if baseline_similarity < 0.5:
        print(f"  ⚠ Low similarity - sessions likely had different finger configurations")
    else:
        print(f"  ✓ Reasonable similarity - baseline transfer may be meaningful")

    # What would be ideal?
    print(f"\n--- IMPLICATIONS FOR 'REST BASELINE' CONCEPT ---")
    print(f"""
For a "Rest Baseline" to work, we need:
1. A consistent finger position during baseline capture (e.g., fingers extended)
2. The baseline to be captured WITHOUT finger movement variation
3. Cross-session consistency (same device position on wrist)

Session 1 characteristics:
  - Lower variation (CV = {mag3(*r1_std) / mag3(*r1_mean):.2f})
  - Mean residual: {mag3(*r1_mean):.1f} µT

Session 2 characteristics:
  - Higher variation (CV = {mag3(*r2_std) / mag3(*r2_mean):.2f})
  - Mean residual: {mag3(*r2_mean):.1f} µT

The {mag3(*baseline_diff):.1f} µT difference between session means suggests
{"different finger configurations" if mag3(*baseline_diff) > 100 else "possibly similar rest positions"}.
""")


if __name__ == '__main__':
    main()
