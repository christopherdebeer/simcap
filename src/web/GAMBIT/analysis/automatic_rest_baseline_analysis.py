#!/usr/bin/env python3
"""
Automatic Rest Baseline Analysis

Investigation: Can we automatically capture a "rest baseline" from the
initial period of a session, without requiring explicit user calibration?

Hypothesis: Users naturally have fingers at rest when a session starts.
The first N samples could serve as an automatic baseline.

Test conditions:
1. Use first N samples as baseline (N = 50, 100, 200)
2. Compare to: no baseline, self-centering, full-session baseline
3. Analyze both sessions to understand variability

Key questions:
1. Does early-period baseline improve or hurt signal?
2. What happens if user is NOT at rest during initial period?
3. What's the optimal baseline window size?
4. How does this compare to explicit calibration?
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Tuple


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


def calc_snr(mags: List[float]) -> float:
    """Calculate SNR as 95th/25th percentile ratio."""
    if len(mags) < 10:
        return 0
    baseline = percentile(mags, 25)
    peak = percentile(mags, 95)
    return peak / baseline if baseline > 0 else 0


def compute_earth_residuals(samples: List[Dict]) -> Tuple[List[List[float]], List[float]]:
    """
    Compute Earth-subtracted residuals.
    Returns (residual_vectors, earth_world_final)
    """
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

    return residuals, earth_world


def analyze_baseline_strategy(residuals: List[List[float]],
                               baseline_start: int,
                               baseline_end: int,
                               eval_start: int = None,
                               eval_end: int = None) -> Dict:
    """
    Analyze a specific baseline strategy.

    Args:
        residuals: List of [x, y, z] residual vectors
        baseline_start: Start index for baseline computation
        baseline_end: End index for baseline computation
        eval_start: Start index for evaluation (default: after baseline)
        eval_end: End index for evaluation (default: end of session)

    Returns:
        Dict with baseline, SNR metrics, and analysis
    """
    if eval_start is None:
        eval_start = baseline_end
    if eval_end is None:
        eval_end = len(residuals)

    # Compute baseline from specified window
    baseline_residuals = residuals[baseline_start:baseline_end]
    baseline = [
        mean([r[0] for r in baseline_residuals]),
        mean([r[1] for r in baseline_residuals]),
        mean([r[2] for r in baseline_residuals])
    ]
    baseline_std = [
        std([r[0] for r in baseline_residuals]),
        std([r[1] for r in baseline_residuals]),
        std([r[2] for r in baseline_residuals])
    ]

    # Apply baseline correction to evaluation period
    eval_residuals = residuals[eval_start:eval_end]

    # No correction (Earth-only)
    raw_mags = [mag3(*r) for r in eval_residuals]

    # With baseline correction
    corrected = [[r[i] - baseline[i] for i in range(3)] for r in eval_residuals]
    corrected_mags = [mag3(*r) for r in corrected]

    return {
        'baseline': baseline,
        'baseline_mag': mag3(*baseline),
        'baseline_std': baseline_std,
        'baseline_std_mag': mag3(*baseline_std),
        'baseline_cv': mag3(*baseline_std) / mag3(*baseline) if mag3(*baseline) > 0 else float('inf'),
        'raw_snr': calc_snr(raw_mags),
        'corrected_snr': calc_snr(corrected_mags),
        'snr_change': calc_snr(corrected_mags) - calc_snr(raw_mags),
        'n_baseline': baseline_end - baseline_start,
        'n_eval': eval_end - eval_start,
        'raw_mean': mean(raw_mags),
        'corrected_mean': mean(corrected_mags),
        'raw_baseline_pct': percentile(raw_mags, 25),
        'corrected_baseline_pct': percentile(corrected_mags, 25),
    }


def analyze_session_baselines(filepath: str) -> Dict:
    """Comprehensive baseline analysis for a session."""
    with open(filepath) as f:
        data = json.load(f)

    samples = data.get('samples', [])
    name = Path(filepath).name

    print(f"\n{'='*80}")
    print(f"AUTOMATIC REST BASELINE ANALYSIS: {name}")
    print(f"{'='*80}")

    # Compute residuals (skip first 100 for Earth warmup)
    all_residuals, _ = compute_earth_residuals(samples)
    residuals = all_residuals[100:]  # Post-warmup residuals

    print(f"Total samples: {len(samples)}")
    print(f"Post-warmup residuals: {len(residuals)}")

    # Analyze different baseline strategies
    strategies = {}

    # Strategy 1: No baseline (Earth-only) - reference
    strategies['earth_only'] = analyze_baseline_strategy(
        residuals, 0, 0, eval_start=0, eval_end=len(residuals)
    )
    # For earth-only, raw_snr is the actual SNR
    strategies['earth_only']['corrected_snr'] = strategies['earth_only']['raw_snr']
    strategies['earth_only']['snr_change'] = 0

    # Strategy 2: Self-centering (full session baseline)
    strategies['self_center'] = analyze_baseline_strategy(
        residuals, 0, len(residuals), eval_start=0, eval_end=len(residuals)
    )

    # Strategy 3: Early baseline (first 50 samples = ~1 second @ 50Hz)
    if len(residuals) > 100:
        strategies['early_50'] = analyze_baseline_strategy(
            residuals, 0, 50, eval_start=50, eval_end=len(residuals)
        )

    # Strategy 4: Early baseline (first 100 samples = ~2 seconds)
    if len(residuals) > 150:
        strategies['early_100'] = analyze_baseline_strategy(
            residuals, 0, 100, eval_start=100, eval_end=len(residuals)
        )

    # Strategy 5: Early baseline (first 200 samples = ~4 seconds)
    if len(residuals) > 250:
        strategies['early_200'] = analyze_baseline_strategy(
            residuals, 0, 200, eval_start=200, eval_end=len(residuals)
        )

    # Strategy 6: Late baseline (last 200 samples) - for comparison
    if len(residuals) > 250:
        strategies['late_200'] = analyze_baseline_strategy(
            residuals, len(residuals)-200, len(residuals),
            eval_start=0, eval_end=len(residuals)-200
        )

    # Print results
    print(f"\n--- BASELINE STRATEGY COMPARISON ---")
    print(f"{'Strategy':<20} {'Baseline|':>10} {'CV':>8} {'Earth SNR':>12} {'Corrected':>12} {'Change':>10}")
    print("-" * 74)

    for name, s in strategies.items():
        print(f"{name:<20} {s['baseline_mag']:>9.1f} {s['baseline_cv']:>8.2f} "
              f"{s['raw_snr']:>12.2f}x {s['corrected_snr']:>11.2f}x {s['snr_change']:>+10.2f}x")

    # Analyze early period characteristics
    print(f"\n--- EARLY PERIOD ANALYSIS ---")
    print("Is the early period suitable as a rest baseline?")

    early = residuals[:100] if len(residuals) >= 100 else residuals
    late = residuals[-100:] if len(residuals) >= 100 else residuals

    early_mean = [mean([r[i] for r in early]) for i in range(3)]
    early_std = [std([r[i] for r in early]) for i in range(3)]
    late_mean = [mean([r[i] for r in late]) for i in range(3)]
    late_std = [std([r[i] for r in late]) for i in range(3)]

    print(f"\n  Early period (first 100 samples):")
    print(f"    Mean: [{early_mean[0]:.1f}, {early_mean[1]:.1f}, {early_mean[2]:.1f}] |{mag3(*early_mean):.1f}| µT")
    print(f"    Std:  [{early_std[0]:.1f}, {early_std[1]:.1f}, {early_std[2]:.1f}] |{mag3(*early_std):.1f}| µT")
    print(f"    CV:   {mag3(*early_std)/mag3(*early_mean):.2f}")

    print(f"\n  Late period (last 100 samples):")
    print(f"    Mean: [{late_mean[0]:.1f}, {late_mean[1]:.1f}, {late_mean[2]:.1f}] |{mag3(*late_mean):.1f}| µT")
    print(f"    Std:  [{late_std[0]:.1f}, {late_std[1]:.1f}, {late_std[2]:.1f}] |{mag3(*late_std):.1f}| µT")
    print(f"    CV:   {mag3(*late_std)/mag3(*late_mean):.2f}")

    # Key metric: is early period lower variance (more "at rest")?
    early_cv = mag3(*early_std) / mag3(*early_mean) if mag3(*early_mean) > 0 else float('inf')
    late_cv = mag3(*late_std) / mag3(*late_mean) if mag3(*late_mean) > 0 else float('inf')

    print(f"\n  Comparison:")
    if early_cv < late_cv * 0.8:
        print(f"    ✓ Early period is MORE STABLE (CV {early_cv:.2f} vs {late_cv:.2f})")
        print(f"      → Suggests user was at rest during session start")
    elif early_cv > late_cv * 1.2:
        print(f"    ✗ Early period is LESS STABLE (CV {early_cv:.2f} vs {late_cv:.2f})")
        print(f"      → User was moving at session start")
    else:
        print(f"    ○ Early and late periods have SIMILAR stability")
        print(f"      (CV {early_cv:.2f} vs {late_cv:.2f})")

    return strategies


def main():
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    print("=" * 80)
    print("AUTOMATIC REST BASELINE INVESTIGATION")
    print("=" * 80)
    print("""
Question: Can we automatically capture a "rest baseline" from the initial
period of a session without explicit user calibration?

Approach: Compare baseline strategies using different time windows.

Key metrics:
- CV (Coefficient of Variation) = std/mean - lower indicates more stability
- SNR change - positive means baseline helps, negative means it hurts
""")

    sessions = sorted(data_dir.glob('2025-12-15T22*.json'))

    all_results = []
    for session in sessions:
        if 'gambit' not in session.name.lower():
            results = analyze_session_baselines(str(session))
            all_results.append(results)

    # Summary comparison
    if len(all_results) >= 2:
        print(f"\n{'='*80}")
        print("CROSS-SESSION SUMMARY")
        print("=" * 80)

        print(f"\n--- SNR CHANGE BY STRATEGY ---")
        print(f"{'Strategy':<20} {'Session 1':>12} {'Session 2':>12} {'Average':>12}")
        print("-" * 58)

        for strategy in ['earth_only', 'self_center', 'early_50', 'early_100', 'early_200']:
            if strategy in all_results[0] and strategy in all_results[1]:
                s1 = all_results[0][strategy]['snr_change']
                s2 = all_results[1][strategy]['snr_change']
                avg = (s1 + s2) / 2
                print(f"{strategy:<20} {s1:>+12.2f}x {s2:>+12.2f}x {avg:>+12.2f}x")

        print(f"\n--- CONCLUSIONS ---")

        # Find best automatic strategy
        auto_strategies = ['early_50', 'early_100', 'early_200']
        best_auto = None
        best_avg = float('-inf')

        for strategy in auto_strategies:
            if strategy in all_results[0] and strategy in all_results[1]:
                avg = (all_results[0][strategy]['snr_change'] +
                       all_results[1][strategy]['snr_change']) / 2
                if avg > best_avg:
                    best_avg = avg
                    best_auto = strategy

        self_avg = (all_results[0]['self_center']['snr_change'] +
                    all_results[1]['self_center']['snr_change']) / 2

        print(f"""
1. SELF-CENTERING (sliding window average):
   Average SNR change: {self_avg:+.2f}x
   → {"HARMFUL - removes signal variance" if self_avg < -1 else "MARGINAL" if self_avg < 1 else "HELPFUL"}

2. AUTOMATIC EARLY BASELINE (first N samples):
   Best strategy: {best_auto} (average {best_avg:+.2f}x)
   → {"HELPFUL - preserves more signal" if best_avg > 0.5 else "MARGINAL" if best_avg > -0.5 else "HARMFUL"}

3. COMPARISON:
   Early baseline vs self-centering: {best_avg - self_avg:+.2f}x improvement
""")

        # Key finding
        if best_avg > self_avg + 1:
            print("""
KEY FINDING: An automatic early baseline phase IS beneficial compared to
self-centering, even without explicit user calibration.

RECOMMENDATION: Implement a "startup baseline" phase that:
1. Captures residual mean during first ~2-4 seconds (100-200 samples)
2. Uses this as the reference offset for subsequent samples
3. Does NOT require explicit user action (automatic)

CAVEAT: Signal quality depends on user having fingers relatively still
during the startup period. Consider:
- UI prompt: "Hold still for 2 seconds..."
- Gyroscope monitoring to detect if user was moving
- Quality indicator based on baseline CV
""")
        else:
            print("""
KEY FINDING: Automatic early baseline does NOT reliably improve signal
over Earth-only calibration.

RECOMMENDATION: Keep Earth-only as default. If baseline calibration is
desired, require explicit user action (fingers extended, hold still).
""")


if __name__ == '__main__':
    main()
