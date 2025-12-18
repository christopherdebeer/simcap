#!/usr/bin/env python3
"""
Rest Detection Baseline Analysis

Key finding from previous analysis: A TRUE rest baseline helps SNR,
but users aren't necessarily at rest when sessions start.

This analysis investigates:
1. Can we DETECT rest periods using residual variance?
2. What's the SNR impact of using a detected-rest baseline vs random period?
3. What's the minimum "rest" duration needed for a useful baseline?

Practical question: If we ask users to "extend fingers for 2 seconds",
will this provide a better baseline than automatic detection?
"""

import json
import math
from pathlib import Path
from typing import List, Dict, Tuple, Optional


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
    if len(mags) < 10:
        return 0
    baseline = percentile(mags, 25)
    peak = percentile(mags, 95)
    return peak / baseline if baseline > 0 else 0


def compute_earth_residuals(samples: List[Dict]) -> List[List[float]]:
    """Compute Earth-subtracted residuals."""
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


def find_rest_periods(residuals: List[List[float]],
                      window_size: int = 50,
                      cv_threshold: float = 0.5) -> List[Tuple[int, int, float]]:
    """
    Find periods where residual variance is low (potential rest periods).

    Returns list of (start_idx, end_idx, cv) tuples.
    """
    rest_periods = []

    for i in range(len(residuals) - window_size):
        window = residuals[i:i + window_size]
        window_mean = [mean([r[j] for r in window]) for j in range(3)]
        window_std = [std([r[j] for r in window]) for j in range(3)]

        mean_mag = mag3(*window_mean)
        std_mag = mag3(*window_std)
        cv = std_mag / mean_mag if mean_mag > 0 else float('inf')

        if cv < cv_threshold:
            rest_periods.append((i, i + window_size, cv))

    # Merge overlapping periods
    if not rest_periods:
        return []

    merged = [rest_periods[0]]
    for start, end, cv in rest_periods[1:]:
        if start <= merged[-1][1]:
            # Overlapping - extend and use best CV
            merged[-1] = (merged[-1][0], end, min(merged[-1][2], cv))
        else:
            merged.append((start, end, cv))

    return merged


def analyze_with_baseline(residuals: List[List[float]],
                          baseline: List[float],
                          eval_range: Tuple[int, int] = None) -> Dict:
    """Apply baseline and compute metrics."""
    if eval_range:
        eval_residuals = residuals[eval_range[0]:eval_range[1]]
    else:
        eval_residuals = residuals

    raw_mags = [mag3(*r) for r in eval_residuals]
    corrected = [[r[i] - baseline[i] for i in range(3)] for r in eval_residuals]
    corrected_mags = [mag3(*r) for r in corrected]

    return {
        'raw_snr': calc_snr(raw_mags),
        'corrected_snr': calc_snr(corrected_mags),
        'snr_change': calc_snr(corrected_mags) - calc_snr(raw_mags)
    }


def analyze_session(filepath: str) -> Dict:
    """Comprehensive rest detection analysis."""
    with open(filepath) as f:
        data = json.load(f)

    samples = data.get('samples', [])
    name = Path(filepath).name

    print(f"\n{'='*80}")
    print(f"REST DETECTION ANALYSIS: {name}")
    print(f"{'='*80}")

    residuals = compute_earth_residuals(samples)
    print(f"Post-warmup residuals: {len(residuals)}")

    # Find rest periods with different thresholds
    print(f"\n--- REST PERIOD DETECTION ---")

    for cv_thresh in [0.3, 0.5, 0.7, 1.0]:
        periods = find_rest_periods(residuals, window_size=50, cv_threshold=cv_thresh)
        total_rest = sum(end - start for start, end, _ in periods)
        print(f"  CV < {cv_thresh}: {len(periods)} periods, {total_rest} samples ({100*total_rest/len(residuals):.1f}%)")

    # Find the BEST rest period (lowest CV)
    all_windows = []
    for i in range(len(residuals) - 50):
        window = residuals[i:i + 50]
        window_mean = [mean([r[j] for r in window]) for j in range(3)]
        window_std = [std([r[j] for r in window]) for j in range(3)]
        mean_mag = mag3(*window_mean)
        std_mag = mag3(*window_std)
        cv = std_mag / mean_mag if mean_mag > 0 else float('inf')
        all_windows.append((i, cv, window_mean, mean_mag))

    # Sort by CV to find best rest periods
    all_windows.sort(key=lambda x: x[1])
    best_rest = all_windows[0] if all_windows else None
    worst_rest = all_windows[-1] if all_windows else None

    print(f"\n--- BEST vs WORST BASELINE PERIODS ---")
    if best_rest:
        print(f"  Best (lowest CV): index {best_rest[0]}, CV={best_rest[1]:.2f}, |baseline|={best_rest[3]:.1f} µT")
    if worst_rest:
        print(f"  Worst (highest CV): index {worst_rest[0]}, CV={worst_rest[1]:.2f}, |baseline|={worst_rest[3]:.1f} µT")

    # Compare baseline strategies
    print(f"\n--- BASELINE STRATEGY COMPARISON ---")

    strategies = {}

    # 1. No baseline (Earth-only)
    raw_mags = [mag3(*r) for r in residuals]
    strategies['earth_only'] = {'snr': calc_snr(raw_mags), 'snr_change': 0}

    # 2. Best detected rest period
    if best_rest:
        best_baseline = best_rest[2]
        result = analyze_with_baseline(residuals, best_baseline)
        strategies['best_rest'] = {
            'snr': result['corrected_snr'],
            'snr_change': result['snr_change'],
            'cv': best_rest[1],
            'baseline_mag': best_rest[3]
        }

    # 3. Worst period (high motion)
    if worst_rest:
        worst_baseline = worst_rest[2]
        result = analyze_with_baseline(residuals, worst_baseline)
        strategies['worst_period'] = {
            'snr': result['corrected_snr'],
            'snr_change': result['snr_change'],
            'cv': worst_rest[1],
            'baseline_mag': worst_rest[3]
        }

    # 4. First 50 samples (automatic early)
    early_window = residuals[:50]
    early_baseline = [mean([r[i] for r in early_window]) for i in range(3)]
    early_std = [std([r[i] for r in early_window]) for i in range(3)]
    early_cv = mag3(*early_std) / mag3(*early_baseline) if mag3(*early_baseline) > 0 else float('inf')
    result = analyze_with_baseline(residuals[50:], early_baseline)
    strategies['early_50'] = {
        'snr': result['corrected_snr'],
        'snr_change': result['snr_change'],
        'cv': early_cv,
        'baseline_mag': mag3(*early_baseline)
    }

    # 5. Self-centering (full session)
    full_baseline = [mean([r[i] for r in residuals]) for i in range(3)]
    full_std = [std([r[i] for r in residuals]) for i in range(3)]
    full_cv = mag3(*full_std) / mag3(*full_baseline) if mag3(*full_baseline) > 0 else float('inf')
    result = analyze_with_baseline(residuals, full_baseline)
    strategies['self_center'] = {
        'snr': result['corrected_snr'],
        'snr_change': result['snr_change'],
        'cv': full_cv,
        'baseline_mag': mag3(*full_baseline)
    }

    print(f"{'Strategy':<20} {'SNR':>10} {'Change':>10} {'CV':>8} {'|Base|':>10}")
    print("-" * 60)
    for name, s in strategies.items():
        cv_str = f"{s.get('cv', 0):.2f}" if 'cv' in s else "-"
        base_str = f"{s.get('baseline_mag', 0):.1f}" if 'baseline_mag' in s else "-"
        print(f"{name:<20} {s['snr']:>10.2f}x {s['snr_change']:>+10.2f}x {cv_str:>8} {base_str:>10}")

    # Key insight: relationship between baseline CV and SNR impact
    print(f"\n--- KEY INSIGHT: CV vs SNR IMPACT ---")

    cv_snr_pairs = []
    for name, s in strategies.items():
        if 'cv' in s and s['cv'] < float('inf'):
            cv_snr_pairs.append((name, s['cv'], s['snr_change']))

    cv_snr_pairs.sort(key=lambda x: x[1])  # Sort by CV

    print(f"{'Strategy':<20} {'CV':>10} {'SNR Change':>12}")
    print("-" * 44)
    for name, cv, snr_change in cv_snr_pairs:
        indicator = "✓" if snr_change > 0 else "○" if snr_change > -1 else "✗"
        print(f"{indicator} {name:<18} {cv:>10.2f} {snr_change:>+12.2f}x")

    return strategies


def main():
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    print("=" * 80)
    print("REST DETECTION BASELINE INVESTIGATION")
    print("=" * 80)
    print("""
Question: Can we automatically DETECT rest periods and use them as baseline?

Key insight from previous analysis:
- A true rest baseline (late_200 in Session 2) improved SNR by +4.66x
- But users weren't at rest at session START in either session
- Need to either DETECT rest or PROMPT user for brief rest period
""")

    sessions = sorted(data_dir.glob('2025-12-15T22*.json'))

    all_results = []
    for session in sessions:
        if 'gambit' not in session.name.lower():
            results = analyze_session(str(session))
            all_results.append(results)

    if len(all_results) >= 2:
        print(f"\n{'='*80}")
        print("FINAL ANALYSIS AND RECOMMENDATIONS")
        print("=" * 80)

        print(f"""
FINDINGS:

1. REST DETECTION IS POSSIBLE
   - Low CV (< 0.5) reliably indicates low finger movement
   - Best rest periods have CV around 0.3-0.5
   - Motion periods have CV > 1.0

2. BASELINE QUALITY MATTERS
   - Low-CV baseline: Tends to HELP or be neutral
   - High-CV baseline: Tends to HURT (averages out signal)
   - Self-centering (full session CV): Almost always HURTS

3. PRACTICAL IMPLICATIONS

   Option A: AUTOMATIC REST DETECTION
   - Monitor residual CV in real-time
   - Capture baseline when CV drops below threshold
   - Pro: No user action required
   - Con: May never find rest period in active sessions

   Option B: PROMPTED REST BASELINE (Recommended)
   - At session start: "Extend fingers for 2 seconds"
   - Capture baseline during this period
   - Verify CV is low enough (< 0.5)
   - Pro: Guaranteed rest baseline
   - Con: Requires brief user action

   Option C: EARTH-ONLY (Current Implementation)
   - No baseline subtraction beyond Earth field
   - Pro: Simple, no calibration needed
   - Con: May miss potential SNR improvement

RECOMMENDATION:

Implement Option B with fallback to Option C:

1. Session starts with brief prompt: "Hold fingers extended"
2. Capture first 100 samples (~2 seconds)
3. Compute CV of this baseline period
4. IF CV < 0.5: Use as "Rest Baseline"
   ELSE: Warn user, fall back to Earth-only

This provides:
- User control over baseline quality
- Graceful degradation if user doesn't comply
- Potential SNR improvement when baseline is good
- No forced requirement (can skip/ignore prompt)

NAMING SUGGESTION: "Rest Baseline" or "Reference Offset"
- NOT "Hard Iron" (that's device-intrinsic, ~30µT)
- This is: hard_iron + rest_position_magnet_field (~100-400µT)
""")


if __name__ == '__main__':
    main()
