#!/usr/bin/env python3
"""
Baseline Magnitude Analysis

Key insight from previous analysis: What matters is not CV (stability)
but baseline MAGNITUDE (finger distance from sensor).

- Low baseline magnitude (~20-50 µT) = fingers extended/far = HELPS SNR
- High baseline magnitude (~150-400 µT) = fingers close = HURTS SNR

This makes physical sense:
- "Rest baseline" should be FINGERS EXTENDED (magnets far from sensor)
- This gives low baseline magnitude
- Subtracting it shifts signal DOWN, increasing dynamic range

The CV was a red herring - it measures stability, not finger position.
"""

import json
import math
from pathlib import Path
from typing import List, Dict


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


def analyze_baseline_by_magnitude(residuals: List[List[float]],
                                  window_size: int = 50) -> List[Dict]:
    """Analyze all possible baseline windows, sorted by magnitude."""
    results = []

    for i in range(0, len(residuals) - window_size, 10):  # Step by 10 for efficiency
        window = residuals[i:i + window_size]
        baseline = [mean([r[j] for r in window]) for j in range(3)]
        baseline_std = [std([r[j] for r in window]) for j in range(3)]
        baseline_mag = mag3(*baseline)
        baseline_cv = mag3(*baseline_std) / baseline_mag if baseline_mag > 0 else float('inf')

        # Apply this baseline to REST of session (excluding baseline period)
        eval_residuals = residuals[:i] + residuals[i + window_size:]
        if len(eval_residuals) < 100:
            continue

        raw_mags = [mag3(*r) for r in eval_residuals]
        corrected = [[r[j] - baseline[j] for j in range(3)] for r in eval_residuals]
        corrected_mags = [mag3(*r) for r in corrected]

        raw_snr = calc_snr(raw_mags)
        corrected_snr = calc_snr(corrected_mags)

        results.append({
            'index': i,
            'baseline': baseline,
            'baseline_mag': baseline_mag,
            'baseline_cv': baseline_cv,
            'raw_snr': raw_snr,
            'corrected_snr': corrected_snr,
            'snr_change': corrected_snr - raw_snr
        })

    return results


def analyze_session(filepath: str) -> Dict:
    """Analyze relationship between baseline magnitude and SNR impact."""
    with open(filepath) as f:
        data = json.load(f)

    samples = data.get('samples', [])
    name = Path(filepath).name

    print(f"\n{'='*80}")
    print(f"BASELINE MAGNITUDE ANALYSIS: {name}")
    print(f"{'='*80}")

    residuals = compute_earth_residuals(samples)
    print(f"Post-warmup residuals: {len(residuals)}")

    # Analyze all baseline windows
    results = analyze_baseline_by_magnitude(residuals)

    # Sort by baseline magnitude
    results.sort(key=lambda x: x['baseline_mag'])

    print(f"\n--- BASELINE MAGNITUDE vs SNR IMPACT ---")
    print(f"(Lower magnitude = fingers farther from sensor)")
    print(f"\n{'|Baseline|':>12} {'CV':>8} {'Raw SNR':>10} {'Corrected':>12} {'Change':>10}")
    print("-" * 54)

    # Show samples at different magnitude levels
    n = len(results)
    indices = [0, n//4, n//2, 3*n//4, n-1]  # 5 samples across range

    for idx in indices:
        r = results[idx]
        marker = "✓" if r['snr_change'] > 0.5 else "○" if r['snr_change'] > -0.5 else "✗"
        print(f"{marker} {r['baseline_mag']:>10.1f} {r['baseline_cv']:>8.2f} "
              f"{r['raw_snr']:>10.2f}x {r['corrected_snr']:>11.2f}x {r['snr_change']:>+10.2f}x")

    # Find the BEST baseline (highest SNR improvement)
    best = max(results, key=lambda x: x['snr_change'])
    worst = min(results, key=lambda x: x['snr_change'])

    print(f"\n--- OPTIMAL BASELINE ---")
    print(f"Best:  |baseline|={best['baseline_mag']:.1f} µT, SNR change={best['snr_change']:+.2f}x")
    print(f"Worst: |baseline|={worst['baseline_mag']:.1f} µT, SNR change={worst['snr_change']:+.2f}x")

    # Correlation analysis
    import statistics
    mags = [r['baseline_mag'] for r in results]
    snr_changes = [r['snr_change'] for r in results]

    # Simple correlation (Pearson)
    n = len(mags)
    mean_mag = mean(mags)
    mean_snr = mean(snr_changes)
    numerator = sum((mags[i] - mean_mag) * (snr_changes[i] - mean_snr) for i in range(n))
    denom = math.sqrt(sum((mags[i] - mean_mag)**2 for i in range(n)) *
                      sum((snr_changes[i] - mean_snr)**2 for i in range(n)))
    correlation = numerator / denom if denom > 0 else 0

    print(f"\n--- CORRELATION ---")
    print(f"Baseline magnitude vs SNR change: r = {correlation:.3f}")
    if correlation < -0.3:
        print("→ NEGATIVE correlation: Lower magnitude = Better SNR")
        print("→ Confirms: Fingers EXTENDED (far from sensor) = optimal baseline")
    elif correlation > 0.3:
        print("→ POSITIVE correlation: Higher magnitude = Better SNR")
    else:
        print("→ WEAK correlation: Magnitude alone doesn't predict SNR impact")

    # What's the threshold?
    good_results = [r for r in results if r['snr_change'] > 0]
    if good_results:
        avg_good_mag = mean([r['baseline_mag'] for r in good_results])
        print(f"\nAverage magnitude of HELPFUL baselines: {avg_good_mag:.1f} µT")
    bad_results = [r for r in results if r['snr_change'] < -1]
    if bad_results:
        avg_bad_mag = mean([r['baseline_mag'] for r in bad_results])
        print(f"Average magnitude of HARMFUL baselines: {avg_bad_mag:.1f} µT")

    return {
        'best': best,
        'worst': worst,
        'correlation': correlation,
        'results': results
    }


def main():
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    print("=" * 80)
    print("BASELINE MAGNITUDE INVESTIGATION")
    print("=" * 80)
    print("""
HYPOTHESIS: What matters for a good baseline is not CV (stability)
but MAGNITUDE (finger distance from sensor).

Physical model:
- Fingers extended (far from sensor) → Low residual magnitude → Good baseline
- Fingers flexed (close to sensor) → High residual magnitude → Bad baseline

Why? Subtracting a LOW magnitude baseline:
- Shifts the signal DOWN uniformly
- Preserves the RANGE of the signal (peak - baseline)
- SNR = peak/baseline benefits from lower baseline

Subtracting a HIGH magnitude baseline:
- Shifts the signal in a way that may reduce dynamic range
- If baseline > minimum signal, we get negative/inverted values
""")

    sessions = sorted(data_dir.glob('2025-12-15T22*.json'))

    all_results = []
    for session in sessions:
        if 'gambit' not in session.name.lower():
            results = analyze_session(str(session))
            all_results.append(results)

    print(f"\n{'='*80}")
    print("CONCLUSIONS")
    print("=" * 80)

    if all_results:
        correlations = [r['correlation'] for r in all_results]
        avg_corr = mean(correlations)

        print(f"""
AVERAGE CORRELATION (magnitude vs SNR change): {avg_corr:.3f}

INTERPRETATION:
{
"CONFIRMED: Lower baseline magnitude (fingers extended) consistently improves SNR."
if avg_corr < -0.3 else
"PARTIAL: Magnitude matters but other factors also contribute."
if avg_corr < 0 else
"UNEXPECTED: Higher magnitude correlates with better SNR (needs investigation)."
}

PRACTICAL IMPLICATION FOR "REST BASELINE":

The term "Rest Baseline" implies stability/stillness, but what actually
matters is FINGER POSITION (extended vs flexed), not motion.

Better terminology: "Extended Baseline" or "Reference Offset"

For calibration:
1. Prompt user: "Extend all fingers away from wrist"
2. Capture baseline during this position
3. Verify magnitude is LOW (< 100 µT suggested threshold)
4. Use this low-magnitude baseline for session

This works because:
- Extended fingers = magnets far from sensor = low residual
- Flexed fingers = magnets close to sensor = high residual
- Subtracting LOW baseline preserves more dynamic range

Hand movement during baseline capture is OK (helps Earth estimation)
as long as FINGERS remain extended.
""")


if __name__ == '__main__':
    main()
