#!/usr/bin/env python3
"""
Analyze Bootstrap Calibration Impact

Compares calibration performance:
1. Without bootstrap (starting from zero)
2. With current hardcoded bootstrap (from 2025-12-29 session)
3. With optimal bootstrap (computed offline from all sessions)

Goal: Determine if offline optimization can provide better bootstrap values.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from scipy.optimize import least_squares
import matplotlib.pyplot as plt

# Edinburgh geomagnetic reference
EARTH_H = 16.0  # µT
EARTH_V = 47.8  # µT
EARTH_MAG = np.sqrt(EARTH_H**2 + EARTH_V**2)
EARTH_WORLD = np.array([EARTH_H, 0, EARTH_V])  # NED

# Current hardcoded bootstrap values (from 2025-12-29 session)
CURRENT_BOOTSTRAP = {
    'hard_iron': np.array([-33.0, -69.1, -50.8]),
    'ranges': np.array([79.6, 99.0, 115.4])  # 2x half-ranges
}

print("=" * 80)
print("BOOTSTRAP CALIBRATION IMPACT ANALYSIS")
print("=" * 80)
print(f"\nCurrent hardcoded bootstrap (from 2025-12-29):")
print(f"  Hard iron: {CURRENT_BOOTSTRAP['hard_iron']}")
print(f"  Ranges: {CURRENT_BOOTSTRAP['ranges']}")


def load_session(filepath: Path) -> Optional[Dict]:
    """Load and validate session data."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        samples = data.get('samples', [])
        if len(samples) < 100:
            return None
        if 'mx_ut' not in samples[0] and 'mx' not in samples[0]:
            return None
        return data
    except:
        return None


def extract_and_filter(samples: List[Dict], max_mag=200, max_comp=150):
    """Extract and filter magnetometer/accelerometer data."""
    mx = np.array([s.get('mx_ut', s.get('mx', 0)) for s in samples])
    my = np.array([s.get('my_ut', s.get('my', 0)) for s in samples])
    mz = np.array([s.get('mz_ut', s.get('mz', 0)) for s in samples])
    ax = np.array([s.get('ax_g', s.get('ax', 0)) for s in samples])
    ay = np.array([s.get('ay_g', s.get('ay', 0)) for s in samples])
    az = np.array([s.get('az_g', s.get('az', 0)) for s in samples])

    mag = np.sqrt(mx**2 + my**2 + mz**2)
    valid = (mag <= max_mag) & (np.abs(mx) <= max_comp) & (np.abs(my) <= max_comp) & (np.abs(mz) <= max_comp)

    return mx[valid], my[valid], mz[valid], ax[valid], ay[valid], az[valid]


def accel_to_roll_pitch(ax, ay, az):
    a_norm = np.sqrt(ax**2 + ay**2 + az**2)
    if a_norm < 0.1:
        return 0, 0
    ax, ay, az = ax/a_norm, ay/a_norm, az/a_norm
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    return roll, pitch


def tilt_compensate(mx, my, mz, roll, pitch):
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    mx_h = mx * cos_p + my * sin_r * sin_p + mz * cos_r * sin_p
    my_h = my * cos_r - mz * sin_r
    mz_h = -mx * sin_p + my * cos_r * sin_p + mz * cos_r * cos_p
    return mx_h, my_h, mz_h


def euler_to_rotation(roll, pitch, yaw):
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])


def scipy_calibration(samples: List[Dict], initial_offset: np.ndarray = None) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run scipy least_squares calibration with optional initial offset."""
    if initial_offset is None:
        # Compute from min-max
        mx = np.array([s['mx'] for s in samples])
        my = np.array([s['my'] for s in samples])
        mz = np.array([s['mz'] for s in samples])
        initial_offset = np.array([
            (mx.max() + mx.min()) / 2,
            (my.max() + my.min()) / 2,
            (mz.max() + mz.min()) / 2
        ])

    def residual_func(params):
        offset = params[:3]
        S = params[3:12].reshape(3, 3)

        residuals = []
        for sample in samples:
            centered = np.array([
                sample['mx'] - offset[0],
                sample['my'] - offset[1],
                sample['mz'] - offset[2]
            ])
            corrected = S @ centered

            roll, pitch = accel_to_roll_pitch(sample['ax'], sample['ay'], sample['az'])
            mx_h, my_h, _ = tilt_compensate(corrected[0], corrected[1], corrected[2], roll, pitch)
            yaw = np.atan2(-my_h, mx_h)

            R = euler_to_rotation(roll, pitch, yaw)
            earth_device = R.T @ EARTH_WORLD

            diff = corrected - earth_device
            residuals.extend(diff.tolist())

        return np.array(residuals)

    x0 = np.concatenate([initial_offset, np.eye(3).flatten()])
    result = least_squares(residual_func, x0, method='lm', max_nfev=10000)

    offset = result.x[:3]
    S = result.x[3:12].reshape(3, 3)
    residual = np.sqrt(np.mean(result.fun**2))

    return offset, S, residual


def compute_metrics(offset, S, samples):
    """Compute H/V ratio and magnitude after calibration."""
    h_comp, v_comp, mags = [], [], []

    for sample in samples:
        centered = np.array([
            sample['mx'] - offset[0],
            sample['my'] - offset[1],
            sample['mz'] - offset[2]
        ])
        corrected = S @ centered
        mags.append(np.linalg.norm(corrected))

        roll, pitch = accel_to_roll_pitch(sample['ax'], sample['ay'], sample['az'])
        mx_h, my_h, mz_h = tilt_compensate(corrected[0], corrected[1], corrected[2], roll, pitch)
        h_comp.append(np.sqrt(mx_h**2 + my_h**2))
        v_comp.append(mz_h)

    h_mean = np.mean(h_comp)
    v_mean = np.mean(v_comp)
    hv_ratio = abs(h_mean / v_mean) if abs(v_mean) > 0.1 else float('inf')

    return np.mean(mags), np.std(mags), hv_ratio


def analyze_session_bootstrap(filepath: Path, verbose: bool = True) -> Optional[Dict]:
    """Analyze a session with different bootstrap strategies."""
    data = load_session(filepath)
    if data is None:
        return None

    samples = data['samples']
    mx, my, mz, ax, ay, az = extract_and_filter(samples)

    if len(mx) < 100:
        return None

    # Prepare samples
    cal_samples = [{'mx': mx[i], 'my': my[i], 'mz': mz[i],
                    'ax': ax[i], 'ay': ay[i], 'az': az[i]}
                   for i in range(min(300, len(mx)))]

    if verbose:
        print(f"\n{'='*70}")
        print(f"Session: {filepath.name}")
        print(f"{'='*70}")
        print(f"Samples: {len(mx)} (filtered)")

    results = {'session': filepath.name, 'n_samples': len(mx)}

    # Get session's actual min-max offset (ground truth for this session)
    session_offset = np.array([
        (mx.max() + mx.min()) / 2,
        (my.max() + my.min()) / 2,
        (mz.max() + mz.min()) / 2
    ])
    session_ranges = np.array([
        mx.max() - mx.min(),
        my.max() - my.min(),
        mz.max() - mz.min()
    ])

    results['session_offset'] = session_offset.tolist()
    results['session_ranges'] = session_ranges.tolist()

    if verbose:
        print(f"\nSession's actual offset: [{session_offset[0]:.1f}, {session_offset[1]:.1f}, {session_offset[2]:.1f}]")
        print(f"Session's actual ranges: [{session_ranges[0]:.1f}, {session_ranges[1]:.1f}, {session_ranges[2]:.1f}]")

    # Strategy 1: No bootstrap (start from zero, let min-max find offset)
    if verbose:
        print(f"\n--- Strategy 1: NO BOOTSTRAP (min-max from scratch) ---")
    offset1, S1, res1 = scipy_calibration(cal_samples, initial_offset=session_offset)
    mag1, std1, hv1 = compute_metrics(offset1, S1, cal_samples)
    results['no_bootstrap'] = {
        'residual': res1,
        'magnitude': mag1,
        'hv_ratio': hv1,
        'offset': offset1.tolist()
    }
    if verbose:
        print(f"  Residual: {res1:.1f} µT")
        print(f"  Magnitude: {mag1:.1f} ± {std1:.1f} µT")
        print(f"  H/V ratio: {hv1:.2f}")

    # Strategy 2: Current hardcoded bootstrap
    if verbose:
        print(f"\n--- Strategy 2: CURRENT BOOTSTRAP (hardcoded 2025-12-29) ---")
    offset2, S2, res2 = scipy_calibration(cal_samples, initial_offset=CURRENT_BOOTSTRAP['hard_iron'])
    mag2, std2, hv2 = compute_metrics(offset2, S2, cal_samples)
    results['current_bootstrap'] = {
        'residual': res2,
        'magnitude': mag2,
        'hv_ratio': hv2,
        'offset': offset2.tolist()
    }
    if verbose:
        print(f"  Residual: {res2:.1f} µT")
        print(f"  Magnitude: {mag2:.1f} ± {std2:.1f} µT")
        print(f"  H/V ratio: {hv2:.2f}")

    # Compute offset difference between bootstrap and actual
    offset_diff = np.linalg.norm(CURRENT_BOOTSTRAP['hard_iron'] - session_offset)
    results['bootstrap_offset_error'] = offset_diff
    if verbose:
        print(f"  Bootstrap offset error: {offset_diff:.1f} µT from session's actual")

    return results


def compute_optimal_bootstrap(all_results: List[Dict]) -> Dict:
    """Compute optimal bootstrap values from all sessions."""
    # Collect all session offsets
    offsets = np.array([r['session_offset'] for r in all_results])
    ranges = np.array([r['session_ranges'] for r in all_results])

    # Compute statistics
    mean_offset = np.mean(offsets, axis=0)
    std_offset = np.std(offsets, axis=0)
    median_offset = np.median(offsets, axis=0)

    mean_ranges = np.mean(ranges, axis=0)
    std_ranges = np.std(ranges, axis=0)

    # Robust estimate using median (less sensitive to outliers)
    robust_offset = median_offset
    robust_ranges = np.median(ranges, axis=0)

    return {
        'mean_offset': mean_offset,
        'std_offset': std_offset,
        'median_offset': median_offset,
        'mean_ranges': mean_ranges,
        'std_ranges': std_ranges,
        'robust_offset': robust_offset,
        'robust_ranges': robust_ranges
    }


def validate_optimal_bootstrap(all_results: List[Dict], optimal: Dict):
    """Validate optimal bootstrap on all sessions."""
    print("\n" + "=" * 80)
    print("VALIDATING OPTIMAL BOOTSTRAP")
    print("=" * 80)

    data_dir = Path('data/GAMBIT')
    sessions = sorted([f for f in data_dir.glob('*.json') if f.name != 'manifest.json'])

    optimal_offset = optimal['robust_offset']
    print(f"\nOptimal bootstrap offset: [{optimal_offset[0]:.1f}, {optimal_offset[1]:.1f}, {optimal_offset[2]:.1f}]")

    results = []
    for session_path in sessions:
        data = load_session(session_path)
        if data is None:
            continue

        samples = data['samples']
        mx, my, mz, ax, ay, az = extract_and_filter(samples)
        if len(mx) < 100:
            continue

        cal_samples = [{'mx': mx[i], 'my': my[i], 'mz': mz[i],
                        'ax': ax[i], 'ay': ay[i], 'az': az[i]}
                       for i in range(min(300, len(mx)))]

        # Session's actual offset
        session_offset = np.array([
            (mx.max() + mx.min()) / 2,
            (my.max() + my.min()) / 2,
            (mz.max() + mz.min()) / 2
        ])

        # Calibrate with optimal bootstrap
        offset, S, res = scipy_calibration(cal_samples, initial_offset=optimal_offset)
        mag, std, hv = compute_metrics(offset, S, cal_samples)

        # Compare to no bootstrap (session's own offset)
        offset_nb, S_nb, res_nb = scipy_calibration(cal_samples, initial_offset=session_offset)

        results.append({
            'session': session_path.name,
            'optimal_residual': res,
            'no_bootstrap_residual': res_nb,
            'optimal_hv': hv,
            'offset_error': np.linalg.norm(optimal_offset - session_offset)
        })

    # Summary
    print(f"\n{'Session':<30} {'Optimal Res':>12} {'No-Boot Res':>12} {'Offset Err':>12}")
    print("-" * 70)
    for r in results:
        print(f"{r['session'][:28]:<30} {r['optimal_residual']:>10.1f}µT {r['no_bootstrap_residual']:>10.1f}µT {r['offset_error']:>10.1f}µT")

    # Overall statistics
    opt_res = [r['optimal_residual'] for r in results]
    nb_res = [r['no_bootstrap_residual'] for r in results]

    print(f"\n{'='*70}")
    print(f"Mean residual (optimal bootstrap): {np.mean(opt_res):.1f} µT")
    print(f"Mean residual (no bootstrap):      {np.mean(nb_res):.1f} µT")
    print(f"Mean offset error from optimal:    {np.mean([r['offset_error'] for r in results]):.1f} µT")

    return results


def main():
    data_dir = Path('data/GAMBIT')
    sessions = sorted([f for f in data_dir.glob('*.json') if f.name != 'manifest.json'])

    print(f"\nAnalyzing {len(sessions)} sessions...\n")

    all_results = []
    for session_path in sessions:
        result = analyze_session_bootstrap(session_path, verbose=True)
        if result:
            all_results.append(result)

    if not all_results:
        print("No valid sessions found!")
        return

    # Compute optimal bootstrap
    print("\n" + "=" * 80)
    print("COMPUTING OPTIMAL BOOTSTRAP VALUES")
    print("=" * 80)

    optimal = compute_optimal_bootstrap(all_results)

    print(f"\nFrom {len(all_results)} sessions:")
    print(f"\n--- Offset Statistics ---")
    print(f"Mean offset:   [{optimal['mean_offset'][0]:.1f}, {optimal['mean_offset'][1]:.1f}, {optimal['mean_offset'][2]:.1f}] µT")
    print(f"Std offset:    [{optimal['std_offset'][0]:.1f}, {optimal['std_offset'][1]:.1f}, {optimal['std_offset'][2]:.1f}] µT")
    print(f"Median offset: [{optimal['median_offset'][0]:.1f}, {optimal['median_offset'][1]:.1f}, {optimal['median_offset'][2]:.1f}] µT")

    print(f"\n--- Range Statistics ---")
    print(f"Mean ranges:   [{optimal['mean_ranges'][0]:.1f}, {optimal['mean_ranges'][1]:.1f}, {optimal['mean_ranges'][2]:.1f}] µT")
    print(f"Std ranges:    [{optimal['std_ranges'][0]:.1f}, {optimal['std_ranges'][1]:.1f}, {optimal['std_ranges'][2]:.1f}] µT")

    print(f"\n--- RECOMMENDED BOOTSTRAP VALUES ---")
    print(f"(Using robust median to reduce outlier sensitivity)")
    print(f"\nHard iron offset: [{optimal['robust_offset'][0]:.1f}, {optimal['robust_offset'][1]:.1f}, {optimal['robust_offset'][2]:.1f}] µT")
    print(f"Initial ranges:   [{optimal['robust_ranges'][0]:.1f}, {optimal['robust_ranges'][1]:.1f}, {optimal['robust_ranges'][2]:.1f}] µT")

    # Compare to current hardcoded values
    print(f"\n--- COMPARISON TO CURRENT HARDCODED VALUES ---")
    print(f"Current:     [{CURRENT_BOOTSTRAP['hard_iron'][0]:.1f}, {CURRENT_BOOTSTRAP['hard_iron'][1]:.1f}, {CURRENT_BOOTSTRAP['hard_iron'][2]:.1f}] µT")
    print(f"Recommended: [{optimal['robust_offset'][0]:.1f}, {optimal['robust_offset'][1]:.1f}, {optimal['robust_offset'][2]:.1f}] µT")

    diff = np.linalg.norm(CURRENT_BOOTSTRAP['hard_iron'] - optimal['robust_offset'])
    print(f"Difference:  {diff:.1f} µT")

    # Validate optimal bootstrap
    validation_results = validate_optimal_bootstrap(all_results, optimal)

    # Generate TypeScript code for new bootstrap values
    print("\n" + "=" * 80)
    print("TYPESCRIPT CODE FOR UPDATED BOOTSTRAP")
    print("=" * 80)
    half_ranges = optimal['robust_ranges'] / 2
    print(f"""
// Updated bootstrap values from offline analysis of {len(all_results)} sessions
// Generated: {np.datetime64('today')}
if (this._autoHardIronEnabled) {{
    this._autoHardIronEstimate = {{ x: {optimal['robust_offset'][0]:.1f}, y: {optimal['robust_offset'][1]:.1f}, z: {optimal['robust_offset'][2]:.1f} }};
    this._autoHardIronMin = {{ x: {optimal['robust_offset'][0]:.1f} - {half_ranges[0]:.1f}, y: {optimal['robust_offset'][1]:.1f} - {half_ranges[1]:.1f}, z: {optimal['robust_offset'][2]:.1f} - {half_ranges[2]:.1f} }};
    this._autoHardIronMax = {{ x: {optimal['robust_offset'][0]:.1f} + {half_ranges[0]:.1f}, y: {optimal['robust_offset'][1]:.1f} + {half_ranges[1]:.1f}, z: {optimal['robust_offset'][2]:.1f} + {half_ranges[2]:.1f} }};
}}
""")

    # Save results
    output = {
        'sessions_analyzed': len(all_results),
        'current_bootstrap': {
            'hard_iron': CURRENT_BOOTSTRAP['hard_iron'].tolist(),
            'ranges': CURRENT_BOOTSTRAP['ranges'].tolist()
        },
        'optimal_bootstrap': {
            'hard_iron': optimal['robust_offset'].tolist(),
            'ranges': optimal['robust_ranges'].tolist(),
            'mean_offset': optimal['mean_offset'].tolist(),
            'std_offset': optimal['std_offset'].tolist()
        },
        'session_results': all_results,
        'validation_results': validation_results
    }

    output_path = Path('ml/bootstrap_analysis_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Generate comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Bootstrap Calibration Analysis', fontsize=14)

    # Plot 1: Session offsets scatter
    ax = axes[0, 0]
    offsets = np.array([r['session_offset'] for r in all_results])
    ax.scatter(offsets[:, 0], offsets[:, 1], c=offsets[:, 2], cmap='coolwarm', s=50, alpha=0.7)
    ax.scatter(CURRENT_BOOTSTRAP['hard_iron'][0], CURRENT_BOOTSTRAP['hard_iron'][1],
               c='red', s=200, marker='*', label='Current bootstrap', zorder=5)
    ax.scatter(optimal['robust_offset'][0], optimal['robust_offset'][1],
               c='green', s=200, marker='*', label='Optimal bootstrap', zorder=5)
    ax.set_xlabel('X offset (µT)')
    ax.set_ylabel('Y offset (µT)')
    ax.set_title('Session Hard Iron Offsets (color=Z)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 2: Bootstrap offset errors
    ax = axes[0, 1]
    current_errors = [r['bootstrap_offset_error'] for r in all_results]
    optimal_errors = [np.linalg.norm(optimal['robust_offset'] - np.array(r['session_offset']))
                      for r in all_results]
    x = range(len(all_results))
    ax.bar([i-0.2 for i in x], current_errors, 0.4, label='Current bootstrap', alpha=0.7)
    ax.bar([i+0.2 for i in x], optimal_errors, 0.4, label='Optimal bootstrap', alpha=0.7)
    ax.set_xlabel('Session')
    ax.set_ylabel('Offset error (µT)')
    ax.set_title('Bootstrap Offset Error per Session')
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels([r['session'][:10] for r in all_results], rotation=45, ha='right')

    # Plot 3: Residual comparison
    ax = axes[1, 0]
    no_boot = [r['no_bootstrap']['residual'] for r in all_results]
    with_boot = [r['current_bootstrap']['residual'] for r in all_results]
    ax.bar([i-0.2 for i in x], no_boot, 0.4, label='No bootstrap', alpha=0.7)
    ax.bar([i+0.2 for i in x], with_boot, 0.4, label='Current bootstrap', alpha=0.7)
    ax.set_xlabel('Session')
    ax.set_ylabel('Residual (µT)')
    ax.set_title('Calibration Residual: No Bootstrap vs Current Bootstrap')
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels([r['session'][:10] for r in all_results], rotation=45, ha='right')

    # Plot 4: Offset components histogram
    ax = axes[1, 1]
    ax.hist(offsets[:, 0], bins=10, alpha=0.5, label='X')
    ax.hist(offsets[:, 1], bins=10, alpha=0.5, label='Y')
    ax.hist(offsets[:, 2], bins=10, alpha=0.5, label='Z')
    ax.axvline(optimal['robust_offset'][0], color='C0', linestyle='--')
    ax.axvline(optimal['robust_offset'][1], color='C1', linestyle='--')
    ax.axvline(optimal['robust_offset'][2], color='C2', linestyle='--')
    ax.set_xlabel('Offset (µT)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Session Offsets (dashed=optimal)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path('ml/bootstrap_analysis_plot.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")
    plt.close()


if __name__ == '__main__':
    main()
