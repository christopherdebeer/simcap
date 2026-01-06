#!/usr/bin/env python3
"""
Validate Proposed Calibration Fixes

Compares current (buggy) orientation-aware calibration with proposed fixes:
1. Current: epsilon=0.5 for both offset and matrix, 50 iterations
2. Proposed: epsilon_offset=0.5, epsilon_matrix=0.01, 200 iterations

Tests on all available session data to ensure fixes work across sessions.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt

# Edinburgh geomagnetic reference
EARTH_H = 16.0  # µT
EARTH_V = 47.8  # µT
EARTH_MAG = np.sqrt(EARTH_H**2 + EARTH_V**2)
EARTH_WORLD = np.array([EARTH_H, 0, EARTH_V])  # NED


def load_session(filepath: Path) -> Optional[Dict]:
    """Load and validate session data."""
    try:
        with open(filepath, 'r') as f:
            data = json.load(f)
        samples = data.get('samples', [])
        if len(samples) < 100:
            return None
        # Check for required fields
        if 'mx_ut' not in samples[0] and 'mx' not in samples[0]:
            return None
        return data
    except Exception as e:
        print(f"  Error loading {filepath.name}: {e}")
        return None


def extract_arrays(samples: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract numpy arrays from samples."""
    return {
        'mx': np.array([s.get('mx_ut', s.get('mx', 0)) for s in samples]),
        'my': np.array([s.get('my_ut', s.get('my', 0)) for s in samples]),
        'mz': np.array([s.get('mz_ut', s.get('mz', 0)) for s in samples]),
        'ax': np.array([s.get('ax_g', s.get('ax', 0)) for s in samples]),
        'ay': np.array([s.get('ay_g', s.get('ay', 0)) for s in samples]),
        'az': np.array([s.get('az_g', s.get('az', 0)) for s in samples]),
    }


def filter_outliers(mx, my, mz, ax, ay, az, max_mag=200, max_comp=150):
    """Apply firmware-style outlier filtering."""
    mag = np.sqrt(mx**2 + my**2 + mz**2)
    valid = (mag <= max_mag) & (np.abs(mx) <= max_comp) & (np.abs(my) <= max_comp) & (np.abs(mz) <= max_comp)
    return mx[valid], my[valid], mz[valid], ax[valid], ay[valid], az[valid]


def accel_to_roll_pitch(ax, ay, az):
    """Get roll and pitch from accelerometer."""
    a_norm = np.sqrt(ax**2 + ay**2 + az**2)
    if a_norm < 0.1:
        return 0, 0
    ax, ay, az = ax/a_norm, ay/a_norm, az/a_norm
    roll = np.arctan2(ay, az)
    pitch = np.arctan2(-ax, np.sqrt(ay**2 + az**2))
    return roll, pitch


def tilt_compensate(mx, my, mz, roll, pitch):
    """Tilt-compensate magnetometer."""
    cos_r, sin_r = np.cos(roll), np.sin(roll)
    cos_p, sin_p = np.cos(pitch), np.sin(pitch)
    mx_h = mx * cos_p + my * sin_r * sin_p + mz * cos_r * sin_p
    my_h = my * cos_r - mz * sin_r
    mz_h = -mx * sin_p + my * cos_r * sin_p + mz * cos_r * cos_p
    return mx_h, my_h, mz_h


def euler_to_rotation(roll, pitch, yaw):
    """ZYX rotation matrix."""
    cy, sy = np.cos(yaw), np.sin(yaw)
    cp, sp = np.cos(pitch), np.sin(pitch)
    cr, sr = np.cos(roll), np.sin(roll)
    return np.array([
        [cy*cp, cy*sp*sr - sy*cr, cy*sp*cr + sy*sr],
        [sy*cp, sy*sp*sr + cy*cr, sy*sp*cr - cy*sr],
        [-sp, cp*sr, cp*cr]
    ])


def compute_residual(offset, S, samples, earth_world):
    """Compute RMS residual for given calibration parameters."""
    total_residual = 0
    n = len(samples)

    for sample in samples:
        # Apply calibration
        centered = np.array([
            sample['mx'] - offset[0],
            sample['my'] - offset[1],
            sample['mz'] - offset[2]
        ])
        corrected = S @ centered

        # Get orientation from accelerometer
        roll, pitch = accel_to_roll_pitch(sample['ax'], sample['ay'], sample['az'])
        mx_h, my_h, _ = tilt_compensate(corrected[0], corrected[1], corrected[2], roll, pitch)
        yaw = np.atan2(-my_h, mx_h)

        # Expected Earth field in device frame
        R = euler_to_rotation(roll, pitch, yaw)
        earth_device = R.T @ earth_world

        # Residual
        diff = corrected - earth_device
        total_residual += np.sum(diff**2)

    return np.sqrt(total_residual / n)


def orientation_aware_calibration(
    samples: List[Dict],
    earth_world: np.ndarray,
    epsilon_offset: float = 0.5,
    epsilon_matrix: float = 0.5,  # Current buggy value
    max_iterations: int = 50,
    learning_rate_offset: float = 0.1,
    learning_rate_matrix: float = 0.01,
    clamp_diagonal: Tuple[float, float] = (0.5, 2.0),
    clamp_offdiag: Tuple[float, float] = (-0.5, 0.5),
    verbose: bool = False
) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Orientation-aware calibration matching firmware implementation.

    Returns: (offset, soft_iron_matrix, final_residual)
    """
    # Initial guess from min-max
    mx = np.array([s['mx'] for s in samples])
    my = np.array([s['my'] for s in samples])
    mz = np.array([s['mz'] for s in samples])

    offset = np.array([
        (mx.max() + mx.min()) / 2,
        (my.max() + my.min()) / 2,
        (mz.max() + mz.min()) / 2
    ])

    # Start with identity matrix
    S = np.eye(3)

    best_residual = compute_residual(offset, S, samples, earth_world)
    best_offset = offset.copy()
    best_S = S.copy()

    prev_residual = best_residual

    for iteration in range(max_iterations):
        # Compute gradients for offset
        grad_offset = np.zeros(3)
        for i in range(3):
            offset_plus = offset.copy()
            offset_minus = offset.copy()
            offset_plus[i] += epsilon_offset
            offset_minus[i] -= epsilon_offset
            grad_offset[i] = (compute_residual(offset_plus, S, samples, earth_world) -
                             compute_residual(offset_minus, S, samples, earth_world)) / (2 * epsilon_offset)

        # Compute gradients for soft iron matrix
        grad_S = np.zeros((3, 3))
        for i in range(3):
            for j in range(3):
                S_plus = S.copy()
                S_minus = S.copy()
                S_plus[i, j] += epsilon_matrix
                S_minus[i, j] -= epsilon_matrix
                grad_S[i, j] = (compute_residual(offset, S_plus, samples, earth_world) -
                               compute_residual(offset, S_minus, samples, earth_world)) / (2 * epsilon_matrix)

        # Gradient clipping
        max_grad = 10
        grad_offset = np.clip(grad_offset, -max_grad, max_grad)
        grad_S = np.clip(grad_S, -max_grad, max_grad)

        # Update parameters
        offset -= learning_rate_offset * grad_offset
        S -= learning_rate_matrix * grad_S

        # Regularization: clamp values
        for i in range(3):
            S[i, i] = np.clip(S[i, i], clamp_diagonal[0], clamp_diagonal[1])
            for j in range(3):
                if i != j:
                    S[i, j] = np.clip(S[i, j], clamp_offdiag[0], clamp_offdiag[1])

        # Track best
        new_residual = compute_residual(offset, S, samples, earth_world)
        if new_residual < best_residual:
            best_residual = new_residual
            best_offset = offset.copy()
            best_S = S.copy()

        # Convergence check
        if abs(prev_residual - new_residual) < 0.1:
            if verbose:
                print(f"    Converged at iteration {iteration}")
            break
        prev_residual = new_residual

    return best_offset, best_S, best_residual


def compute_hv_ratio(offset, S, samples):
    """Compute H/V component ratio after calibration."""
    h_components = []
    v_components = []
    magnitudes = []

    for sample in samples:
        centered = np.array([
            sample['mx'] - offset[0],
            sample['my'] - offset[1],
            sample['mz'] - offset[2]
        ])
        corrected = S @ centered
        magnitudes.append(np.linalg.norm(corrected))

        roll, pitch = accel_to_roll_pitch(sample['ax'], sample['ay'], sample['az'])
        mx_h, my_h, mz_h = tilt_compensate(corrected[0], corrected[1], corrected[2], roll, pitch)

        h_components.append(np.sqrt(mx_h**2 + my_h**2))
        v_components.append(mz_h)

    h_mean = np.mean(h_components)
    v_mean = np.mean(v_components)
    mag_mean = np.mean(magnitudes)
    mag_std = np.std(magnitudes)

    # Avoid division by zero
    if abs(v_mean) < 0.1:
        hv_ratio = float('inf')
    else:
        hv_ratio = abs(h_mean / v_mean)

    return h_mean, v_mean, hv_ratio, mag_mean, mag_std


def analyze_session(filepath: Path, verbose: bool = True) -> Optional[Dict]:
    """Analyze a single session with both calibration methods."""
    data = load_session(filepath)
    if data is None:
        return None

    samples = data['samples']
    arrays = extract_arrays(samples)

    # Filter outliers
    mx, my, mz, ax, ay, az = filter_outliers(
        arrays['mx'], arrays['my'], arrays['mz'],
        arrays['ax'], arrays['ay'], arrays['az']
    )

    if len(mx) < 100:
        return None

    # Prepare samples for calibration
    cal_samples = [{'mx': mx[i], 'my': my[i], 'mz': mz[i],
                    'ax': ax[i], 'ay': ay[i], 'az': az[i]}
                   for i in range(min(300, len(mx)))]

    if verbose:
        print(f"\n{'='*70}")
        print(f"Session: {filepath.name}")
        print(f"{'='*70}")
        print(f"Samples: {len(mx)} (after filtering)")

    results = {'session': filepath.name, 'n_samples': len(mx)}

    # Method 1: Current (buggy) implementation
    if verbose:
        print(f"\n--- Method 1: CURRENT (epsilon=0.5, 50 iter) ---")
    offset1, S1, res1 = orientation_aware_calibration(
        cal_samples, EARTH_WORLD,
        epsilon_offset=0.5,
        epsilon_matrix=0.5,  # BUG: same as offset
        max_iterations=50,
        verbose=verbose
    )
    h1, v1, hv1, mag1, std1 = compute_hv_ratio(offset1, S1, cal_samples)

    results['current'] = {
        'residual': res1,
        'magnitude': mag1,
        'magnitude_std': std1,
        'h_component': h1,
        'v_component': v1,
        'hv_ratio': hv1,
        'offset': offset1.tolist(),
        'soft_iron': S1.tolist()
    }

    if verbose:
        print(f"  Residual: {res1:.1f} µT")
        print(f"  Magnitude: {mag1:.1f} ± {std1:.1f} µT (expected {EARTH_MAG:.1f})")
        print(f"  H/V ratio: {hv1:.2f} (expected {EARTH_H/EARTH_V:.2f})")
        print(f"  Soft iron diagonal: [{S1[0,0]:.3f}, {S1[1,1]:.3f}, {S1[2,2]:.3f}]")
        if S1[0,0] == 0.5 or S1[0,0] == 2.0 or abs(S1[0,1]) == 0.5:
            print(f"  ⚠️  Values hit clamp limits!")

    # Method 2: Proposed fix
    if verbose:
        print(f"\n--- Method 2: PROPOSED (epsilon_matrix=0.01, 200 iter) ---")
    offset2, S2, res2 = orientation_aware_calibration(
        cal_samples, EARTH_WORLD,
        epsilon_offset=0.5,
        epsilon_matrix=0.01,  # FIX: appropriate for matrix
        max_iterations=200,   # FIX: more iterations
        verbose=verbose
    )
    h2, v2, hv2, mag2, std2 = compute_hv_ratio(offset2, S2, cal_samples)

    results['proposed'] = {
        'residual': res2,
        'magnitude': mag2,
        'magnitude_std': std2,
        'h_component': h2,
        'v_component': v2,
        'hv_ratio': hv2,
        'offset': offset2.tolist(),
        'soft_iron': S2.tolist()
    }

    if verbose:
        print(f"  Residual: {res2:.1f} µT")
        print(f"  Magnitude: {mag2:.1f} ± {std2:.1f} µT (expected {EARTH_MAG:.1f})")
        print(f"  H/V ratio: {hv2:.2f} (expected {EARTH_H/EARTH_V:.2f})")
        print(f"  Soft iron diagonal: [{S2[0,0]:.3f}, {S2[1,1]:.3f}, {S2[2,2]:.3f}]")

    # Method 3: scipy least_squares (reference)
    if verbose:
        print(f"\n--- Method 3: SCIPY least_squares (reference) ---")
    try:
        from scipy.optimize import least_squares

        def residual_func(params):
            offset = params[:3]
            S = params[3:12].reshape(3, 3)

            residuals = []
            for sample in cal_samples:
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

        # Initial guess
        x0 = np.concatenate([offset1, np.eye(3).flatten()])
        result = least_squares(residual_func, x0, method='lm', max_nfev=10000)

        offset3 = result.x[:3]
        S3 = result.x[3:12].reshape(3, 3)
        res3 = np.sqrt(np.mean(result.fun**2))
        h3, v3, hv3, mag3, std3 = compute_hv_ratio(offset3, S3, cal_samples)

        results['scipy'] = {
            'residual': res3,
            'magnitude': mag3,
            'magnitude_std': std3,
            'h_component': h3,
            'v_component': v3,
            'hv_ratio': hv3,
            'offset': offset3.tolist(),
            'soft_iron': S3.tolist()
        }

        if verbose:
            print(f"  Residual: {res3:.1f} µT")
            print(f"  Magnitude: {mag3:.1f} ± {std3:.1f} µT (expected {EARTH_MAG:.1f})")
            print(f"  H/V ratio: {hv3:.2f} (expected {EARTH_H/EARTH_V:.2f})")
    except Exception as e:
        if verbose:
            print(f"  Error: {e}")
        results['scipy'] = None

    # Summary
    if verbose:
        print(f"\n--- COMPARISON ---")
        print(f"{'Method':<20} {'Residual':>10} {'Mag Error':>12} {'H/V Ratio':>10}")
        print(f"{'-'*52}")
        print(f"{'Current (buggy)':<20} {res1:>10.1f} {abs(mag1-EARTH_MAG)/EARTH_MAG*100:>11.1f}% {hv1:>10.2f}")
        print(f"{'Proposed (fixed)':<20} {res2:>10.1f} {abs(mag2-EARTH_MAG)/EARTH_MAG*100:>11.1f}% {hv2:>10.2f}")
        if results.get('scipy'):
            print(f"{'Scipy (reference)':<20} {res3:>10.1f} {abs(mag3-EARTH_MAG)/EARTH_MAG*100:>11.1f}% {hv3:>10.2f}")
        print(f"{'Expected':<20} {'<10':>10} {'<5%':>12} {EARTH_H/EARTH_V:>10.2f}")

        # Improvement
        if res2 < res1:
            print(f"\n✅ Proposed fix improves residual by {(res1-res2)/res1*100:.0f}%")
        else:
            print(f"\n⚠️ Proposed fix did not improve residual")

    return results


def main():
    """Analyze all sessions and generate summary."""
    data_dir = Path('data/GAMBIT')
    sessions = sorted([f for f in data_dir.glob('*.json') if f.name != 'manifest.json'])

    print("=" * 70)
    print("CALIBRATION FIX VALIDATION")
    print("=" * 70)
    print(f"Testing proposed fixes on {len(sessions)} sessions")
    print(f"\nExpected values:")
    print(f"  Earth magnitude: {EARTH_MAG:.1f} µT")
    print(f"  H/V ratio: {EARTH_H/EARTH_V:.2f}")
    print(f"  Target residual: <10 µT")

    all_results = []

    for session_path in sessions:
        result = analyze_session(session_path, verbose=True)
        if result:
            all_results.append(result)

    if not all_results:
        print("\nNo valid sessions found!")
        return

    # Summary statistics
    print("\n" + "=" * 70)
    print("OVERALL SUMMARY")
    print("=" * 70)

    current_residuals = [r['current']['residual'] for r in all_results]
    proposed_residuals = [r['proposed']['residual'] for r in all_results]
    scipy_residuals = [r['scipy']['residual'] for r in all_results if r.get('scipy')]

    current_hv = [r['current']['hv_ratio'] for r in all_results if r['current']['hv_ratio'] < 100]
    proposed_hv = [r['proposed']['hv_ratio'] for r in all_results if r['proposed']['hv_ratio'] < 100]

    print(f"\n{'Metric':<25} {'Current':<15} {'Proposed':<15} {'Scipy':<15}")
    print("-" * 70)
    print(f"{'Mean residual (µT)':<25} {np.mean(current_residuals):>12.1f} {np.mean(proposed_residuals):>14.1f} {np.mean(scipy_residuals) if scipy_residuals else 'N/A':>14}")
    print(f"{'Median residual (µT)':<25} {np.median(current_residuals):>12.1f} {np.median(proposed_residuals):>14.1f} {np.median(scipy_residuals) if scipy_residuals else 'N/A':>14}")
    print(f"{'Mean H/V ratio':<25} {np.mean(current_hv):>12.2f} {np.mean(proposed_hv):>14.2f}")

    # Count improvements
    improved = sum(1 for i in range(len(all_results))
                   if all_results[i]['proposed']['residual'] < all_results[i]['current']['residual'])

    print(f"\n{'Sessions improved:':<25} {improved}/{len(all_results)} ({improved/len(all_results)*100:.0f}%)")

    # Check H/V ratio quality gate
    hv_threshold = 0.8
    current_bad_hv = sum(1 for r in all_results if r['current']['hv_ratio'] > hv_threshold)
    proposed_bad_hv = sum(1 for r in all_results if r['proposed']['hv_ratio'] > hv_threshold)

    print(f"\n{'Sessions with H/V > 0.8 (inverted):'}")
    print(f"  Current: {current_bad_hv}/{len(all_results)}")
    print(f"  Proposed: {proposed_bad_hv}/{len(all_results)}")

    # Save results
    output_path = Path('ml/calibration_validation_results.json')
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDetailed results saved to: {output_path}")

    # Generate comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Calibration Fix Validation', fontsize=14)

    # Residual comparison
    ax = axes[0, 0]
    x = range(len(all_results))
    ax.bar([i-0.2 for i in x], current_residuals, 0.4, label='Current', alpha=0.7)
    ax.bar([i+0.2 for i in x], proposed_residuals, 0.4, label='Proposed', alpha=0.7)
    ax.axhline(10, color='g', linestyle='--', label='Target <10µT')
    ax.set_ylabel('Residual (µT)')
    ax.set_xlabel('Session')
    ax.set_title('Residual Comparison')
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels([r['session'][:10] for r in all_results], rotation=45, ha='right')

    # H/V ratio comparison
    ax = axes[0, 1]
    ax.bar([i-0.2 for i in x], [min(r['current']['hv_ratio'], 5) for r in all_results], 0.4, label='Current', alpha=0.7)
    ax.bar([i+0.2 for i in x], [min(r['proposed']['hv_ratio'], 5) for r in all_results], 0.4, label='Proposed', alpha=0.7)
    ax.axhline(EARTH_H/EARTH_V, color='g', linestyle='--', label=f'Expected {EARTH_H/EARTH_V:.2f}')
    ax.axhline(0.8, color='r', linestyle='--', label='Reject threshold 0.8')
    ax.set_ylabel('H/V Ratio')
    ax.set_xlabel('Session')
    ax.set_title('H/V Ratio Comparison')
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels([r['session'][:10] for r in all_results], rotation=45, ha='right')

    # Improvement scatter
    ax = axes[1, 0]
    ax.scatter(current_residuals, proposed_residuals, alpha=0.7)
    max_val = max(max(current_residuals), max(proposed_residuals))
    ax.plot([0, max_val], [0, max_val], 'k--', alpha=0.5, label='No change')
    ax.set_xlabel('Current Residual (µT)')
    ax.set_ylabel('Proposed Residual (µT)')
    ax.set_title('Residual: Current vs Proposed')
    ax.legend()

    # Magnitude error comparison
    ax = axes[1, 1]
    current_mag_err = [abs(r['current']['magnitude'] - EARTH_MAG) / EARTH_MAG * 100 for r in all_results]
    proposed_mag_err = [abs(r['proposed']['magnitude'] - EARTH_MAG) / EARTH_MAG * 100 for r in all_results]
    ax.bar([i-0.2 for i in x], current_mag_err, 0.4, label='Current', alpha=0.7)
    ax.bar([i+0.2 for i in x], proposed_mag_err, 0.4, label='Proposed', alpha=0.7)
    ax.axhline(5, color='g', linestyle='--', label='Target <5%')
    ax.set_ylabel('Magnitude Error (%)')
    ax.set_xlabel('Session')
    ax.set_title('Magnitude Error Comparison')
    ax.legend()
    ax.set_xticks(x)
    ax.set_xticklabels([r['session'][:10] for r in all_results], rotation=45, ha='right')

    plt.tight_layout()
    plot_path = Path('ml/calibration_validation_comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Comparison plot saved to: {plot_path}")
    plt.close()


if __name__ == '__main__':
    main()
