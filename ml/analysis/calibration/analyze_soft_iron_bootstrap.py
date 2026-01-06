#!/usr/bin/env python3
"""
Analyze soft iron calibration across sessions to compute optimal bootstrap values.

Soft iron distortion causes the magnetometer reading ellipsoid to be stretched/rotated.
This script computes optimal diagonal scale factors as bootstrap values.
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
from typing import List, Dict, Tuple, Optional
import matplotlib.pyplot as plt

# Expected Earth field magnitude at Edinburgh
EXPECTED_MAGNITUDE = 50.4  # µT


def load_session(path: Path) -> Optional[Dict]:
    """Load and parse session JSON."""
    try:
        with open(path) as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return None


def extract_and_filter(samples: List[Dict]) -> Tuple[np.ndarray, ...]:
    """Extract magnetometer and accelerometer data with outlier filtering."""
    mx = np.array([s.get('mx', s.get('magX', 0)) for s in samples])
    my = np.array([s.get('my', s.get('magY', 0)) for s in samples])
    mz = np.array([s.get('mz', s.get('magZ', 0)) for s in samples])
    ax = np.array([s.get('ax', s.get('accelX', 0)) for s in samples])
    ay = np.array([s.get('ay', s.get('accelY', 0)) for s in samples])
    az = np.array([s.get('az', s.get('accelZ', 0)) for s in samples])

    # Filter outliers (per-axis threshold of 150µT from median)
    valid = np.ones(len(mx), dtype=bool)
    for data in [mx, my, mz]:
        median = np.median(data)
        valid &= np.abs(data - median) < 150

    return mx[valid], my[valid], mz[valid], ax[valid], ay[valid], az[valid]


def compute_soft_iron_from_ranges(mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> np.ndarray:
    """
    Compute diagonal soft iron scale factors from min-max ranges.
    This normalizes the ellipsoid axes to have equal length.
    """
    ranges = np.array([
        mx.max() - mx.min(),
        my.max() - my.min(),
        mz.max() - mz.min()
    ])

    # Target: normalize to average range (spherical)
    avg_range = np.mean(ranges)

    # Scale factors to make each axis have the average range
    # scale * range = avg_range => scale = avg_range / range
    scales = avg_range / ranges

    return scales


def scipy_calibration(samples: List[Dict], initial_offset: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Run scipy least_squares optimization to find offset and full 3x3 soft iron matrix.
    Returns: (offset, soft_iron_matrix, residual)

    Uses regularization to keep soft iron matrix close to identity.
    """
    mx = np.array([s['mx'] for s in samples])
    my = np.array([s['my'] for s in samples])
    mz = np.array([s['mz'] for s in samples])

    # Initial parameters: offset (3) + soft iron matrix (9)
    # Start with identity matrix for soft iron
    initial_S = np.eye(3).flatten()
    x0 = np.concatenate([initial_offset, initial_S])

    # Regularization weight - keeps soft iron near identity
    reg_weight = 0.1

    def residual_func(params):
        offset = params[:3]
        S = params[3:].reshape(3, 3)

        centered = np.column_stack([mx - offset[0], my - offset[1], mz - offset[2]])
        corrected = (S @ centered.T).T
        magnitudes = np.linalg.norm(corrected, axis=1)

        # Magnitude residuals
        mag_residuals = magnitudes - EXPECTED_MAGNITUDE

        # Regularization: penalize deviation from identity matrix
        identity = np.eye(3)
        reg_residuals = reg_weight * (S - identity).flatten() * 10  # Scale to similar magnitude

        return np.concatenate([mag_residuals, reg_residuals])

    # Add bounds to prevent degenerate solutions
    # Soft iron diagonal should be between 0.5 and 2.0
    # Off-diagonal should be between -0.5 and 0.5
    lower_bounds = np.concatenate([
        [-np.inf, -np.inf, -np.inf],  # offset: no bounds
        [0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5]  # S matrix
    ])
    upper_bounds = np.concatenate([
        [np.inf, np.inf, np.inf],  # offset: no bounds
        [2.0, 0.5, 0.5, 0.5, 2.0, 0.5, 0.5, 0.5, 2.0]  # S matrix
    ])

    result = least_squares(residual_func, x0, bounds=(lower_bounds, upper_bounds), max_nfev=500)

    offset = result.x[:3]
    S = result.x[3:].reshape(3, 3)

    # Compute actual magnitude residual (without regularization)
    centered = np.column_stack([mx - offset[0], my - offset[1], mz - offset[2]])
    corrected = (S @ centered.T).T
    magnitudes = np.linalg.norm(corrected, axis=1)
    residual = np.std(magnitudes - EXPECTED_MAGNITUDE)

    return offset, S, residual


def analyze_session(session_path: Path, verbose: bool = True) -> Optional[Dict]:
    """Analyze soft iron for a single session."""
    data = load_session(session_path)
    if data is None:
        return None

    samples = data.get('samples', [])
    mx, my, mz, ax, ay, az = extract_and_filter(samples)

    if len(mx) < 100:
        return None

    if verbose:
        print(f"\n{'='*70}")
        print(f"Session: {session_path.name}")
        print(f"{'='*70}")
        print(f"Samples: {len(mx)} (filtered)")

    results = {'session': session_path.name, 'n_samples': len(mx)}

    # Method 1: Diagonal scale factors from ranges
    diag_scales = compute_soft_iron_from_ranges(mx, my, mz)
    results['diagonal_scales'] = diag_scales.tolist()

    if verbose:
        print(f"\nDiagonal soft iron scales (from ranges): [{diag_scales[0]:.3f}, {diag_scales[1]:.3f}, {diag_scales[2]:.3f}]")

    # Compute min-max offset
    offset = np.array([
        (mx.max() + mx.min()) / 2,
        (my.max() + my.min()) / 2,
        (mz.max() + mz.min()) / 2
    ])

    # Method 2: Full 3x3 matrix from scipy optimization
    cal_samples = [{'mx': mx[i], 'my': my[i], 'mz': mz[i],
                    'ax': ax[i], 'ay': ay[i], 'az': az[i]}
                   for i in range(min(300, len(mx)))]

    try:
        opt_offset, opt_S, residual = scipy_calibration(cal_samples, offset)
        results['optimized_matrix'] = opt_S.tolist()
        results['optimized_diagonal'] = np.diag(opt_S).tolist()
        results['optimized_residual'] = residual

        if verbose:
            print(f"\nOptimized soft iron matrix (scipy):")
            print(f"  Diagonal: [{opt_S[0,0]:.3f}, {opt_S[1,1]:.3f}, {opt_S[2,2]:.3f}]")
            print(f"  Off-diagonal max: {np.max(np.abs(opt_S - np.diag(np.diag(opt_S)))):.3f}")
            print(f"  Residual: {residual:.2f} µT")
    except Exception as e:
        print(f"  Optimization failed: {e}")
        results['optimized_matrix'] = None
        results['optimized_residual'] = None

    return results


def compute_optimal_soft_iron(all_results: List[Dict]) -> Dict:
    """Compute optimal soft iron bootstrap from all sessions."""
    # Collect diagonal scales
    diag_scales = np.array([r['diagonal_scales'] for r in all_results])

    # Collect optimized diagonals (where available)
    opt_diags = []
    opt_matrices = []
    for r in all_results:
        if r.get('optimized_diagonal'):
            opt_diags.append(r['optimized_diagonal'])
        if r.get('optimized_matrix'):
            opt_matrices.append(r['optimized_matrix'])

    opt_diags = np.array(opt_diags) if opt_diags else None
    opt_matrices = np.array(opt_matrices) if opt_matrices else None

    return {
        'range_based': {
            'mean': np.mean(diag_scales, axis=0).tolist(),
            'std': np.std(diag_scales, axis=0).tolist(),
            'median': np.median(diag_scales, axis=0).tolist()
        },
        'optimized': {
            'mean_diagonal': np.mean(opt_diags, axis=0).tolist() if opt_diags is not None else None,
            'std_diagonal': np.std(opt_diags, axis=0).tolist() if opt_diags is not None else None,
            'median_diagonal': np.median(opt_diags, axis=0).tolist() if opt_diags is not None else None,
            'mean_matrix': np.mean(opt_matrices, axis=0).tolist() if opt_matrices is not None else None,
            'median_matrix': np.median(opt_matrices, axis=0).tolist() if opt_matrices is not None else None
        }
    }


def main():
    print("=" * 80)
    print("SOFT IRON BOOTSTRAP ANALYSIS")
    print("=" * 80)

    data_dir = Path('data/GAMBIT')
    sessions = sorted([f for f in data_dir.glob('*.json') if f.name != 'manifest.json'])

    print(f"\nAnalyzing {len(sessions)} sessions...\n")

    all_results = []
    for session_path in sessions:
        result = analyze_session(session_path, verbose=True)
        if result:
            all_results.append(result)

    if not all_results:
        print("No valid sessions found!")
        return

    # Compute optimal soft iron
    print("\n" + "=" * 80)
    print("COMPUTING OPTIMAL SOFT IRON BOOTSTRAP")
    print("=" * 80)

    optimal = compute_optimal_soft_iron(all_results)

    print(f"\nFrom {len(all_results)} sessions:")

    print(f"\n--- Range-Based Diagonal Scales ---")
    print(f"Mean:   [{optimal['range_based']['mean'][0]:.4f}, {optimal['range_based']['mean'][1]:.4f}, {optimal['range_based']['mean'][2]:.4f}]")
    print(f"Std:    [{optimal['range_based']['std'][0]:.4f}, {optimal['range_based']['std'][1]:.4f}, {optimal['range_based']['std'][2]:.4f}]")
    print(f"Median: [{optimal['range_based']['median'][0]:.4f}, {optimal['range_based']['median'][1]:.4f}, {optimal['range_based']['median'][2]:.4f}]")

    if optimal['optimized']['median_diagonal']:
        print(f"\n--- Optimized Diagonal (scipy LM) ---")
        print(f"Mean:   [{optimal['optimized']['mean_diagonal'][0]:.4f}, {optimal['optimized']['mean_diagonal'][1]:.4f}, {optimal['optimized']['mean_diagonal'][2]:.4f}]")
        print(f"Std:    [{optimal['optimized']['std_diagonal'][0]:.4f}, {optimal['optimized']['std_diagonal'][1]:.4f}, {optimal['optimized']['std_diagonal'][2]:.4f}]")
        print(f"Median: [{optimal['optimized']['median_diagonal'][0]:.4f}, {optimal['optimized']['median_diagonal'][1]:.4f}, {optimal['optimized']['median_diagonal'][2]:.4f}]")

    if optimal['optimized']['median_matrix']:
        print(f"\n--- Optimized Full 3x3 Matrix (median) ---")
        M = np.array(optimal['optimized']['median_matrix'])
        print(f"  [{M[0,0]:7.4f}, {M[0,1]:7.4f}, {M[0,2]:7.4f}]")
        print(f"  [{M[1,0]:7.4f}, {M[1,1]:7.4f}, {M[1,2]:7.4f}]")
        print(f"  [{M[2,0]:7.4f}, {M[2,1]:7.4f}, {M[2,2]:7.4f}]")

    # Generate TypeScript code
    print("\n" + "=" * 80)
    print("TYPESCRIPT CODE FOR SOFT IRON BOOTSTRAP")
    print("=" * 80)

    if optimal['optimized']['median_diagonal']:
        diag = optimal['optimized']['median_diagonal']
        print(f"""
// Soft iron diagonal scale bootstrap from offline analysis of {len(all_results)} sessions
// These are scale factors to normalize ellipsoid to sphere
private _autoSoftIronScale: Vector3 = {{ x: {diag[0]:.4f}, y: {diag[1]:.4f}, z: {diag[2]:.4f} }};
""")

    if optimal['optimized']['median_matrix']:
        M = np.array(optimal['optimized']['median_matrix'])
        print(f"""
// Full 3x3 soft iron matrix bootstrap (if using full matrix mode)
// Generated from median of {len(all_results)} sessions
private _autoSoftIronMatrix: Matrix3 = new Matrix3([
    {M[0,0]:.4f}, {M[0,1]:.4f}, {M[0,2]:.4f},
    {M[1,0]:.4f}, {M[1,1]:.4f}, {M[1,2]:.4f},
    {M[2,0]:.4f}, {M[2,1]:.4f}, {M[2,2]:.4f}
]);
""")

    # Save results
    output = {
        'sessions_analyzed': len(all_results),
        'optimal_soft_iron': optimal,
        'session_results': all_results
    }

    output_path = Path('ml/soft_iron_analysis_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    # Generate comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Soft Iron Scale Factor Analysis', fontsize=14)

    # Plot 1: Diagonal scales distribution
    diag_scales = np.array([r['diagonal_scales'] for r in all_results])
    axes[0, 0].boxplot([diag_scales[:, 0], diag_scales[:, 1], diag_scales[:, 2]],
                        labels=['X', 'Y', 'Z'])
    axes[0, 0].axhline(y=1.0, color='r', linestyle='--', label='Identity')
    axes[0, 0].set_ylabel('Scale Factor')
    axes[0, 0].set_title('Range-Based Diagonal Scales')
    axes[0, 0].legend()

    # Plot 2: Optimized diagonals
    opt_diags = np.array([r['optimized_diagonal'] for r in all_results if r.get('optimized_diagonal')])
    if len(opt_diags) > 0:
        axes[0, 1].boxplot([opt_diags[:, 0], opt_diags[:, 1], opt_diags[:, 2]],
                            labels=['X', 'Y', 'Z'])
        axes[0, 1].axhline(y=1.0, color='r', linestyle='--', label='Identity')
        axes[0, 1].set_ylabel('Scale Factor')
        axes[0, 1].set_title('Optimized Diagonal Scales (scipy)')
        axes[0, 1].legend()

    # Plot 3: Off-diagonal magnitudes
    if optimal['optimized']['median_matrix']:
        opt_matrices = np.array([r['optimized_matrix'] for r in all_results if r.get('optimized_matrix')])
        off_diag_max = []
        for M in opt_matrices:
            M = np.array(M)
            off_diag = M - np.diag(np.diag(M))
            off_diag_max.append(np.max(np.abs(off_diag)))

        axes[1, 0].hist(off_diag_max, bins=20, edgecolor='black')
        axes[1, 0].axvline(x=np.median(off_diag_max), color='r', linestyle='--',
                           label=f'Median: {np.median(off_diag_max):.3f}')
        axes[1, 0].set_xlabel('Max Off-Diagonal Magnitude')
        axes[1, 0].set_ylabel('Count')
        axes[1, 0].set_title('Cross-Axis Coupling Strength')
        axes[1, 0].legend()

    # Plot 4: Residuals
    residuals = [r['optimized_residual'] for r in all_results if r.get('optimized_residual')]
    if residuals:
        axes[1, 1].hist(residuals, bins=20, edgecolor='black')
        axes[1, 1].axvline(x=np.median(residuals), color='r', linestyle='--',
                           label=f'Median: {np.median(residuals):.1f} µT')
        axes[1, 1].set_xlabel('Residual (µT)')
        axes[1, 1].set_ylabel('Count')
        axes[1, 1].set_title('Calibration Residuals')
        axes[1, 1].legend()

    plt.tight_layout()
    plot_path = Path('ml/soft_iron_analysis_plot.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Plot saved to: {plot_path}")


if __name__ == '__main__':
    main()
