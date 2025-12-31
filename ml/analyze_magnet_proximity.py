#!/usr/bin/env python3
"""
Deep analysis of magnetometer session with finger magnet proximity testing.
Investigates the new LM calibration and magnet detection behavior.
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional

# Expected Earth field at Edinburgh
EXPECTED_MAG = 50.4  # µT
EXPECTED_H = 18.9    # Horizontal component
EXPECTED_V = 46.7    # Vertical component (pointing down)


def load_session(path: Path) -> Dict:
    """Load session JSON."""
    with open(path) as f:
        return json.load(f)


def extract_data(samples: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract all sensor data from samples."""
    data = {
        'mx': np.array([s.get('mx', s.get('magX', 0)) for s in samples]),
        'my': np.array([s.get('my', s.get('magY', 0)) for s in samples]),
        'mz': np.array([s.get('mz', s.get('magZ', 0)) for s in samples]),
        'ax': np.array([s.get('ax', s.get('accelX', 0)) for s in samples]),
        'ay': np.array([s.get('ay', s.get('accelY', 0)) for s in samples]),
        'az': np.array([s.get('az', s.get('accelZ', 0)) for s in samples]),
        'timestamp': np.array([s.get('timestamp', i) for i, s in enumerate(samples)])
    }

    # Compute derived quantities
    data['mag_total'] = np.sqrt(data['mx']**2 + data['my']**2 + data['mz']**2)

    return data


def filter_outliers(data: Dict[str, np.ndarray], threshold: float = 150) -> Dict[str, np.ndarray]:
    """Filter outliers based on per-axis threshold from median."""
    valid = np.ones(len(data['mx']), dtype=bool)

    for key in ['mx', 'my', 'mz']:
        median = np.median(data[key])
        valid &= np.abs(data[key] - median) < threshold

    return {k: v[valid] for k, v in data.items()}


def compute_hard_iron_offset(data: Dict[str, np.ndarray]) -> np.ndarray:
    """Compute hard iron offset using min-max method."""
    return np.array([
        (data['mx'].max() + data['mx'].min()) / 2,
        (data['my'].max() + data['my'].min()) / 2,
        (data['mz'].max() + data['mz'].min()) / 2
    ])


def identify_quiet_periods(data: Dict[str, np.ndarray], magnitude_threshold: float = 100) -> np.ndarray:
    """
    Identify samples where only Earth field is present (no magnet nearby).
    Returns boolean mask of quiet samples.
    """
    # Total magnitude should be near Earth field (~50 µT) when no magnet present
    # With some margin for sensor noise and slight miscalibration
    return data['mag_total'] < magnitude_threshold


def lm_calibration(data: Dict[str, np.ndarray], initial_offset: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Levenberg-Marquardt calibration matching TypeScript implementation.
    Returns: (offset, soft_iron_matrix, rms_residual)
    """
    mx, my, mz = data['mx'], data['my'], data['mz']
    n_samples = min(500, len(mx))  # Match TS max samples

    # Subsample if needed
    if len(mx) > n_samples:
        indices = np.linspace(0, len(mx)-1, n_samples, dtype=int)
        mx, my, mz = mx[indices], my[indices], mz[indices]

    # Initial soft iron - start closer to identity to respect bounds
    # Compute from ranges but clamp to valid range
    ranges = np.array([mx.max() - mx.min(), my.max() - my.min(), mz.max() - mz.min()])
    expected_range = 2 * EXPECTED_MAG
    diag_init = expected_range / np.maximum(ranges, 20)
    # Clamp diagonal to bounds [0.5, 2.0]
    diag_init = np.clip(diag_init, 0.5, 2.0)
    initial_S = np.diag(diag_init)

    # Pack parameters
    x0 = np.concatenate([initial_offset, initial_S.flatten()])

    def residual_func(params):
        offset = params[:3]
        S = params[3:].reshape(3, 3)

        centered = np.column_stack([mx - offset[0], my - offset[1], mz - offset[2]])
        corrected = (S @ centered.T).T
        magnitudes = np.linalg.norm(corrected, axis=1)

        return magnitudes - EXPECTED_MAG

    # Bounds matching TS implementation
    lower = np.concatenate([[-np.inf]*3, [0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5]])
    upper = np.concatenate([[np.inf]*3, [2.0, 0.5, 0.5, 0.5, 2.0, 0.5, 0.5, 0.5, 2.0]])

    result = least_squares(residual_func, x0, bounds=(lower, upper), method='trf', max_nfev=1000)

    offset = result.x[:3]
    S = result.x[3:].reshape(3, 3)
    residuals = residual_func(result.x)
    rms = np.sqrt(np.mean(residuals**2))

    return offset, S, rms


def detect_magnet_events(data: Dict[str, np.ndarray], offset: np.ndarray, S: np.ndarray) -> Dict:
    """
    Detect magnet proximity events by analyzing residual magnitude.
    Returns analysis of magnet detection.
    """
    mx, my, mz = data['mx'], data['my'], data['mz']

    # Apply calibration
    centered = np.column_stack([mx - offset[0], my - offset[1], mz - offset[2]])
    corrected = (S @ centered.T).T

    # Corrected magnitude
    mag_corrected = np.linalg.norm(corrected, axis=1)

    # Residual from expected Earth field
    residual = mag_corrected - EXPECTED_MAG

    # Detect peaks (magnet approaches)
    # Finger magnets typically add 50-200+ µT
    peaks_pos, props_pos = find_peaks(residual, height=20, distance=50, prominence=10)
    peaks_neg, props_neg = find_peaks(-residual, height=20, distance=50, prominence=10)

    return {
        'corrected': corrected,
        'mag_corrected': mag_corrected,
        'residual': residual,
        'peaks_positive': peaks_pos,
        'peak_heights_pos': props_pos.get('peak_heights', []),
        'peaks_negative': peaks_neg,
        'peak_heights_neg': props_neg.get('peak_heights', []),
        'residual_mean': np.mean(residual),
        'residual_std': np.std(residual),
        'residual_max': np.max(residual),
        'residual_min': np.min(residual)
    }


def analyze_hv_components(data: Dict[str, np.ndarray], offset: np.ndarray, S: np.ndarray) -> Dict:
    """Analyze horizontal and vertical components after calibration."""
    mx, my, mz = data['mx'], data['my'], data['mz']
    ax, ay, az = data['ax'], data['ay'], data['az']

    # Apply calibration
    centered = np.column_stack([mx - offset[0], my - offset[1], mz - offset[2]])
    corrected = (S @ centered.T).T

    # Get roll/pitch from accelerometer
    a_mag = np.sqrt(ax**2 + ay**2 + az**2)
    ax_n, ay_n, az_n = ax/a_mag, ay/a_mag, az/a_mag

    roll = np.arctan2(ay_n, az_n)
    pitch = np.arctan2(-ax_n, np.sqrt(ay_n**2 + az_n**2))

    # Tilt compensate
    cr, sr = np.cos(roll), np.sin(roll)
    cp, sp = np.cos(pitch), np.sin(pitch)

    mx_h = corrected[:, 0] * cp + corrected[:, 1] * sr * sp + corrected[:, 2] * cr * sp
    my_h = corrected[:, 1] * cr - corrected[:, 2] * sr
    mz_h = -corrected[:, 0] * sp + corrected[:, 1] * cr * sp + corrected[:, 2] * cr * cp

    h_mag = np.sqrt(mx_h**2 + my_h**2)
    v_mag = mz_h

    return {
        'h_component': h_mag,
        'v_component': v_mag,
        'h_mean': np.mean(h_mag),
        'v_mean': np.mean(v_mag),
        'h_std': np.std(h_mag),
        'v_std': np.std(v_mag),
        'hv_ratio': np.mean(h_mag) / np.abs(np.mean(v_mag)) if np.mean(v_mag) != 0 else float('inf')
    }


def main():
    session_path = Path('data/GAMBIT/2025-12-31T12_34_54.116Z.json')

    print("=" * 80)
    print("FINGER MAGNET PROXIMITY ANALYSIS")
    print(f"Session: {session_path.name}")
    print("=" * 80)

    # Load data
    session = load_session(session_path)
    samples = session.get('samples', [])
    metadata = session.get('metadata', {})

    print(f"\nTotal samples: {len(samples)}")
    print(f"Session metadata: {json.dumps(metadata, indent=2)[:500]}")

    # Extract data
    data = extract_data(samples)
    print(f"\nRaw data statistics:")
    print(f"  Mag X: {data['mx'].min():.1f} to {data['mx'].max():.1f} µT (range: {data['mx'].max()-data['mx'].min():.1f})")
    print(f"  Mag Y: {data['my'].min():.1f} to {data['my'].max():.1f} µT (range: {data['my'].max()-data['my'].min():.1f})")
    print(f"  Mag Z: {data['mz'].min():.1f} to {data['mz'].max():.1f} µT (range: {data['mz'].max()-data['mz'].min():.1f})")
    print(f"  Total magnitude: {data['mag_total'].min():.1f} to {data['mag_total'].max():.1f} µT")

    # Identify quiet periods (no magnet) for calibration
    # Try different thresholds to find quiet samples
    quiet_thresholds = [80, 100, 150, 200, 300]
    quiet_counts = {}
    for thresh in quiet_thresholds:
        mask = identify_quiet_periods(data, thresh)
        quiet_counts[thresh] = np.sum(mask)

    print(f"\nQuiet period detection (magnitude < threshold):")
    for thresh, count in quiet_counts.items():
        pct = 100 * count / len(data['mx'])
        print(f"  < {thresh} µT: {count} samples ({pct:.1f}%)")

    # Use best threshold that gives enough samples
    best_thresh = 80
    for thresh in quiet_thresholds:
        if quiet_counts[thresh] >= 200:
            best_thresh = thresh
            break

    quiet_mask = identify_quiet_periods(data, best_thresh)
    n_quiet = np.sum(quiet_mask)

    if n_quiet < 50:
        print(f"\n⚠️  Very few quiet samples ({n_quiet})! Session dominated by magnet activity.")
        print("   Using outlier filtering as fallback...")
        data_for_cal = filter_outliers(data)
        n_quiet = len(data_for_cal['mx'])
    else:
        # Use quiet samples for calibration
        data_for_cal = {k: v[quiet_mask] for k, v in data.items()}

    print(f"\nUsing {n_quiet} quiet samples (threshold: {best_thresh} µT) for calibration")

    # Compute calibration
    print("\n" + "=" * 80)
    print("LEVENBERG-MARQUARDT CALIBRATION")
    print("=" * 80)

    offset = compute_hard_iron_offset(data_for_cal)
    print(f"\nMin-max hard iron offset: [{offset[0]:.1f}, {offset[1]:.1f}, {offset[2]:.1f}] µT")

    lm_offset, lm_S, lm_residual = lm_calibration(data_for_cal, offset)
    print(f"\nLM calibration results:")
    print(f"  Hard iron offset: [{lm_offset[0]:.2f}, {lm_offset[1]:.2f}, {lm_offset[2]:.2f}] µT")
    print(f"  Soft iron matrix:")
    print(f"    [{lm_S[0,0]:.4f}, {lm_S[0,1]:.4f}, {lm_S[0,2]:.4f}]")
    print(f"    [{lm_S[1,0]:.4f}, {lm_S[1,1]:.4f}, {lm_S[1,2]:.4f}]")
    print(f"    [{lm_S[2,0]:.4f}, {lm_S[2,1]:.4f}, {lm_S[2,2]:.4f}]")
    print(f"  RMS residual: {lm_residual:.2f} µT")

    # H/V analysis
    hv = analyze_hv_components(data_for_cal, lm_offset, lm_S)
    print(f"\nH/V component analysis:")
    print(f"  Horizontal: {hv['h_mean']:.1f} ± {hv['h_std']:.1f} µT (expected: {EXPECTED_H:.1f})")
    print(f"  Vertical: {hv['v_mean']:.1f} ± {hv['v_std']:.1f} µT (expected: {EXPECTED_V:.1f})")
    print(f"  H/V ratio: {hv['hv_ratio']:.2f} (expected: {EXPECTED_H/EXPECTED_V:.2f})")

    # Magnet detection on full dataset
    print("\n" + "=" * 80)
    print("MAGNET PROXIMITY DETECTION")
    print("=" * 80)

    magnet = detect_magnet_events(data, lm_offset, lm_S)
    print(f"\nResidual statistics (full dataset):")
    print(f"  Mean: {magnet['residual_mean']:.2f} µT")
    print(f"  Std: {magnet['residual_std']:.2f} µT")
    print(f"  Range: [{magnet['residual_min']:.1f}, {magnet['residual_max']:.1f}] µT")

    print(f"\nDetected magnet events:")
    print(f"  Positive peaks (magnet approach): {len(magnet['peaks_positive'])}")
    if len(magnet['peak_heights_pos']) > 0:
        print(f"    Peak heights: {', '.join([f'{h:.1f}µT' for h in magnet['peak_heights_pos'][:10]])}")
    print(f"  Negative peaks (opposite polarity): {len(magnet['peaks_negative'])}")
    if len(magnet['peak_heights_neg']) > 0:
        print(f"    Peak heights: {', '.join([f'{h:.1f}µT' for h in magnet['peak_heights_neg'][:10]])}")

    # Time-based analysis
    print("\n" + "=" * 80)
    print("TEMPORAL ANALYSIS")
    print("=" * 80)

    # Divide into windows and analyze stability
    window_size = 500
    n_windows = len(data['mx']) // window_size

    print(f"\nAnalysis by {window_size}-sample windows:")
    print(f"{'Window':<10} {'Samples':<10} {'Residual Mean':<15} {'Residual Std':<15} {'Max Deviation':<15}")
    print("-" * 65)

    window_stats = []
    for i in range(n_windows):
        start = i * window_size
        end = start + window_size
        window_data = {k: v[start:end] for k, v in data.items()}

        # Apply calibration to window
        mx, my, mz = window_data['mx'], window_data['my'], window_data['mz']
        centered = np.column_stack([mx - lm_offset[0], my - lm_offset[1], mz - lm_offset[2]])
        corrected = (lm_S @ centered.T).T
        mag = np.linalg.norm(corrected, axis=1)
        residual = mag - EXPECTED_MAG

        stats = {
            'window': i,
            'n_samples': len(mx),
            'residual_mean': np.mean(residual),
            'residual_std': np.std(residual),
            'max_deviation': np.max(np.abs(residual))
        }
        window_stats.append(stats)

        print(f"{i:<10} {stats['n_samples']:<10} {stats['residual_mean']:<15.2f} {stats['residual_std']:<15.2f} {stats['max_deviation']:<15.1f}")

    # Identify windows with magnet activity
    print("\n\nWindows with significant magnet activity (max deviation > 50 µT):")
    for stats in window_stats:
        if stats['max_deviation'] > 50:
            print(f"  Window {stats['window']}: max deviation {stats['max_deviation']:.1f} µT")

    # Generate visualization
    print("\n" + "=" * 80)
    print("GENERATING VISUALIZATION")
    print("=" * 80)

    fig, axes = plt.subplots(4, 1, figsize=(14, 16))
    fig.suptitle(f'Finger Magnet Proximity Analysis\n{session_path.name}', fontsize=14)

    # Plot 1: Raw magnetometer data
    ax1 = axes[0]
    sample_idx = np.arange(len(data['mx']))
    ax1.plot(sample_idx, data['mx'], 'r-', alpha=0.7, label='Mag X')
    ax1.plot(sample_idx, data['my'], 'g-', alpha=0.7, label='Mag Y')
    ax1.plot(sample_idx, data['mz'], 'b-', alpha=0.7, label='Mag Z')
    ax1.set_xlabel('Sample')
    ax1.set_ylabel('Magnetic Field (µT)')
    ax1.set_title('Raw Magnetometer Data')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Calibrated magnitude and residual
    ax2 = axes[1]
    ax2.plot(sample_idx, magnet['mag_corrected'], 'b-', alpha=0.7, label='Corrected Magnitude')
    ax2.axhline(y=EXPECTED_MAG, color='r', linestyle='--', label=f'Expected ({EXPECTED_MAG} µT)')
    ax2.fill_between(sample_idx, EXPECTED_MAG - 5, EXPECTED_MAG + 5, alpha=0.2, color='green', label='±5 µT band')
    ax2.set_xlabel('Sample')
    ax2.set_ylabel('Magnitude (µT)')
    ax2.set_title('Calibrated Magnitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Residual with peaks marked
    ax3 = axes[2]
    ax3.plot(sample_idx, magnet['residual'], 'b-', alpha=0.7)
    ax3.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
    ax3.axhline(y=20, color='r', linestyle='--', alpha=0.5, label='Detection threshold (20 µT)')
    ax3.axhline(y=-20, color='r', linestyle='--', alpha=0.5)

    # Mark detected peaks
    if len(magnet['peaks_positive']) > 0:
        ax3.scatter(magnet['peaks_positive'], magnet['residual'][magnet['peaks_positive']],
                   c='red', s=50, zorder=5, label=f"Positive peaks ({len(magnet['peaks_positive'])})")
    if len(magnet['peaks_negative']) > 0:
        ax3.scatter(magnet['peaks_negative'], magnet['residual'][magnet['peaks_negative']],
                   c='blue', s=50, zorder=5, label=f"Negative peaks ({len(magnet['peaks_negative'])})")

    ax3.set_xlabel('Sample')
    ax3.set_ylabel('Residual (µT)')
    ax3.set_title('Magnitude Residual (Magnet Detection Signal)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: H/V components
    ax4 = axes[3]
    ax4.plot(sample_idx[:len(hv['h_component'])], hv['h_component'], 'r-', alpha=0.7, label='Horizontal')
    ax4.plot(sample_idx[:len(hv['v_component'])], hv['v_component'], 'b-', alpha=0.7, label='Vertical')
    ax4.axhline(y=EXPECTED_H, color='r', linestyle='--', alpha=0.5, label=f'Expected H ({EXPECTED_H} µT)')
    ax4.axhline(y=EXPECTED_V, color='b', linestyle='--', alpha=0.5, label=f'Expected V ({EXPECTED_V} µT)')
    ax4.set_xlabel('Sample')
    ax4.set_ylabel('Component (µT)')
    ax4.set_title('Horizontal and Vertical Components')
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = Path('ml/magnet_proximity_analysis.png')
    plt.savefig(plot_path, dpi=150)
    print(f"\nPlot saved to: {plot_path}")

    # Save detailed results
    results = {
        'session': session_path.name,
        'total_samples': len(samples),
        'filtered_samples': len(data_for_cal['mx']),
        'calibration': {
            'offset': lm_offset.tolist(),
            'soft_iron': lm_S.tolist(),
            'rms_residual': float(lm_residual)
        },
        'hv_analysis': {
            'h_mean': float(hv['h_mean']),
            'v_mean': float(hv['v_mean']),
            'hv_ratio': float(hv['hv_ratio'])
        },
        'magnet_detection': {
            'residual_mean': float(magnet['residual_mean']),
            'residual_std': float(magnet['residual_std']),
            'residual_range': [float(magnet['residual_min']), float(magnet['residual_max'])],
            'n_positive_peaks': len(magnet['peaks_positive']),
            'n_negative_peaks': len(magnet['peaks_negative']),
            'peak_heights_pos': [float(h) for h in magnet['peak_heights_pos']],
            'peak_heights_neg': [float(h) for h in magnet['peak_heights_neg']]
        },
        'window_stats': window_stats
    }

    results_path = Path('ml/magnet_proximity_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
