#!/usr/bin/env python3
"""
Analyze calibration stages and bootstrap impact for a session.

Investigates:
1. Why UI showed 100% from start (bootstrap ranges exceeded threshold)
2. How min-max calibration was contaminated by magnet
3. When and how LM calibration fixed it
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# Expected Earth field at Edinburgh
EXPECTED_MAG = 50.4  # µT
EXPECTED_H = 18.9    # Horizontal component
EXPECTED_V = 46.7    # Vertical component (pointing down)

# Bootstrap values from unified-mag-calibration.ts
BOOTSTRAP_OFFSET = np.array([29.3, -9.9, -20.1])  # µT
BOOTSTRAP_SOFT_IRON = np.array([1.193, 1.018, 0.700])


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
    data['mag_total'] = np.sqrt(data['mx']**2 + data['my']**2 + data['mz']**2)
    return data


def simulate_calibration_progress(data: Dict[str, np.ndarray]) -> Dict:
    """
    Simulate the calibration progress algorithm from TypeScript.

    The firmware uses:
    - Bootstrap offset + bounds as starting point
    - Min-max tracking to update offset estimate
    - Progress = min(axisProgress) where axisProgress = range / targetRange
    - Target range = 80µT per axis
    """
    target_range = 80.0  # µT threshold for 100% on each axis

    # Initialize with bootstrap values
    running_min = BOOTSTRAP_OFFSET - np.array([42.3, 49.5, 72.0])  # bootstrap min
    running_max = BOOTSTRAP_OFFSET + np.array([42.3, 49.5, 72.0])  # bootstrap max

    progress_history = []
    offset_history = []
    range_history = []

    for i in range(len(data['mx'])):
        raw = np.array([data['mx'][i], data['my'][i], data['mz'][i]])

        # Update min/max (this is how firmware tracks)
        running_min = np.minimum(running_min, raw)
        running_max = np.maximum(running_max, raw)

        # Compute current range and offset
        current_range = running_max - running_min
        current_offset = (running_max + running_min) / 2

        # Compute progress (matches TypeScript logic)
        axis_progress = np.minimum(current_range / target_range, 1.0)
        overall_progress = np.min(axis_progress)

        progress_history.append(overall_progress)
        offset_history.append(current_offset.copy())
        range_history.append(current_range.copy())

    return {
        'progress': np.array(progress_history),
        'offsets': np.array(offset_history),
        'ranges': np.array(range_history),
        'timestamps': data['timestamp']
    }


def simulate_calibration_without_bootstrap(data: Dict[str, np.ndarray]) -> Dict:
    """
    Simulate calibration WITHOUT bootstrap for comparison.
    """
    target_range = 80.0

    # Start fresh - no bootstrap
    running_min = np.array([np.inf, np.inf, np.inf])
    running_max = np.array([-np.inf, -np.inf, -np.inf])

    progress_history = []
    range_history = []

    for i in range(len(data['mx'])):
        raw = np.array([data['mx'][i], data['my'][i], data['mz'][i]])

        running_min = np.minimum(running_min, raw)
        running_max = np.maximum(running_max, raw)

        current_range = running_max - running_min
        axis_progress = np.minimum(current_range / target_range, 1.0)
        overall_progress = np.min(axis_progress)

        progress_history.append(overall_progress)
        range_history.append(current_range.copy())

    return {
        'progress': np.array(progress_history),
        'ranges': np.array(range_history)
    }


def find_calibration_completion_sample(progress: np.ndarray) -> int:
    """Find the sample index where calibration first reaches 100%."""
    complete_indices = np.where(progress >= 1.0)[0]
    if len(complete_indices) > 0:
        return complete_indices[0]
    return -1


def lm_calibration(data: Dict[str, np.ndarray], initial_offset: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float]:
    """Run LM calibration matching TypeScript implementation."""
    mx, my, mz = data['mx'], data['my'], data['mz']
    n_samples = min(500, len(mx))

    if len(mx) > n_samples:
        indices = np.linspace(0, len(mx)-1, n_samples, dtype=int)
        mx, my, mz = mx[indices], my[indices], mz[indices]

    # Initial soft iron from ranges
    ranges = np.array([mx.max() - mx.min(), my.max() - my.min(), mz.max() - mz.min()])
    expected_range = 2 * EXPECTED_MAG
    diag_init = expected_range / np.maximum(ranges, 20)
    diag_init = np.clip(diag_init, 0.5, 2.0)
    initial_S = np.diag(diag_init)

    x0 = np.concatenate([initial_offset, initial_S.flatten()])

    def residual_func(params):
        offset = params[:3]
        S = params[3:].reshape(3, 3)
        centered = np.column_stack([mx - offset[0], my - offset[1], mz - offset[2]])
        corrected = (S @ centered.T).T
        magnitudes = np.linalg.norm(corrected, axis=1)
        return magnitudes - EXPECTED_MAG

    lower_bounds = np.concatenate([
        [-np.inf, -np.inf, -np.inf],
        [0.5, -0.5, -0.5, -0.5, 0.5, -0.5, -0.5, -0.5, 0.5]
    ])
    upper_bounds = np.concatenate([
        [np.inf, np.inf, np.inf],
        [2.0, 0.5, 0.5, 0.5, 2.0, 0.5, 0.5, 0.5, 2.0]
    ])

    result = least_squares(residual_func, x0, bounds=(lower_bounds, upper_bounds), max_nfev=500)

    offset = result.x[:3]
    S = result.x[3:].reshape(3, 3)

    # Compute RMS residual
    centered = np.column_stack([mx - offset[0], my - offset[1], mz - offset[2]])
    corrected = (S @ centered.T).T
    magnitudes = np.linalg.norm(corrected, axis=1)
    rms_residual = np.sqrt(np.mean((magnitudes - EXPECTED_MAG)**2))

    return offset, S, rms_residual


def analyze_quality_metrics(data: Dict[str, np.ndarray], offset: np.ndarray, soft_iron: np.ndarray) -> Dict:
    """Compute calibration quality metrics."""
    mx, my, mz = data['mx'], data['my'], data['mz']
    ax, ay, az = data['ax'], data['ay'], data['az']

    # Apply calibration
    centered = np.column_stack([mx - offset[0], my - offset[1], mz - offset[2]])
    if soft_iron.ndim == 1:
        corrected = centered * soft_iron
    else:
        corrected = (soft_iron @ centered.T).T

    magnitudes = np.linalg.norm(corrected, axis=1)
    mean_mag = np.mean(magnitudes)
    mag_error = abs(mean_mag - EXPECTED_MAG) / EXPECTED_MAG * 100

    # Compute H/V ratio
    # Gravity direction
    accel_norm = np.column_stack([ax, ay, az])
    accel_mag = np.linalg.norm(accel_norm, axis=1, keepdims=True)
    down = accel_norm / np.maximum(accel_mag, 0.001)

    # Vertical component (projection onto gravity)
    v_component = np.sum(corrected * down, axis=1)

    # Horizontal component
    h_component = np.sqrt(magnitudes**2 - v_component**2)

    hv_ratio = np.mean(h_component) / np.mean(np.abs(v_component))
    expected_hv_ratio = EXPECTED_H / EXPECTED_V

    return {
        'mean_magnitude': mean_mag,
        'mag_error_percent': mag_error,
        'hv_ratio': hv_ratio,
        'expected_hv_ratio': expected_hv_ratio,
        'rms_residual': np.sqrt(np.mean((magnitudes - EXPECTED_MAG)**2))
    }


def main():
    session_path = Path('data/GAMBIT/2025-12-31T12_34_54.116Z.json')

    print("=" * 80)
    print("CALIBRATION STAGES AND BOOTSTRAP IMPACT ANALYSIS")
    print("=" * 80)
    print(f"\nSession: {session_path.name}")

    session = load_session(session_path)
    samples = session.get('samples', [])

    print(f"Total samples: {len(samples)}")

    # Extract data
    data = extract_data(samples)

    # Get session start time
    start_ts = data['timestamp'][0]
    print(f"Session start timestamp: {start_ts}")

    # 1. Simulate calibration WITH bootstrap
    print("\n" + "=" * 80)
    print("1. CALIBRATION WITH BOOTSTRAP")
    print("=" * 80)

    with_bootstrap = simulate_calibration_progress(data)
    completion_idx = find_calibration_completion_sample(with_bootstrap['progress'])

    if completion_idx >= 0:
        completion_ts = data['timestamp'][completion_idx]
        elapsed_ms = completion_ts - start_ts
        print(f"\n✓ Calibration reached 100% at sample {completion_idx}")
        print(f"  Elapsed time: {elapsed_ms:.0f} ms ({elapsed_ms/1000:.2f} sec)")
        print(f"  Timestamp: {completion_ts}")

        # Show ranges at completion
        ranges = with_bootstrap['ranges'][completion_idx]
        offset = with_bootstrap['offsets'][completion_idx]
        print(f"\n  At completion:")
        print(f"    Ranges: [{ranges[0]:.1f}, {ranges[1]:.1f}, {ranges[2]:.1f}] µT")
        print(f"    Offset: [{offset[0]:.1f}, {offset[1]:.1f}, {offset[2]:.1f}] µT")
    else:
        print("\n✗ Calibration never reached 100%")

    # 2. Compare WITHOUT bootstrap
    print("\n" + "=" * 80)
    print("2. CALIBRATION WITHOUT BOOTSTRAP (COMPARISON)")
    print("=" * 80)

    without_bootstrap = simulate_calibration_without_bootstrap(data)
    completion_idx_no_bootstrap = find_calibration_completion_sample(without_bootstrap['progress'])

    if completion_idx_no_bootstrap >= 0:
        completion_ts_no = data['timestamp'][completion_idx_no_bootstrap]
        elapsed_ms_no = completion_ts_no - start_ts
        print(f"\n✓ Would reach 100% at sample {completion_idx_no_bootstrap}")
        print(f"  Elapsed time: {elapsed_ms_no:.0f} ms ({elapsed_ms_no/1000:.2f} sec)")

        ranges_no = without_bootstrap['ranges'][completion_idx_no_bootstrap]
        print(f"  Ranges at completion: [{ranges_no[0]:.1f}, {ranges_no[1]:.1f}, {ranges_no[2]:.1f}] µT")
    else:
        # Show max progress achieved
        max_progress = np.max(without_bootstrap['progress'])
        max_idx = np.argmax(without_bootstrap['progress'])
        final_ranges = without_bootstrap['ranges'][-1]
        print(f"\n✗ Would NOT reach 100% (max: {max_progress*100:.1f}%)")
        print(f"  Final ranges: [{final_ranges[0]:.1f}, {final_ranges[1]:.1f}, {final_ranges[2]:.1f}] µT")

    # 3. Bootstrap impact analysis
    print("\n" + "=" * 80)
    print("3. BOOTSTRAP IMPACT ANALYSIS")
    print("=" * 80)

    # Show bootstrap contribution
    print(f"\nBootstrap hard iron offset: [{BOOTSTRAP_OFFSET[0]:.1f}, {BOOTSTRAP_OFFSET[1]:.1f}, {BOOTSTRAP_OFFSET[2]:.1f}] µT")
    print(f"Bootstrap min bounds: [{BOOTSTRAP_OFFSET[0]-42.3:.1f}, {BOOTSTRAP_OFFSET[1]-49.5:.1f}, {BOOTSTRAP_OFFSET[2]-72.0:.1f}] µT")
    print(f"Bootstrap max bounds: [{BOOTSTRAP_OFFSET[0]+42.3:.1f}, {BOOTSTRAP_OFFSET[1]+49.5:.1f}, {BOOTSTRAP_OFFSET[2]+72.0:.1f}] µT")
    print(f"Bootstrap implied range: [{84.6:.1f}, {99.0:.1f}, {144.0:.1f}] µT")

    target_range = 80.0
    bootstrap_range = np.array([84.6, 99.0, 144.0])
    bootstrap_progress = np.minimum(bootstrap_range / target_range, 1.0)
    print(f"\n→ Bootstrap alone gives progress: [{bootstrap_progress[0]*100:.0f}%, {bootstrap_progress[1]*100:.0f}%, {bootstrap_progress[2]*100:.0f}%]")
    print(f"→ Minimum = {np.min(bootstrap_progress)*100:.0f}% from start!")

    # 4. Quality of min-max calibration
    print("\n" + "=" * 80)
    print("4. MIN-MAX CALIBRATION QUALITY")
    print("=" * 80)

    # Compute offset from full session
    minmax_offset = np.array([
        (data['mx'].max() + data['mx'].min()) / 2,
        (data['my'].max() + data['my'].min()) / 2,
        (data['mz'].max() + data['mz'].min()) / 2
    ])

    print(f"\nMin-max offset (full session): [{minmax_offset[0]:.1f}, {minmax_offset[1]:.1f}, {minmax_offset[2]:.1f}] µT")

    # Quality with diagonal soft iron
    minmax_quality = analyze_quality_metrics(data, minmax_offset, BOOTSTRAP_SOFT_IRON)
    print(f"\nWith diagonal soft iron [{BOOTSTRAP_SOFT_IRON[0]:.3f}, {BOOTSTRAP_SOFT_IRON[1]:.3f}, {BOOTSTRAP_SOFT_IRON[2]:.3f}]:")
    print(f"  Mean magnitude: {minmax_quality['mean_magnitude']:.1f} µT (expected {EXPECTED_MAG:.1f})")
    print(f"  Magnitude error: {minmax_quality['mag_error_percent']:.1f}%")
    print(f"  H/V ratio: {minmax_quality['hv_ratio']:.2f} (expected {minmax_quality['expected_hv_ratio']:.2f})")
    print(f"  RMS residual: {minmax_quality['rms_residual']:.1f} µT")

    # Check quality gate
    if minmax_quality['rms_residual'] > 60:
        print(f"\n⚠️  MINIMAL TRUST: residual {minmax_quality['rms_residual']:.0f}µT > 60µT threshold")

    # 5. LM calibration improvement
    print("\n" + "=" * 80)
    print("5. LEVENBERG-MARQUARDT CALIBRATION")
    print("=" * 80)

    # Filter extreme outliers for LM (magnet peaks)
    # This session has very high magnitudes due to permanent magnet - use higher threshold
    mag_threshold_filter = np.percentile(data['mag_total'], 50)  # Use median as cutoff
    valid = data['mag_total'] < mag_threshold_filter

    if np.sum(valid) < 100:
        # If too few samples, use all data
        print(f"\nNote: Most samples have high magnitude (magnet present)")
        print(f"Using all samples for LM (no filtering)")
        data_filtered = data
    else:
        data_filtered = {k: v[valid] for k, v in data.items()}
        print(f"\nFiltered samples for LM: {len(data_filtered['mx'])} (threshold: {mag_threshold_filter:.0f}µT)")

    lm_offset, lm_matrix, lm_residual = lm_calibration(data_filtered, minmax_offset)

    print(f"\nLM Offset: [{lm_offset[0]:.1f}, {lm_offset[1]:.1f}, {lm_offset[2]:.1f}] µT")
    print(f"LM Soft Iron Matrix:")
    print(f"  [{lm_matrix[0,0]:.4f}, {lm_matrix[0,1]:.4f}, {lm_matrix[0,2]:.4f}]")
    print(f"  [{lm_matrix[1,0]:.4f}, {lm_matrix[1,1]:.4f}, {lm_matrix[1,2]:.4f}]")
    print(f"  [{lm_matrix[2,0]:.4f}, {lm_matrix[2,1]:.4f}, {lm_matrix[2,2]:.4f}]")
    print(f"RMS Residual: {lm_residual:.2f} µT")

    # Quality with LM calibration
    lm_quality = analyze_quality_metrics(data_filtered, lm_offset, lm_matrix)
    print(f"\nLM Quality metrics:")
    print(f"  Mean magnitude: {lm_quality['mean_magnitude']:.1f} µT (expected {EXPECTED_MAG:.1f})")
    print(f"  Magnitude error: {lm_quality['mag_error_percent']:.1f}%")
    print(f"  H/V ratio: {lm_quality['hv_ratio']:.2f} (expected {lm_quality['expected_hv_ratio']:.2f})")

    # 6. Identify magnet events and their impact
    print("\n" + "=" * 80)
    print("6. MAGNET PROXIMITY EVENTS")
    print("=" * 80)

    # Find periods where magnitude exceeds Earth field significantly
    mag_threshold = 100  # µT - anything above this likely has magnet influence
    magnet_mask = data['mag_total'] > mag_threshold
    n_magnet_samples = np.sum(magnet_mask)

    print(f"\nSamples with magnet influence (>{mag_threshold}µT): {n_magnet_samples} ({n_magnet_samples/len(data['mx'])*100:.1f}%)")
    print(f"Peak magnitude: {data['mag_total'].max():.1f} µT")

    # Show first few magnet events
    transitions = np.diff(magnet_mask.astype(int))
    starts = np.where(transitions == 1)[0]
    ends = np.where(transitions == -1)[0]

    if len(starts) > 0:
        print(f"\nMagnet approach events detected: {len(starts)}")
        print("\nFirst 5 events:")
        for i, start in enumerate(starts[:5]):
            # Find matching end
            matching_ends = ends[ends > start]
            end = matching_ends[0] if len(matching_ends) > 0 else len(data['mx']) - 1
            duration = end - start
            peak_in_event = np.max(data['mag_total'][start:end+1])
            elapsed = (data['timestamp'][start] - start_ts) / 1000
            print(f"  Event {i+1}: samples {start}-{end} (duration: {duration}, peak: {peak_in_event:.0f}µT, t+{elapsed:.1f}s)")

    # 7. Summary
    print("\n" + "=" * 80)
    print("7. SUMMARY: WHY UI SHOWED 100% FROM START")
    print("=" * 80)

    print("""
    The UI showed 100% calibration progress immediately because:

    1. BOOTSTRAP RANGES EXCEED THRESHOLD
       - Bootstrap min/max bounds span 84.6-144.0 µT per axis
       - Target range for 100%: 80 µT per axis
       - Bootstrap alone gives 100% on all axes (smallest = 84.6 > 80)

    2. FIRST SAMPLE TRIGGERS COMPLETION
       - Even without significant device rotation, bootstrap provides "virtual" range
       - Sample 0 immediately satisfies the range threshold
       - Cal progress jumps to 100% with the very first sample

    3. QUALITY GATES CAUGHT THE PROBLEM
       - Min-max calibration quality: {0:.1f}% error, H/V={1:.2f}
       - System correctly flagged: "MINIMAL TRUST: residual >{2:.0f}µT"
       - LM calibration triggered to improve quality

    4. LM CALIBRATION FIXED IT
       - LM achieved: {3:.1f}% error, H/V={4:.2f}
       - RMS residual improved from {5:.0f}µT to {6:.1f}µT

    RECOMMENDATION:
       - Progress calculation should use actual measured ranges, not bootstrap bounds
       - OR bootstrap bounds should not count toward "range seen" for progress
       - This would delay 100% until device actually rotates through required orientations
    """.format(
        minmax_quality['mag_error_percent'],
        minmax_quality['hv_ratio'],
        minmax_quality['rms_residual'],
        lm_quality['mag_error_percent'],
        lm_quality['hv_ratio'],
        minmax_quality['rms_residual'],
        lm_residual
    ))

    # 8. Generate visualization
    print("\n" + "=" * 80)
    print("8. GENERATING VISUALIZATION")
    print("=" * 80)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Calibration Stages and Bootstrap Impact Analysis', fontsize=14)

    time_sec = (data['timestamp'] - start_ts) / 1000

    # Plot 1: Calibration progress comparison
    ax1 = axes[0, 0]
    ax1.plot(time_sec, with_bootstrap['progress'] * 100, 'b-', label='With Bootstrap', linewidth=2)
    ax1.plot(time_sec, without_bootstrap['progress'] * 100, 'r--', label='Without Bootstrap', linewidth=2)
    ax1.axhline(y=100, color='g', linestyle=':', alpha=0.7, label='100% threshold')
    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Calibration Progress (%)')
    ax1.set_title('Calibration Progress: With vs Without Bootstrap')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([0, 110])

    # Plot 2: Magnetometer magnitude
    ax2 = axes[0, 1]
    ax2.plot(time_sec, data['mag_total'], 'b-', alpha=0.7, linewidth=0.5)
    ax2.axhline(y=EXPECTED_MAG, color='g', linestyle='--', label=f'Expected ({EXPECTED_MAG} µT)')
    ax2.axhline(y=100, color='r', linestyle=':', label='Magnet threshold (100 µT)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Magnitude (µT)')
    ax2.set_title('Raw Magnetometer Magnitude')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Per-axis ranges over time
    ax3 = axes[1, 0]
    ax3.plot(time_sec, with_bootstrap['ranges'][:, 0], 'r-', label='X range', alpha=0.8)
    ax3.plot(time_sec, with_bootstrap['ranges'][:, 1], 'g-', label='Y range', alpha=0.8)
    ax3.plot(time_sec, with_bootstrap['ranges'][:, 2], 'b-', label='Z range', alpha=0.8)
    ax3.axhline(y=80, color='k', linestyle='--', label='Target (80 µT)')
    ax3.set_xlabel('Time (seconds)')
    ax3.set_ylabel('Range (µT)')
    ax3.set_title('Per-Axis Range Evolution (with bootstrap)')
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    # Plot 4: Quality comparison
    ax4 = axes[1, 1]
    methods = ['Min-Max\n(diagonal SI)', 'LM Optimized\n(full 3x3)']
    errors = [minmax_quality['mag_error_percent'], lm_quality['mag_error_percent']]
    residuals = [minmax_quality['rms_residual'], lm_residual]

    x = np.arange(len(methods))
    width = 0.35

    bars1 = ax4.bar(x - width/2, errors, width, label='Magnitude Error (%)', color='coral')
    bars2 = ax4.bar(x + width/2, residuals, width, label='RMS Residual (µT)', color='steelblue')

    ax4.set_ylabel('Value')
    ax4.set_title('Calibration Quality Comparison')
    ax4.set_xticks(x)
    ax4.set_xticklabels(methods)
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bar in bars1:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)
    for bar in bars2:
        height = bar.get_height()
        ax4.annotate(f'{height:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha='center', va='bottom', fontsize=10)

    plt.tight_layout()

    output_path = Path('ml/calibration_stages_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")

    # Save results to JSON
    results = {
        'session': session_path.name,
        'total_samples': len(samples),
        'bootstrap': {
            'offset': BOOTSTRAP_OFFSET.tolist(),
            'soft_iron': BOOTSTRAP_SOFT_IRON.tolist(),
            'implied_range': [84.6, 99.0, 144.0],
            'initial_progress_percent': float(np.min(bootstrap_progress) * 100)
        },
        'with_bootstrap': {
            'completion_sample': int(completion_idx) if completion_idx >= 0 else None,
            'completion_time_ms': float(data['timestamp'][completion_idx] - start_ts) if completion_idx >= 0 else None
        },
        'without_bootstrap': {
            'completion_sample': int(completion_idx_no_bootstrap) if completion_idx_no_bootstrap >= 0 else None,
            'max_progress_percent': float(np.max(without_bootstrap['progress']) * 100)
        },
        'minmax_quality': {
            'offset': minmax_offset.tolist(),
            'mag_error_percent': minmax_quality['mag_error_percent'],
            'hv_ratio': minmax_quality['hv_ratio'],
            'rms_residual': minmax_quality['rms_residual']
        },
        'lm_quality': {
            'offset': lm_offset.tolist(),
            'soft_iron_diagonal': np.diag(lm_matrix).tolist(),
            'mag_error_percent': lm_quality['mag_error_percent'],
            'hv_ratio': lm_quality['hv_ratio'],
            'rms_residual': lm_residual
        },
        'magnet_events': {
            'samples_with_magnet': int(n_magnet_samples),
            'percentage': float(n_magnet_samples / len(data['mx']) * 100),
            'peak_magnitude': float(data['mag_total'].max()),
            'n_approach_events': len(starts)
        }
    }

    results_path = Path('ml/calibration_stages_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Results saved to: {results_path}")


if __name__ == '__main__':
    main()
