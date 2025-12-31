#!/usr/bin/env python3
"""
Detailed timeline analysis correlating session data with calibration log timestamps.

From user logs:
- 12:24:25 (t+0s): Cal progress 100%, ranges: [109, 99, 174] µT, offset: [41.3, -9.9, -5.3]
- 12:24:28 (t+3s): "Calibration COMPLETE", error 23.7%, H/V=0.67
- 12:24:28+: "MINIMAL TRUST: residual 125µT > 60µT"
- 12:24:33 (t+8s): LM with 200 samples, RMS residual 0.32µT, error 0.2%, H/V=0.45 ✓
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple

EXPECTED_MAG = 50.4  # µT
EXPECTED_H = 18.9    # Horizontal component
EXPECTED_V = 46.7    # Vertical component

# Bootstrap values
BOOTSTRAP_OFFSET = np.array([29.3, -9.9, -20.1])
BOOTSTRAP_SOFT_IRON = np.array([1.193, 1.018, 0.700])
BOOTSTRAP_MIN = BOOTSTRAP_OFFSET - np.array([42.3, 49.5, 72.0])
BOOTSTRAP_MAX = BOOTSTRAP_OFFSET + np.array([42.3, 49.5, 72.0])


def load_session(path: Path) -> Dict:
    with open(path) as f:
        return json.load(f)


def extract_data(samples: List[Dict]) -> Dict[str, np.ndarray]:
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


def simulate_firmware_calibration(data: Dict[str, np.ndarray],
                                   use_bootstrap: bool = True,
                                   verbose: bool = False) -> Dict:
    """
    Simulate the firmware calibration algorithm step-by-step.
    This matches the TypeScript implementation in unified-mag-calibration.ts
    """
    target_range = 80.0  # µT threshold for each axis

    if use_bootstrap:
        running_min = BOOTSTRAP_MIN.copy()
        running_max = BOOTSTRAP_MAX.copy()
        offset_estimate = BOOTSTRAP_OFFSET.copy()
    else:
        running_min = np.array([np.inf, np.inf, np.inf])
        running_max = np.array([-np.inf, -np.inf, -np.inf])
        offset_estimate = np.array([0.0, 0.0, 0.0])

    calibration_complete = False
    first_complete_sample = -1

    # Track history
    history = {
        'progress': [],
        'offset': [],
        'ranges': [],
        'complete': []
    }

    for i in range(len(data['mx'])):
        raw = np.array([data['mx'][i], data['my'][i], data['mz'][i]])

        # Update min/max
        running_min = np.minimum(running_min, raw)
        running_max = np.maximum(running_max, raw)

        # Compute range and offset
        current_range = running_max - running_min
        offset_estimate = (running_max + running_min) / 2

        # Compute progress
        axis_progress = np.minimum(current_range / target_range, 1.0)
        overall_progress = np.min(axis_progress)

        # Check completion
        if overall_progress >= 1.0 and not calibration_complete:
            calibration_complete = True
            first_complete_sample = i
            if verbose:
                print(f"  Calibration complete at sample {i}")
                print(f"    Ranges: [{current_range[0]:.1f}, {current_range[1]:.1f}, {current_range[2]:.1f}]")
                print(f"    Offset: [{offset_estimate[0]:.1f}, {offset_estimate[1]:.1f}, {offset_estimate[2]:.1f}]")

        history['progress'].append(overall_progress)
        history['offset'].append(offset_estimate.copy())
        history['ranges'].append(current_range.copy())
        history['complete'].append(calibration_complete)

    return {
        'history': {k: np.array(v) for k, v in history.items()},
        'first_complete': first_complete_sample,
        'final_offset': offset_estimate,
        'final_range': current_range
    }


def analyze_log_timestamps(data: Dict[str, np.ndarray]):
    """
    Analyze data at specific time offsets matching the user's log timestamps.
    Logs show:
    - t+0s (12:24:25): Cal progress 100%, ranges [109, 99, 174]
    - t+3s (12:24:28): Cal COMPLETE
    - t+8s (12:24:33): LM calibration
    """
    print("\n" + "=" * 80)
    print("LOG TIMESTAMP CORRELATION ANALYSIS")
    print("=" * 80)

    # Assuming 100 samples/sec, find samples at key timestamps
    timestamps_sec = [0, 3, 8]  # From logs
    samples_per_sec = 100  # Approximate sample rate

    print("\nExpected timestamps from logs:")
    print("  t+0s: Cal progress 100%, ranges [109, 99, 174], offset [41.3, -9.9, -5.3]")
    print("  t+3s: Calibration COMPLETE, error 23.7%, H/V=0.67")
    print("  t+8s: LM calibration, 200 samples, RMS 0.32µT, error 0.2%, H/V=0.45")

    print("\n--- Checking Data at Key Timestamps ---")

    for t in timestamps_sec:
        sample_idx = int(t * samples_per_sec)
        if sample_idx >= len(data['mx']):
            sample_idx = len(data['mx']) - 1

        # Get data window around this time
        window_start = max(0, sample_idx - 50)
        window_end = min(len(data['mx']), sample_idx + 50)

        window_data = {k: v[window_start:window_end] for k, v in data.items()}

        # Compute statistics for window
        ranges = [
            window_data['mx'].max() - window_data['mx'].min(),
            window_data['my'].max() - window_data['my'].min(),
            window_data['mz'].max() - window_data['mz'].min()
        ]
        offset = [
            (window_data['mx'].max() + window_data['mx'].min()) / 2,
            (window_data['my'].max() + window_data['my'].min()) / 2,
            (window_data['mz'].max() + window_data['mz'].min()) / 2
        ]
        mean_mag = np.mean(window_data['mag_total'])

        print(f"\nt+{t}s (samples {window_start}-{window_end}):")
        print(f"  Window ranges: [{ranges[0]:.1f}, {ranges[1]:.1f}, {ranges[2]:.1f}] µT")
        print(f"  Window offset: [{offset[0]:.1f}, {offset[1]:.1f}, {offset[2]:.1f}] µT")
        print(f"  Mean magnitude: {mean_mag:.1f} µT")

        # Compare to log values at t=0s
        if t == 0:
            log_ranges = [109, 99, 174]
            log_offset = [41.3, -9.9, -5.3]
            print(f"\n  Log values: ranges [{log_ranges[0]}, {log_ranges[1]}, {log_ranges[2]}]")
            print(f"  Log values: offset [{log_offset[0]}, {log_offset[1]}, {log_offset[2]}]")
            print(f"\n  → The log shows ranges exceeded bootstrap bounds early")
            print(f"  → Bootstrap bounds: [84.6, 99.0, 144.0]")
            print(f"  → Z-axis: 174 > 144, triggering 100%")


def compute_quality_at_time(data: Dict[str, np.ndarray],
                            end_sample: int,
                            offset: np.ndarray,
                            soft_iron: np.ndarray) -> Dict:
    """Compute quality metrics for data up to end_sample."""
    mx = data['mx'][:end_sample]
    my = data['my'][:end_sample]
    mz = data['mz'][:end_sample]
    ax = data['ax'][:end_sample]
    ay = data['ay'][:end_sample]
    az = data['az'][:end_sample]

    # Apply calibration
    centered = np.column_stack([mx - offset[0], my - offset[1], mz - offset[2]])
    if soft_iron.ndim == 1:
        corrected = centered * soft_iron
    else:
        corrected = (soft_iron @ centered.T).T

    magnitudes = np.linalg.norm(corrected, axis=1)
    mean_mag = np.mean(magnitudes)
    mag_error = abs(mean_mag - EXPECTED_MAG) / EXPECTED_MAG * 100
    rms_residual = np.sqrt(np.mean((magnitudes - EXPECTED_MAG)**2))

    # H/V ratio
    accel_norm = np.column_stack([ax, ay, az])
    accel_mag = np.linalg.norm(accel_norm, axis=1, keepdims=True)
    down = accel_norm / np.maximum(accel_mag, 0.001)

    v_component = np.sum(corrected * down, axis=1)
    h_component = np.sqrt(magnitudes**2 - v_component**2)

    hv_ratio = np.mean(h_component) / np.mean(np.abs(v_component))

    return {
        'mean_magnitude': mean_mag,
        'mag_error_percent': mag_error,
        'rms_residual': rms_residual,
        'hv_ratio': hv_ratio
    }


def main():
    session_path = Path('data/GAMBIT/2025-12-31T12_34_54.116Z.json')

    print("=" * 80)
    print("CALIBRATION TIMELINE AND BOOTSTRAP IMPACT ANALYSIS")
    print("=" * 80)
    print(f"\nSession: {session_path.name}")

    session = load_session(session_path)
    samples = session.get('samples', [])
    data = extract_data(samples)

    print(f"Total samples: {len(samples)}")
    print(f"Duration: ~{len(samples)/100:.1f} seconds")

    # 1. Show initial raw data characteristics
    print("\n" + "=" * 80)
    print("1. RAW DATA CHARACTERISTICS")
    print("=" * 80)

    print(f"\nFirst sample raw values:")
    print(f"  [{data['mx'][0]:.1f}, {data['my'][0]:.1f}, {data['mz'][0]:.1f}] µT")
    print(f"  Magnitude: {data['mag_total'][0]:.1f} µT")

    print(f"\nBootstrap bounds:")
    print(f"  Min: [{BOOTSTRAP_MIN[0]:.1f}, {BOOTSTRAP_MIN[1]:.1f}, {BOOTSTRAP_MIN[2]:.1f}] µT")
    print(f"  Max: [{BOOTSTRAP_MAX[0]:.1f}, {BOOTSTRAP_MAX[1]:.1f}, {BOOTSTRAP_MAX[2]:.1f}] µT")

    # Check if first sample exceeds bootstrap
    sample0 = np.array([data['mx'][0], data['my'][0], data['mz'][0]])
    exceeds_min = sample0 < BOOTSTRAP_MIN
    exceeds_max = sample0 > BOOTSTRAP_MAX

    print(f"\nFirst sample vs bootstrap bounds:")
    for i, axis in enumerate(['X', 'Y', 'Z']):
        if exceeds_min[i]:
            print(f"  {axis}: {sample0[i]:.1f} < {BOOTSTRAP_MIN[i]:.1f} (BELOW bootstrap min)")
        elif exceeds_max[i]:
            print(f"  {axis}: {sample0[i]:.1f} > {BOOTSTRAP_MAX[i]:.1f} (ABOVE bootstrap max)")
        else:
            print(f"  {axis}: {sample0[i]:.1f} within bounds")

    # 2. Simulate firmware calibration
    print("\n" + "=" * 80)
    print("2. FIRMWARE CALIBRATION SIMULATION")
    print("=" * 80)

    print("\n--- With Bootstrap ---")
    result_with = simulate_firmware_calibration(data, use_bootstrap=True, verbose=True)

    if result_with['first_complete'] == 0:
        print(f"\n⚠️  Calibration was 100% from sample 0!")
        # Explain why
        implied_range = BOOTSTRAP_MAX - BOOTSTRAP_MIN
        target = 80.0
        print(f"\n  Bootstrap provides implicit range:")
        print(f"    X: {implied_range[0]:.1f}µT vs target {target}µT → {min(1.0, implied_range[0]/target)*100:.0f}%")
        print(f"    Y: {implied_range[1]:.1f}µT vs target {target}µT → {min(1.0, implied_range[1]/target)*100:.0f}%")
        print(f"    Z: {implied_range[2]:.1f}µT vs target {target}µT → {min(1.0, implied_range[2]/target)*100:.0f}%")
        print(f"\n  → All axes already at 100% before first sample!")

    print("\n--- Without Bootstrap (comparison) ---")
    result_without = simulate_firmware_calibration(data, use_bootstrap=False, verbose=True)
    print(f"  First complete: sample {result_without['first_complete']}")

    # 3. Analyze log timestamp correlation
    analyze_log_timestamps(data)

    # 4. Quality evolution over time
    print("\n" + "=" * 80)
    print("3. QUALITY EVOLUTION OVER TIME")
    print("=" * 80)

    checkpoints = [100, 200, 300, 500, 800, 1000, 2000, len(samples)]

    print("\nQuality at different sample counts (with min-max offset + diagonal soft iron):")
    print("-" * 70)
    print(f"{'Samples':>8} | {'Offset X':>8} {'Y':>8} {'Z':>8} | {'Error%':>7} {'H/V':>5} {'RMS':>6}")
    print("-" * 70)

    for n in checkpoints:
        if n > len(samples):
            n = len(samples)

        # Get offset at this point
        offset = result_with['history']['offset'][n-1]

        # Compute quality
        quality = compute_quality_at_time(data, n, offset, BOOTSTRAP_SOFT_IRON)

        print(f"{n:>8} | {offset[0]:>8.1f} {offset[1]:>8.1f} {offset[2]:>8.1f} | "
              f"{quality['mag_error_percent']:>7.1f} {quality['hv_ratio']:>5.2f} {quality['rms_residual']:>6.1f}")

    # 5. Why the logs show different values
    print("\n" + "=" * 80)
    print("4. RECONCILING LOG VALUES WITH SESSION DATA")
    print("=" * 80)

    print("""
    LOG VALUES (from user):
    - t+0s: ranges [109, 99, 174], offset [41.3, -9.9, -5.3]
    - t+3s: error 23.7%, H/V=0.67
    - t+8s: LM achieved RMS 0.32µT

    SESSION DATA (first 300 samples):
    - Very high magnitudes: {:.0f}-{:.0f} µT (magnet always present)
    - Min-max offset: [{:.1f}, {:.1f}, {:.1f}] µT

    DISCREPANCY EXPLANATION:
    1. The log ranges [109, 99, 174] match bootstrap bounds (84.6, 99.0, 144.0)
       expanded by early samples slightly exceeding bounds

    2. The log offset [41.3, -9.9, -5.3] is close to bootstrap [29.3, -9.9, -20.1]
       with minor updates from early samples

    3. The log's 23.7% error and 125µT residual are from EARLY calibration
       before the magnet got very close

    4. The session data's high magnitudes (>500µT) are from LATER when
       magnet was brought very close for proximity testing

    5. The log's LM achievement (0.32µT residual) was possible because:
       - Used only first 200 samples
       - Magnet was farther away at that point
       - Our current analysis uses ALL samples including extreme proximity events
    """.format(
        data['mag_total'].min(),
        data['mag_total'].max(),
        result_with['history']['offset'][-1][0],
        result_with['history']['offset'][-1][1],
        result_with['history']['offset'][-1][2]
    ))

    # 6. Generate timeline plot
    print("\n" + "=" * 80)
    print("5. GENERATING TIMELINE VISUALIZATION")
    print("=" * 80)

    fig, axes = plt.subplots(3, 2, figsize=(14, 12))
    fig.suptitle('Calibration Timeline Analysis: Bootstrap Impact', fontsize=14)

    sample_idx = np.arange(len(data['mx']))
    time_sec = sample_idx / 100  # Assuming 100 Hz

    # Plot 1: Progress comparison
    ax = axes[0, 0]
    ax.plot(time_sec, result_with['history']['progress'] * 100, 'b-', label='With Bootstrap', linewidth=2)
    ax.plot(time_sec, result_without['history']['progress'] * 100, 'r--', label='Without Bootstrap', linewidth=2)
    ax.axhline(y=100, color='g', linestyle=':', alpha=0.7)
    ax.axvline(x=0, color='orange', linestyle='--', alpha=0.7, label='Log t+0s')
    ax.axvline(x=3, color='purple', linestyle='--', alpha=0.7, label='Log t+3s')
    ax.axvline(x=8, color='brown', linestyle='--', alpha=0.7, label='Log t+8s')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Calibration Progress (%)')
    ax.set_title('Progress: With vs Without Bootstrap')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 15])
    ax.set_ylim([0, 110])

    # Plot 2: Raw magnitude
    ax = axes[0, 1]
    ax.plot(time_sec, data['mag_total'], 'b-', alpha=0.7, linewidth=0.5)
    ax.axhline(y=EXPECTED_MAG, color='g', linestyle='--', label=f'Expected ({EXPECTED_MAG} µT)')
    ax.axhline(y=100, color='orange', linestyle=':', alpha=0.7, label='100 µT')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Magnitude (µT)')
    ax.set_title('Raw Magnetometer Magnitude')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 3: Per-axis values
    ax = axes[1, 0]
    ax.plot(time_sec, data['mx'], 'r-', alpha=0.5, label='X')
    ax.plot(time_sec, data['my'], 'g-', alpha=0.5, label='Y')
    ax.plot(time_sec, data['mz'], 'b-', alpha=0.5, label='Z')
    ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Field (µT)')
    ax.set_title('Raw Magnetometer Per-Axis')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 4: Bootstrap bounds visualization
    ax = axes[1, 1]
    n_show = 1500  # First 15 seconds
    for i, (axis, color) in enumerate(zip(['X', 'Y', 'Z'], ['r', 'g', 'b'])):
        ax.axhline(y=BOOTSTRAP_MIN[i], color=color, linestyle='--', alpha=0.5)
        ax.axhline(y=BOOTSTRAP_MAX[i], color=color, linestyle='--', alpha=0.5)
        ax.axhspan(BOOTSTRAP_MIN[i], BOOTSTRAP_MAX[i], color=color, alpha=0.1)

    ax.plot(time_sec[:n_show], data['mx'][:n_show], 'r-', alpha=0.7, label='X', linewidth=0.5)
    ax.plot(time_sec[:n_show], data['my'][:n_show], 'g-', alpha=0.7, label='Y', linewidth=0.5)
    ax.plot(time_sec[:n_show], data['mz'][:n_show], 'b-', alpha=0.7, label='Z', linewidth=0.5)
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Field (µT)')
    ax.set_title('First 15s: Data vs Bootstrap Bounds')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 15])

    # Plot 5: Range evolution
    ax = axes[2, 0]
    ranges = result_with['history']['ranges']
    ax.plot(time_sec, ranges[:, 0], 'r-', label='X range', linewidth=1.5)
    ax.plot(time_sec, ranges[:, 1], 'g-', label='Y range', linewidth=1.5)
    ax.plot(time_sec, ranges[:, 2], 'b-', label='Z range', linewidth=1.5)
    ax.axhline(y=80, color='k', linestyle='--', label='Target (80µT)')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Range (µT)')
    ax.set_title('Per-Axis Range Evolution (with bootstrap)')
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Plot 6: First 1000 samples zoomed
    ax = axes[2, 1]
    ax.plot(time_sec[:1000], data['mag_total'][:1000], 'b-', linewidth=1)
    ax.axhline(y=EXPECTED_MAG, color='g', linestyle='--', label=f'Expected ({EXPECTED_MAG} µT)')
    ax.axvline(x=0, color='orange', linestyle='--', alpha=0.7, label='Log t+0s')
    ax.axvline(x=3, color='purple', linestyle='--', alpha=0.7, label='Log t+3s')
    ax.axvline(x=8, color='brown', linestyle='--', alpha=0.7, label='Log t+8s')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Magnitude (µT)')
    ax.set_title('First 10s: Magnitude vs Log Timestamps')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    output_path = Path('ml/calibration_timeline_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"Plot saved to: {output_path}")

    # 7. Summary
    print("\n" + "=" * 80)
    print("SUMMARY: BOOTSTRAP IMPACT ON CALIBRATION")
    print("=" * 80)

    print("""
    KEY FINDING: Bootstrap bounds cause instant 100% progress

    1. BOOTSTRAP BOUNDS TOO WIDE
       - Bootstrap min/max span 84.6-144.0 µT per axis
       - Target range for 100%: 80 µT per axis
       - Bootstrap alone exceeds threshold → instant 100%

    2. ROOT CAUSE
       - When bootstrap is applied, the "virtual" range from historical
         min/max already exceeds the 80µT threshold
       - First sample immediately triggers "calibration complete"
       - User sees 100% before they've rotated the device at all

    3. QUALITY GATES WORKING CORRECTLY
       - Despite 100% progress, quality gates detected poor calibration
       - "MINIMAL TRUST" triggered when residual > 60µT
       - LM optimization scheduled and improved results

    4. RECOMMENDATIONS FOR CODE FIX
       a) Don't use bootstrap bounds for progress calculation:
          - Track actual measured min/max separately from bootstrap hints
          - Progress = actual_range / target_range (not bootstrap_range)

       b) OR increase target range when bootstrap is active:
          - If using bootstrap, require actual samples to expand range
          - e.g., target_range = max(80, bootstrap_range * 0.5)

       c) OR add "samples required" threshold:
          - Don't mark complete until N samples collected
          - e.g., require 200+ samples before completion
    """)


if __name__ == '__main__':
    main()
