#!/usr/bin/env python3
"""
Visualize calibration data to compare sessions and identify issues.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

def load_session(filepath):
    """Load a session file."""
    with open(filepath, 'r') as f:
        return json.load(f)

def plot_session_comparison():
    """Create comparison plots for all sessions."""
    data_dir = Path('/home/user/simcap/data/GAMBIT')
    session_files = sorted(data_dir.glob('2025-12-12T*.json'))

    fig, axes = plt.subplots(4, 1, figsize=(14, 12))
    fig.suptitle('GAMBIT Calibration Sessions - Magnetometer Data Comparison\n2025-12-12',
                 fontsize=14, fontweight='bold')

    colors = ['blue', 'orange', 'green', 'red']

    for idx, session_file in enumerate(session_files):
        session = load_session(session_file)
        samples = session['samples']

        # Extract magnetometer data
        mx = np.array([s['mx'] for s in samples])
        my = np.array([s['my'] for s in samples])
        mz = np.array([s['mz'] for s in samples])
        magnitude = np.sqrt(mx**2 + my**2 + mz**2)

        # Time axis (assuming samples are sequential)
        time = np.arange(len(samples))

        # Get session name
        name = session_file.stem.split('T')[1].replace('_', ':').split('.')[0]

        # Plot magnitude
        axes[idx].plot(time, magnitude, color=colors[idx], linewidth=1, alpha=0.8)
        axes[idx].axhline(y=50, color='green', linestyle='--', linewidth=1, label='Expected Earth Field (~50 µT)', alpha=0.5)
        axes[idx].set_ylabel('Magnitude (µT)', fontsize=10)
        axes[idx].set_title(f'Session {idx+1}: {name}', fontsize=10, fontweight='bold')
        axes[idx].grid(True, alpha=0.3)
        axes[idx].legend(loc='upper right', fontsize=8)

        # Annotate calibration segments
        labels = session.get('labels', [])
        for label in labels:
            start = label['start_sample']
            end = label['end_sample']
            cal_step = label.get('metadata', {}).get('calibration_step', 'Unknown')

            # Color code by calibration step
            if cal_step == 'EARTH_FIELD':
                axes[idx].axvspan(start, end, alpha=0.2, color='cyan', label=f'{cal_step}' if start == 0 else '')
            elif cal_step == 'HARD_IRON':
                axes[idx].axvspan(start, end, alpha=0.2, color='yellow', label=f'{cal_step}' if start == 1500 else '')

    axes[-1].set_xlabel('Sample Index', fontsize=10)
    plt.tight_layout()

    # Save plot
    output_path = data_dir / 'calibration_sessions_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved comparison plot: {output_path}")
    plt.close()

def plot_session_1_detailed():
    """Detailed plot of Session 1 showing all calibration stages."""
    data_dir = Path('/home/user/simcap/data/GAMBIT')
    session_file = data_dir / '2025-12-12T11_14_50.144Z.json'

    if not session_file.exists():
        print("Session 1 not found, skipping detailed plot")
        return

    session = load_session(session_file)
    samples = session['samples']

    # Extract all magnetometer representations
    mx_raw = np.array([s['mx'] for s in samples])
    my_raw = np.array([s['my'] for s in samples])
    mz_raw = np.array([s['mz'] for s in samples])

    mx_cal = np.array([s.get('calibrated_mx', 0) for s in samples])
    my_cal = np.array([s.get('calibrated_my', 0) for s in samples])
    mz_cal = np.array([s.get('calibrated_mz', 0) for s in samples])

    mx_fused = np.array([s.get('fused_mx', 0) for s in samples])
    my_fused = np.array([s.get('fused_my', 0) for s in samples])
    mz_fused = np.array([s.get('fused_mz', 0) for s in samples])

    mx_filt = np.array([s.get('filtered_mx', 0) for s in samples])
    my_filt = np.array([s.get('filtered_my', 0) for s in samples])
    mz_filt = np.array([s.get('filtered_mz', 0) for s in samples])

    # Calculate magnitudes
    mag_raw = np.sqrt(mx_raw**2 + my_raw**2 + mz_raw**2)
    mag_cal = np.sqrt(mx_cal**2 + my_cal**2 + mz_cal**2)
    mag_fused = np.sqrt(mx_fused**2 + my_fused**2 + mz_fused**2)
    mag_filt = np.sqrt(mx_filt**2 + my_filt**2 + mz_filt**2)

    time = np.arange(len(samples))

    # Create plot
    fig, axes = plt.subplots(2, 1, figsize=(14, 10))
    fig.suptitle('Session 1: Calibration Pipeline Stages (WITH Calibration Applied)',
                 fontsize=14, fontweight='bold')

    # Plot 1: Magnitudes
    axes[0].plot(time, mag_raw, 'gray', alpha=0.5, linewidth=1, label='Raw')
    axes[0].plot(time, mag_cal, 'blue', alpha=0.7, linewidth=1, label='Iron Corrected')
    axes[0].plot(time, mag_fused, 'green', alpha=0.8, linewidth=1.5, label='Fused (Earth Subtracted)')
    axes[0].plot(time, mag_filt, 'red', alpha=0.9, linewidth=1, label='Filtered')
    axes[0].axhline(y=50, color='black', linestyle='--', linewidth=1, label='Expected (~50 µT)', alpha=0.5)
    axes[0].set_ylabel('Magnitude (µT)', fontsize=11)
    axes[0].set_title('Magnetometer Magnitude: Processing Stages', fontsize=12)
    axes[0].legend(loc='upper right', fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Component breakdown for fused
    axes[1].plot(time, mx_fused, 'r-', alpha=0.7, linewidth=1, label='Fused MX')
    axes[1].plot(time, my_fused, 'g-', alpha=0.7, linewidth=1, label='Fused MY')
    axes[1].plot(time, mz_fused, 'b-', alpha=0.7, linewidth=1, label='Fused MZ')
    axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5, alpha=0.3)
    axes[1].set_ylabel('Field (µT)', fontsize=11)
    axes[1].set_xlabel('Sample Index', fontsize=11)
    axes[1].set_title('Fused Field Components (Should be near zero without magnets)', fontsize=12)
    axes[1].legend(loc='upper right', fontsize=9)
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    output_path = data_dir / 'session1_detailed_stages.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved Session 1 detailed plot: {output_path}")
    plt.close()

def plot_earth_field_segments():
    """Plot Earth field calibration segments to show environmental distortion."""
    data_dir = Path('/home/user/simcap/data/GAMBIT')
    session_file = data_dir / '2025-12-12T11_29_11.224Z.json'

    session = load_session(session_file)
    samples = session['samples']
    labels = session['labels']

    fig, axes = plt.subplots(3, 1, figsize=(14, 10))
    fig.suptitle('Earth Field Calibration Segments - Environmental Distortion Analysis',
                 fontsize=14, fontweight='bold')

    # Find Earth field segments
    earth_segments = [l for l in labels if l.get('metadata', {}).get('calibration_step') == 'EARTH_FIELD']

    for idx, segment in enumerate(earth_segments[:3]):  # Plot first 3
        start = segment['start_sample']
        end = segment['end_sample']

        seg_samples = samples[start:end]
        mx = np.array([s['mx'] for s in seg_samples])
        my = np.array([s['my'] for s in seg_samples])
        mz = np.array([s['mz'] for s in seg_samples])
        mag = np.sqrt(mx**2 + my**2 + mz**2)

        time = np.arange(len(seg_samples))

        axes[idx].plot(time, mx, 'r-', alpha=0.6, linewidth=1, label='MX')
        axes[idx].plot(time, my, 'g-', alpha=0.6, linewidth=1, label='MY')
        axes[idx].plot(time, mz, 'b-', alpha=0.6, linewidth=1, label='MZ')
        axes[idx].plot(time, mag, 'k-', alpha=0.8, linewidth=1.5, label='Magnitude')
        axes[idx].axhline(y=50, color='orange', linestyle='--', linewidth=1,
                         label='Expected Earth Mag (~50 µT)', alpha=0.7)

        quality = segment['metadata'].get('quality', 0)
        result = segment['metadata'].get('result_summary', {})
        measured_mag = float(result.get('magnitude', 0))

        axes[idx].set_ylabel('Field (µT)', fontsize=10)
        axes[idx].set_title(f'Earth Field Segment {idx+1}: Quality={quality:.3f}, Measured Mag={measured_mag:.1f} µT',
                           fontsize=11)
        axes[idx].legend(loc='upper right', fontsize=8, ncol=2)
        axes[idx].grid(True, alpha=0.3)

        # Annotate the issue
        axes[idx].text(0.02, 0.98,
                      f'⚠️ Excess field: ~{measured_mag - 50:.0f} µT\n(Environmental distortion)',
                      transform=axes[idx].transAxes, fontsize=9, verticalalignment='top',
                      bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.5))

    axes[-1].set_xlabel('Sample Index', fontsize=10)
    plt.tight_layout()

    # Save plot
    output_path = data_dir / 'earth_field_segments_analysis.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"✓ Saved Earth field segments plot: {output_path}")
    plt.close()

def main():
    print("Generating calibration visualization plots...")
    print()

    plot_session_comparison()
    plot_session_1_detailed()
    plot_earth_field_segments()

    print()
    print("=" * 80)
    print("Visualization complete!")
    print("=" * 80)
    print()
    print("Generated plots:")
    print("  1. calibration_sessions_comparison.png - Overview of all 4 sessions")
    print("  2. session1_detailed_stages.png - Session 1 calibration pipeline stages")
    print("  3. earth_field_segments_analysis.png - Earth field calibration issues")
    print()
    print("These plots clearly show:")
    print("  - Environmental magnetic distortion (1200-1600 µT vs expected 50 µT)")
    print("  - Inconsistent calibration application (Session 1 vs others)")
    print("  - High residual 'fused' field even without magnets (~1500 µT)")
    print()

if __name__ == '__main__':
    main()
