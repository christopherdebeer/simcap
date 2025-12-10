#!/usr/bin/env python3
"""
Comprehensive SNR Analysis for All GAMBIT Sessions

Analyzes signal-to-noise ratio across all sessions, comparing
raw vs calibrated/filtered data where available.

Generates:
- Console report with summary statistics
- PNG visualization saved to visualizations/
- JSON results file
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec


def compute_snr_metrics(mx, my, mz, name="signal"):
    """Compute comprehensive SNR metrics for magnetometer data."""
    mag = np.sqrt(mx**2 + my**2 + mz**2)

    mean_mag = np.mean(mag)
    std_mag = np.std(mag)

    # SNR = mean / std
    snr = mean_mag / std_mag if std_mag > 0 else float('inf')
    snr_db = 20 * np.log10(snr) if snr > 0 and snr != float('inf') else 0

    # Per-axis SNR
    snr_x = np.mean(np.abs(mx)) / np.std(mx) if np.std(mx) > 0 else 0
    snr_y = np.mean(np.abs(my)) / np.std(my) if np.std(my) > 0 else 0
    snr_z = np.mean(np.abs(mz)) / np.std(mz) if np.std(mz) > 0 else 0

    # Drift = cumulative deviation
    drift = np.max(np.abs(np.cumsum(mag - mean_mag))) / len(mag) if len(mag) > 0 else 0

    # Noise floor estimate (std is proxy)
    noise_floor = std_mag

    return {
        'name': name,
        'mean_mag': float(mean_mag),
        'std_mag': float(std_mag),
        'snr': float(snr),
        'snr_db': float(snr_db),
        'snr_x': float(snr_x),
        'snr_y': float(snr_y),
        'snr_z': float(snr_z),
        'noise_floor': float(noise_floor),
        'drift': float(drift),
        'min_mag': float(np.min(mag)),
        'max_mag': float(np.max(mag)),
        'range': float(np.max(mag) - np.min(mag)),
    }


def analyze_session(json_path):
    """Analyze a single session file."""
    with open(json_path, 'r') as f:
        data = json.load(f)

    if not data:
        return None

    n_samples = len(data)
    duration = n_samples / 50.0  # 50Hz

    # Extract raw data
    mx = np.array([s.get('mx', 0) for s in data])
    my = np.array([s.get('my', 0) for s in data])
    mz = np.array([s.get('mz', 0) for s in data])

    result = {
        'filename': json_path.name,
        'timestamp': json_path.stem,
        'n_samples': n_samples,
        'duration': duration,
        'stages': {}
    }

    # Raw metrics
    result['stages']['raw'] = compute_snr_metrics(mx, my, mz, 'Raw')

    # Calibrated (iron corrected)
    if 'calibrated_mx' in data[0]:
        cal_mx = np.array([s.get('calibrated_mx', 0) for s in data])
        cal_my = np.array([s.get('calibrated_my', 0) for s in data])
        cal_mz = np.array([s.get('calibrated_mz', 0) for s in data])
        result['stages']['calibrated'] = compute_snr_metrics(cal_mx, cal_my, cal_mz, 'Iron Corrected')

    # Fused (Earth field subtracted)
    if 'fused_mx' in data[0]:
        fused_mx = np.array([s.get('fused_mx', 0) for s in data])
        fused_my = np.array([s.get('fused_my', 0) for s in data])
        fused_mz = np.array([s.get('fused_mz', 0) for s in data])
        result['stages']['fused'] = compute_snr_metrics(fused_mx, fused_my, fused_mz, 'Fused')

    # Filtered (Kalman smoothed)
    if 'filtered_mx' in data[0]:
        filt_mx = np.array([s.get('filtered_mx', 0) for s in data])
        filt_my = np.array([s.get('filtered_my', 0) for s in data])
        filt_mz = np.array([s.get('filtered_mz', 0) for s in data])
        result['stages']['filtered'] = compute_snr_metrics(filt_mx, filt_my, filt_mz, 'Filtered')

    return result


def generate_report(results, output_dir):
    """Generate comprehensive SNR analysis report and visualizations."""

    # Aggregate statistics
    all_raw_snr = [r['stages']['raw']['snr_db'] for r in results]
    all_raw_noise = [r['stages']['raw']['noise_floor'] for r in results]

    sessions_with_filtered = [r for r in results if 'filtered' in r['stages']]
    sessions_with_calibrated = [r for r in results if 'calibrated' in r['stages']]
    sessions_with_fused = [r for r in results if 'fused' in r['stages']]

    print("=" * 70)
    print("MAGNETOMETER SNR ANALYSIS REPORT")
    print("=" * 70)
    print(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Total Sessions: {len(results)}")
    print(f"Total Duration: {sum(r['duration'] for r in results):.1f}s")
    print(f"Total Samples: {sum(r['n_samples'] for r in results)}")
    print()

    print("CALIBRATION COVERAGE:")
    print(f"  Sessions with raw data:       {len(results)}")
    print(f"  Sessions with iron correction: {len(sessions_with_calibrated)}")
    print(f"  Sessions with fusion:          {len(sessions_with_fused)}")
    print(f"  Sessions with filtering:       {len(sessions_with_filtered)}")
    print()

    print("=" * 70)
    print("RAW MAGNETOMETER STATISTICS (All Sessions)")
    print("=" * 70)
    print(f"  SNR (dB):      {np.mean(all_raw_snr):.1f} ± {np.std(all_raw_snr):.1f}")
    print(f"  Noise Floor:   {np.mean(all_raw_noise):.2f} ± {np.std(all_raw_noise):.2f} μT")
    print()

    # Quality assessment
    def assess_snr(snr_db):
        if snr_db >= 20:
            return "EXCELLENT", "✓"
        elif snr_db >= 10:
            return "GOOD", "✓"
        elif snr_db >= 6:
            return "MARGINAL", "⚠"
        else:
            return "POOR", "✗"

    avg_snr = np.mean(all_raw_snr)
    quality, symbol = assess_snr(avg_snr)
    print(f"  Overall Quality: {symbol} {quality}")
    print()

    # Filtered sessions analysis
    if sessions_with_filtered:
        print("=" * 70)
        print("FILTERED SESSIONS ANALYSIS")
        print("=" * 70)

        for r in sessions_with_filtered:
            raw = r['stages']['raw']
            filt = r['stages']['filtered']

            noise_reduction = ((raw['noise_floor'] - filt['noise_floor']) / raw['noise_floor'] * 100) if raw['noise_floor'] > 0 else 0
            snr_improvement = filt['snr_db'] - raw['snr_db']

            print(f"\n  Session: {r['filename']}")
            print(f"  Duration: {r['duration']:.1f}s ({r['n_samples']} samples)")
            print()
            print(f"  {'Metric':<20} {'Raw':>12} {'Filtered':>12} {'Change':>12}")
            print(f"  {'-'*56}")
            print(f"  {'SNR (dB)':<20} {raw['snr_db']:>12.1f} {filt['snr_db']:>12.1f} {snr_improvement:>+12.1f}")
            print(f"  {'Noise Floor (μT)':<20} {raw['noise_floor']:>12.2f} {filt['noise_floor']:>12.2f} {-noise_reduction:>+12.1f}%")
            print(f"  {'Signal Range (μT)':<20} {raw['range']:>12.1f} {filt['range']:>12.1f}")
            print(f"  {'Drift (μT/sample)':<20} {raw['drift']:>12.3f} {filt['drift']:>12.3f}")

    print()
    print("=" * 70)
    print("PER-SESSION SNR BREAKDOWN")
    print("=" * 70)
    print(f"{'Session':<45} {'Samples':>8} {'SNR(dB)':>10} {'Noise':>10} {'Quality':>12}")
    print("-" * 85)

    for r in sorted(results, key=lambda x: x['stages']['raw']['snr_db'], reverse=True):
        raw = r['stages']['raw']
        quality, symbol = assess_snr(raw['snr_db'])
        short_name = r['filename'][:40] + '...' if len(r['filename']) > 43 else r['filename']
        print(f"{short_name:<45} {r['n_samples']:>8} {raw['snr_db']:>10.1f} {raw['noise_floor']:>10.2f} {symbol:>2} {quality:<10}")

    print("=" * 70)

    # Generate visualization
    fig = plt.figure(figsize=(16, 12))
    gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

    fig.suptitle('Magnetometer SNR Analysis Report\n'
                 f'{len(results)} sessions | {sum(r["n_samples"] for r in results)} total samples',
                 fontsize=14, fontweight='bold')

    # 1. SNR Distribution
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(all_raw_snr, bins=15, color='#3498db', alpha=0.7, edgecolor='black')
    ax1.axvline(x=10, color='orange', linestyle='--', linewidth=2, label='Good threshold (10 dB)')
    ax1.axvline(x=20, color='green', linestyle='--', linewidth=2, label='Excellent threshold (20 dB)')
    ax1.axvline(x=np.mean(all_raw_snr), color='red', linestyle='-', linewidth=2, label=f'Mean ({np.mean(all_raw_snr):.1f} dB)')
    ax1.set_xlabel('SNR (dB)')
    ax1.set_ylabel('Number of Sessions')
    ax1.set_title('SNR Distribution (Raw Data)')
    ax1.legend(fontsize=8)
    ax1.grid(True, alpha=0.3)

    # 2. Noise Floor Distribution
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.hist(all_raw_noise, bins=15, color='#e74c3c', alpha=0.7, edgecolor='black')
    ax2.axvline(x=1.0, color='green', linestyle='--', linewidth=2, label='Excellent (<1 μT)')
    ax2.axvline(x=3.0, color='orange', linestyle='--', linewidth=2, label='Good (<3 μT)')
    ax2.axvline(x=np.mean(all_raw_noise), color='blue', linestyle='-', linewidth=2, label=f'Mean ({np.mean(all_raw_noise):.2f} μT)')
    ax2.set_xlabel('Noise Floor (μT)')
    ax2.set_ylabel('Number of Sessions')
    ax2.set_title('Noise Floor Distribution (Raw Data)')
    ax2.legend(fontsize=8)
    ax2.grid(True, alpha=0.3)

    # 3. SNR vs Duration
    ax3 = fig.add_subplot(gs[1, 0])
    durations = [r['duration'] for r in results]
    ax3.scatter(durations, all_raw_snr, c='#3498db', alpha=0.6, s=50)
    ax3.set_xlabel('Session Duration (s)')
    ax3.set_ylabel('SNR (dB)')
    ax3.set_title('SNR vs Session Duration')
    ax3.axhline(y=10, color='orange', linestyle='--', alpha=0.5)
    ax3.axhline(y=20, color='green', linestyle='--', alpha=0.5)
    ax3.grid(True, alpha=0.3)

    # 4. Per-axis SNR comparison
    ax4 = fig.add_subplot(gs[1, 1])
    snr_x = [r['stages']['raw']['snr_x'] for r in results]
    snr_y = [r['stages']['raw']['snr_y'] for r in results]
    snr_z = [r['stages']['raw']['snr_z'] for r in results]

    x_pos = np.arange(3)
    means = [np.mean(snr_x), np.mean(snr_y), np.mean(snr_z)]
    stds = [np.std(snr_x), np.std(snr_y), np.std(snr_z)]

    bars = ax4.bar(x_pos, means, yerr=stds, capsize=5, color=['#e74c3c', '#2ecc71', '#3498db'], alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(['X-axis', 'Y-axis', 'Z-axis'])
    ax4.set_ylabel('SNR (linear)')
    ax4.set_title('Per-Axis SNR Comparison')
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels on bars
    for bar, mean in zip(bars, means):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                f'{mean:.1f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

    # 5. Raw vs Filtered comparison (if available)
    ax5 = fig.add_subplot(gs[2, 0])
    if sessions_with_filtered:
        raw_snr_filt_sessions = [r['stages']['raw']['snr_db'] for r in sessions_with_filtered]
        filt_snr = [r['stages']['filtered']['snr_db'] for r in sessions_with_filtered]

        x = np.arange(len(sessions_with_filtered))
        width = 0.35
        ax5.bar(x - width/2, raw_snr_filt_sessions, width, label='Raw', color='#95a5a6', alpha=0.7)
        ax5.bar(x + width/2, filt_snr, width, label='Filtered', color='#e74c3c', alpha=0.7)
        ax5.set_xlabel('Session')
        ax5.set_ylabel('SNR (dB)')
        ax5.set_title('Raw vs Filtered SNR')
        ax5.set_xticks(x)
        ax5.set_xticklabels([f'S{i+1}' for i in range(len(sessions_with_filtered))])
        ax5.legend()
        ax5.grid(True, alpha=0.3, axis='y')
    else:
        ax5.text(0.5, 0.5, 'No filtered sessions available',
                ha='center', va='center', transform=ax5.transAxes, fontsize=12)
        ax5.set_title('Raw vs Filtered SNR')

    # 6. Summary stats box
    ax6 = fig.add_subplot(gs[2, 1])
    ax6.axis('off')

    avg_quality, avg_symbol = assess_snr(np.mean(all_raw_snr))

    summary_text = f"""
SUMMARY STATISTICS
{'='*40}

Total Sessions Analyzed: {len(results)}
Total Recording Time:    {sum(r['duration'] for r in results):.1f} seconds
Total Samples:           {sum(r['n_samples'] for r in results):,}

RAW MAGNETOMETER:
  Mean SNR:        {np.mean(all_raw_snr):.1f} dB
  SNR Range:       {min(all_raw_snr):.1f} - {max(all_raw_snr):.1f} dB
  Mean Noise:      {np.mean(all_raw_noise):.2f} μT
  Noise Range:     {min(all_raw_noise):.2f} - {max(all_raw_noise):.2f} μT

QUALITY ASSESSMENT: {avg_symbol} {avg_quality}

CALIBRATION STATUS:
  Iron Corrected:  {len(sessions_with_calibrated)} sessions
  Fused:           {len(sessions_with_fused)} sessions
  Filtered:        {len(sessions_with_filtered)} sessions

{'='*40}
"""

    ax6.text(0.05, 0.95, summary_text.strip(), transform=ax6.transAxes,
            fontsize=10, verticalalignment='top', fontfamily='monospace',
            bbox=dict(boxstyle='round,pad=0.5', facecolor='#f8f9fa', alpha=0.9, edgecolor='#dee2e6'))

    # Save visualization
    output_file = output_dir / 'snr_analysis_report.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"\nVisualization saved to: {output_file}")

    # Save JSON results
    json_file = output_dir / 'snr_analysis_results.json'
    with open(json_file, 'w') as f:
        json.dump({
            'analysis_date': datetime.now().isoformat(),
            'summary': {
                'total_sessions': len(results),
                'total_samples': sum(r['n_samples'] for r in results),
                'total_duration': sum(r['duration'] for r in results),
                'mean_snr_db': np.mean(all_raw_snr),
                'mean_noise_floor': np.mean(all_raw_noise),
                'quality': avg_quality,
                'sessions_with_calibrated': len(sessions_with_calibrated),
                'sessions_with_fused': len(sessions_with_fused),
                'sessions_with_filtered': len(sessions_with_filtered),
            },
            'sessions': results
        }, f, indent=2)
    print(f"JSON results saved to: {json_file}")

    return results


def main():
    data_dir = Path('data/GAMBIT')
    output_dir = Path('visualizations')
    output_dir.mkdir(exist_ok=True)

    print("Analyzing all GAMBIT sessions...")
    print()

    results = []
    for json_path in sorted(data_dir.glob('*.json')):
        if json_path.name.endswith('.meta.json'):
            continue

        result = analyze_session(json_path)
        if result:
            results.append(result)

    if not results:
        print("No sessions found!")
        return

    generate_report(results, output_dir)


if __name__ == '__main__':
    main()
