#!/usr/bin/env python3
"""
SNR Analysis Tool for Magnetic Finger Tracking

Analyzes signal-to-noise ratio for magnetic finger signals to validate
tracking feasibility before full data collection.

Usage:
    python -m ml.analyze_snr --data-dir data/GAMBIT --finger index --plot
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import numpy as np

from .data_loader import GambitDataset, load_session_data, load_session_metadata
from .schema import Gesture, FingerState


def compute_field_magnitude(mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> np.ndarray:
    """Compute magnetic field magnitude |B| = sqrt(mx² + my² + mz²)"""
    return np.sqrt(mx**2 + my**2 + mz**2)


def analyze_snr_session(
    session_path: Path,
    finger: str = 'index',
    use_calibrated: bool = True,
    use_filtered: bool = True
) -> Optional[Dict]:
    """
    Analyze SNR for a single session marked with custom label 'snr_test'.
    
    Args:
        session_path: Path to .json session file
        finger: Which finger to analyze ('thumb', 'index', 'middle', 'ring', 'pinky')
        use_calibrated: Use calibrated magnetometer data if available
        use_filtered: Use filtered magnetometer data if available
    
    Returns:
        Dict with SNR analysis results, or None if session not suitable
    """
    # Load metadata
    meta = load_session_metadata(session_path)
    if not meta:
        return None
    
    # Check for SNR test label
    has_snr_label = 'snr_test' in meta.custom_label_definitions
    if not has_snr_label and not any('snr' in label.lower() for seg in meta.labels_v2 for label in seg.labels.custom):
        return None
    
    # Load data with calibration/filtering
    data = load_session_data(
        session_path,
        apply_calibration=use_calibrated,
        apply_filtering=use_filtered
    )
    
    # Determine which magnetometer fields to use
    # Priority: filtered > calibrated > raw
    with open(session_path, 'r') as f:
        raw_data = json.load(f)
    
    if use_filtered and 'filtered_mx' in raw_data[0]:
        mx = np.array([s.get('filtered_mx', s['mx']) for s in raw_data])
        my = np.array([s.get('filtered_my', s['my']) for s in raw_data])
        mz = np.array([s.get('filtered_mz', s['mz']) for s in raw_data])
        data_type = 'filtered'
    elif use_calibrated and 'calibrated_mx' in raw_data[0]:
        mx = np.array([s.get('calibrated_mx', s['mx']) for s in raw_data])
        my = np.array([s.get('calibrated_my', s['my']) for s in raw_data])
        mz = np.array([s.get('calibrated_mz', s['mz']) for s in raw_data])
        data_type = 'calibrated'
    else:
        mx = np.array([s['mx'] for s in raw_data])
        my = np.array([s['my'] for s in raw_data])
        mz = np.array([s['mz'] for s in raw_data])
        data_type = 'raw'
    
    # Compute field magnitude
    field_mag = compute_field_magnitude(mx, my, mz)
    
    # Extract segments for target finger
    extended_segments = []
    flexed_segments = []
    
    for seg in meta.labels_v2:
        if seg.labels.motion.value != 'static':
            continue  # Only analyze static poses
        
        if not seg.labels.fingers:
            continue
        
        finger_state = getattr(seg.labels.fingers, finger)
        start = seg.start_sample
        end = seg.end_sample
        
        if finger_state == FingerState.EXTENDED:
            extended_segments.append(field_mag[start:end])
        elif finger_state == FingerState.FLEXED:
            flexed_segments.append(field_mag[start:end])
    
    if not extended_segments and not flexed_segments:
        return None
    
    # Compute statistics
    results = {
        'session': session_path.name,
        'finger': finger,
        'data_type': data_type,
        'num_samples': len(field_mag)
    }
    
    if extended_segments:
        extended_all = np.concatenate(extended_segments)
        results['signal_extended'] = {
            'mean': float(np.mean(extended_all)),
            'std': float(np.std(extended_all)),
            'min': float(np.min(extended_all)),
            'max': float(np.max(extended_all)),
            'num_segments': len(extended_segments),
            'num_samples': len(extended_all)
        }
    
    if flexed_segments:
        flexed_all = np.concatenate(flexed_segments)
        results['signal_flexed'] = {
            'mean': float(np.mean(flexed_all)),
            'std': float(np.std(flexed_all)),
            'min': float(np.min(flexed_all)),
            'max': float(np.max(flexed_all)),
            'num_segments': len(flexed_segments),
            'num_samples': len(flexed_all)
        }
    
    # Compute signal delta and SNR
    if extended_segments and flexed_segments:
        extended_mean = results['signal_extended']['mean']
        flexed_mean = results['signal_flexed']['mean']
        
        results['signal_delta'] = float(abs(flexed_mean - extended_mean))
        
        # Noise floor: use std during static holds
        # Average std from both extended and flexed segments
        noise_extended = results['signal_extended']['std']
        noise_flexed = results['signal_flexed']['std']
        noise_floor = (noise_extended + noise_flexed) / 2
        
        results['noise_floor'] = float(noise_floor)
        
        # SNR = signal / noise
        results['snr_extended'] = float(extended_mean / noise_floor) if noise_floor > 0 else float('inf')
        results['snr_flexed'] = float(flexed_mean / noise_floor) if noise_floor > 0 else float('inf')
        results['snr_delta'] = float(results['signal_delta'] / noise_floor) if noise_floor > 0 else float('inf')
    
    return results


def analyze_dataset(
    data_dir: Path,
    finger: str = 'index',
    use_calibrated: bool = True,
    use_filtered: bool = True
) -> List[Dict]:
    """
    Analyze all SNR test sessions in a dataset.
    
    Returns:
        List of analysis results, one per session
    """
    results = []
    
    for json_path in sorted(data_dir.glob('*.json')):
        if json_path.name.endswith('.meta.json'):
            continue
        
        result = analyze_snr_session(json_path, finger, use_calibrated, use_filtered)
        if result:
            results.append(result)
    
    return results


def print_results(results: List[Dict], verbose: bool = False):
    """Print SNR analysis results in a readable format."""
    if not results:
        print("No SNR test sessions found.")
        print("\nTo collect SNR test data:")
        print("  1. In collector.html, add custom label 'snr_test'")
        print("  2. Label segments with finger states (extended/flexed) and motion='static'")
        print("  3. Hold each pose for 3-5 seconds")
        print("  4. Repeat 10 times")
        return
    
    print("=" * 70)
    print("MAGNETIC FINGER TRACKING - SNR ANALYSIS")
    print("=" * 70)
    print()
    
    # Aggregate statistics
    all_signal_extended = []
    all_signal_flexed = []
    all_signal_delta = []
    all_noise = []
    all_snr_extended = []
    all_snr_flexed = []
    all_snr_delta = []
    
    for r in results:
        if 'signal_extended' in r:
            all_signal_extended.append(r['signal_extended']['mean'])
        if 'signal_flexed' in r:
            all_signal_flexed.append(r['signal_flexed']['mean'])
        if 'signal_delta' in r:
            all_signal_delta.append(r['signal_delta'])
        if 'noise_floor' in r:
            all_noise.append(r['noise_floor'])
        if 'snr_extended' in r:
            all_snr_extended.append(r['snr_extended'])
        if 'snr_flexed' in r:
            all_snr_flexed.append(r['snr_flexed'])
        if 'snr_delta' in r:
            all_snr_delta.append(r['snr_delta'])
    
    # Print summary
    print(f"Sessions analyzed: {len(results)}")
    print(f"Finger: {results[0]['finger']}")
    print(f"Data type: {results[0]['data_type']}")
    print()
    
    if all_signal_extended:
        print(f"Signal (extended):  {np.mean(all_signal_extended):.1f} ± {np.std(all_signal_extended):.1f} μT")
    if all_signal_flexed:
        print(f"Signal (flexed):    {np.mean(all_signal_flexed):.1f} ± {np.std(all_signal_flexed):.1f} μT")
    if all_signal_delta:
        print(f"Signal delta:       {np.mean(all_signal_delta):.1f} ± {np.std(all_signal_delta):.1f} μT")
    if all_noise:
        print(f"Noise floor:        {np.mean(all_noise):.2f} ± {np.std(all_noise):.2f} μT")
    print()
    
    if all_snr_extended:
        snr_ext = np.mean(all_snr_extended)
        status_ext = "✓ GOOD" if snr_ext > 10 else "✗ POOR"
        print(f"SNR (extended):     {snr_ext:.1f}:1  {status_ext}")
    
    if all_snr_flexed:
        snr_flx = np.mean(all_snr_flexed)
        status_flx = "✓ EXCELLENT" if snr_flx > 50 else "✓ GOOD" if snr_flx > 20 else "⚠ MARGINAL"
        print(f"SNR (flexed):       {snr_flx:.1f}:1  {status_flx}")
    
    if all_snr_delta:
        snr_dlt = np.mean(all_snr_delta)
        status_dlt = "✓ EXCELLENT" if snr_dlt > 50 else "✓ GOOD" if snr_dlt > 20 else "⚠ MARGINAL"
        print(f"SNR (delta):        {snr_dlt:.1f}:1  {status_dlt}")
    
    print()
    print("=" * 70)
    
    # Recommendation
    if all_snr_extended and np.mean(all_snr_extended) < 10:
        print("⚠ WARNING: SNR at extended position is low (<10:1)")
        print("  - Use larger magnet (6mm x 3mm instead of 5mm x 2mm)")
        print("  - Improve calibration (run calibration wizard)")
        print("  - Check for environmental magnetic interference")
    elif all_snr_delta and np.mean(all_snr_delta) < 20:
        print("⚠ CAUTION: SNR delta is marginal (20-50:1)")
        print("  - Proceed with single-finger tracking (Phase 1)")
        print("  - Multi-finger tracking may be challenging")
    else:
        print("✓ RECOMMENDATION: SNR is adequate for magnetic finger tracking")
        print("  - Proceed to Phase 1 (single-finger validation)")
        print("  - Then scale to Phase 2 (multi-finger poses)")
    
    print("=" * 70)
    
    # Verbose per-session details
    if verbose:
        print("\nPer-Session Details:")
        print("-" * 70)
        for r in results:
            print(f"\n{r['session']}:")
            if 'signal_extended' in r:
                print(f"  Extended: {r['signal_extended']['mean']:.1f} μT "
                      f"({r['signal_extended']['num_segments']} segments, "
                      f"{r['signal_extended']['num_samples']} samples)")
            if 'signal_flexed' in r:
                print(f"  Flexed:   {r['signal_flexed']['mean']:.1f} μT "
                      f"({r['signal_flexed']['num_segments']} segments, "
                      f"{r['signal_flexed']['num_samples']} samples)")
            if 'snr_delta' in r:
                print(f"  SNR:      {r['snr_delta']:.1f}:1")


def plot_results(results: List[Dict], output_path: Optional[Path] = None):
    """Generate plots of SNR analysis."""
    try:
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed. Run: pip install matplotlib")
        return
    
    if not results:
        print("No results to plot")
        return
    
    # Collect data for plotting
    sessions = [r['session'] for r in results]
    signal_ext = [r.get('signal_extended', {}).get('mean', 0) for r in results]
    signal_flx = [r.get('signal_flexed', {}).get('mean', 0) for r in results]
    noise = [r.get('noise_floor', 0) for r in results]
    snr_ext = [r.get('snr_extended', 0) for r in results]
    snr_flx = [r.get('snr_flexed', 0) for r in results]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'Magnetic Finger Tracking - SNR Analysis ({results[0]["finger"]} finger)', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Signal magnitude
    ax1 = axes[0, 0]
    x = np.arange(len(sessions))
    width = 0.35
    ax1.bar(x - width/2, signal_ext, width, label='Extended', color='#4CAF50')
    ax1.bar(x + width/2, signal_flx, width, label='Flexed', color='#2196F3')
    ax1.axhline(y=np.mean(noise) if noise else 0, color='r', linestyle='--', 
                label=f'Noise floor ({np.mean(noise):.1f} μT)', linewidth=2)
    ax1.set_ylabel('Field Magnitude (μT)')
    ax1.set_title('Magnetic Field Strength')
    ax1.set_xticks(x)
    ax1.set_xticklabels([s[:10] for s in sessions], rotation=45, ha='right', fontsize=8)
    ax1.legend()
    ax1.grid(axis='y', alpha=0.3)
    
    # Plot 2: SNR
    ax2 = axes[0, 1]
    ax2.bar(x - width/2, snr_ext, width, label='Extended', color='#4CAF50')
    ax2.bar(x + width/2, snr_flx, width, label='Flexed', color='#2196F3')
    ax2.axhline(y=10, color='orange', linestyle='--', label='Minimum (10:1)', linewidth=2)
    ax2.axhline(y=20, color='green', linestyle='--', label='Good (20:1)', linewidth=2)
    ax2.set_ylabel('SNR (signal:noise)')
    ax2.set_title('Signal-to-Noise Ratio')
    ax2.set_xticks(x)
    ax2.set_xticklabels([s[:10] for s in sessions], rotation=45, ha='right', fontsize=8)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)
    
    # Plot 3: Signal distribution
    ax3 = axes[1, 0]
    all_ext = [r.get('signal_extended', {}).get('mean', 0) for r in results if 'signal_extended' in r]
    all_flx = [r.get('signal_flexed', {}).get('mean', 0) for r in results if 'signal_flexed' in r]
    if all_ext and all_flx:
        ax3.hist([all_ext, all_flx], bins=15, label=['Extended', 'Flexed'], 
                 color=['#4CAF50', '#2196F3'], alpha=0.7)
        ax3.set_xlabel('Field Magnitude (μT)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Signal Distribution')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)
    
    # Plot 4: Summary metrics
    ax4 = axes[1, 1]
    metrics = []
    values = []
    colors = []
    
    if signal_ext:
        metrics.append('Signal\n(extended)')
        values.append(np.mean(signal_ext))
        colors.append('#4CAF50')
    
    if signal_flx:
        metrics.append('Signal\n(flexed)')
        values.append(np.mean(signal_flx))
        colors.append('#2196F3')
    
    if noise:
        metrics.append('Noise\nFloor')
        values.append(np.mean(noise))
        colors.append('#FF5722')
    
    ax4.bar(metrics, values, color=colors, alpha=0.7)
    ax4.set_ylabel('Magnitude (μT)')
    ax4.set_title('Average Metrics')
    ax4.grid(axis='y', alpha=0.3)
    
    # Add SNR annotations
    for i, (metric, value) in enumerate(zip(metrics, values)):
        if 'Noise' not in metric and noise:
            snr = value / np.mean(noise)
            ax4.text(i, value + 2, f'SNR: {snr:.1f}:1', 
                    ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        default_path = Path('snr_analysis.png')
        plt.savefig(default_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {default_path}")
    
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Analyze SNR for magnetic finger tracking'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/GAMBIT',
        help='Path to data directory'
    )
    parser.add_argument(
        '--finger', type=str, default='index',
        choices=['thumb', 'index', 'middle', 'ring', 'pinky'],
        help='Which finger to analyze'
    )
    parser.add_argument(
        '--no-calibration', action='store_true',
        help='Use raw magnetometer data (no calibration)'
    )
    parser.add_argument(
        '--no-filtering', action='store_true',
        help='Do not apply Kalman filtering'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate plots'
    )
    parser.add_argument(
        '--output', type=str,
        help='Output path for plot (default: snr_analysis.png)'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show per-session details'
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Output results as JSON'
    )
    
    args = parser.parse_args()
    
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)
    
    # Run analysis
    results = analyze_dataset(
        data_dir,
        finger=args.finger,
        use_calibrated=not args.no_calibration,
        use_filtered=not args.no_filtering
    )
    
    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_results(results, verbose=args.verbose)
    
    if args.plot:
        output_path = Path(args.output) if args.output else None
        plot_results(results, output_path)


if __name__ == '__main__':
    main()
