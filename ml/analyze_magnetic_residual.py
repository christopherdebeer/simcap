#!/usr/bin/env python3
"""
Magnetic Residual Analysis Tool

Analyzes magnetic residual signal from sessions recorded WITHOUT finger magnets.
The expected residual after orientation correction and earth field subtraction
should be near zero (< 5 µT).

This script helps validate:
1. Calibration quality
2. Orientation compensation effectiveness
3. Baseline noise floor before adding finger magnets

Usage:
    python -m ml.analyze_magnetic_residual --data-dir data/GAMBIT --plot
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np

# For plotting
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_session(json_path: Path) -> Tuple[Optional[Dict], List[Dict]]:
    """
    Load a session file and return metadata and samples.

    Returns:
        Tuple of (metadata_dict, samples_list)
    """
    with open(json_path, 'r') as f:
        data = json.load(f)

    # Handle both V1 (array) and V2+ (object with samples) formats
    if isinstance(data, list):
        return {}, data
    elif isinstance(data, dict) and 'samples' in data:
        return {
            'version': data.get('version', '2.0'),
            'timestamp': data.get('timestamp'),
            'labels': data.get('labels', []),
            'metadata': data.get('metadata', {})
        }, data['samples']
    else:
        return {}, []


def analyze_session(json_path: Path) -> Optional[Dict]:
    """
    Analyze magnetic residual for a single session.

    Returns dict with statistics for each magnetometer field type.
    """
    meta, samples = load_session(json_path)

    if not samples:
        return None

    # Check what fields are available
    sample0 = samples[0]
    has_raw = 'mx' in sample0
    has_ut = 'mx_ut' in sample0
    has_calibrated = 'calibrated_mx' in sample0
    has_fused = 'fused_mx' in sample0
    has_residual = 'residual_magnitude' in sample0
    has_ahrs_residual = 'ahrs_mag_residual_magnitude' in sample0
    has_filtered = 'filtered_mx' in sample0

    result = {
        'filename': json_path.name,
        'num_samples': len(samples),
        'version': meta.get('version', 'unknown'),
        'fields_available': {
            'raw': has_raw,
            'ut': has_ut,
            'calibrated': has_calibrated,
            'fused': has_fused,
            'residual_magnitude': has_residual,
            'ahrs_residual': has_ahrs_residual,
            'filtered': has_filtered
        }
    }

    # Extract arrays for each field type
    if has_raw:
        mx_raw = np.array([s.get('mx', 0) for s in samples])
        my_raw = np.array([s.get('my', 0) for s in samples])
        mz_raw = np.array([s.get('mz', 0) for s in samples])
        mag_raw = np.sqrt(mx_raw**2 + my_raw**2 + mz_raw**2)
        result['raw_lsb'] = {
            'mean': float(np.mean(mag_raw)),
            'std': float(np.std(mag_raw)),
            'min': float(np.min(mag_raw)),
            'max': float(np.max(mag_raw))
        }

    if has_ut:
        mx_ut = np.array([s.get('mx_ut', 0) for s in samples])
        my_ut = np.array([s.get('my_ut', 0) for s in samples])
        mz_ut = np.array([s.get('mz_ut', 0) for s in samples])
        mag_ut = np.sqrt(mx_ut**2 + my_ut**2 + mz_ut**2)
        result['converted_ut'] = {
            'mean': float(np.mean(mag_ut)),
            'std': float(np.std(mag_ut)),
            'min': float(np.min(mag_ut)),
            'max': float(np.max(mag_ut)),
            'mx_mean': float(np.mean(mx_ut)),
            'my_mean': float(np.mean(my_ut)),
            'mz_mean': float(np.mean(mz_ut))
        }

    if has_calibrated:
        mx_cal = np.array([s.get('calibrated_mx', 0) for s in samples])
        my_cal = np.array([s.get('calibrated_my', 0) for s in samples])
        mz_cal = np.array([s.get('calibrated_mz', 0) for s in samples])
        mag_cal = np.sqrt(mx_cal**2 + my_cal**2 + mz_cal**2)
        result['calibrated'] = {
            'mean': float(np.mean(mag_cal)),
            'std': float(np.std(mag_cal)),
            'min': float(np.min(mag_cal)),
            'max': float(np.max(mag_cal)),
            'mx_mean': float(np.mean(mx_cal)),
            'my_mean': float(np.mean(my_cal)),
            'mz_mean': float(np.mean(mz_cal))
        }

    if has_fused:
        mx_fus = np.array([s.get('fused_mx', 0) for s in samples])
        my_fus = np.array([s.get('fused_my', 0) for s in samples])
        mz_fus = np.array([s.get('fused_mz', 0) for s in samples])
        mag_fus = np.sqrt(mx_fus**2 + my_fus**2 + mz_fus**2)
        result['fused'] = {
            'mean': float(np.mean(mag_fus)),
            'std': float(np.std(mag_fus)),
            'min': float(np.min(mag_fus)),
            'max': float(np.max(mag_fus)),
            'mx_mean': float(np.mean(mx_fus)),
            'my_mean': float(np.mean(my_fus)),
            'mz_mean': float(np.mean(mz_fus))
        }

    if has_residual:
        residual_mag = np.array([s.get('residual_magnitude', 0) for s in samples])
        result['residual_magnitude'] = {
            'mean': float(np.mean(residual_mag)),
            'std': float(np.std(residual_mag)),
            'min': float(np.min(residual_mag)),
            'max': float(np.max(residual_mag)),
            'median': float(np.median(residual_mag))
        }

    if has_ahrs_residual:
        ahrs_res = np.array([s.get('ahrs_mag_residual_magnitude', 0) for s in samples])
        result['ahrs_residual'] = {
            'mean': float(np.mean(ahrs_res)),
            'std': float(np.std(ahrs_res)),
            'min': float(np.min(ahrs_res)),
            'max': float(np.max(ahrs_res)),
            'median': float(np.median(ahrs_res))
        }

    if has_filtered:
        mx_filt = np.array([s.get('filtered_mx', 0) for s in samples])
        my_filt = np.array([s.get('filtered_my', 0) for s in samples])
        mz_filt = np.array([s.get('filtered_mz', 0) for s in samples])
        mag_filt = np.sqrt(mx_filt**2 + my_filt**2 + mz_filt**2)
        result['filtered'] = {
            'mean': float(np.mean(mag_filt)),
            'std': float(np.std(mag_filt)),
            'min': float(np.min(mag_filt)),
            'max': float(np.max(mag_filt))
        }

    return result


def analyze_dataset(data_dir: Path) -> List[Dict]:
    """
    Analyze all sessions in a data directory.
    """
    results = []

    for json_path in sorted(data_dir.glob('*.json')):
        # Skip non-session files
        if (json_path.name.endswith('.meta.json') or
            'calibration' in json_path.name.lower() or
            json_path.name == 'manifest.json'):
            continue

        result = analyze_session(json_path)
        if result:
            results.append(result)

    return results


def print_summary(results: List[Dict]):
    """Print summary statistics across all sessions."""
    if not results:
        print("No sessions found to analyze.")
        return

    print("=" * 80)
    print("MAGNETIC RESIDUAL ANALYSIS - BASELINE (NO FINGER MAGNETS)")
    print("=" * 80)
    print()

    print(f"Sessions analyzed: {len(results)}")
    total_samples = sum(r['num_samples'] for r in results)
    print(f"Total samples: {total_samples:,}")
    print()

    # Check what fields are available across all sessions
    fields_available = {}
    for r in results:
        for field, available in r['fields_available'].items():
            fields_available[field] = fields_available.get(field, 0) + (1 if available else 0)

    print("Fields available across sessions:")
    for field, count in fields_available.items():
        print(f"  {field}: {count}/{len(results)} sessions")
    print()

    # Aggregate statistics for each field type
    field_types = ['converted_ut', 'calibrated', 'fused', 'residual_magnitude', 'ahrs_residual']

    for field_type in field_types:
        values = [r.get(field_type, {}).get('mean', None) for r in results]
        values = [v for v in values if v is not None]

        if not values:
            continue

        print(f"\n{field_type.upper()}:")
        print("-" * 40)

        if field_type in ['residual_magnitude', 'ahrs_residual']:
            # These are the key metrics we care about
            print(f"  Mean across sessions: {np.mean(values):.2f} µT")
            print(f"  Std across sessions:  {np.std(values):.2f} µT")
            print(f"  Min session mean:     {np.min(values):.2f} µT")
            print(f"  Max session mean:     {np.max(values):.2f} µT")

            # Expected value
            expected = 5.0  # Expected < 5 µT for no finger magnets
            actual_mean = np.mean(values)

            if actual_mean < expected:
                status = "GOOD"
                symbol = "✓"
            elif actual_mean < expected * 10:
                status = "MARGINAL"
                symbol = "⚠"
            else:
                status = "ISSUE"
                symbol = "✗"

            print(f"\n  Status: {symbol} {status}")
            print(f"  Expected (no magnets): < {expected} µT")
            print(f"  Actual mean: {actual_mean:.2f} µT")

            if actual_mean > expected * 10:
                print(f"\n  ⚠ WARNING: Residual is {actual_mean/expected:.0f}x higher than expected!")
                print("    Possible causes:")
                print("    1. Units mismatch in calibration pipeline")
                print("    2. Earth field calibration incorrect")
                print("    3. Orientation estimation error")
                print("    4. Hard/soft iron calibration incomplete")
        else:
            print(f"  Mean across sessions: {np.mean(values):.2f}")
            print(f"  Std across sessions:  {np.std(values):.2f}")

    print()
    print("=" * 80)


def print_per_session_details(results: List[Dict]):
    """Print detailed stats per session."""
    print("\nPER-SESSION DETAILS:")
    print("-" * 80)

    for r in results:
        print(f"\n{r['filename']}:")
        print(f"  Samples: {r['num_samples']}")

        if 'converted_ut' in r:
            ut = r['converted_ut']
            print(f"  Raw (µT): mean={ut['mean']:.1f}, std={ut['std']:.1f}")

        if 'calibrated' in r:
            cal = r['calibrated']
            print(f"  Calibrated: mean={cal['mean']:.1f}, std={cal['std']:.1f}")

        if 'fused' in r:
            fus = r['fused']
            print(f"  Fused: mean={fus['mean']:.1f}, std={fus['std']:.1f}")

        if 'residual_magnitude' in r:
            res = r['residual_magnitude']
            print(f"  Residual: mean={res['mean']:.1f}, std={res['std']:.1f}, median={res['median']:.1f}")


def plot_results(results: List[Dict], output_path: Optional[Path] = None):
    """Generate plots of residual analysis."""
    if not HAS_MATPLOTLIB:
        print("matplotlib not installed. Run: pip install matplotlib")
        return

    if not results:
        print("No results to plot")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Magnetic Residual Analysis - Baseline (No Finger Magnets)', fontsize=14, fontweight='bold')

    # Collect data
    sessions = [r['filename'][:15] for r in results]

    # Get residual values
    residual_means = [r.get('residual_magnitude', {}).get('mean', None) for r in results]
    residual_stds = [r.get('residual_magnitude', {}).get('std', None) for r in results]
    ahrs_residual_means = [r.get('ahrs_residual', {}).get('mean', None) for r in results]

    # Get converted/calibrated/fused values
    converted_means = [r.get('converted_ut', {}).get('mean', None) for r in results]
    calibrated_means = [r.get('calibrated', {}).get('mean', None) for r in results]
    fused_means = [r.get('fused', {}).get('mean', None) for r in results]

    x = np.arange(len(sessions))

    # Plot 1: Residual magnitude per session
    ax1 = axes[0, 0]
    if any(v is not None for v in residual_means):
        residual_means_clean = [v if v is not None else 0 for v in residual_means]
        residual_stds_clean = [v if v is not None else 0 for v in residual_stds]
        bars = ax1.bar(x, residual_means_clean, yerr=residual_stds_clean,
                       capsize=3, color='#e74c3c', alpha=0.7)
        ax1.axhline(y=5, color='green', linestyle='--', linewidth=2, label='Expected (<5 µT)')
        ax1.set_ylabel('Residual Magnitude (µT)')
        ax1.set_title('Residual Magnitude by Session')
        ax1.set_xticks(x)
        ax1.set_xticklabels(sessions, rotation=45, ha='right', fontsize=7)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

    # Plot 2: Field magnitudes comparison
    ax2 = axes[0, 1]
    width = 0.25
    if any(v is not None for v in converted_means):
        ax2.bar(x - width, [v if v else 0 for v in converted_means], width,
                label='Raw (µT)', color='#3498db', alpha=0.7)
    if any(v is not None for v in calibrated_means):
        ax2.bar(x, [v if v else 0 for v in calibrated_means], width,
                label='Calibrated', color='#2ecc71', alpha=0.7)
    if any(v is not None for v in fused_means):
        ax2.bar(x + width, [v if v else 0 for v in fused_means], width,
                label='Fused', color='#9b59b6', alpha=0.7)
    ax2.set_ylabel('Magnitude')
    ax2.set_title('Field Magnitude at Each Stage')
    ax2.set_xticks(x)
    ax2.set_xticklabels(sessions, rotation=45, ha='right', fontsize=7)
    ax2.legend()
    ax2.grid(axis='y', alpha=0.3)

    # Plot 3: Residual distribution histogram
    ax3 = axes[1, 0]
    all_residuals = [v for v in residual_means if v is not None]
    if all_residuals:
        ax3.hist(all_residuals, bins=15, color='#e74c3c', alpha=0.7, edgecolor='black')
        ax3.axvline(x=5, color='green', linestyle='--', linewidth=2, label='Expected (<5 µT)')
        ax3.axvline(x=np.mean(all_residuals), color='blue', linestyle='-', linewidth=2,
                    label=f'Mean ({np.mean(all_residuals):.1f} µT)')
        ax3.set_xlabel('Residual Magnitude (µT)')
        ax3.set_ylabel('Frequency')
        ax3.set_title('Distribution of Session Mean Residuals')
        ax3.legend()
        ax3.grid(axis='y', alpha=0.3)

    # Plot 4: Summary comparison
    ax4 = axes[1, 1]
    summary_labels = []
    summary_values = []
    summary_colors = []

    if converted_means and any(v is not None for v in converted_means):
        summary_labels.append('Raw\n(µT)')
        summary_values.append(np.mean([v for v in converted_means if v is not None]))
        summary_colors.append('#3498db')

    if calibrated_means and any(v is not None for v in calibrated_means):
        summary_labels.append('Calibrated')
        summary_values.append(np.mean([v for v in calibrated_means if v is not None]))
        summary_colors.append('#2ecc71')

    if fused_means and any(v is not None for v in fused_means):
        summary_labels.append('Fused')
        summary_values.append(np.mean([v for v in fused_means if v is not None]))
        summary_colors.append('#9b59b6')

    if residual_means and any(v is not None for v in residual_means):
        summary_labels.append('Residual')
        summary_values.append(np.mean([v for v in residual_means if v is not None]))
        summary_colors.append('#e74c3c')

    ax4.bar(summary_labels, summary_values, color=summary_colors, alpha=0.7)
    ax4.axhline(y=5, color='green', linestyle='--', linewidth=2, label='Expected residual (<5 µT)')
    ax4.set_ylabel('Average Magnitude')
    ax4.set_title('Average Values Across All Sessions')
    ax4.legend()
    ax4.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {output_path}")
    else:
        default_path = Path('magnetic_residual_analysis.png')
        plt.savefig(default_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to: {default_path}")

    plt.close()


def check_calibration_file(data_dir: Path):
    """Check and report on calibration file contents."""
    cal_path = data_dir / 'gambit_calibration.json'

    if not cal_path.exists():
        print("\n⚠ No calibration file found at:", cal_path)
        return

    print("\n" + "=" * 80)
    print("CALIBRATION FILE ANALYSIS")
    print("=" * 80)

    with open(cal_path, 'r') as f:
        cal = json.load(f)

    print(f"\nCalibration file: {cal_path}")
    print(f"Timestamp: {cal.get('timestamp', 'unknown')}")
    print()

    # Hard iron offset
    hi = cal.get('hardIronOffset', {})
    print("Hard Iron Offset:")
    print(f"  x: {hi.get('x', 0):.3f}")
    print(f"  y: {hi.get('y', 0):.3f}")
    print(f"  z: {hi.get('z', 0):.3f}")
    hi_mag = np.sqrt(hi.get('x', 0)**2 + hi.get('y', 0)**2 + hi.get('z', 0)**2)
    print(f"  magnitude: {hi_mag:.3f}")

    # Earth field
    ef = cal.get('earthField', {})
    print("\nEarth Field:")
    print(f"  x: {ef.get('x', 0):.3f}")
    print(f"  y: {ef.get('y', 0):.3f}")
    print(f"  z: {ef.get('z', 0):.3f}")

    ef_mag = cal.get('earthFieldMagnitude', 0)
    print(f"  magnitude: {ef_mag:.3f}")

    # Check if Earth field magnitude is reasonable (25-65 µT typical)
    units = cal.get('units', {})
    print(f"\nUnits: {units}")

    if ef_mag < 20:
        print(f"\n⚠ WARNING: Earth field magnitude ({ef_mag:.1f}) seems very low!")
        print("  Expected: 25-65 µT (depending on location)")
        print("  This could indicate:")
        print("    1. Units mismatch (calibration done in different units)")
        print("    2. Indoor environment with magnetic shielding")
        print("    3. Calibration error")
    elif ef_mag > 80:
        print(f"\n⚠ WARNING: Earth field magnitude ({ef_mag:.1f}) seems high!")
        print("  Expected: 25-65 µT (depending on location)")


def main():
    parser = argparse.ArgumentParser(
        description='Analyze magnetic residual for sessions without finger magnets'
    )
    parser.add_argument(
        '--data-dir', type=str, default='data/GAMBIT',
        help='Path to data directory'
    )
    parser.add_argument(
        '--plot', action='store_true',
        help='Generate plots'
    )
    parser.add_argument(
        '--output', type=str,
        help='Output path for plot'
    )
    parser.add_argument(
        '--verbose', '-v', action='store_true',
        help='Show per-session details'
    )
    parser.add_argument(
        '--json', action='store_true',
        help='Output results as JSON'
    )
    parser.add_argument(
        '--check-calibration', action='store_true',
        help='Analyze calibration file'
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory not found: {data_dir}")
        sys.exit(1)

    # Analyze sessions
    results = analyze_dataset(data_dir)

    if args.json:
        print(json.dumps(results, indent=2))
    else:
        print_summary(results)

        if args.verbose:
            print_per_session_details(results)

        if args.check_calibration:
            check_calibration_file(data_dir)

    if args.plot:
        output_path = Path(args.output) if args.output else None
        plot_results(results, output_path)


if __name__ == '__main__':
    main()
