#!/usr/bin/env python3
"""
Magnetometer Calibration Validation Script

Analyzes GAMBIT session data to:
1. Estimate hard iron offset using min-max method
2. Calculate corrected field magnitude
3. Compare against expected Earth field
4. Simulate yaw drift with different calibration strategies

Usage:
    python scripts/validate-mag-calibration.py [--session <name>] [--all]
"""

import json
import numpy as np
from pathlib import Path
import argparse
import sys

# Expected Earth field (Edinburgh UK)
EXPECTED_FIELD = 50.5  # µT

def load_session(filepath: Path) -> dict:
    """Load and validate session data."""
    with open(filepath) as f:
        data = json.load(f)

    if 'samples' not in data or len(data['samples']) < 50:
        raise ValueError(f"Insufficient samples: {len(data.get('samples', []))}")

    return data


def estimate_hard_iron(mx: np.ndarray, my: np.ndarray, mz: np.ndarray) -> tuple:
    """Estimate hard iron offset using min-max method."""
    offset_x = (np.min(mx) + np.max(mx)) / 2
    offset_y = (np.min(my) + np.max(my)) / 2
    offset_z = (np.min(mz) + np.max(mz)) / 2

    range_x = np.max(mx) - np.min(mx)
    range_y = np.max(my) - np.min(my)
    range_z = np.max(mz) - np.min(mz)

    return (
        np.array([offset_x, offset_y, offset_z]),
        np.array([range_x, range_y, range_z])
    )


def analyze_session(data: dict, verbose: bool = True) -> dict:
    """Analyze magnetometer calibration for a session."""
    samples = data['samples']

    # Extract magnetometer data (already in µT)
    mx = np.array([s.get('mx_ut', 0) for s in samples if 'mx_ut' in s])
    my = np.array([s.get('my_ut', 0) for s in samples if 'my_ut' in s])
    mz = np.array([s.get('mz_ut', 0) for s in samples if 'mz_ut' in s])

    if len(mx) < 50:
        return {'error': 'Insufficient mag data'}

    # Raw magnitude
    raw_mag = np.sqrt(mx**2 + my**2 + mz**2)

    # Estimate hard iron offset
    offset, ranges = estimate_hard_iron(mx, my, mz)

    # Corrected magnitude
    corr_mx = mx - offset[0]
    corr_my = my - offset[1]
    corr_mz = mz - offset[2]
    corr_mag = np.sqrt(corr_mx**2 + corr_my**2 + corr_mz**2)

    # Get expected field from metadata
    meta = data.get('metadata', {})
    loc = meta.get('location', {}).get('geomagnetic_field', {})
    expected = loc.get('total_intensity', EXPECTED_FIELD)

    # Calculate quality metrics
    raw_error = abs(np.mean(raw_mag) - expected)
    corr_error = abs(np.mean(corr_mag) - expected)

    # Check if calibration is viable
    min_range = np.min(ranges)
    calibration_viable = min_range > 30  # Need at least 30µT range for good estimate

    # Yaw drift analysis (if orientation data available)
    yaw = np.array([s.get('euler_yaw', 0) for s in samples if 'euler_yaw' in s])
    yaw_drift_rate = None
    if len(yaw) > 100:
        # Unwrap and fit linear trend
        yaw_unwrapped = np.unwrap(yaw * np.pi / 180) * 180 / np.pi
        x = np.arange(len(yaw_unwrapped))
        slope, _ = np.polyfit(x, yaw_unwrapped, 1)
        # Assuming 26 Hz sample rate
        yaw_drift_rate = slope * 26 * 60  # degrees per minute

    result = {
        'samples': len(mx),
        'expected_field': expected,
        'raw_mag_mean': np.mean(raw_mag),
        'raw_mag_std': np.std(raw_mag),
        'hard_iron_offset': offset.tolist(),
        'offset_magnitude': float(np.linalg.norm(offset)),
        'ranges': ranges.tolist(),
        'min_range': float(min_range),
        'calibration_viable': calibration_viable,
        'corrected_mag_mean': np.mean(corr_mag),
        'corrected_mag_std': np.std(corr_mag),
        'raw_error': raw_error,
        'corrected_error': corr_error,
        'improvement': raw_error - corr_error,
        'yaw_drift_rate': yaw_drift_rate,
    }

    if verbose:
        print(f"\n{'='*60}")
        print(f"MAGNETOMETER CALIBRATION ANALYSIS")
        print(f"{'='*60}")
        print(f"\nSamples: {result['samples']}")
        print(f"Expected |B|: {result['expected_field']:.1f} µT")

        print(f"\n--- Raw Magnetometer ---")
        print(f"Mean |B|: {result['raw_mag_mean']:.1f} ± {result['raw_mag_std']:.1f} µT")
        print(f"Error: {result['raw_error']:.1f} µT")

        print(f"\n--- Hard Iron Estimation ---")
        print(f"Offset: [{offset[0]:.1f}, {offset[1]:.1f}, {offset[2]:.1f}] µT")
        print(f"|Offset|: {result['offset_magnitude']:.1f} µT")
        print(f"Ranges: [{ranges[0]:.0f}, {ranges[1]:.0f}, {ranges[2]:.0f}] µT")
        print(f"Min range: {min_range:.0f} µT (need >30 for good cal)")
        print(f"Calibration viable: {'Yes' if calibration_viable else 'No'}")

        print(f"\n--- Corrected Magnetometer ---")
        print(f"Mean |B|: {result['corrected_mag_mean']:.1f} ± {result['corrected_mag_std']:.1f} µT")
        print(f"Error: {result['corrected_error']:.1f} µT")
        print(f"Improvement: {result['improvement']:.1f} µT")

        if yaw_drift_rate is not None:
            print(f"\n--- Yaw Drift ---")
            print(f"Drift rate: {yaw_drift_rate:.1f}°/min")
            status = "Good" if abs(yaw_drift_rate) < 5 else "Moderate" if abs(yaw_drift_rate) < 30 else "Severe"
            print(f"Status: {status}")

        # Recommendation
        print(f"\n--- Recommendation ---")
        if result['corrected_error'] < 5:
            print("✓ Calibration excellent - use 9-DOF fusion")
        elif result['corrected_error'] < 15:
            print("~ Calibration acceptable - use 9-DOF with reduced trust")
        elif calibration_viable:
            print("⚠ Calibration marginal - consider 6-DOF fallback")
        else:
            print("✗ Insufficient rotation for calibration - use 6-DOF only")

    return result


def main():
    parser = argparse.ArgumentParser(description='Validate magnetometer calibration')
    parser.add_argument('--session', type=str, help='Specific session file to analyze')
    parser.add_argument('--all', action='store_true', help='Analyze all sessions')
    parser.add_argument('--quiet', action='store_true', help='Minimal output')
    args = parser.parse_args()

    data_dir = Path(__file__).parent.parent / 'data' / 'GAMBIT'

    if not data_dir.exists():
        # Try worktree path
        data_dir = Path(__file__).parent.parent / '.worktrees' / 'data' / 'GAMBIT'

    if not data_dir.exists():
        print(f"Data directory not found: {data_dir}")
        sys.exit(1)

    if args.session:
        # Analyze specific session
        filepath = data_dir / args.session
        if not filepath.exists():
            # Try adding .json
            filepath = data_dir / f"{args.session}.json"

        if not filepath.exists():
            print(f"Session not found: {args.session}")
            sys.exit(1)

        data = load_session(filepath)
        analyze_session(data, verbose=not args.quiet)

    elif args.all:
        # Analyze all sessions
        results = []
        for f in sorted(data_dir.glob('*.json')):
            if f.name == 'manifest.json' or 'generate' in f.name:
                continue

            try:
                data = load_session(f)
                result = analyze_session(data, verbose=False)
                result['session'] = f.name
                results.append(result)
            except Exception as e:
                if not args.quiet:
                    print(f"Error loading {f.name}: {e}")

        # Summary table
        print(f"\n{'='*100}")
        print(f"SUMMARY: {len(results)} sessions analyzed")
        print(f"{'='*100}")
        print(f"\n{'Session':<28} {'Raw':>8} {'Corr':>8} {'Err':>6} {'Drift':>8} {'Status':<12}")
        print("-" * 100)

        for r in sorted(results, key=lambda x: x.get('corrected_error', 999)):
            drift = f"{r['yaw_drift_rate']:.0f}°/m" if r.get('yaw_drift_rate') else "N/A"

            if r.get('corrected_error', 999) < 5:
                status = "✓ Excellent"
            elif r.get('corrected_error', 999) < 15:
                status = "~ Acceptable"
            elif r.get('calibration_viable'):
                status = "⚠ Marginal"
            else:
                status = "✗ Poor"

            print(f"{r['session'][:27]:<28} {r['raw_mag_mean']:>7.1f}µ {r['corrected_mag_mean']:>7.1f}µ {r['corrected_error']:>5.1f}µ {drift:>8} {status:<12}")

        # Statistics
        good = [r for r in results if r.get('corrected_error', 999) < 5]
        acceptable = [r for r in results if 5 <= r.get('corrected_error', 999) < 15]
        marginal = [r for r in results if 15 <= r.get('corrected_error', 999) < 30]
        poor = [r for r in results if r.get('corrected_error', 999) >= 30]

        print(f"\n{'='*40}")
        print(f"Excellent (<5µT error): {len(good)}")
        print(f"Acceptable (5-15µT):    {len(acceptable)}")
        print(f"Marginal (15-30µT):     {len(marginal)}")
        print(f"Poor (>30µT):           {len(poor)}")

    else:
        parser.print_help()


if __name__ == '__main__':
    main()
