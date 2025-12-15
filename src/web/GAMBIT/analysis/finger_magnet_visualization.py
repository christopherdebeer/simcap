#!/usr/bin/env python3
"""
GAMBIT Finger Magnet Visualization Script

Creates ASCII visualizations of magnetic field data for terminal-based analysis.
Also generates detailed spectral and temporal analysis.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple
from datetime import datetime
import math
from collections import defaultdict


def magnitude_3d(x: float, y: float, z: float) -> float:
    return math.sqrt(x*x + y*y + z*z)


def mean(arr: List[float]) -> float:
    return sum(arr) / len(arr) if arr else 0.0


def std(arr: List[float]) -> float:
    if len(arr) < 2:
        return 0.0
    m = mean(arr)
    return math.sqrt(sum((x - m) ** 2 for x in arr) / len(arr))


def ascii_histogram(data: List[float], bins: int = 20, width: int = 50, title: str = "") -> str:
    """Create an ASCII histogram."""
    if not data:
        return "No data"

    min_val, max_val = min(data), max(data)
    if max_val == min_val:
        return f"{title}: All values = {min_val:.2f}"

    bin_width = (max_val - min_val) / bins
    counts = [0] * bins

    for v in data:
        bin_idx = min(int((v - min_val) / bin_width), bins - 1)
        counts[bin_idx] += 1

    max_count = max(counts)
    scale = width / max_count if max_count > 0 else 1

    lines = [f"\n{title}" if title else ""]
    lines.append(f"Range: [{min_val:.2f}, {max_val:.2f}], Samples: {len(data)}")
    lines.append("-" * (width + 25))

    for i, count in enumerate(counts):
        bin_start = min_val + i * bin_width
        bar_len = int(count * scale)
        bar = "#" * bar_len
        lines.append(f"{bin_start:8.1f} |{bar:<{width}} {count}")

    return "\n".join(lines)


def ascii_time_series(data: List[float], width: int = 80, height: int = 15, title: str = "") -> str:
    """Create an ASCII time series plot."""
    if not data:
        return "No data"

    # Downsample if needed
    if len(data) > width:
        step = len(data) // width
        data = [mean(data[i:i+step]) for i in range(0, len(data), step)][:width]

    min_val, max_val = min(data), max(data)
    if max_val == min_val:
        return f"{title}: Constant value = {min_val:.2f}"

    range_val = max_val - min_val

    lines = [f"\n{title}" if title else ""]
    lines.append(f"Range: [{min_val:.2f}, {max_val:.2f}]")

    # Create plot area
    plot = [[" " for _ in range(width)] for _ in range(height)]

    for i, v in enumerate(data[:width]):
        y = int((v - min_val) / range_val * (height - 1))
        y = height - 1 - y  # Flip for display
        plot[y][i] = "*"

    # Add axis
    for row_idx, row in enumerate(plot):
        val = max_val - (row_idx / (height - 1)) * range_val
        lines.append(f"{val:8.1f} |{''.join(row)}")

    lines.append(" " * 9 + "+" + "-" * width)
    lines.append(" " * 9 + f"0{' ' * (width//2 - 2)}time{' ' * (width//2 - 3)}{len(data)}")

    return "\n".join(lines)


def analyze_frequency_content(data: List[float], sample_rate: float = 50.0) -> Dict:
    """
    Simple frequency analysis using autocorrelation.
    (Approximation without FFT libraries)
    """
    if len(data) < 20:
        return {'error': 'Insufficient data'}

    # Remove mean
    m = mean(data)
    centered = [x - m for x in data]

    # Autocorrelation for different lags
    max_lag = min(len(data) // 4, 100)
    autocorr = []

    var = sum(x*x for x in centered)
    if var == 0:
        return {'error': 'Zero variance'}

    for lag in range(max_lag):
        corr = sum(centered[i] * centered[i + lag] for i in range(len(centered) - lag))
        autocorr.append(corr / var)

    # Find peaks in autocorrelation (indicates periodicity)
    peaks = []
    for i in range(2, len(autocorr) - 1):
        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.2:
            period_samples = i
            frequency = sample_rate / period_samples if period_samples > 0 else 0
            peaks.append({
                'lag': i,
                'correlation': autocorr[i],
                'period_samples': period_samples,
                'period_seconds': period_samples / sample_rate,
                'frequency_hz': frequency,
            })

    # Dominant frequency
    dominant = max(peaks, key=lambda x: x['correlation']) if peaks else None

    return {
        'autocorrelation': autocorr[:20],  # First 20 lags
        'peaks': peaks[:5],  # Top 5 peaks
        'dominant_frequency': dominant,
        'has_periodicity': len(peaks) > 0,
    }


def segment_by_magnitude(samples: List[Dict], threshold_ratio: float = 1.5) -> Dict:
    """
    Segment the session into high/low magnetic field regions.
    Useful for identifying finger approach events.
    """
    # Calculate magnitude
    magnitudes = []
    for s in samples:
        mx = s.get('filtered_mx', s.get('mx_ut', 0))
        my = s.get('filtered_my', s.get('my_ut', 0))
        mz = s.get('filtered_mz', s.get('mz_ut', 0))
        magnitudes.append(magnitude_3d(mx, my, mz))

    # Find baseline (lower quartile)
    sorted_mag = sorted(magnitudes)
    baseline = sorted_mag[len(sorted_mag) // 4]
    threshold = baseline * threshold_ratio

    # Segment
    segments = []
    in_high = False
    segment_start = 0

    for i, mag in enumerate(magnitudes):
        if mag > threshold and not in_high:
            # Start of high region
            if i > 0:
                segments.append({
                    'type': 'low',
                    'start': segment_start,
                    'end': i - 1,
                    'duration': i - segment_start,
                    'mean_magnitude': mean(magnitudes[segment_start:i])
                })
            segment_start = i
            in_high = True
        elif mag <= threshold and in_high:
            # End of high region
            segments.append({
                'type': 'high',
                'start': segment_start,
                'end': i - 1,
                'duration': i - segment_start,
                'mean_magnitude': mean(magnitudes[segment_start:i])
            })
            segment_start = i
            in_high = False

    # Final segment
    if segment_start < len(magnitudes):
        segments.append({
            'type': 'high' if in_high else 'low',
            'start': segment_start,
            'end': len(magnitudes) - 1,
            'duration': len(magnitudes) - segment_start,
            'mean_magnitude': mean(magnitudes[segment_start:])
        })

    # Count and stats
    high_segments = [s for s in segments if s['type'] == 'high']
    low_segments = [s for s in segments if s['type'] == 'low']

    return {
        'threshold': threshold,
        'baseline': baseline,
        'total_segments': len(segments),
        'high_segments': len(high_segments),
        'low_segments': len(low_segments),
        'avg_high_duration': mean([s['duration'] for s in high_segments]) if high_segments else 0,
        'avg_low_duration': mean([s['duration'] for s in low_segments]) if low_segments else 0,
        'high_time_percentage': 100 * sum(s['duration'] for s in high_segments) / len(magnitudes) if magnitudes else 0,
        'segments': segments[:20],  # First 20 for display
    }


def analyze_axis_dominance(samples: List[Dict]) -> Dict:
    """
    Analyze which magnetometer axis is most sensitive to finger magnets.
    """
    mx_vals = [abs(s.get('filtered_mx', s.get('mx_ut', 0))) for s in samples]
    my_vals = [abs(s.get('filtered_my', s.get('my_ut', 0))) for s in samples]
    mz_vals = [abs(s.get('filtered_mz', s.get('mz_ut', 0))) for s in samples]

    mx_var = std(mx_vals) ** 2
    my_var = std(my_vals) ** 2
    mz_var = std(mz_vals) ** 2
    total_var = mx_var + my_var + mz_var

    if total_var == 0:
        return {'error': 'No variance'}

    return {
        'mx_variance_contribution': 100 * mx_var / total_var,
        'my_variance_contribution': 100 * my_var / total_var,
        'mz_variance_contribution': 100 * mz_var / total_var,
        'dominant_axis': max([('mx', mx_var), ('my', my_var), ('mz', mz_var)], key=lambda x: x[1])[0],
        'axis_ranking': sorted([('mx', mx_var), ('my', my_var), ('mz', mz_var)], key=lambda x: -x[1]),
    }


def analyze_polarity_signature(samples: List[Dict]) -> Dict:
    """
    Deep analysis of magnetic polarity patterns for alternating finger magnets.
    """
    mx = [s.get('filtered_mx', s.get('mx_ut', 0)) for s in samples]
    my = [s.get('filtered_my', s.get('my_ut', 0)) for s in samples]
    mz = [s.get('filtered_mz', s.get('mz_ut', 0)) for s in samples]

    # Analyze each axis for sign patterns
    def analyze_sign_pattern(values: List[float], axis_name: str) -> Dict:
        if not values:
            return {}

        positive_count = sum(1 for v in values if v > 0)
        negative_count = sum(1 for v in values if v < 0)
        zero_count = sum(1 for v in values if v == 0)

        # Track sign transitions
        transitions = []
        for i in range(1, len(values)):
            if (values[i] > 0) != (values[i-1] > 0) and values[i] != 0 and values[i-1] != 0:
                transitions.append(i)

        return {
            'axis': axis_name,
            'positive_percentage': 100 * positive_count / len(values),
            'negative_percentage': 100 * negative_count / len(values),
            'transitions': len(transitions),
            'first_10_transitions': transitions[:10],
            'dominant_sign': 'positive' if positive_count > negative_count else 'negative',
        }

    mx_pattern = analyze_sign_pattern(mx, 'mx')
    my_pattern = analyze_sign_pattern(my, 'my')
    mz_pattern = analyze_sign_pattern(mz, 'mz')

    # Combined polarity signature (8 octants)
    octant_counts = defaultdict(int)
    for i in range(len(samples)):
        octant = (
            ('+' if mx[i] > 0 else '-') +
            ('+' if my[i] > 0 else '-') +
            ('+' if mz[i] > 0 else '-')
        )
        octant_counts[octant] += 1

    dominant_octant = max(octant_counts.items(), key=lambda x: x[1])

    return {
        'axis_patterns': {
            'mx': mx_pattern,
            'my': my_pattern,
            'mz': mz_pattern,
        },
        'octant_distribution': dict(octant_counts),
        'dominant_octant': dominant_octant[0],
        'dominant_octant_percentage': 100 * dominant_octant[1] / len(samples) if samples else 0,
        'unique_octants_visited': len(octant_counts),
    }


def deep_temporal_analysis(samples: List[Dict]) -> Dict:
    """
    Deep temporal analysis of magnetic field evolution.
    """
    if len(samples) < 50:
        return {'error': 'Insufficient samples'}

    # Extract filtered magnetometer
    mx = [s.get('filtered_mx', s.get('mx_ut', 0)) for s in samples]
    my = [s.get('filtered_my', s.get('my_ut', 0)) for s in samples]
    mz = [s.get('filtered_mz', s.get('mz_ut', 0)) for s in samples]
    mag = [magnitude_3d(mx[i], my[i], mz[i]) for i in range(len(samples))]

    # Window analysis (50-sample windows)
    window_size = 50
    windows = []

    for start in range(0, len(samples) - window_size, window_size // 2):
        end = start + window_size
        window_mag = mag[start:end]
        window_mx = mx[start:end]
        window_my = my[start:end]
        window_mz = mz[start:end]

        windows.append({
            'start': start,
            'end': end,
            'mag_mean': mean(window_mag),
            'mag_std': std(window_mag),
            'mx_mean': mean(window_mx),
            'my_mean': mean(window_my),
            'mz_mean': mean(window_mz),
        })

    # Trend analysis
    mag_means = [w['mag_mean'] for w in windows]
    if len(mag_means) > 1:
        trend_direction = 'increasing' if mag_means[-1] > mag_means[0] else 'decreasing'
        trend_magnitude = abs(mag_means[-1] - mag_means[0])
    else:
        trend_direction = 'stable'
        trend_magnitude = 0

    # Volatility (std of rolling std)
    stds = [w['mag_std'] for w in windows]
    volatility = std(stds) if stds else 0

    return {
        'window_count': len(windows),
        'window_size': window_size,
        'trend_direction': trend_direction,
        'trend_magnitude': trend_magnitude,
        'volatility': volatility,
        'max_window_magnitude': max(w['mag_mean'] for w in windows) if windows else 0,
        'min_window_magnitude': min(w['mag_mean'] for w in windows) if windows else 0,
        'windows': windows[:10],  # First 10 for display
    }


def load_session(filepath: str) -> Tuple[Dict, List[Dict]]:
    """Load a GAMBIT session file."""
    with open(filepath, 'r') as f:
        data = json.load(f)

    metadata = {
        'version': data.get('version', 'unknown'),
        'timestamp': data.get('timestamp', ''),
        'filepath': filepath,
    }

    samples = data.get('samples', [])
    return metadata, samples


def print_visualization_report(filepath: str) -> None:
    """Generate and print visualization report."""
    metadata, samples = load_session(filepath)

    if not samples:
        print(f"No samples in {filepath}")
        return

    print("=" * 80)
    print(f"VISUALIZATION REPORT: {Path(filepath).name}")
    print("=" * 80)
    print(f"Samples: {len(samples)}")

    # Extract time series
    mx = [s.get('filtered_mx', s.get('mx_ut', 0)) for s in samples]
    my = [s.get('filtered_my', s.get('my_ut', 0)) for s in samples]
    mz = [s.get('filtered_mz', s.get('mz_ut', 0)) for s in samples]
    mag = [magnitude_3d(mx[i], my[i], mz[i]) for i in range(len(samples))]

    # Histograms
    print(ascii_histogram(mx, bins=15, width=40, title="Magnetometer X (uT)"))
    print(ascii_histogram(my, bins=15, width=40, title="Magnetometer Y (uT)"))
    print(ascii_histogram(mz, bins=15, width=40, title="Magnetometer Z (uT)"))
    print(ascii_histogram(mag, bins=15, width=40, title="Magnetic Field Magnitude (uT)"))

    # Time series
    print(ascii_time_series(mag, width=60, height=12, title="Magnetic Field Magnitude over Time"))

    # Axis dominance
    print("\n--- AXIS DOMINANCE ANALYSIS ---")
    axis_dom = analyze_axis_dominance(samples)
    print(f"  MX variance contribution: {axis_dom.get('mx_variance_contribution', 0):.1f}%")
    print(f"  MY variance contribution: {axis_dom.get('my_variance_contribution', 0):.1f}%")
    print(f"  MZ variance contribution: {axis_dom.get('mz_variance_contribution', 0):.1f}%")
    print(f"  Dominant axis: {axis_dom.get('dominant_axis', 'N/A')}")

    # Polarity signature
    print("\n--- POLARITY SIGNATURE ANALYSIS ---")
    pol_sig = analyze_polarity_signature(samples)
    print(f"  Unique octants visited: {pol_sig.get('unique_octants_visited', 0)}/8")
    print(f"  Dominant octant: {pol_sig.get('dominant_octant', 'N/A')} ({pol_sig.get('dominant_octant_percentage', 0):.1f}%)")
    print("\n  Octant distribution:")
    for octant, count in sorted(pol_sig.get('octant_distribution', {}).items()):
        pct = 100 * count / len(samples) if samples else 0
        bar = "#" * int(pct / 2)
        print(f"    {octant}: {pct:5.1f}% {bar}")

    for axis in ['mx', 'my', 'mz']:
        pattern = pol_sig.get('axis_patterns', {}).get(axis, {})
        print(f"\n  {axis.upper()}: {pattern.get('positive_percentage', 0):.1f}% positive, "
              f"{pattern.get('negative_percentage', 0):.1f}% negative, "
              f"{pattern.get('transitions', 0)} transitions")

    # Segmentation
    print("\n--- MAGNITUDE SEGMENTATION ---")
    segs = segment_by_magnitude(samples)
    print(f"  Threshold: {segs.get('threshold', 0):.1f} uT (baseline: {segs.get('baseline', 0):.1f} uT)")
    print(f"  High-field segments: {segs.get('high_segments', 0)}")
    print(f"  Low-field segments: {segs.get('low_segments', 0)}")
    print(f"  Average high duration: {segs.get('avg_high_duration', 0):.0f} samples")
    print(f"  Average low duration: {segs.get('avg_low_duration', 0):.0f} samples")
    print(f"  Time in high field: {segs.get('high_time_percentage', 0):.1f}%")

    # Frequency analysis
    print("\n--- FREQUENCY ANALYSIS ---")
    freq_mag = analyze_frequency_content(mag, sample_rate=50.0)
    if 'error' not in freq_mag:
        print(f"  Has periodicity: {freq_mag.get('has_periodicity', False)}")
        if freq_mag.get('dominant_frequency'):
            dom = freq_mag['dominant_frequency']
            print(f"  Dominant frequency: {dom.get('frequency_hz', 0):.2f} Hz")
            print(f"  Period: {dom.get('period_seconds', 0):.3f} seconds")
        print(f"  Autocorrelation (first 10 lags): {[round(x, 3) for x in freq_mag.get('autocorrelation', [])[:10]]}")
    else:
        print(f"  {freq_mag.get('error')}")

    # Temporal analysis
    print("\n--- TEMPORAL EVOLUTION ---")
    temporal = deep_temporal_analysis(samples)
    if 'error' not in temporal:
        print(f"  Trend: {temporal.get('trend_direction', 'N/A')} (magnitude: {temporal.get('trend_magnitude', 0):.1f} uT)")
        print(f"  Volatility: {temporal.get('volatility', 0):.2f}")
        print(f"  Peak window magnitude: {temporal.get('max_window_magnitude', 0):.1f} uT")
        print(f"  Trough window magnitude: {temporal.get('min_window_magnitude', 0):.1f} uT")
    else:
        print(f"  {temporal.get('error')}")

    print("\n" + "=" * 80)


def main():
    """Main entry point."""
    data_dir = Path('/home/user/simcap/data/GAMBIT')

    # Find today's sessions after 22:00
    target_date = "2025-12-15"
    target_hour = 22

    json_files = list(data_dir.glob(f'{target_date}*.json'))

    # Filter by time
    filtered_files = []
    for f in json_files:
        name = f.name
        try:
            ts_str = name.replace('.json', '').replace('_', ':')
            ts = datetime.fromisoformat(ts_str.replace('Z', '+00:00'))
            if ts.hour >= target_hour:
                filtered_files.append((f, ts))
        except ValueError:
            continue

    if not filtered_files:
        print(f"No sessions found for {target_date} after {target_hour}:00")
        return

    filtered_files.sort(key=lambda x: x[1])

    for filepath, _ in filtered_files:
        print_visualization_report(str(filepath))


if __name__ == '__main__':
    main()
