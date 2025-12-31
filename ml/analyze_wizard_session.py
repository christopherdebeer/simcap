#!/usr/bin/env python3
"""
Deep analysis of wizard-labeled session data with ground truth labels.

This session contains wizard-guided data collection with known poses and finger states,
providing ground truth for evaluating calibration quality and ML model training.
"""

import json
import numpy as np
from pathlib import Path
from scipy.optimize import least_squares
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Any, Optional
from collections import defaultdict

# Expected Earth field at Edinburgh
EXPECTED_MAG = 50.4  # µT
EXPECTED_H = 18.9    # Horizontal component
EXPECTED_V = 46.7    # Vertical component


def load_session(path: Path) -> Dict:
    """Load session JSON."""
    with open(path) as f:
        return json.load(f)


def extract_samples(samples: List[Dict]) -> Dict[str, np.ndarray]:
    """Extract all sensor data from samples."""
    data = {
        'mx': np.array([s.get('mx', s.get('magX', 0)) for s in samples]),
        'my': np.array([s.get('my', s.get('magY', 0)) for s in samples]),
        'mz': np.array([s.get('mz', s.get('magZ', 0)) for s in samples]),
        'ax': np.array([s.get('ax', s.get('accelX', 0)) for s in samples]),
        'ay': np.array([s.get('ay', s.get('accelY', 0)) for s in samples]),
        'az': np.array([s.get('az', s.get('accelZ', 0)) for s in samples]),
        'gx': np.array([s.get('gx', s.get('gyroX', 0)) for s in samples]),
        'gy': np.array([s.get('gy', s.get('gyroY', 0)) for s in samples]),
        'gz': np.array([s.get('gz', s.get('gyroZ', 0)) for s in samples]),
        'timestamp': np.array([s.get('timestamp', i) for i, s in enumerate(samples)])
    }

    # Calibrated mag if available
    if 'calibrated_mx' in samples[0]:
        data['cal_mx'] = np.array([s.get('calibrated_mx', 0) for s in samples])
        data['cal_my'] = np.array([s.get('calibrated_my', 0) for s in samples])
        data['cal_mz'] = np.array([s.get('calibrated_mz', 0) for s in samples])

    # Residual mag if available
    if 'residual_mx' in samples[0]:
        data['res_mx'] = np.array([s.get('residual_mx', 0) for s in samples])
        data['res_my'] = np.array([s.get('residual_my', 0) for s in samples])
        data['res_mz'] = np.array([s.get('residual_mz', 0) for s in samples])

    # Compute derived quantities
    data['mag_total'] = np.sqrt(data['mx']**2 + data['my']**2 + data['mz']**2)

    return data


def parse_labels(labels: List[Dict]) -> List[Dict]:
    """Parse wizard labels into a consistent format."""
    parsed = []
    for label in labels:
        # Handle both nested (wizard) and flat (collector) formats
        start = label.get('start_sample', label.get('startIndex', 0))
        end = label.get('end_sample', label.get('endIndex', 0))

        # Get label content
        if 'labels' in label:
            content = label['labels']
        else:
            content = label

        parsed.append({
            'start': start,
            'end': end,
            'pose': content.get('pose', None),
            'fingers': content.get('fingers', {}),
            'calibration': content.get('calibration', 'none'),
            'motion': content.get('motion', 'static'),
            'custom': content.get('custom', [])
        })

    return parsed


def finger_state_to_code(fingers: Dict) -> str:
    """Convert finger states to a 5-character code (e.g., '00222' for closed fist)."""
    code = ''
    for f in ['thumb', 'index', 'middle', 'ring', 'pinky']:
        state = fingers.get(f, 'unknown')
        if state == 'extended':
            code += '0'
        elif state == 'partial':
            code += '1'
        elif state == 'flexed':
            code += '2'
        else:
            code += '?'
    return code


def analyze_label_distribution(labels: List[Dict]) -> Dict:
    """Analyze the distribution of labels."""
    stats = {
        'total_labels': len(labels),
        'poses': defaultdict(int),
        'finger_codes': defaultdict(int),
        'calibration': defaultdict(int),
        'motion': defaultdict(int),
        'total_samples': 0
    }

    for label in labels:
        if label['pose']:
            stats['poses'][label['pose']] += 1

        code = finger_state_to_code(label['fingers'])
        if code != '?????':
            stats['finger_codes'][code] += 1

        stats['calibration'][label['calibration']] += 1
        stats['motion'][label['motion']] += 1
        stats['total_samples'] += label['end'] - label['start']

    return stats


def extract_labeled_segments(data: Dict[str, np.ndarray], labels: List[Dict]) -> Dict[str, List[Dict]]:
    """Extract data segments for each finger configuration."""
    segments_by_code = defaultdict(list)

    for label in labels:
        code = finger_state_to_code(label['fingers'])
        if code == '?????':
            continue

        start, end = label['start'], label['end']
        if end <= start or end > len(data['mx']):
            continue

        segment = {
            'label': label,
            'code': code,
            'mx': data['mx'][start:end],
            'my': data['my'][start:end],
            'mz': data['mz'][start:end],
            'ax': data['ax'][start:end],
            'ay': data['ay'][start:end],
            'az': data['az'][start:end],
        }

        # Add calibrated if available
        if 'cal_mx' in data:
            segment['cal_mx'] = data['cal_mx'][start:end]
            segment['cal_my'] = data['cal_my'][start:end]
            segment['cal_mz'] = data['cal_mz'][start:end]

        # Add residual if available
        if 'res_mx' in data:
            segment['res_mx'] = data['res_mx'][start:end]
            segment['res_my'] = data['res_my'][start:end]
            segment['res_mz'] = data['res_mz'][start:end]

        segments_by_code[code].append(segment)

    return dict(segments_by_code)


def compute_segment_statistics(segments_by_code: Dict[str, List[Dict]]) -> Dict:
    """Compute statistics for each finger configuration."""
    stats = {}

    for code, segments in segments_by_code.items():
        all_mx, all_my, all_mz = [], [], []
        all_cal_mx, all_cal_my, all_cal_mz = [], [], []
        all_res_mx, all_res_my, all_res_mz = [], [], []

        for seg in segments:
            all_mx.extend(seg['mx'])
            all_my.extend(seg['my'])
            all_mz.extend(seg['mz'])

            if 'cal_mx' in seg:
                all_cal_mx.extend(seg['cal_mx'])
                all_cal_my.extend(seg['cal_my'])
                all_cal_mz.extend(seg['cal_mz'])

            if 'res_mx' in seg:
                all_res_mx.extend(seg['res_mx'])
                all_res_my.extend(seg['res_my'])
                all_res_mz.extend(seg['res_mz'])

        stats[code] = {
            'n_segments': len(segments),
            'n_samples': len(all_mx),
            'raw': {
                'mean': [np.mean(all_mx), np.mean(all_my), np.mean(all_mz)],
                'std': [np.std(all_mx), np.std(all_my), np.std(all_mz)],
                'magnitude': np.mean(np.sqrt(np.array(all_mx)**2 + np.array(all_my)**2 + np.array(all_mz)**2))
            }
        }

        if all_cal_mx:
            stats[code]['calibrated'] = {
                'mean': [np.mean(all_cal_mx), np.mean(all_cal_my), np.mean(all_cal_mz)],
                'std': [np.std(all_cal_mx), np.std(all_cal_my), np.std(all_cal_mz)],
                'magnitude': np.mean(np.sqrt(np.array(all_cal_mx)**2 + np.array(all_cal_my)**2 + np.array(all_cal_mz)**2))
            }

        if all_res_mx:
            stats[code]['residual'] = {
                'mean': [np.mean(all_res_mx), np.mean(all_res_my), np.mean(all_res_mz)],
                'std': [np.std(all_res_mx), np.std(all_res_my), np.std(all_res_mz)],
                'magnitude': np.mean(np.sqrt(np.array(all_res_mx)**2 + np.array(all_res_my)**2 + np.array(all_res_mz)**2))
            }

    return stats


def compute_class_separability(stats: Dict) -> Dict:
    """Compute class separability metrics between finger configurations."""
    codes = list(stats.keys())
    n = len(codes)

    if n < 2:
        return {'error': 'Need at least 2 classes for separability analysis'}

    # Use residual or calibrated or raw (in order of preference)
    def get_mean(code):
        s = stats[code]
        if 'residual' in s:
            return np.array(s['residual']['mean'])
        elif 'calibrated' in s:
            return np.array(s['calibrated']['mean'])
        else:
            return np.array(s['raw']['mean'])

    def get_std(code):
        s = stats[code]
        if 'residual' in s:
            return np.array(s['residual']['std'])
        elif 'calibrated' in s:
            return np.array(s['calibrated']['std'])
        else:
            return np.array(s['raw']['std'])

    # Compute pairwise distances
    distances = {}
    for i in range(n):
        for j in range(i + 1, n):
            c1, c2 = codes[i], codes[j]
            mean1, mean2 = get_mean(c1), get_mean(c2)
            std1, std2 = get_std(c1), get_std(c2)

            euclidean = np.linalg.norm(mean1 - mean2)

            # Fisher's discriminant ratio (simplified)
            pooled_std = np.sqrt((std1**2 + std2**2) / 2)
            fisher = euclidean / (np.mean(pooled_std) + 1e-6)

            distances[f'{c1}_vs_{c2}'] = {
                'euclidean': euclidean,
                'fisher': fisher
            }

    return {
        'pairwise_distances': distances,
        'avg_euclidean': np.mean([d['euclidean'] for d in distances.values()]),
        'avg_fisher': np.mean([d['fisher'] for d in distances.values()])
    }


def main():
    session_path = Path('data/GAMBIT/2025-12-31T14_06_18.270Z.json')

    print("=" * 80)
    print("WIZARD-LABELED SESSION ANALYSIS")
    print("Ground Truth Data for ML Training Validation")
    print("=" * 80)
    print(f"\nSession: {session_path.name}")

    # Load session
    session = load_session(session_path)
    samples = session.get('samples', [])
    labels = session.get('labels', [])
    metadata = session.get('metadata', {})

    print(f"\nTotal samples: {len(samples)}")
    print(f"Total labels: {len(labels)}")
    print(f"Duration: ~{len(samples)/26:.1f} seconds (at 26 Hz)")

    # Show metadata
    if metadata:
        print(f"\nMetadata:")
        for key, value in metadata.items():
            if isinstance(value, dict):
                print(f"  {key}: (nested object)")
            else:
                print(f"  {key}: {value}")

    # Extract data
    data = extract_samples(samples)

    # Check what processed fields are available
    print(f"\n--- Available Processed Fields ---")
    print(f"Calibrated magnetometer: {'cal_mx' in data}")
    print(f"Residual magnetometer: {'res_mx' in data}")

    # Parse labels
    parsed_labels = parse_labels(labels)

    # 1. Label Distribution Analysis
    print("\n" + "=" * 80)
    print("1. LABEL DISTRIBUTION ANALYSIS")
    print("=" * 80)

    label_stats = analyze_label_distribution(parsed_labels)

    print(f"\nTotal labeled samples: {label_stats['total_samples']} / {len(samples)} "
          f"({label_stats['total_samples']/len(samples)*100:.1f}%)")

    print(f"\nPose distribution:")
    for pose, count in sorted(label_stats['poses'].items(), key=lambda x: -x[1]):
        print(f"  {pose}: {count} segments")

    print(f"\nFinger configuration distribution:")
    for code, count in sorted(label_stats['finger_codes'].items(), key=lambda x: -x[1]):
        # Decode the finger state
        fingers = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
        states = {'0': 'ext', '1': 'part', '2': 'flex'}
        decoded = ', '.join([f"{fingers[i]}:{states.get(c, '?')}" for i, c in enumerate(code)])
        print(f"  {code}: {count} segments ({decoded})")

    print(f"\nCalibration poses:")
    for cal, count in label_stats['calibration'].items():
        print(f"  {cal}: {count}")

    print(f"\nMotion types:")
    for motion, count in label_stats['motion'].items():
        print(f"  {motion}: {count}")

    # 2. Extract and analyze labeled segments
    print("\n" + "=" * 80)
    print("2. LABELED SEGMENT STATISTICS")
    print("=" * 80)

    segments = extract_labeled_segments(data, parsed_labels)
    segment_stats = compute_segment_statistics(segments)

    print(f"\nStatistics per finger configuration:")
    print("-" * 70)

    for code in sorted(segment_stats.keys()):
        stats = segment_stats[code]
        print(f"\n{code} ({stats['n_segments']} segments, {stats['n_samples']} samples):")

        if 'residual' in stats:
            r = stats['residual']
            print(f"  Residual: mean=[{r['mean'][0]:.1f}, {r['mean'][1]:.1f}, {r['mean'][2]:.1f}] µT, "
                  f"std=[{r['std'][0]:.1f}, {r['std'][1]:.1f}, {r['std'][2]:.1f}], mag={r['magnitude']:.1f}")
        elif 'calibrated' in stats:
            c = stats['calibrated']
            print(f"  Calibrated: mean=[{c['mean'][0]:.1f}, {c['mean'][1]:.1f}, {c['mean'][2]:.1f}] µT, "
                  f"std=[{c['std'][0]:.1f}, {c['std'][1]:.1f}, {c['std'][2]:.1f}], mag={c['magnitude']:.1f}")
        else:
            r = stats['raw']
            print(f"  Raw: mean=[{r['mean'][0]:.1f}, {r['mean'][1]:.1f}, {r['mean'][2]:.1f}] µT, "
                  f"std=[{r['std'][0]:.1f}, {r['std'][1]:.1f}, {r['std'][2]:.1f}], mag={r['magnitude']:.1f}")

    # 3. Class Separability
    print("\n" + "=" * 80)
    print("3. CLASS SEPARABILITY ANALYSIS")
    print("=" * 80)

    separability = compute_class_separability(segment_stats)

    if 'error' not in separability:
        print(f"\nAverage Euclidean distance between classes: {separability['avg_euclidean']:.2f} µT")
        print(f"Average Fisher discriminant ratio: {separability['avg_fisher']:.2f}")

        print(f"\nTop 5 most separable class pairs:")
        sorted_pairs = sorted(separability['pairwise_distances'].items(),
                             key=lambda x: -x[1]['fisher'])[:5]
        for pair, metrics in sorted_pairs:
            print(f"  {pair}: euclidean={metrics['euclidean']:.1f}µT, fisher={metrics['fisher']:.2f}")

        print(f"\nTop 5 least separable class pairs (hardest to distinguish):")
        sorted_pairs = sorted(separability['pairwise_distances'].items(),
                             key=lambda x: x[1]['fisher'])[:5]
        for pair, metrics in sorted_pairs:
            print(f"  {pair}: euclidean={metrics['euclidean']:.1f}µT, fisher={metrics['fisher']:.2f}")
    else:
        print(f"\n{separability['error']}")

    # 4. Comparison with simulated training data
    print("\n" + "=" * 80)
    print("4. COMPARISON WITH SIMULATED TRAINING DATA")
    print("=" * 80)

    # Check if we have any simulated/generated training data
    training_data_paths = [
        Path('ml/training_data'),
        Path('ml/simulated_data'),
        Path('ml/generated_data'),
        Path('data/training'),
    ]

    found_training = False
    for path in training_data_paths:
        if path.exists():
            print(f"\nFound training data at: {path}")
            found_training = True

    if not found_training:
        print("\nNo pre-existing simulated training data found.")
        print("This session can serve as ground truth for generating synthetic training data.")

    # 5. Generate visualizations
    print("\n" + "=" * 80)
    print("5. GENERATING VISUALIZATIONS")
    print("=" * 80)

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(16, 12))

    # Plot 1: Raw magnetometer over time with labels
    ax1 = fig.add_subplot(3, 2, 1)
    time_sec = np.arange(len(data['mx'])) / 26  # 26 Hz
    ax1.plot(time_sec, data['mx'], 'r-', alpha=0.5, linewidth=0.5, label='X')
    ax1.plot(time_sec, data['my'], 'g-', alpha=0.5, linewidth=0.5, label='Y')
    ax1.plot(time_sec, data['mz'], 'b-', alpha=0.5, linewidth=0.5, label='Z')

    # Shade labeled regions
    for label in parsed_labels[:20]:  # First 20 labels
        start_t = label['start'] / 26
        end_t = label['end'] / 26
        ax1.axvspan(start_t, end_t, alpha=0.2, color='yellow')

    ax1.set_xlabel('Time (seconds)')
    ax1.set_ylabel('Magnetic Field (µT)')
    ax1.set_title('Raw Magnetometer with Labeled Regions')
    ax1.legend(loc='upper right')
    ax1.grid(True, alpha=0.3)

    # Plot 2: Magnitude over time
    ax2 = fig.add_subplot(3, 2, 2)
    ax2.plot(time_sec, data['mag_total'], 'b-', alpha=0.7, linewidth=0.5)
    ax2.axhline(y=EXPECTED_MAG, color='g', linestyle='--', label=f'Expected ({EXPECTED_MAG} µT)')
    ax2.set_xlabel('Time (seconds)')
    ax2.set_ylabel('Magnitude (µT)')
    ax2.set_title('Magnetometer Magnitude Over Time')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: 3D scatter of finger configurations
    ax3 = fig.add_subplot(3, 2, 3, projection='3d')

    colors = plt.cm.tab10(np.linspace(0, 1, len(segment_stats)))
    for i, (code, stats) in enumerate(segment_stats.items()):
        # Get all samples for this code
        if code in segments:
            for seg in segments[code][:3]:  # Limit segments per class for clarity
                ax3.scatter(seg['mx'], seg['my'], seg['mz'],
                           c=[colors[i]], alpha=0.3, s=2, label=code if seg == segments[code][0] else '')

    ax3.set_xlabel('Mx (µT)')
    ax3.set_ylabel('My (µT)')
    ax3.set_zlabel('Mz (µT)')
    ax3.set_title('3D Scatter by Finger Configuration')

    # Plot 4: Class centroids
    ax4 = fig.add_subplot(3, 2, 4)
    codes = list(segment_stats.keys())
    x_pos = np.arange(len(codes))

    means = []
    stds = []
    for code in codes:
        s = segment_stats[code]
        if 'residual' in s:
            means.append(s['residual']['magnitude'])
            stds.append(np.mean(s['residual']['std']))
        elif 'calibrated' in s:
            means.append(s['calibrated']['magnitude'])
            stds.append(np.mean(s['calibrated']['std']))
        else:
            means.append(s['raw']['magnitude'])
            stds.append(np.mean(s['raw']['std']))

    ax4.bar(x_pos, means, yerr=stds, capsize=3, color='steelblue', alpha=0.7)
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(codes, rotation=45, ha='right')
    ax4.set_ylabel('Mean Magnitude (µT)')
    ax4.set_title('Mean Magnitude per Finger Configuration')
    ax4.grid(True, alpha=0.3, axis='y')

    # Plot 5: Label timeline
    ax5 = fig.add_subplot(3, 2, 5)

    # Color code by finger configuration
    unique_codes = list(set(finger_state_to_code(l['fingers']) for l in parsed_labels))
    code_colors = {code: plt.cm.tab10(i / len(unique_codes)) for i, code in enumerate(unique_codes)}

    for i, label in enumerate(parsed_labels):
        code = finger_state_to_code(label['fingers'])
        start_t = label['start'] / 26
        end_t = label['end'] / 26
        color = code_colors.get(code, 'gray')
        ax5.barh(0, end_t - start_t, left=start_t, height=0.8, color=color, alpha=0.7)

    ax5.set_xlabel('Time (seconds)')
    ax5.set_yticks([])
    ax5.set_title('Label Timeline (color = finger configuration)')

    # Plot 6: Separability matrix
    ax6 = fig.add_subplot(3, 2, 6)

    if 'pairwise_distances' in separability:
        n = len(codes)
        matrix = np.zeros((n, n))
        for i, c1 in enumerate(codes):
            for j, c2 in enumerate(codes):
                if i == j:
                    matrix[i, j] = 0
                else:
                    key = f'{c1}_vs_{c2}' if f'{c1}_vs_{c2}' in separability['pairwise_distances'] else f'{c2}_vs_{c1}'
                    if key in separability['pairwise_distances']:
                        matrix[i, j] = separability['pairwise_distances'][key]['fisher']

        im = ax6.imshow(matrix, cmap='YlOrRd')
        ax6.set_xticks(range(n))
        ax6.set_yticks(range(n))
        ax6.set_xticklabels(codes, rotation=45, ha='right')
        ax6.set_yticklabels(codes)
        ax6.set_title('Fisher Separability Matrix')
        plt.colorbar(im, ax=ax6, label='Fisher Ratio')

    plt.tight_layout()

    output_path = Path('ml/wizard_session_analysis.png')
    plt.savefig(output_path, dpi=150)
    print(f"\nVisualization saved to: {output_path}")

    # 6. Summary for ML Training
    print("\n" + "=" * 80)
    print("6. SUMMARY FOR ML TRAINING")
    print("=" * 80)

    print(f"""
    SESSION SUMMARY:
    ----------------
    - Total samples: {len(samples)}
    - Labeled samples: {label_stats['total_samples']}
    - Unique finger configurations: {len(segment_stats)}
    - Label segments: {label_stats['total_labels']}

    DATA QUALITY FOR TRAINING:
    --------------------------
    - Has calibrated magnetometer: {'cal_mx' in data}
    - Has residual (Earth-subtracted): {'res_mx' in data}
    - Average samples per class: {np.mean([s['n_samples'] for s in segment_stats.values()]):.0f}
    - Class balance: {'Good' if np.std([s['n_samples'] for s in segment_stats.values()]) / np.mean([s['n_samples'] for s in segment_stats.values()]) < 0.5 else 'Imbalanced'}

    SEPARABILITY ASSESSMENT:
    ------------------------
    - Average Fisher ratio: {separability.get('avg_fisher', 'N/A'):.2f}
    - Interpretation: {'Good' if separability.get('avg_fisher', 0) > 2 else 'Moderate' if separability.get('avg_fisher', 0) > 1 else 'Low'} separability

    RECOMMENDATIONS:
    ----------------
    1. {'Use residual magnetometer for training (Earth field subtracted)' if 'res_mx' in data else 'Apply calibration before training'}
    2. {'Classes are well-separated - direct classification should work' if separability.get('avg_fisher', 0) > 2 else 'Consider data augmentation to improve separability'}
    3. Use labeled segments as ground truth anchors for simulated data generation
    """)

    # Save results to JSON
    results = {
        'session': session_path.name,
        'total_samples': len(samples),
        'total_labels': len(labels),
        'label_distribution': {
            'poses': dict(label_stats['poses']),
            'finger_codes': dict(label_stats['finger_codes']),
            'calibration': dict(label_stats['calibration']),
            'motion': dict(label_stats['motion'])
        },
        'segment_statistics': {
            code: {
                'n_segments': s['n_segments'],
                'n_samples': s['n_samples'],
                'raw_magnitude': s['raw']['magnitude'],
                'residual_magnitude': s.get('residual', {}).get('magnitude'),
                'calibrated_magnitude': s.get('calibrated', {}).get('magnitude')
            }
            for code, s in segment_stats.items()
        },
        'separability': {
            'avg_euclidean': separability.get('avg_euclidean'),
            'avg_fisher': separability.get('avg_fisher')
        },
        'has_calibrated': 'cal_mx' in data,
        'has_residual': 'res_mx' in data
    }

    results_path = Path('ml/wizard_session_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")


if __name__ == '__main__':
    main()
