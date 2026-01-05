#!/usr/bin/env python3
"""
Residual Signal Analysis for Orientation-Independent Finger Classification

This script analyzes whether properly computed magnetic residual (finger magnets only)
provides better orientation generalization than raw magnetometer readings.

Key hypothesis: If residual = raw_mag - Earth_field_in_sensor_frame is computed correctly,
it should be orientation-independent because:
1. Finger magnets are rigidly attached to fingers (move with hand)
2. Sensor is rigidly attached to hand
3. Magnet-to-sensor relationship only changes with finger flexion

The residual computation requires:
1. Earth field estimate in WORLD frame
2. Orientation quaternion for each sample
3. Rotation of Earth field TO sensor frame before subtraction
"""

import json
import numpy as np
from collections import defaultdict
from scipy.spatial.transform import Rotation as R
from typing import Dict, List, Tuple
from pathlib import Path


def load_labeled_session(data_dir: str = "data/GAMBIT") -> Dict:
    """Load the main labeled session with orientation data."""

    # Find the session with most labels (Dec 31 session)
    session_path = Path(data_dir) / "2025-12-31T14_06_18.270Z.json"

    if not session_path.exists():
        # Find any session with labels
        for fname in Path(data_dir).glob("*.json"):
            if fname.name == "manifest.json":
                continue
            with open(fname) as f:
                data = json.load(f)
            if data.get('labels'):
                session_path = fname
                break

    with open(session_path) as f:
        data = json.load(f)

    print(f"Loaded session: {session_path.name}")
    print(f"  Samples: {len(data.get('samples', []))}")
    print(f"  Labels: {len(data.get('labels', []))}")

    return data


def extract_samples_with_orientation(data: Dict) -> Dict[str, List[Dict]]:
    """Extract labeled samples with magnetometer and orientation."""

    samples = data.get('samples', [])
    labels = data.get('labels', [])

    samples_by_code = defaultdict(list)
    residual_count = 0

    for lbl in labels:
        fingers = lbl.get('labels', {}).get('fingers', lbl.get('fingers', {}))
        start = lbl.get('start_sample', lbl.get('startIndex', 0))
        end = lbl.get('end_sample', lbl.get('endIndex', 0))

        states = []
        for f in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            state = fingers.get(f, 'unknown')
            states.append('0' if state == 'extended' else '2' if state == 'flexed' else '?')
        code = ''.join(states)

        if '?' not in code:
            for idx in range(start, min(end + 1, len(samples))):
                s = samples[idx]
                if 'mx_ut' in s and 'orientation_w' in s:
                    # Use iron-corrected if available, else raw
                    if 'iron_mx' in s:
                        mag = np.array([s['iron_mx'], s['iron_my'], s['iron_mz']])
                    else:
                        mag = np.array([s['mx_ut'], s['my_ut'], s['mz_ut']])

                    sample_data = {
                        'mag': mag,
                        'raw_mag': np.array([s['mx_ut'], s['my_ut'], s['mz_ut']]),
                        'quat': np.array([s['orientation_x'], s['orientation_y'],
                                         s['orientation_z'], s['orientation_w']]),
                        'pitch': s.get('euler_pitch', 0),
                        'roll': s.get('euler_roll', 0),
                        'yaw': s.get('euler_yaw', 0),
                    }

                    # Include pre-computed residual from calibration pipeline if available
                    if 'residual_mx' in s:
                        sample_data['pre_residual'] = np.array([
                            s['residual_mx'], s['residual_my'], s['residual_mz']
                        ])
                        sample_data['earth_magnitude'] = s.get('mag_cal_earth_magnitude', 0)
                        residual_count += 1

                    samples_by_code[code].append(sample_data)

    print(f"  Samples with pre-computed residual: {residual_count}")
    return dict(samples_by_code)


def estimate_earth_field_world(samples_by_code: Dict[str, List[Dict]],
                               reference_code: str = '00000') -> np.ndarray:
    """
    Estimate Earth field in world frame from reference pose samples.

    Uses a robust estimation that accounts for orientation variance:
    1. For each sample, rotate magnetometer to world frame
    2. Average all world-frame readings

    This gives Earth field + some finger magnet contribution in world frame.
    For reference pose (all extended), finger magnets should be minimal.

    Rotation convention (matching TypeScript):
    - R @ world = sensor (R rotates world vectors to sensor frame)
    - R^T @ sensor = world (R^T rotates sensor vectors to world frame)
    - In scipy: rot.apply(v) = R @ v, rot.inv().apply(v) = R^T @ v
    """

    if reference_code not in samples_by_code:
        raise ValueError(f"Reference code {reference_code} not found in data")

    world_readings = []

    for s in samples_by_code[reference_code]:
        rot = R.from_quat(s['quat'])
        # sensor→world: use R^T = rot.inv().apply()
        mag_world = rot.inv().apply(s['mag'])
        world_readings.append(mag_world)

    # Use median for robustness against outliers
    earth_estimate = np.median(world_readings, axis=0)

    std = np.std(world_readings, axis=0)
    print(f"\nEarth field estimate (from {reference_code}):")
    print(f"  World frame: [{earth_estimate[0]:.1f}, {earth_estimate[1]:.1f}, {earth_estimate[2]:.1f}] μT")
    print(f"  Magnitude: {np.linalg.norm(earth_estimate):.1f} μT")
    print(f"  Std: [{std[0]:.1f}, {std[1]:.1f}, {std[2]:.1f}] μT")

    return earth_estimate


def compute_residual(mag: np.ndarray, quat: np.ndarray, earth_world: np.ndarray) -> np.ndarray:
    """
    Compute orientation-aware magnetic residual.

    residual = mag_sensor - (R @ earth_world)

    where R is the rotation from world to sensor frame.
    Matching TypeScript: earthSensor = R @ earthWorld
    """
    rot = R.from_quat(quat)
    # Earth field in sensor frame: world→sensor uses R = rot.apply()
    earth_sensor = rot.apply(earth_world)
    return mag - earth_sensor


def add_residuals(samples_by_code: Dict[str, List[Dict]], earth_world: np.ndarray):
    """Add residual field to all samples."""

    for code, samples in samples_by_code.items():
        for s in samples:
            s['residual'] = compute_residual(s['mag'], s['quat'], earth_world)


def knn_classify(test_vec: np.ndarray, train_data: Dict[str, np.ndarray], k: int = 5) -> str:
    """k-NN classification."""

    distances = []
    for code, vecs in train_data.items():
        for v in vecs:
            d = np.linalg.norm(test_vec - v)
            distances.append((d, code))

    distances.sort(key=lambda x: x[0])
    votes = defaultdict(int)
    for d, code in distances[:k]:
        votes[code] += 1

    return max(votes.keys(), key=lambda c: votes[c])


def evaluate_cross_orientation(samples_by_code: Dict[str, List[Dict]],
                               feature_key: str = 'mag',
                               split_axis: str = 'pitch',
                               k: int = 5) -> Tuple[float, float]:
    """
    Evaluate cross-orientation accuracy.

    Returns (Q4->Q1 accuracy, Q1->Q4 accuracy)
    """

    # Get split points
    all_angles = []
    for samples in samples_by_code.values():
        for s in samples:
            all_angles.append(s[split_axis])

    q1, q3 = np.percentile(all_angles, [25, 75])

    # Split data
    train_q4 = defaultdict(list)  # High angles (Q4)
    train_q1 = defaultdict(list)  # Low angles (Q1)

    for code, samples in samples_by_code.items():
        for s in samples:
            angle = s[split_axis]
            if angle >= q3:
                train_q4[code].append(s)
            elif angle <= q1:
                train_q1[code].append(s)

    # Evaluate Q4 -> Q1
    def evaluate_split(train_dict, test_dict):
        # Z-score normalize
        all_train = np.array([s[feature_key] for samps in train_dict.values() for s in samps])
        mean = np.mean(all_train, axis=0)
        std = np.std(all_train, axis=0)
        std = np.where(std > 0, std, 1)

        train_norm = {c: [(s[feature_key] - mean) / std for s in samps]
                      for c, samps in train_dict.items()}

        correct = 0
        total = 0

        for true_code, samples in test_dict.items():
            for s in samples:
                test_vec = (s[feature_key] - mean) / std
                pred = knn_classify(test_vec, train_norm, k)
                if pred == true_code:
                    correct += 1
                total += 1

        return correct / total if total > 0 else 0

    acc_q4_to_q1 = evaluate_split(train_q4, train_q1)
    acc_q1_to_q4 = evaluate_split(train_q1, train_q4)

    return acc_q4_to_q1, acc_q1_to_q4


def analyze_residual_variance(samples_by_code: Dict[str, List[Dict]]):
    """
    Analyze variance of raw vs residual across orientations.

    If residual is truly orientation-independent, its variance within each class
    should be lower than raw magnetometer.
    """

    print("\n" + "="*70)
    print("VARIANCE ANALYSIS: Raw vs Residual")
    print("="*70)

    print(f"\n{'Class':<8} {'Raw Var':>12} {'Residual Var':>14} {'Reduction':>12}")
    print("-" * 50)

    total_raw_var = 0
    total_res_var = 0

    for code in sorted(samples_by_code.keys()):
        samples = samples_by_code[code]

        raw_vecs = np.array([s['mag'] for s in samples])
        res_vecs = np.array([s['residual'] for s in samples])

        # Total variance (sum of per-axis variance)
        raw_var = np.sum(np.var(raw_vecs, axis=0))
        res_var = np.sum(np.var(res_vecs, axis=0))

        reduction = (raw_var - res_var) / raw_var * 100 if raw_var > 0 else 0

        total_raw_var += raw_var
        total_res_var += res_var

        print(f"{code:<8} {raw_var:>12.0f} {res_var:>14.0f} {reduction:>+11.1f}%")

    overall_reduction = (total_raw_var - total_res_var) / total_raw_var * 100
    print("-" * 50)
    print(f"{'TOTAL':<8} {total_raw_var:>12.0f} {total_res_var:>14.0f} {overall_reduction:>+11.1f}%")

    return overall_reduction


def estimate_earth_with_known_magnitude(samples_by_code: Dict[str, List[Dict]],
                                        reference_code: str = '00000',
                                        target_magnitude: float = 50.0) -> np.ndarray:
    """
    Estimate Earth field using known magnitude constraint.

    Uses the world-frame direction from data but scales to expected Earth magnitude.
    This helps when the reference class has magnet contribution that biases the estimate.
    """
    if reference_code not in samples_by_code:
        raise ValueError(f"Reference code {reference_code} not found in data")

    world_readings = []
    for s in samples_by_code[reference_code]:
        rot = R.from_quat(s['quat'])
        # sensor→world: R^T = rot.inv().apply()
        mag_world = rot.inv().apply(s['mag'])
        world_readings.append(mag_world)

    # Get direction (median)
    earth_direction = np.median(world_readings, axis=0)
    earth_direction = earth_direction / np.linalg.norm(earth_direction)

    # Scale to target magnitude
    earth_estimate = earth_direction * target_magnitude

    print(f"\nEarth field estimate (scaled to {target_magnitude} μT):")
    print(f"  World frame: [{earth_estimate[0]:.1f}, {earth_estimate[1]:.1f}, {earth_estimate[2]:.1f}] μT")

    return earth_estimate


def add_world_frame_mag(samples_by_code: Dict[str, List[Dict]]):
    """Add world-frame magnetometer to all samples."""
    for code, samples in samples_by_code.items():
        for s in samples:
            rot = R.from_quat(s['quat'])
            # sensor→world: R^T = rot.inv().apply()
            s['mag_world'] = rot.inv().apply(s['mag'])


def run_analysis():
    """Run full residual analysis."""

    print("="*70)
    print("RESIDUAL SIGNAL ANALYSIS")
    print("Testing if orientation-aware Earth subtraction improves generalization")
    print("="*70)

    # Load data
    data = load_labeled_session()
    samples_by_code = extract_samples_with_orientation(data)

    total_samples = sum(len(v) for v in samples_by_code.values())
    print(f"\nExtracted {total_samples} samples across {len(samples_by_code)} classes")

    # Check if we have iron-corrected
    s0 = list(samples_by_code.values())[0][0]
    has_iron = 'iron_mx' in data.get('samples', [{}])[0]
    print(f"Using iron-corrected: {has_iron}")

    # Method 1: Estimate Earth from data
    earth_from_data = estimate_earth_field_world(samples_by_code, reference_code='00000')

    # Method 2: Use known Earth magnitude (~50 μT for most locations)
    earth_scaled = estimate_earth_with_known_magnitude(samples_by_code, target_magnitude=50.0)

    # Add world frame mag
    add_world_frame_mag(samples_by_code)

    # Test with different Earth estimates
    print("\n" + "="*70)
    print("TESTING MULTIPLE EARTH FIELD ESTIMATES")
    print("="*70)

    earth_estimates = {
        'from_data': earth_from_data,
        'scaled_50uT': earth_scaled,
    }

    results = {}

    for earth_name, earth_world in earth_estimates.items():
        print(f"\n--- Earth estimate: {earth_name} ---")

        # Add residuals
        for code, samples in samples_by_code.items():
            for s in samples:
                s['residual'] = compute_residual(s['mag'], s['quat'], earth_world)

        # Verify
        residuals_00000 = np.array([s['residual'] for s in samples_by_code['00000']])
        print(f"  Reference class residual mean: [{residuals_00000[:,0].mean():.1f}, {residuals_00000[:,1].mean():.1f}, {residuals_00000[:,2].mean():.1f}]")
        print(f"  Reference class residual std:  [{residuals_00000[:,0].std():.1f}, {residuals_00000[:,1].std():.1f}, {residuals_00000[:,2].std():.1f}]")

        # Cross-pitch accuracy
        acc_q4_q1, acc_q1_q4 = evaluate_cross_orientation(
            samples_by_code, feature_key='residual', split_axis='pitch'
        )
        print(f"  Cross-pitch: Q4→Q1 {acc_q4_q1:.1%}, Q1→Q4 {acc_q1_q4:.1%}")
        results[f'{earth_name}_residual'] = min(acc_q4_q1, acc_q1_q4)

    # Also test raw iron-corrected and world-frame
    acc_q4_q1, acc_q1_q4 = evaluate_cross_orientation(
        samples_by_code, feature_key='mag', split_axis='pitch'
    )
    print(f"\n--- Iron-corrected (sensor frame) ---")
    print(f"  Cross-pitch: Q4→Q1 {acc_q4_q1:.1%}, Q1→Q4 {acc_q1_q4:.1%}")
    results['iron_sensor'] = min(acc_q4_q1, acc_q1_q4)

    acc_q4_q1, acc_q1_q4 = evaluate_cross_orientation(
        samples_by_code, feature_key='mag_world', split_axis='pitch'
    )
    print(f"\n--- Iron-corrected (world frame) ---")
    print(f"  Cross-pitch: Q4→Q1 {acc_q4_q1:.1%}, Q1→Q4 {acc_q1_q4:.1%}")
    results['iron_world'] = min(acc_q4_q1, acc_q1_q4)

    # Test pre-computed residual from calibration pipeline (if available)
    has_pre_residual = all(
        'pre_residual' in s
        for samples in samples_by_code.values()
        for s in samples
    )
    if has_pre_residual:
        acc_q4_q1, acc_q1_q4 = evaluate_cross_orientation(
            samples_by_code, feature_key='pre_residual', split_axis='pitch'
        )
        print(f"\n--- Pre-computed residual (from calibration) ---")
        print(f"  Cross-pitch: Q4→Q1 {acc_q4_q1:.1%}, Q1→Q4 {acc_q1_q4:.1%}")
        results['pre_residual'] = min(acc_q4_q1, acc_q1_q4)

        # Check Earth magnitude used by calibration
        earth_mags = [s['earth_magnitude'] for samples in samples_by_code.values() for s in samples]
        print(f"  Earth magnitude range: {min(earth_mags):.1f} - {max(earth_mags):.1f} μT")

        # Check residual variance for reference class
        pre_res_00000 = np.array([s['pre_residual'] for s in samples_by_code.get('00000', [])])
        if len(pre_res_00000) > 0:
            print(f"  Reference class (00000) residual mean: {np.mean(pre_res_00000, axis=0)}")
            print(f"  Reference class (00000) residual std: {np.std(pre_res_00000, axis=0)}")
    else:
        print(f"\n--- Pre-computed residual: NOT AVAILABLE ---")

    # Analyze variance reduction with best Earth estimate
    add_residuals(samples_by_code, earth_scaled)
    variance_reduction = analyze_residual_variance(samples_by_code)

    # Summary
    print("\n" + "="*70)
    print("CROSS-PITCH ACCURACY SUMMARY")
    print("="*70)

    for name, acc in sorted(results.items(), key=lambda x: -x[1]):
        print(f"  {name:25s}: {acc:.1%}")

    best_method = max(results.items(), key=lambda x: x[1])
    print(f"\nBest method: {best_method[0]} ({best_method[1]:.1%})")
    print(f"Variance reduction: {variance_reduction:.1f}%")

    # Diagnosis
    print("\n" + "="*70)
    print("DIAGNOSIS")
    print("="*70)

    if results.get('from_data_residual', 0) < results.get('iron_sensor', 0):
        print("WARNING: Residual performs WORSE than sensor-frame iron-corrected")
        print("  → This suggests the Earth field estimate is contaminated")
        print("  → The magnets contribute even when fingers are 'extended'")
        print("  → Need magnet-free baseline calibration or physics model")
    elif results.get('scaled_50uT_residual', 0) > results.get('iron_sensor', 0) + 0.05:
        print("SUCCESS: Residual with scaled Earth improves generalization")
    else:
        print("NEUTRAL: No significant improvement from residual computation")
        print("  → Orientation variance may not be the main issue")
        print("  → Consider finger-specific calibration or different features")

    return results


def run_residual_only_analysis():
    """
    Run analysis using ONLY samples with pre-computed residual.
    Tests if training on residual (despite contamination) provides any benefit.
    """
    print("="*70)
    print("RESIDUAL-ONLY ANALYSIS")
    print("Training on samples with pre-computed residual from calibration")
    print("="*70)

    # Load data
    data = load_labeled_session()
    samples_by_code = extract_samples_with_orientation(data)

    # Filter to only samples with pre-computed residual
    filtered = {}
    for code, samples in samples_by_code.items():
        with_residual = [s for s in samples if 'pre_residual' in s]
        if with_residual:
            filtered[code] = with_residual

    total_original = sum(len(v) for v in samples_by_code.values())
    total_filtered = sum(len(v) for v in filtered.values())
    print(f"\nFiltered to {total_filtered}/{total_original} samples with residual")
    print(f"Classes: {len(filtered)}")

    if total_filtered == 0:
        print("ERROR: No samples with pre-computed residual!")
        return {}

    # Show per-class counts
    print(f"\n{'Class':<10} {'Count':>8}")
    print("-" * 20)
    for code in sorted(filtered.keys()):
        print(f"{code:<10} {len(filtered[code]):>8}")

    # Add world frame mag
    for code, samples in filtered.items():
        for s in samples:
            rot = R.from_quat(s['quat'])
            s['mag_world'] = rot.inv().apply(s['mag'])

    # Compare different features for cross-pitch accuracy
    print("\n" + "="*70)
    print("CROSS-PITCH ACCURACY COMPARISON")
    print("="*70)

    features_to_test = [
        ('Iron-corrected (sensor)', 'mag'),
        ('Pre-computed residual', 'pre_residual'),
        ('World frame', 'mag_world'),
    ]

    results = {}

    for name, key in features_to_test:
        acc_q4_q1, acc_q1_q4 = evaluate_cross_orientation(
            filtered, feature_key=key, split_axis='pitch'
        )
        print(f"\n{name}:")
        print(f"  Q4→Q1: {acc_q4_q1:.1%}, Q1→Q4: {acc_q1_q4:.1%}")
        print(f"  Worst: {min(acc_q4_q1, acc_q1_q4):.1%}")
        results[key] = {
            'q4_to_q1': acc_q4_q1,
            'q1_to_q4': acc_q1_q4,
            'worst': min(acc_q4_q1, acc_q1_q4),
        }

    # Check variance within classes
    print("\n" + "="*70)
    print("WITHIN-CLASS VARIANCE COMPARISON")
    print("="*70)
    print(f"\n{'Class':<10} {'Iron Var':>12} {'Residual Var':>14} {'Reduction':>12}")
    print("-" * 50)

    total_iron_var = 0
    total_res_var = 0

    for code in sorted(filtered.keys()):
        samples = filtered[code]
        iron_vecs = np.array([s['mag'] for s in samples])
        res_vecs = np.array([s['pre_residual'] for s in samples])

        iron_var = np.sum(np.var(iron_vecs, axis=0))
        res_var = np.sum(np.var(res_vecs, axis=0))
        reduction = (iron_var - res_var) / iron_var * 100 if iron_var > 0 else 0

        total_iron_var += iron_var
        total_res_var += res_var

        print(f"{code:<10} {iron_var:>12.0f} {res_var:>14.0f} {reduction:>+11.1f}%")

    overall_reduction = (total_iron_var - total_res_var) / total_iron_var * 100
    print("-" * 50)
    print(f"{'TOTAL':<10} {total_iron_var:>12.0f} {total_res_var:>14.0f} {overall_reduction:>+11.1f}%")

    # Check if residual provides class separation
    print("\n" + "="*70)
    print("CLASS SEPARATION ANALYSIS")
    print("="*70)

    # Compute between-class vs within-class variance ratio (Fisher criterion)
    for feat_name, feat_key in [('Iron-corrected', 'mag'), ('Residual', 'pre_residual')]:
        all_vecs = []
        labels = []
        for code, samples in filtered.items():
            for s in samples:
                all_vecs.append(s[feat_key])
                labels.append(code)

        all_vecs = np.array(all_vecs)
        labels = np.array(labels)

        # Overall mean
        overall_mean = np.mean(all_vecs, axis=0)

        # Within-class scatter
        within_scatter = 0
        between_scatter = 0

        for code in set(labels):
            class_vecs = all_vecs[labels == code]
            class_mean = np.mean(class_vecs, axis=0)

            # Within-class variance
            within_scatter += np.sum((class_vecs - class_mean) ** 2)

            # Between-class variance
            n_class = len(class_vecs)
            between_scatter += n_class * np.sum((class_mean - overall_mean) ** 2)

        fisher = between_scatter / within_scatter if within_scatter > 0 else 0
        print(f"\n{feat_name}:")
        print(f"  Within-class scatter:  {within_scatter:.0f}")
        print(f"  Between-class scatter: {between_scatter:.0f}")
        print(f"  Fisher ratio (higher=better): {fisher:.3f}")

    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)

    iron_worst = results['mag']['worst']
    res_worst = results['pre_residual']['worst']

    print(f"\nWorst-case cross-pitch accuracy:")
    print(f"  Iron-corrected: {iron_worst:.1%}")
    print(f"  Residual:       {res_worst:.1%}")
    print(f"  Difference:     {res_worst - iron_worst:+.1%}")

    if res_worst > iron_worst:
        print("\n✓ Residual IMPROVES cross-orientation generalization!")
        print("  → Despite calibration contamination, residual provides value")
    else:
        print("\n✗ Residual does NOT improve generalization")
        print(f"  → Variance reduction: {overall_reduction:.1f}%")

    return results


if __name__ == "__main__":
    # Run both analyses
    print("\n" + "#"*70)
    print("# PART 1: Full Analysis (all samples)")
    print("#"*70 + "\n")
    results = run_analysis()

    print("\n\n" + "#"*70)
    print("# PART 2: Residual-Only Analysis (samples with calibration)")
    print("#"*70 + "\n")
    residual_results = run_residual_only_analysis()
