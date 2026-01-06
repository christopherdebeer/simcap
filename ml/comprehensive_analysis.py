#!/usr/bin/env python3
"""
Comprehensive Model Analysis: Beyond Cross-Orientation Accuracy

Explores multiple dimensions:
1. Per-finger and per-state accuracy patterns
2. Confusion matrices and error clustering
3. Temporal consistency and transition behavior
4. Feature importance and magnetometer axis analysis
5. Robustness to noise levels
6. Sample efficiency curves
7. Model calibration and confidence analysis
8. State transition confusion analysis

Author: Claude
Date: January 2026
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict, Counter
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.calibration import calibration_curve
import scipy.stats as stats


# ============================================================================
# DATA LOADING (reuse from cross_orientation_ablation)
# ============================================================================

@dataclass
class FingerStateData:
    combo: str
    samples: np.ndarray  # (n, 9) ax,ay,az,gx,gy,gz,mx,my,mz
    pitch_angles: np.ndarray
    raw_samples: List[Dict] = None  # Original sample dicts for extended analysis


def load_session_with_full_data() -> Dict[str, FingerStateData]:
    """Load session data with all available fields."""
    session_path = Path('data/GAMBIT/2025-12-31T14_06_18.270Z.json')
    if not session_path.exists():
        session_path = Path('.worktrees/data/GAMBIT/2025-12-31T14_06_18.270Z.json')

    with open(session_path) as f:
        data = json.load(f)

    samples = data['samples']
    labels = data['labels']

    combo_data = {}

    for lbl in labels:
        if 'labels' in lbl and isinstance(lbl['labels'], dict):
            fingers = lbl['labels'].get('fingers', {})
            start = lbl.get('start_sample', 0)
            end = lbl.get('end_sample', 0)
        else:
            fingers = lbl.get('fingers', {})
            start = lbl.get('startIndex', 0)
            end = lbl.get('endIndex', 0)

        if not fingers or all(v == 'unknown' for v in fingers.values()):
            continue

        segment_samples = samples[start:end]
        if len(segment_samples) < 5:
            continue

        sensor_data = []
        pitch_data = []
        raw_segment = []

        for s in segment_samples:
            ax = s.get('ax', 0) / 8192.0
            ay = s.get('ay', 0) / 8192.0
            az = s.get('az', 0) / 8192.0
            gx = s.get('gx', 0) / 114.28
            gy = s.get('gy', 0) / 114.28
            gz = s.get('gz', 0) / 114.28

            if 'mx_ut' in s:
                mx, my, mz = s['mx_ut'], s['my_ut'], s['mz_ut']
            else:
                mx = s.get('mx', 0) / 10.24
                my = s.get('my', 0) / 10.24
                mz = s.get('mz', 0) / 10.24

            pitch = s.get('euler_pitch', 0)

            sensor_data.append([ax, ay, az, gx, gy, gz, mx, my, mz])
            pitch_data.append(pitch)
            raw_segment.append(s)

        if not sensor_data:
            continue

        combo = ''.join([
            'e' if fingers.get(f, '?') == 'extended' else
            'f' if fingers.get(f, '?') == 'flexed' else '?'
            for f in ['thumb', 'index', 'middle', 'ring', 'pinky']
        ])

        if '?' in combo:
            continue

        if combo not in combo_data:
            combo_data[combo] = FingerStateData(
                combo=combo,
                samples=np.array(sensor_data),
                pitch_angles=np.array(pitch_data),
                raw_samples=raw_segment
            )
        else:
            existing = combo_data[combo]
            combo_data[combo] = FingerStateData(
                combo=combo,
                samples=np.vstack([existing.samples, sensor_data]),
                pitch_angles=np.concatenate([existing.pitch_angles, pitch_data]),
                raw_samples=(existing.raw_samples or []) + raw_segment
            )

    return combo_data


# ============================================================================
# SYNTHETIC GENERATOR
# ============================================================================

class SyntheticGenerator:
    def __init__(self, real_data: Dict[str, FingerStateData]):
        self.real_data = real_data
        self.baseline = real_data.get('eeeee')
        self._compute_finger_effects()

    def _compute_finger_effects(self):
        self.finger_effects = {}
        if not self.baseline:
            return

        baseline_mag = self.baseline.samples[:, 6:9].mean(axis=0)
        single_finger_combos = {
            'thumb': 'feeee', 'index': 'efeee', 'middle': 'eefee',
            'ring': 'eeefe', 'pinky': 'eeeef'
        }

        for finger, combo in single_finger_combos.items():
            if combo in self.real_data:
                data = self.real_data[combo]
                self.finger_effects[finger] = {
                    'mag_delta': data.samples[:, 6:9].mean(axis=0) - baseline_mag,
                    'mag_std': data.samples[:, 6:9].std(axis=0),
                }
            else:
                self.finger_effects[finger] = {
                    'mag_delta': np.array([200, 200, 200]),
                    'mag_std': np.array([50, 50, 50]),
                }

    def generate_combo(self, combo: str, n_samples: int,
                       include_orientation_variance: bool = False) -> np.ndarray:
        if combo in self.real_data:
            return self._generate_from_real(combo, n_samples, include_orientation_variance)
        else:
            return self._generate_interpolated(combo, n_samples, include_orientation_variance)

    def _generate_from_real(self, combo: str, n_samples: int,
                            include_orientation_variance: bool) -> np.ndarray:
        real = self.real_data[combo]
        mag_mean = real.samples[:, 6:9].mean(axis=0)
        mag_std = real.samples[:, 6:9].std(axis=0)

        samples = []
        for _ in range(n_samples):
            if include_orientation_variance:
                mag_sample = mag_mean + np.random.randn(3) * mag_std * 2.0
            else:
                mag_sample = mag_mean + np.random.randn(3) * mag_std

            ax = np.random.normal(0, 0.05)
            ay = np.random.normal(0, 0.05)
            az = np.random.normal(-1, 0.05)
            gx = np.random.normal(0, 2.0)
            gy = np.random.normal(0, 2.0)
            gz = np.random.normal(0, 2.0)

            samples.append([ax, ay, az, gx, gy, gz, mag_sample[0], mag_sample[1], mag_sample[2]])

        return np.array(samples)

    def _generate_interpolated(self, combo: str, n_samples: int,
                               include_orientation_variance: bool) -> np.ndarray:
        if self.baseline:
            base_mag = self.baseline.samples[:, 6:9].mean(axis=0)
            base_std = self.baseline.samples[:, 6:9].std(axis=0)
        else:
            base_mag = np.array([46, -46, 31])
            base_std = np.array([25, 40, 50])

        fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
        total_delta = np.zeros(3)
        total_var = base_std ** 2

        for i, state in enumerate(combo):
            if state == 'f':
                finger = fingers[i]
                if finger in self.finger_effects:
                    total_delta += self.finger_effects[finger]['mag_delta']
                    total_var += self.finger_effects[finger]['mag_std'] ** 2

        mag_mean = base_mag + total_delta
        mag_std = np.sqrt(total_var)

        samples = []
        for _ in range(n_samples):
            if include_orientation_variance:
                mag_sample = mag_mean + np.random.randn(3) * mag_std * 2.0
            else:
                mag_sample = mag_mean + np.random.randn(3) * mag_std

            ax = np.random.normal(0, 0.05)
            ay = np.random.normal(0, 0.05)
            az = np.random.normal(-1, 0.05)
            gx = np.random.normal(0, 2.0)
            gy = np.random.normal(0, 2.0)
            gz = np.random.normal(0, 2.0)

            samples.append([ax, ay, az, gx, gy, gz, mag_sample[0], mag_sample[1], mag_sample[2]])

        return np.array(samples)


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def extract_features(samples: np.ndarray, feature_set: str) -> np.ndarray:
    if feature_set == '9dof':
        return samples
    elif feature_set == 'mag_only':
        return samples[:, 6:9]
    elif feature_set == 'accel_gyro':
        return samples[:, 0:6]
    else:
        return samples


def create_windows(samples: np.ndarray, window_size: int, stride: int = None) -> np.ndarray:
    if stride is None:
        stride = max(1, window_size // 2)

    n_samples = len(samples)
    if n_samples < window_size:
        padding = np.zeros((window_size - n_samples, samples.shape[1]))
        samples = np.vstack([samples, padding])
        n_samples = window_size

    windows = []
    for i in range(0, n_samples - window_size + 1, stride):
        windows.append(samples[i:i+window_size])

    if not windows:
        windows.append(samples[:window_size])

    return np.array(windows)


def combo_to_label(combo: str) -> np.ndarray:
    return np.array([0 if c == 'e' else 1 for c in combo], dtype=np.float32)


def combo_to_class_id(combo: str) -> int:
    """Convert combo to single class ID (0-31)."""
    return sum(1 << i for i, c in enumerate(combo) if c == 'f')


def class_id_to_combo(class_id: int) -> str:
    """Convert class ID back to combo string."""
    return ''.join('f' if (class_id >> i) & 1 else 'e' for i in range(5))


def build_model(window_size: int, n_features: int) -> keras.Model:
    if window_size == 1:
        inputs = keras.layers.Input(shape=(1, n_features))
        x = keras.layers.Flatten()(inputs)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(5, activation='sigmoid')(x)
    elif window_size <= 5:
        inputs = keras.layers.Input(shape=(window_size, n_features))
        x = keras.layers.Conv1D(32, min(3, window_size), activation='relu', padding='same')(inputs)
        x = keras.layers.GlobalAveragePooling1D()(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        x = keras.layers.Dropout(0.3)(x)
        outputs = keras.layers.Dense(5, activation='sigmoid')(x)
    else:
        inputs = keras.layers.Input(shape=(window_size, n_features))
        x = keras.layers.Conv1D(32, 5, activation='relu', padding='same')(inputs)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling1D(2)(x)
        x = keras.layers.Conv1D(64, 5, activation='relu', padding='same')(x)
        x = keras.layers.BatchNormalization()(x)
        x = keras.layers.MaxPooling1D(2)(x)
        x = keras.layers.LSTM(32)(x)
        x = keras.layers.Dropout(0.3)(x)
        x = keras.layers.Dense(32, activation='relu')(x)
        outputs = keras.layers.Dense(5, activation='sigmoid')(x)

    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model


# ============================================================================
# ANALYSIS 1: PER-STATE ACCURACY ANALYSIS
# ============================================================================

def analyze_per_state_accuracy(real_data: Dict[str, FingerStateData], results: Dict):
    """Analyze accuracy patterns across different finger state combinations."""
    print("\n" + "=" * 80)
    print("ANALYSIS 1: PER-STATE ACCURACY PATTERNS")
    print("=" * 80)

    # Get best config's per-finger results
    best_config = "mag_w10_synth50_augFalse"
    if best_config in results:
        per_finger = results[best_config]['per_finger']

        print("\n### Per-Finger Accuracy (Best Config)")
        print(f"{'Finger':<10} {'Accuracy':>10} {'Bar':<40}")
        print("-" * 60)

        for finger, acc in sorted(per_finger.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(acc * 40)
            print(f"{finger:<10} {acc:>9.1%} {bar}")

    # Analyze by finger count (how many fingers flexed)
    print("\n### Accuracy by Number of Flexed Fingers")

    finger_counts = defaultdict(list)
    for combo in real_data.keys():
        count = combo.count('f')
        finger_counts[count].append(combo)

    print(f"{'# Flexed':<10} {'Combos':<10} {'Examples'}")
    print("-" * 60)
    for count in sorted(finger_counts.keys()):
        examples = finger_counts[count][:3]
        print(f"{count:<10} {len(finger_counts[count]):<10} {', '.join(examples)}")

    # Analyze confusion between similar states
    print("\n### Common State Confusions (Hamming distance = 1)")

    combos = list(real_data.keys())
    similar_pairs = []
    for i, c1 in enumerate(combos):
        for c2 in combos[i+1:]:
            hamming = sum(a != b for a, b in zip(c1, c2))
            if hamming == 1:
                # Find which finger differs
                diff_finger = ['thumb', 'index', 'middle', 'ring', 'pinky'][
                    [a != b for a, b in zip(c1, c2)].index(True)
                ]
                similar_pairs.append((c1, c2, diff_finger))

    print(f"Found {len(similar_pairs)} pairs differing by 1 finger:")
    for c1, c2, diff in similar_pairs[:10]:
        print(f"  {c1} <-> {c2} (differs in {diff})")

    return {
        'per_finger': per_finger if best_config in results else {},
        'combos_by_flexed_count': {k: len(v) for k, v in finger_counts.items()},
        'similar_pairs': len(similar_pairs)
    }


# ============================================================================
# ANALYSIS 2: CONFUSION MATRIX AND ERROR CLUSTERING
# ============================================================================

def analyze_confusion_patterns(real_data: Dict[str, FingerStateData]):
    """Train model and analyze confusion patterns."""
    print("\n" + "=" * 80)
    print("ANALYSIS 2: CONFUSION MATRIX & ERROR PATTERNS")
    print("=" * 80)

    # Prepare data with optimal config
    window_size = 10
    feature_set = 'mag_only'

    # Split by pitch
    all_pitches = []
    for combo_data in real_data.values():
        all_pitches.extend(combo_data.pitch_angles.tolist())
    q1 = np.percentile(all_pitches, 25)
    q3 = np.percentile(all_pitches, 75)

    generator = SyntheticGenerator(real_data)

    train_windows = []
    train_labels = []
    train_combos = []
    test_windows = []
    test_labels = []
    test_combos = []

    for combo, combo_data in real_data.items():
        label = combo_to_label(combo)
        features = extract_features(combo_data.samples, feature_set)

        high_pitch_mask = combo_data.pitch_angles >= q3
        low_pitch_mask = combo_data.pitch_angles <= q1

        high_pitch_samples = features[high_pitch_mask]
        low_pitch_samples = features[low_pitch_mask]

        # Add synthetic (50%, no orientation augmentation)
        synth_samples = generator.generate_combo(combo, 150, False)
        synth_features = extract_features(synth_samples, feature_set)

        if len(high_pitch_samples) > 0:
            high_pitch_samples = np.vstack([high_pitch_samples, synth_features])
        else:
            high_pitch_samples = synth_features

        if len(high_pitch_samples) >= window_size:
            windows = create_windows(high_pitch_samples, window_size)
            for w in windows:
                train_windows.append(w)
                train_labels.append(label)
                train_combos.append(combo)

        if len(low_pitch_samples) >= window_size:
            windows = create_windows(low_pitch_samples, window_size)
            for w in windows:
                test_windows.append(w)
                test_labels.append(label)
                test_combos.append(combo)

    X_train = np.array(train_windows)
    y_train = np.array(train_labels)
    X_test = np.array(test_windows)
    y_test = np.array(test_labels)

    # Normalize
    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    print(f"\nTraining: {len(X_train)} samples, Testing: {len(X_test)} samples")

    # Train model
    model = build_model(window_size, X_train.shape[-1])
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0, validation_split=0.15)

    # Get predictions with probabilities
    y_prob = model.predict(X_test, verbose=0)
    y_pred = (y_prob > 0.5).astype(int)

    # Analyze per-finger confusion
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    print("\n### Per-Finger Confusion Rates")
    print(f"{'Finger':<10} {'FP Rate':>10} {'FN Rate':>10} {'Total Errors':>12}")
    print("-" * 45)

    finger_errors = {}
    for i, finger in enumerate(fingers):
        fp = np.sum((y_pred[:, i] == 1) & (y_test[:, i] == 0))
        fn = np.sum((y_pred[:, i] == 0) & (y_test[:, i] == 1))
        total_neg = np.sum(y_test[:, i] == 0)
        total_pos = np.sum(y_test[:, i] == 1)

        fp_rate = fp / total_neg if total_neg > 0 else 0
        fn_rate = fn / total_pos if total_pos > 0 else 0

        finger_errors[finger] = {'fp_rate': fp_rate, 'fn_rate': fn_rate}
        print(f"{finger:<10} {fp_rate:>9.1%} {fn_rate:>9.1%} {fp + fn:>12}")

    # Analyze combo-level errors
    print("\n### Most Common Misclassifications")

    # Convert predictions to combos
    pred_combos = []
    for p in y_pred:
        pred_combos.append(''.join('f' if v else 'e' for v in p))

    # Count misclassifications
    misclass = defaultdict(int)
    for true, pred, tc in zip(y_test, y_pred, test_combos):
        pred_combo = ''.join('f' if v else 'e' for v in pred)
        if pred_combo != tc:
            misclass[(tc, pred_combo)] += 1

    print(f"{'True State':<12} {'Predicted':<12} {'Count':>8} {'Hamming':>8}")
    print("-" * 45)

    for (true, pred), count in sorted(misclass.items(), key=lambda x: -x[1])[:10]:
        hamming = sum(a != b for a, b in zip(true, pred))
        print(f"{true:<12} {pred:<12} {count:>8} {hamming:>8}")

    # Analyze error by Hamming distance
    print("\n### Error Distribution by Hamming Distance")

    hamming_counts = defaultdict(int)
    total_errors = 0
    for (true, pred), count in misclass.items():
        hamming = sum(a != b for a, b in zip(true, pred))
        hamming_counts[hamming] += count
        total_errors += count

    print(f"{'Hamming Dist':>12} {'Errors':>10} {'Percentage':>12}")
    print("-" * 40)
    for d in sorted(hamming_counts.keys()):
        pct = hamming_counts[d] / total_errors * 100 if total_errors > 0 else 0
        print(f"{d:>12} {hamming_counts[d]:>10} {pct:>11.1f}%")

    tf.keras.backend.clear_session()

    return {
        'finger_errors': finger_errors,
        'top_misclassifications': dict(sorted(misclass.items(), key=lambda x: -x[1])[:10]),
        'hamming_distribution': dict(hamming_counts)
    }


# ============================================================================
# ANALYSIS 3: TEMPORAL CONSISTENCY
# ============================================================================

def analyze_temporal_consistency(real_data: Dict[str, FingerStateData]):
    """Analyze how consistent predictions are over time within a labeled segment."""
    print("\n" + "=" * 80)
    print("ANALYSIS 3: TEMPORAL CONSISTENCY & STABILITY")
    print("=" * 80)

    # Train model with optimal config
    window_size = 10
    feature_set = 'mag_only'

    generator = SyntheticGenerator(real_data)

    # Build training data
    train_windows = []
    train_labels = []

    for combo, combo_data in real_data.items():
        label = combo_to_label(combo)
        features = extract_features(combo_data.samples, feature_set)

        # Add synthetic
        synth_samples = generator.generate_combo(combo, 200, False)
        synth_features = extract_features(synth_samples, feature_set)

        all_features = np.vstack([features, synth_features])

        if len(all_features) >= window_size:
            windows = create_windows(all_features, window_size)
            for w in windows:
                train_windows.append(w)
                train_labels.append(label)

    X_train = np.array(train_windows)
    y_train = np.array(train_labels)

    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    X_train = (X_train - mean) / std

    model = build_model(window_size, X_train.shape[-1])
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

    # Now analyze temporal consistency per segment
    print("\n### Prediction Stability Within Segments")

    stability_results = {}

    for combo, combo_data in real_data.items():
        features = extract_features(combo_data.samples, feature_set)

        if len(features) < window_size:
            continue

        # Create overlapping windows with stride=1
        windows = create_windows(features, window_size, stride=1)
        windows_norm = (windows - mean) / std

        # Get predictions
        probs = model.predict(windows_norm, verbose=0)
        preds = (probs > 0.5).astype(int)

        # Calculate flip rate (how often prediction changes)
        if len(preds) > 1:
            flips = np.sum(preds[1:] != preds[:-1], axis=0)
            flip_rate = flips / (len(preds) - 1)
        else:
            flip_rate = np.zeros(5)

        # Calculate confidence variance
        conf_var = probs.var(axis=0)

        # Prediction consistency (% of windows with correct prediction)
        true_label = combo_to_label(combo)
        consistency = np.mean(preds == true_label, axis=0)

        stability_results[combo] = {
            'flip_rate': flip_rate.tolist(),
            'conf_variance': conf_var.tolist(),
            'consistency': consistency.tolist(),
            'n_windows': len(windows)
        }

    # Summary
    all_flip_rates = []
    all_conf_vars = []
    all_consistency = []

    for r in stability_results.values():
        all_flip_rates.append(r['flip_rate'])
        all_conf_vars.append(r['conf_variance'])
        all_consistency.append(r['consistency'])

    avg_flip = np.mean(all_flip_rates, axis=0)
    avg_conf_var = np.mean(all_conf_vars, axis=0)
    avg_consistency = np.mean(all_consistency, axis=0)

    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']

    print(f"\n{'Finger':<10} {'Flip Rate':>12} {'Conf Var':>12} {'Consistency':>12}")
    print("-" * 50)
    for i, finger in enumerate(fingers):
        print(f"{finger:<10} {avg_flip[i]:>11.1%} {avg_conf_var[i]:>12.4f} {avg_consistency[i]:>11.1%}")

    # Find most unstable combos
    print("\n### Most Unstable States (highest flip rate)")

    sorted_by_flip = sorted(stability_results.items(),
                           key=lambda x: np.mean(x[1]['flip_rate']), reverse=True)

    print(f"{'Combo':<10} {'Avg Flip':>10} {'Thumb':>8} {'Index':>8} {'Middle':>8}")
    print("-" * 50)
    for combo, r in sorted_by_flip[:5]:
        avg = np.mean(r['flip_rate'])
        print(f"{combo:<10} {avg:>9.1%} {r['flip_rate'][0]:>7.1%} {r['flip_rate'][1]:>7.1%} {r['flip_rate'][2]:>7.1%}")

    tf.keras.backend.clear_session()

    return {
        'avg_flip_rate': avg_flip.tolist(),
        'avg_confidence_variance': avg_conf_var.tolist(),
        'avg_consistency': avg_consistency.tolist(),
        'per_combo': stability_results
    }


# ============================================================================
# ANALYSIS 4: FEATURE IMPORTANCE / MAGNETOMETER AXIS ANALYSIS
# ============================================================================

def analyze_feature_importance(real_data: Dict[str, FingerStateData]):
    """Analyze which magnetometer axes are most important for classification."""
    print("\n" + "=" * 80)
    print("ANALYSIS 4: MAGNETOMETER AXIS IMPORTANCE")
    print("=" * 80)

    # Get magnetometer statistics per combo
    print("\n### Magnetometer Signatures by Finger State")

    baseline = real_data.get('eeeee')
    if baseline:
        baseline_mag = baseline.samples[:, 6:9].mean(axis=0)
        print(f"Baseline (eeeee): mx={baseline_mag[0]:.1f}, my={baseline_mag[1]:.1f}, mz={baseline_mag[2]:.1f} μT")
    else:
        baseline_mag = np.zeros(3)

    # Single finger states
    single_fingers = {
        'thumb': 'feeee', 'index': 'efeee', 'middle': 'eefee',
        'ring': 'eeefe', 'pinky': 'eeeef'
    }

    print("\n### Single Finger Deltas from Baseline")
    print(f"{'Finger':<10} {'Δmx':>10} {'Δmy':>10} {'Δmz':>10} {'|Δ|':>10}")
    print("-" * 55)

    finger_deltas = {}
    for finger, combo in single_fingers.items():
        if combo in real_data:
            data = real_data[combo]
            mag_mean = data.samples[:, 6:9].mean(axis=0)
            delta = mag_mean - baseline_mag
            magnitude = np.linalg.norm(delta)
            finger_deltas[finger] = delta
            print(f"{finger:<10} {delta[0]:>9.1f} {delta[1]:>9.1f} {delta[2]:>9.1f} {magnitude:>9.1f}")

    # Analyze which axis varies most across states
    print("\n### Axis Variance Across All States")

    all_mag_means = []
    for combo, data in real_data.items():
        all_mag_means.append(data.samples[:, 6:9].mean(axis=0))

    all_mag_means = np.array(all_mag_means)
    axis_variance = all_mag_means.var(axis=0)
    axis_range = all_mag_means.max(axis=0) - all_mag_means.min(axis=0)

    axes = ['mx', 'my', 'mz']
    print(f"{'Axis':<6} {'Variance':>12} {'Range':>12}")
    print("-" * 35)
    for i, ax in enumerate(axes):
        print(f"{ax:<6} {axis_variance[i]:>12.1f} {axis_range[i]:>12.1f}")

    # Test single-axis models
    print("\n### Single-Axis Model Performance")

    generator = SyntheticGenerator(real_data)

    for axis_idx, axis_name in enumerate(axes):
        train_X = []
        train_y = []
        test_X = []
        test_y = []

        all_pitches = []
        for cd in real_data.values():
            all_pitches.extend(cd.pitch_angles.tolist())
        q1, q3 = np.percentile(all_pitches, [25, 75])

        for combo, combo_data in real_data.items():
            label = combo_to_label(combo)
            axis_values = combo_data.samples[:, 6 + axis_idx:7 + axis_idx]

            high_mask = combo_data.pitch_angles >= q3
            low_mask = combo_data.pitch_angles <= q1

            high_samples = axis_values[high_mask]
            low_samples = axis_values[low_mask]

            synth = generator.generate_combo(combo, 150, False)[:, 6+axis_idx:7+axis_idx]

            if len(high_samples) > 0:
                high_samples = np.vstack([high_samples, synth])
            else:
                high_samples = synth

            if len(high_samples) >= 10:
                windows = create_windows(high_samples, 10)
                for w in windows:
                    train_X.append(w)
                    train_y.append(label)

            if len(low_samples) >= 10:
                windows = create_windows(low_samples, 10)
                for w in windows:
                    test_X.append(w)
                    test_y.append(label)

        if len(train_X) < 20 or len(test_X) < 20:
            print(f"{axis_name}: insufficient data")
            continue

        train_X = np.array(train_X)
        train_y = np.array(train_y)
        test_X = np.array(test_X)
        test_y = np.array(test_y)

        mean = train_X.mean()
        std = train_X.std() + 1e-8
        train_X = (train_X - mean) / std
        test_X = (test_X - mean) / std

        model = build_model(10, 1)
        model.fit(train_X, train_y, epochs=30, batch_size=32, verbose=0)

        pred = (model.predict(test_X, verbose=0) > 0.5).astype(int)
        acc = np.mean(pred == test_y)

        print(f"{axis_name}: Cross-orientation accuracy = {acc:.1%}")

        tf.keras.backend.clear_session()

    # Fisher criterion analysis
    print("\n### Fisher Criterion (Class Separability)")

    class_means = {}
    class_vars = {}

    for combo, data in real_data.items():
        mag = data.samples[:, 6:9]
        class_means[combo] = mag.mean(axis=0)
        class_vars[combo] = mag.var(axis=0)

    # Calculate Fisher criterion for each axis
    for axis_idx, axis_name in enumerate(axes):
        means = [class_means[c][axis_idx] for c in real_data.keys()]
        vars_ = [class_vars[c][axis_idx] for c in real_data.keys()]

        # Between-class variance / within-class variance
        between_var = np.var(means)
        within_var = np.mean(vars_)
        fisher = between_var / (within_var + 1e-8)

        print(f"{axis_name}: Fisher criterion = {fisher:.2f}")

    return {
        'finger_deltas': {k: v.tolist() for k, v in finger_deltas.items()},
        'axis_variance': axis_variance.tolist(),
        'axis_range': axis_range.tolist()
    }


# ============================================================================
# ANALYSIS 5: NOISE ROBUSTNESS
# ============================================================================

def analyze_noise_robustness(real_data: Dict[str, FingerStateData]):
    """Test model robustness to different noise levels."""
    print("\n" + "=" * 80)
    print("ANALYSIS 5: NOISE ROBUSTNESS")
    print("=" * 80)

    # Train model on clean data
    window_size = 10
    feature_set = 'mag_only'

    generator = SyntheticGenerator(real_data)

    train_windows = []
    train_labels = []

    for combo, combo_data in real_data.items():
        label = combo_to_label(combo)
        features = extract_features(combo_data.samples, feature_set)

        synth_samples = generator.generate_combo(combo, 200, False)
        synth_features = extract_features(synth_samples, feature_set)

        all_features = np.vstack([features, synth_features])

        if len(all_features) >= window_size:
            windows = create_windows(all_features, window_size)
            for w in windows:
                train_windows.append(w)
                train_labels.append(label)

    X_train = np.array(train_windows)
    y_train = np.array(train_labels)

    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8

    X_train_norm = (X_train - mean) / std

    model = build_model(window_size, X_train.shape[-1])
    model.fit(X_train_norm, y_train, epochs=30, batch_size=32, verbose=0)

    # Test with different noise levels
    noise_levels = [0, 5, 10, 20, 50, 100]  # μT

    print(f"\n{'Noise (μT)':<12} {'Accuracy':>10} {'Δ from clean':>14}")
    print("-" * 40)

    results = {}
    clean_acc = None

    for noise in noise_levels:
        test_windows = []
        test_labels = []

        for combo, combo_data in real_data.items():
            label = combo_to_label(combo)
            features = extract_features(combo_data.samples, feature_set).copy()

            # Add noise to magnetometer
            if noise > 0:
                features += np.random.normal(0, noise, features.shape)

            if len(features) >= window_size:
                windows = create_windows(features, window_size)
                for w in windows:
                    test_windows.append(w)
                    test_labels.append(label)

        X_test = np.array(test_windows)
        y_test = np.array(test_labels)
        X_test_norm = (X_test - mean) / std

        pred = (model.predict(X_test_norm, verbose=0) > 0.5).astype(int)
        acc = np.mean(pred == y_test)

        if clean_acc is None:
            clean_acc = acc

        delta = acc - clean_acc
        delta_str = f"{delta:+.1%}" if noise > 0 else "baseline"

        print(f"{noise:<12} {acc:>9.1%} {delta_str:>14}")
        results[noise] = acc

    tf.keras.backend.clear_session()

    return results


# ============================================================================
# ANALYSIS 6: SAMPLE EFFICIENCY
# ============================================================================

def analyze_sample_efficiency(real_data: Dict[str, FingerStateData]):
    """Test how accuracy changes with training data size."""
    print("\n" + "=" * 80)
    print("ANALYSIS 6: SAMPLE EFFICIENCY (Learning Curve)")
    print("=" * 80)

    window_size = 10
    feature_set = 'mag_only'

    generator = SyntheticGenerator(real_data)

    # Prepare full training set
    all_train = []
    all_train_y = []
    all_test = []
    all_test_y = []

    all_pitches = []
    for cd in real_data.values():
        all_pitches.extend(cd.pitch_angles.tolist())
    q1, q3 = np.percentile(all_pitches, [25, 75])

    for combo, combo_data in real_data.items():
        label = combo_to_label(combo)
        features = extract_features(combo_data.samples, feature_set)

        high_mask = combo_data.pitch_angles >= q3
        low_mask = combo_data.pitch_angles <= q1

        high_samples = features[high_mask]
        low_samples = features[low_mask]

        synth = generator.generate_combo(combo, 150, False)
        synth_features = extract_features(synth, feature_set)

        if len(high_samples) > 0:
            high_samples = np.vstack([high_samples, synth_features])
        else:
            high_samples = synth_features

        if len(high_samples) >= window_size:
            windows = create_windows(high_samples, window_size)
            for w in windows:
                all_train.append(w)
                all_train_y.append(label)

        if len(low_samples) >= window_size:
            windows = create_windows(low_samples, window_size)
            for w in windows:
                all_test.append(w)
                all_test_y.append(label)

    X_train_full = np.array(all_train)
    y_train_full = np.array(all_train_y)
    X_test = np.array(all_test)
    y_test = np.array(all_test_y)

    # Test with different training sizes
    fractions = [0.1, 0.25, 0.5, 0.75, 1.0]

    print(f"\n{'Train Size':<12} {'Fraction':>10} {'Test Acc':>10}")
    print("-" * 35)

    results = {}

    for frac in fractions:
        n = int(len(X_train_full) * frac)
        indices = np.random.permutation(len(X_train_full))[:n]

        X_train = X_train_full[indices]
        y_train = y_train_full[indices]

        mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
        std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8

        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std

        model = build_model(window_size, X_train.shape[-1])
        model.fit(X_train_norm, y_train, epochs=30, batch_size=32, verbose=0)

        pred = (model.predict(X_test_norm, verbose=0) > 0.5).astype(int)
        acc = np.mean(pred == y_test)

        print(f"{n:<12} {frac:>9.0%} {acc:>9.1%}")
        results[frac] = {'n_samples': n, 'accuracy': acc}

        tf.keras.backend.clear_session()

    return results


# ============================================================================
# ANALYSIS 7: MODEL CALIBRATION
# ============================================================================

def analyze_calibration(real_data: Dict[str, FingerStateData]):
    """Analyze how well model probabilities match actual frequencies."""
    print("\n" + "=" * 80)
    print("ANALYSIS 7: MODEL CALIBRATION (Reliability)")
    print("=" * 80)

    # Train model
    window_size = 10
    feature_set = 'mag_only'

    generator = SyntheticGenerator(real_data)

    train_windows = []
    train_labels = []
    test_windows = []
    test_labels = []

    all_pitches = []
    for cd in real_data.values():
        all_pitches.extend(cd.pitch_angles.tolist())
    q1, q3 = np.percentile(all_pitches, [25, 75])

    for combo, combo_data in real_data.items():
        label = combo_to_label(combo)
        features = extract_features(combo_data.samples, feature_set)

        high_mask = combo_data.pitch_angles >= q3
        low_mask = combo_data.pitch_angles <= q1

        high_samples = features[high_mask]
        low_samples = features[low_mask]

        synth = generator.generate_combo(combo, 150, False)
        synth_features = extract_features(synth, feature_set)

        if len(high_samples) > 0:
            high_samples = np.vstack([high_samples, synth_features])
        else:
            high_samples = synth_features

        if len(high_samples) >= window_size:
            windows = create_windows(high_samples, window_size)
            for w in windows:
                train_windows.append(w)
                train_labels.append(label)

        if len(low_samples) >= window_size:
            windows = create_windows(low_samples, window_size)
            for w in windows:
                test_windows.append(w)
                test_labels.append(label)

    X_train = np.array(train_windows)
    y_train = np.array(train_labels)
    X_test = np.array(test_windows)
    y_test = np.array(test_labels)

    mean = X_train.reshape(-1, X_train.shape[-1]).mean(axis=0)
    std = X_train.reshape(-1, X_train.shape[-1]).std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = build_model(window_size, X_train.shape[-1])
    model.fit(X_train, y_train, epochs=30, batch_size=32, verbose=0)

    probs = model.predict(X_test, verbose=0)

    # Analyze calibration per finger
    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    n_bins = 5

    print("\n### Calibration by Probability Bin")
    print("(A well-calibrated model: predicted prob ≈ actual frequency)")

    calibration_results = {}

    for i, finger in enumerate(fingers):
        y_true = y_test[:, i]
        y_prob = probs[:, i]

        # Calculate calibration curve
        bins = np.linspace(0, 1, n_bins + 1)
        bin_indices = np.digitize(y_prob, bins) - 1
        bin_indices = np.clip(bin_indices, 0, n_bins - 1)

        calibration_data = []
        for b in range(n_bins):
            mask = bin_indices == b
            if mask.sum() > 0:
                mean_pred = y_prob[mask].mean()
                actual_freq = y_true[mask].mean()
                calibration_data.append({
                    'bin': b,
                    'mean_pred': mean_pred,
                    'actual_freq': actual_freq,
                    'count': int(mask.sum())
                })

        # Calculate Expected Calibration Error (ECE)
        ece = 0
        for cd in calibration_data:
            ece += cd['count'] / len(y_prob) * abs(cd['mean_pred'] - cd['actual_freq'])

        calibration_results[finger] = {
            'ece': ece,
            'bins': calibration_data
        }

        print(f"\n{finger}: ECE = {ece:.3f}")
        print(f"  {'Pred Range':<15} {'Mean Pred':>10} {'Actual':>10} {'Count':>8}")
        for cd in calibration_data:
            bin_range = f"{bins[cd['bin']]:.1f}-{bins[cd['bin']+1]:.1f}"
            print(f"  {bin_range:<15} {cd['mean_pred']:>9.2f} {cd['actual_freq']:>9.2f} {cd['count']:>8}")

    # Overall confidence distribution
    print("\n### Confidence Distribution")

    flat_probs = probs.flatten()

    # Convert to "confidence" (distance from 0.5)
    confidence = np.abs(flat_probs - 0.5) * 2

    print(f"Mean confidence: {confidence.mean():.2f}")
    print(f"Std confidence: {confidence.std():.2f}")
    print(f"% very confident (>0.9): {(confidence > 0.9).mean():.1%}")
    print(f"% uncertain (<0.3): {(confidence < 0.3).mean():.1%}")

    tf.keras.backend.clear_session()

    return calibration_results


# ============================================================================
# MAIN
# ============================================================================

def main():
    print("=" * 80)
    print("COMPREHENSIVE MODEL ANALYSIS: BEYOND CROSS-ORIENTATION ACCURACY")
    print("=" * 80)

    # Load existing results
    results_path = Path("ml/cross_orientation_ablation.json")
    if results_path.exists():
        with open(results_path) as f:
            results = json.load(f)
        print(f"\nLoaded existing ablation results from {results_path}")
    else:
        results = {}
        print("\nNo existing results found")

    # Load session data
    print("\nLoading session data...")
    real_data = load_session_with_full_data()
    print(f"Loaded {len(real_data)} finger state combinations")

    all_results = {}

    # Run analyses
    all_results['per_state'] = analyze_per_state_accuracy(real_data, results)
    confusion_results = analyze_confusion_patterns(real_data)
    # Convert tuple keys to string for JSON serialization
    if 'top_misclassifications' in confusion_results:
        confusion_results['top_misclassifications'] = {
            f"{k[0]}->{k[1]}": v for k, v in confusion_results['top_misclassifications'].items()
        }
    all_results['confusion'] = confusion_results
    all_results['temporal'] = analyze_temporal_consistency(real_data)
    all_results['feature_importance'] = analyze_feature_importance(real_data)
    noise_results = analyze_noise_robustness(real_data)
    # Convert int keys to string for JSON
    all_results['noise_robustness'] = {str(k): v for k, v in noise_results.items()}

    sample_results = analyze_sample_efficiency(real_data)
    # Convert float keys to string for JSON
    all_results['sample_efficiency'] = {str(k): v for k, v in sample_results.items()}
    all_results['calibration'] = analyze_calibration(real_data)

    # Save results
    output_path = Path("ml/comprehensive_analysis.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\n\nResults saved to: {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("ANALYSIS COMPLETE")
    print("=" * 80)

    print("""
Key Metrics Explored:
1. Per-State Accuracy: Which finger combos are hardest?
2. Confusion Patterns: What gets misclassified as what?
3. Temporal Consistency: How stable are predictions over time?
4. Feature Importance: Which magnetometer axes matter most?
5. Noise Robustness: How does accuracy degrade with noise?
6. Sample Efficiency: How much training data is needed?
7. Calibration: Are predicted probabilities reliable?
""")

    return all_results


if __name__ == "__main__":
    main()
