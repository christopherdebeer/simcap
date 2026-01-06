#!/usr/bin/env python3
"""
Test Active Self-Training for Finger State Classification

Inspired by ActiveSelfHAR (Wei et al., arXiv:2303.15107, Mar 2023):
Combines active learning + self-training to improve cross-domain accuracy
with minimal labeled data from target domain.

Workflow:
1. Train V4 on Q3 (source domain) - high pitch angles
2. Generate pseudo-labels for Q1 (target domain) - low pitch angles
3. Select high-confidence pseudo-labels (confidence > 0.9)
4. Select high-uncertainty samples for manual labeling (confidence < 0.6)
5. Fine-tune on combined: Q3 labeled + Q1 pseudo-labeled + Q1 manually-labeled

Expected improvements:
- Test accuracy: 72.8% → 80-85%
- Addresses domain shift directly with target-domain data
- Minimal labeling effort (20-50 samples)

Author: Claude
Date: January 2026
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, Tuple
from dataclasses import dataclass

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')


# ============================================================================
# DATA STRUCTURES
# ============================================================================

@dataclass
class FingerStateData:
    combo: str
    samples: np.ndarray
    pitch_angles: np.ndarray


def load_session_with_pitch() -> Dict[str, FingerStateData]:
    """Load session data with pitch angles."""
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
                pitch_angles=np.array(pitch_data)
            )
        else:
            existing = combo_data[combo]
            combo_data[combo] = FingerStateData(
                combo=combo,
                samples=np.vstack([existing.samples, sensor_data]),
                pitch_angles=np.concatenate([existing.pitch_angles, pitch_data])
            )

    return combo_data


# ============================================================================
# SYNTHETIC DATA
# ============================================================================

class SyntheticGenerator:
    """Generate synthetic samples with tight distribution."""

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

    def generate_combo(self, combo: str, n_samples: int) -> np.ndarray:
        """Generate synthetic samples."""
        if combo in self.real_data:
            real = self.real_data[combo]
            mag_mean = real.samples[:, 6:9].mean(axis=0)
            mag_std = real.samples[:, 6:9].std(axis=0)
        else:
            if self.baseline:
                mag_mean = self.baseline.samples[:, 6:9].mean(axis=0)
                mag_std = self.baseline.samples[:, 6:9].std(axis=0)
            else:
                mag_mean = np.array([46, -46, 31])
                mag_std = np.array([25, 40, 50])

            fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
            for i, state in enumerate(combo):
                if state == 'f':
                    finger = fingers[i]
                    if finger in self.finger_effects:
                        mag_mean = mag_mean + self.finger_effects[finger]['mag_delta']

        samples = []
        for _ in range(n_samples):
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
# FEATURE EXTRACTION & WINDOWING
# ============================================================================

def extract_features(samples: np.ndarray, feature_set: str = 'mag_only') -> np.ndarray:
    """Extract magnetometer features."""
    if feature_set == 'mag_only':
        return samples[:, 6:9]
    return samples


def create_windows(samples: np.ndarray, window_size: int, stride: int = None) -> np.ndarray:
    """Create sliding windows."""
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
    """Convert combo string to binary label."""
    return np.array([0 if c == 'e' else 1 for c in combo], dtype=np.float32)


# ============================================================================
# V4 MODEL ARCHITECTURE
# ============================================================================

def build_v4_regularized(window_size: int = 10, n_features: int = 3) -> keras.Model:
    """V4-Regularized architecture."""
    inputs = keras.layers.Input(shape=(window_size, n_features))

    x = keras.layers.Conv1D(32, 3, activation='relu', padding='same')(inputs)
    x = keras.layers.BatchNormalization()(x)
    x = keras.layers.MaxPooling1D(2)(x)
    x = keras.layers.Dropout(0.4)(x)
    x = keras.layers.LSTM(32)(x)
    x = keras.layers.Dropout(0.5)(x)
    x = keras.layers.Dense(32, activation='relu',
                           kernel_regularizer=keras.regularizers.l2(0.01))(x)
    x = keras.layers.Dropout(0.4)(x)
    outputs = keras.layers.Dense(5, activation='sigmoid',
                                 kernel_regularizer=keras.regularizers.l2(0.01))(x)

    model = keras.Model(inputs, outputs, name='V4_Regularized')

    def label_smoothed_loss(y_true, y_pred):
        y_true_smooth = y_true * 0.9 + 0.05
        return keras.losses.binary_crossentropy(y_true_smooth, y_pred)

    model.compile(
        optimizer=keras.optimizers.Adam(0.001),
        loss=label_smoothed_loss,
        metrics=['accuracy']
    )
    return model


# ============================================================================
# DATA PREPARATION
# ============================================================================

def prepare_source_target_split(
    real_data: Dict[str, FingerStateData],
    window_size: int,
    synthetic_ratio: float,
) -> Tuple[np.ndarray, ...]:
    """
    Prepare data with source/target split for active self-training.

    Source domain (Q3 high pitch): Training data
    Target domain (Q1 low pitch): Pseudo-labeling + active sampling

    Returns:
        X_source, y_source,  # Source domain (Q3) with synthetic
        X_target, y_target,  # Target domain (Q1) ground truth (for evaluation)
        mag_mean, mag_std    # Normalization stats
    """
    generator = SyntheticGenerator(real_data)

    all_pitches = []
    for data in real_data.values():
        all_pitches.extend(data.pitch_angles)
    all_pitches = np.array(all_pitches)

    q1 = np.percentile(all_pitches, 25)
    q3 = np.percentile(all_pitches, 75)

    print(f"Pitch quartiles: Q1={q1:.1f}°, Q3={q3:.1f}°")

    source_windows = []
    source_labels = []
    target_windows = []
    target_labels = []

    for combo in real_data.keys():
        combo_data = real_data[combo]
        label = combo_to_label(combo)

        high_pitch_mask = combo_data.pitch_angles >= q3
        low_pitch_mask = combo_data.pitch_angles <= q1

        high_pitch_samples = combo_data.samples[high_pitch_mask]
        low_pitch_samples = combo_data.samples[low_pitch_mask]

        # Source domain (Q3)
        if len(high_pitch_samples) > 0:
            features = extract_features(high_pitch_samples, 'mag_only')

            # Add synthetic
            if synthetic_ratio > 0 and len(features) > 0:
                n_synth = int(len(features) * synthetic_ratio)
                if n_synth > 0:
                    synth = generator.generate_combo(combo, n_synth)
                    if len(synth) > 0 and synth.ndim == 2:
                        synth_feat = extract_features(synth, 'mag_only')
                        if len(synth_feat) > 0 and synth_feat.shape[1] == features.shape[1]:
                            features = np.vstack([features, synth_feat])

            windows = create_windows(features, window_size)
            for w in windows:
                source_windows.append(w)
                source_labels.append(label)

        # Target domain (Q1)
        if len(low_pitch_samples) > 0:
            features = extract_features(low_pitch_samples, 'mag_only')
            windows = create_windows(features, window_size)
            for w in windows:
                target_windows.append(w)
                target_labels.append(label)

    X_source = np.array(source_windows)
    y_source = np.array(source_labels)
    X_target = np.array(target_windows)
    y_target = np.array(target_labels)

    print(f"\nData split:")
    print(f"  Source (Q3): {len(X_source)} windows")
    print(f"  Target (Q1): {len(X_target)} windows")

    # Compute normalization on source
    mag_mean = X_source.reshape(-1, 3).mean(axis=0)
    mag_std = X_source.reshape(-1, 3).std(axis=0) + 1e-8

    return X_source, y_source, X_target, y_target, mag_mean, mag_std


# ============================================================================
# ACTIVE SELF-TRAINING
# ============================================================================

def select_pseudo_labeled_and_active_samples(
    model: keras.Model,
    X_target: np.ndarray,
    y_target_true: np.ndarray,
    high_conf_threshold: float = 0.9,
    low_conf_threshold: float = 0.6,
    max_active_samples: int = 30
) -> Tuple[np.ndarray, ...]:
    """
    Select pseudo-labeled and active samples from target domain.

    Returns:
        X_pseudo, y_pseudo: High-confidence pseudo-labeled samples
        X_active, y_active: Low-confidence samples (simulated manual labels)
        confidence_stats: Dict with confidence statistics
    """
    # Generate predictions
    y_pred_probs = model.predict(X_target, verbose=0)

    # Compute per-sample confidence (mean probability)
    confidence = np.mean(np.maximum(y_pred_probs, 1 - y_pred_probs), axis=1)

    # High-confidence samples (pseudo-labeled)
    high_conf_mask = confidence >= high_conf_threshold
    X_pseudo = X_target[high_conf_mask]
    y_pseudo = (y_pred_probs[high_conf_mask] > 0.5).astype(np.float32)

    # Low-confidence samples (active learning)
    low_conf_mask = confidence < low_conf_threshold
    low_conf_indices = np.where(low_conf_mask)[0]

    # Sample up to max_active_samples
    if len(low_conf_indices) > max_active_samples:
        # Select most uncertain (lowest confidence)
        sorted_indices = low_conf_indices[np.argsort(confidence[low_conf_indices])]
        active_indices = sorted_indices[:max_active_samples]
    else:
        active_indices = low_conf_indices

    X_active = X_target[active_indices]
    y_active = y_target_true[active_indices]  # Simulate manual labeling

    confidence_stats = {
        'mean_confidence': float(confidence.mean()),
        'high_conf_count': int(high_conf_mask.sum()),
        'low_conf_count': int(low_conf_mask.sum()),
        'active_count': len(active_indices),
        'high_conf_threshold': high_conf_threshold,
        'low_conf_threshold': low_conf_threshold
    }

    print(f"\n--- Pseudo-Labeling & Active Sampling ---")
    print(f"Total target samples: {len(X_target)}")
    print(f"High-confidence (pseudo-labeled): {len(X_pseudo)} ({len(X_pseudo)/len(X_target)*100:.1f}%)")
    print(f"Low-confidence (active): {len(X_active)} ({len(X_active)/len(X_target)*100:.1f}%)")
    print(f"Mean confidence: {confidence_stats['mean_confidence']:.3f}")

    return X_pseudo, y_pseudo, X_active, y_active, confidence_stats


# ============================================================================
# EVALUATION
# ============================================================================

def evaluate_model(model, X, y, dataset_name: str):
    """Evaluate model."""
    y_pred = (model.predict(X, verbose=0) > 0.5).astype(int)
    overall_acc = np.mean(y_pred == y)

    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    per_finger = {}
    for i, finger in enumerate(fingers):
        acc = np.mean(y_pred[:, i] == y[:, i])
        per_finger[finger] = acc

    print(f"\n{dataset_name} Accuracy: {overall_acc:.1%}")
    for finger, acc in per_finger.items():
        print(f"  {finger}: {acc:.1%}")

    return overall_acc, per_finger


# ============================================================================
# MAIN EXPERIMENT
# ============================================================================

def main():
    print("=" * 80)
    print("ACTIVE SELF-TRAINING EXPERIMENT (ActiveSelfHAR-Inspired)")
    print("=" * 80)
    print("\nWorkflow:")
    print("  1. Train V4 on Q3 (source domain)")
    print("  2. Generate pseudo-labels for Q1 (target domain)")
    print("  3. Select high-confidence pseudo-labels (conf > 0.9)")
    print("  4. Select high-uncertainty samples for 'manual labeling' (conf < 0.6)")
    print("  5. Fine-tune on combined: Q3 + Q1 pseudo + Q1 manual")

    # Set random seeds
    np.random.seed(42)
    tf.random.set_seed(42)

    # Load data
    print("\n--- Loading Data ---")
    real_data = load_session_with_pitch()
    print(f"Loaded {len(real_data)} finger state combinations")

    # Prepare source/target split
    print("\n--- Preparing Source/Target Split ---")
    X_source, y_source, X_target, y_target, mag_mean, mag_std = prepare_source_target_split(
        real_data=real_data,
        window_size=10,
        synthetic_ratio=0.5
    )

    # Normalize
    X_source = (X_source - mag_mean) / mag_std
    X_target = (X_target - mag_mean) / mag_std

    results = {}

    # ========================================================================
    # BASELINE: V4 trained on source only
    # ========================================================================

    print("\n" + "=" * 80)
    print("BASELINE: V4 Trained on Source (Q3) Only")
    print("=" * 80)

    model_baseline = build_v4_regularized(window_size=10, n_features=3)

    print("\n--- Training Baseline V4 ---")
    # Split source into train/val (80/20)
    n_train = int(0.8 * len(X_source))
    indices = np.random.permutation(len(X_source))
    X_source_train = X_source[indices[:n_train]]
    y_source_train = y_source[indices[:n_train]]
    X_source_val = X_source[indices[n_train:]]
    y_source_val = y_source[indices[n_train:]]

    print(f"Source train: {len(X_source_train)}, Source val: {len(X_source_val)}")

    callbacks = [
        keras.callbacks.EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]

    history_baseline = model_baseline.fit(
        X_source_train, y_source_train,
        validation_data=(X_source_val, y_source_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate baseline on target
    test_acc_baseline, test_per_finger_baseline = evaluate_model(
        model_baseline, X_target, y_target, "Baseline Target (Q1)"
    )

    results['baseline'] = {
        'test_acc': test_acc_baseline,
        'test_per_finger': test_per_finger_baseline
    }

    # ========================================================================
    # ACTIVE SELF-TRAINING: V4 + Pseudo-Labels + Active Samples
    # ========================================================================

    print("\n" + "=" * 80)
    print("ACTIVE SELF-TRAINING: V4 + Pseudo-Labels + Active Samples")
    print("=" * 80)

    # Select pseudo-labeled and active samples
    X_pseudo, y_pseudo, X_active, y_active, conf_stats = select_pseudo_labeled_and_active_samples(
        model=model_baseline,
        X_target=X_target,
        y_target_true=y_target,
        high_conf_threshold=0.9,
        low_conf_threshold=0.6,
        max_active_samples=30
    )

    # Combine datasets
    print(f"\n--- Combining Datasets for Fine-Tuning ---")
    X_combined = np.vstack([X_source_train, X_pseudo, X_active])
    y_combined = np.vstack([y_source_train, y_pseudo, y_active])

    print(f"Combined dataset:")
    print(f"  Source (Q3 labeled): {len(X_source_train)}")
    print(f"  Pseudo (Q1 high-conf): {len(X_pseudo)}")
    print(f"  Active (Q1 manual): {len(X_active)}")
    print(f"  Total: {len(X_combined)}")

    # Build new model for fine-tuning
    model_finetuned = build_v4_regularized(window_size=10, n_features=3)

    print("\n--- Fine-Tuning on Combined Dataset ---")
    history_finetuned = model_finetuned.fit(
        X_combined, y_combined,
        validation_data=(X_source_val, y_source_val),
        epochs=30,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )

    # Evaluate fine-tuned model on target
    test_acc_finetuned, test_per_finger_finetuned = evaluate_model(
        model_finetuned, X_target, y_target, "Fine-Tuned Target (Q1)"
    )

    results['finetuned'] = {
        'test_acc': test_acc_finetuned,
        'test_per_finger': test_per_finger_finetuned,
        'confidence_stats': conf_stats,
        'training_data': {
            'source': len(X_source_train),
            'pseudo': len(X_pseudo),
            'active': len(X_active),
            'total': len(X_combined)
        }
    }

    # ========================================================================
    # COMPARISON
    # ========================================================================

    print("\n" + "=" * 80)
    print("COMPARISON: Baseline vs Active Self-Training")
    print("=" * 80)

    print("\n| Metric | Baseline (Q3 only) | Active Self-Training | Improvement |")
    print("|--------|-------------------|----------------------|-------------|")

    test_diff = (test_acc_finetuned - test_acc_baseline) * 100

    print(f"| Test Accuracy | {test_acc_baseline:.1%} | {test_acc_finetuned:.1%} | {test_diff:+.1f}% |")

    print("\nPer-Finger Test Accuracy:")
    print("| Finger | Baseline | Active Self-Training | Improvement |")
    print("|--------|----------|----------------------|-------------|")

    fingers = ['thumb', 'index', 'middle', 'ring', 'pinky']
    for finger in fingers:
        acc_baseline = test_per_finger_baseline[finger]
        acc_finetuned = test_per_finger_finetuned[finger]
        diff = (acc_finetuned - acc_baseline) * 100
        print(f"| {finger} | {acc_baseline:.1%} | {acc_finetuned:.1%} | {diff:+.1f}% |")

    print(f"\n📊 Training Data Breakdown:")
    print(f"  Source (Q3 labeled): {len(X_source_train)} samples")
    print(f"  Pseudo (Q1 high-conf): {len(X_pseudo)} samples")
    print(f"  Active (Q1 'manual'): {len(X_active)} samples")
    print(f"  Total: {len(X_combined)} samples (+{len(X_combined)-len(X_source_train)} from Q1)")

    # Determine winner
    if test_acc_finetuned > test_acc_baseline:
        print(f"\n🏆 Winner: Active Self-Training")
        print(f"   Improves test accuracy by {test_diff:.1f}%")
        print(f"   With only {len(X_active)} 'manually labeled' Q1 samples")
    else:
        print(f"\n⚠️ No improvement: {test_diff:.1f}%")

    # Save results
    output_path = Path('ml/results/active_self_training_experiment.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=float)

    print(f"\n📊 Results saved to {output_path}")

    # Save fine-tuned model if better
    if test_acc_finetuned > test_acc_baseline:
        print("\n--- Saving Fine-Tuned Model ---")
        model_path = Path('ml/models/finger_v4_finetuned.keras')
        model_path.parent.mkdir(parents=True, exist_ok=True)
        model_finetuned.save(model_path)
        print(f"Model saved to {model_path}")

    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

    return results


if __name__ == '__main__':
    main()
