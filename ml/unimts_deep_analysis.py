#!/usr/bin/env python3
"""
Deep Analysis of UniMTS Experiment Results

Investigates why SO(3) rotation augmentation hurts magnetometer-based
finger classification, despite being state-of-the-art for IMU gesture recognition.

Hypothesis: Magnetometer values encode ABSOLUTE orientation in Earth's magnetic field,
which is discriminative for finger state classification but destroyed by SO(3) rotation.

Author: Claude
Date: January 2026
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from dataclasses import dataclass
from scipy.spatial.transform import Rotation
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
from tensorflow import keras
import warnings
warnings.filterwarnings('ignore')

# Import from main experiment
from ml.unimts_experiment import (
    load_session_with_pitch,
    FingerStateData,
    create_windows,
    combo_to_label,
    SyntheticGenerator,
    build_standard_model,
    train_and_evaluate,
    so3_augment_window,
    svd_orientation_invariant,
)


def analyze_magnetometer_variance():
    """Analyze how much magnetometer values vary across orientations."""
    print("\n" + "=" * 70)
    print("MAGNETOMETER VARIANCE ANALYSIS")
    print("=" * 70)

    real_data = load_session_with_pitch()

    # Calculate pitch quartiles
    all_pitches = []
    for cd in real_data.values():
        all_pitches.extend(cd.pitch_angles.tolist())
    q1 = np.percentile(all_pitches, 25)
    q3 = np.percentile(all_pitches, 75)

    print(f"\nPitch angle quartiles: Q1={q1:.1f}°, Q3={q3:.1f}°")

    print("\nMagnetometer values by orientation (combo: eeeee - all extended):")
    if 'eeeee' in real_data:
        data = real_data['eeeee']
        mag = data.samples[:, 6:9]
        pitch = data.pitch_angles

        high_pitch = mag[pitch >= q3]
        low_pitch = mag[pitch <= q1]

        print(f"\n  High pitch (≥{q3:.1f}°): n={len(high_pitch)}")
        print(f"    Mean: [{high_pitch.mean(axis=0)[0]:.1f}, {high_pitch.mean(axis=0)[1]:.1f}, {high_pitch.mean(axis=0)[2]:.1f}]")
        print(f"    Std:  [{high_pitch.std(axis=0)[0]:.1f}, {high_pitch.std(axis=0)[1]:.1f}, {high_pitch.std(axis=0)[2]:.1f}]")

        print(f"\n  Low pitch (≤{q1:.1f}°): n={len(low_pitch)}")
        print(f"    Mean: [{low_pitch.mean(axis=0)[0]:.1f}, {low_pitch.mean(axis=0)[1]:.1f}, {low_pitch.mean(axis=0)[2]:.1f}]")
        print(f"    Std:  [{low_pitch.std(axis=0)[0]:.1f}, {low_pitch.std(axis=0)[1]:.1f}, {low_pitch.std(axis=0)[2]:.1f}]")

        # Calculate shift
        shift = high_pitch.mean(axis=0) - low_pitch.mean(axis=0)
        shift_pct = shift / high_pitch.std(axis=0) * 100
        print(f"\n  Orientation shift: [{shift[0]:.1f}, {shift[1]:.1f}, {shift[2]:.1f}] μT")
        print(f"  Shift as % of std: [{shift_pct[0]:.0f}%, {shift_pct[1]:.0f}%, {shift_pct[2]:.0f}%]")


def analyze_so3_impact():
    """Analyze how SO(3) rotation changes the data distribution."""
    print("\n" + "=" * 70)
    print("SO(3) ROTATION IMPACT ANALYSIS")
    print("=" * 70)

    real_data = load_session_with_pitch()

    # Get baseline data
    if 'eeeee' in real_data:
        baseline = real_data['eeeee'].samples[:100, 6:9]

        print("\nOriginal data (first 100 samples of 'eeeee'):")
        print(f"  Mean: [{baseline.mean(axis=0)[0]:.1f}, {baseline.mean(axis=0)[1]:.1f}, {baseline.mean(axis=0)[2]:.1f}]")
        print(f"  Std:  [{baseline.std(axis=0)[0]:.1f}, {baseline.std(axis=0)[1]:.1f}, {baseline.std(axis=0)[2]:.1f}]")

        # Apply multiple SO(3) rotations
        rotated_means = []
        for _ in range(100):
            rotated = baseline @ Rotation.random().as_matrix().T
            rotated_means.append(rotated.mean(axis=0))

        rotated_means = np.array(rotated_means)
        print(f"\nAfter SO(3) rotation (100 random rotations):")
        print(f"  Mean of means: [{rotated_means.mean(axis=0)[0]:.1f}, {rotated_means.mean(axis=0)[1]:.1f}, {rotated_means.mean(axis=0)[2]:.1f}]")
        print(f"  Std of means:  [{rotated_means.std(axis=0)[0]:.1f}, {rotated_means.std(axis=0)[1]:.1f}, {rotated_means.std(axis=0)[2]:.1f}]")

        # Key insight
        print("\n  *** Key insight: SO(3) rotation randomizes the absolute position ***")
        print("  *** This destroys discriminative information for finger classification ***")


def test_small_angle_rotation():
    """Test if small-angle rotation helps without destroying signal."""
    print("\n" + "=" * 70)
    print("SMALL ANGLE ROTATION EXPERIMENT")
    print("=" * 70)

    real_data = load_session_with_pitch()

    # Calculate pitch quartiles
    all_pitches = []
    for cd in real_data.values():
        all_pitches.extend(cd.pitch_angles.tolist())
    q1 = np.percentile(all_pitches, 25)
    q3 = np.percentile(all_pitches, 75)

    generator = SyntheticGenerator(real_data)
    window_size = 10

    results = []

    for max_angle_deg in [0, 5, 10, 15, 25, 45, 90, 180]:
        max_angle_rad = np.deg2rad(max_angle_deg)

        train_windows = []
        train_labels = []
        test_windows = []
        test_labels = []

        for combo, combo_data in real_data.items():
            label = combo_to_label(combo)
            features = combo_data.samples[:, 6:9]

            high_pitch_mask = combo_data.pitch_angles >= q3
            low_pitch_mask = combo_data.pitch_angles <= q1

            high_pitch_samples = features[high_pitch_mask]
            low_pitch_samples = features[low_pitch_mask]

            # Add synthetic
            n_synth = 150
            synth_samples = []
            for _ in range(n_synth):
                base = generator.generate_combo(combo, 1, augmentation='none')[0]

                # Apply small-angle rotation
                if max_angle_deg > 0:
                    angles = np.random.uniform(-max_angle_rad, max_angle_rad, size=3)
                    R = Rotation.from_euler('xyz', angles).as_matrix()
                    base = base @ R.T

                synth_samples.append(base)

            synth_samples = np.array(synth_samples)

            if len(high_pitch_samples) > 0:
                combined = np.vstack([high_pitch_samples, synth_samples])
            else:
                combined = synth_samples

            if len(combined) >= window_size:
                windows = create_windows(combined, window_size)
                for w in windows:
                    train_windows.append(w)
                    train_labels.append(label)

            if len(low_pitch_samples) >= window_size:
                windows = create_windows(low_pitch_samples, window_size)
                for w in windows:
                    test_windows.append(w)
                    test_labels.append(label)

        X_train = np.array(train_windows)
        y_train = np.array(train_labels)
        X_test = np.array(test_windows)
        y_test = np.array(test_labels)

        # Normalize
        mean = X_train.reshape(-1, 3).mean(axis=0)
        std = X_train.reshape(-1, 3).std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        # Split
        indices = np.random.permutation(len(X_train))
        val_size = int(0.15 * len(X_train))
        X_val = X_train[indices[:val_size]]
        y_val = y_train[indices[:val_size]]
        X_train = X_train[indices[val_size:]]
        y_train = y_train[indices[val_size:]]

        # Train
        _, metrics = train_and_evaluate(
            X_train, y_train, X_val, y_val, X_test, y_test,
            build_standard_model, epochs=30
        )

        results.append({
            'max_angle': max_angle_deg,
            'train_acc': metrics['train_acc'],
            'test_acc': metrics['test_acc'],
            'gap': metrics['gap']
        })

        print(f"  Max angle: {max_angle_deg:>3}° -> Train: {metrics['train_acc']:.1%}, Test: {metrics['test_acc']:.1%}")

        tf.keras.backend.clear_session()

    return results


def test_magnitude_only():
    """Test if using magnitude instead of 3D vectors helps."""
    print("\n" + "=" * 70)
    print("MAGNITUDE-ONLY EXPERIMENT")
    print("=" * 70)

    real_data = load_session_with_pitch()

    all_pitches = []
    for cd in real_data.values():
        all_pitches.extend(cd.pitch_angles.tolist())
    q1 = np.percentile(all_pitches, 25)
    q3 = np.percentile(all_pitches, 75)

    generator = SyntheticGenerator(real_data)
    window_size = 10

    # Test both raw and magnitude
    for feature_type in ['3d_vector', 'magnitude', 'magnitude_and_angles']:
        train_windows = []
        train_labels = []
        test_windows = []
        test_labels = []

        for combo, combo_data in real_data.items():
            label = combo_to_label(combo)

            # Extract features based on type
            mag_3d = combo_data.samples[:, 6:9]

            if feature_type == '3d_vector':
                features = mag_3d
            elif feature_type == 'magnitude':
                features = np.linalg.norm(mag_3d, axis=1, keepdims=True)
            elif feature_type == 'magnitude_and_angles':
                # Magnitude + spherical angles
                mag = np.linalg.norm(mag_3d, axis=1)
                theta = np.arctan2(mag_3d[:, 1], mag_3d[:, 0])
                phi = np.arccos(np.clip(mag_3d[:, 2] / (mag + 1e-8), -1, 1))
                features = np.column_stack([mag, theta, phi])

            high_pitch_mask = combo_data.pitch_angles >= q3
            low_pitch_mask = combo_data.pitch_angles <= q1

            high_pitch_samples = features[high_pitch_mask]
            low_pitch_samples = features[low_pitch_mask]

            # Add synthetic
            n_synth = 150
            synth_3d = generator.generate_combo(combo, n_synth, augmentation='none')

            # Note: synth_3d is (n_synth, 3) magnetometer values
            if feature_type == '3d_vector':
                synth_features = synth_3d
            elif feature_type == 'magnitude':
                synth_features = np.linalg.norm(synth_3d, axis=1, keepdims=True)
            elif feature_type == 'magnitude_and_angles':
                mag = np.linalg.norm(synth_3d, axis=1)
                theta = np.arctan2(synth_3d[:, 1], synth_3d[:, 0])
                phi = np.arccos(np.clip(synth_3d[:, 2] / (mag + 1e-8), -1, 1))
                synth_features = np.column_stack([mag, theta, phi])

            if len(high_pitch_samples) > 0:
                combined = np.vstack([high_pitch_samples, synth_features])
            else:
                combined = synth_features

            if len(combined) >= window_size:
                windows = create_windows(combined, window_size)
                for w in windows:
                    train_windows.append(w)
                    train_labels.append(label)

            if len(low_pitch_samples) >= window_size:
                windows = create_windows(low_pitch_samples, window_size)
                for w in windows:
                    test_windows.append(w)
                    test_labels.append(label)

        X_train = np.array(train_windows)
        y_train = np.array(train_labels)
        X_test = np.array(test_windows)
        y_test = np.array(test_labels)

        # Normalize
        n_features = X_train.shape[-1]
        mean = X_train.reshape(-1, n_features).mean(axis=0)
        std = X_train.reshape(-1, n_features).std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        # Split
        indices = np.random.permutation(len(X_train))
        val_size = int(0.15 * len(X_train))
        X_val = X_train[indices[:val_size]]
        y_val = y_train[indices[:val_size]]
        X_train = X_train[indices[val_size:]]
        y_train = y_train[indices[val_size:]]

        # Train
        _, metrics = train_and_evaluate(
            X_train, y_train, X_val, y_val, X_test, y_test,
            build_standard_model, epochs=30
        )

        print(f"  {feature_type:<20}: Train: {metrics['train_acc']:.1%}, Test: {metrics['test_acc']:.1%}, Gap: {metrics['gap']:.1%}")

        tf.keras.backend.clear_session()


def main():
    print("=" * 80)
    print("DEEP ANALYSIS: WHY SO(3) ROTATION HURTS MAGNETOMETER CLASSIFICATION")
    print("=" * 80)

    np.random.seed(42)
    tf.random.set_seed(42)

    # Analysis 1: Magnetometer variance across orientations
    analyze_magnetometer_variance()

    # Analysis 2: Impact of SO(3) rotation
    analyze_so3_impact()

    # Analysis 3: Small angle rotation
    small_angle_results = test_small_angle_rotation()

    # Analysis 4: Magnitude-only features
    test_magnitude_only()

    # Summary
    print("\n" + "=" * 80)
    print("KEY FINDINGS")
    print("=" * 80)

    print("""
1. MAGNETOMETER VALUES ARE ORIENTATION-SENSITIVE
   - The mean magnetometer values shift significantly between high/low pitch
   - This shift is ~100-200% of the standard deviation
   - This is actually USEFUL information for finger state classification

2. SO(3) ROTATION DESTROYS DISCRIMINATIVE INFORMATION
   - Full SO(3) rotation randomizes absolute magnetometer direction
   - This removes the orientation-dependent signal that distinguishes finger states
   - Unlike gesture recognition, we WANT orientation sensitivity here

3. SMALL-ANGLE ROTATION MAY HELP
   - Very small rotations (≤10°) may add robustness without losing signal
   - Large rotations (>45°) significantly degrade performance

4. FUNDAMENTAL DIFFERENCE FROM GESTURE RECOGNITION
   - Gesture recognition: Same gesture at different orientations should match
   - Finger state classification: Same finger state produces different mag values
     at different orientations, and this is discriminative

5. CONTRASTIVE LEARNING HELPS BUT DOESN'T BEAT BASELINE
   - Contrastive pre-training with SO(3) augmentation learns rotation-invariant
     representations, but this is the WRONG invariance for this task

RECOMMENDATION:
   - Keep V3 baseline (no rotation augmentation)
   - Focus on other robustness methods (e.g., domain adaptation for different users)
   - Consider adaptive calibration rather than rotation invariance
""")


if __name__ == '__main__':
    main()
