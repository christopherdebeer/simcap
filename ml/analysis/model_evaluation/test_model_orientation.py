#!/usr/bin/env python3
"""
Test deployed model's orientation generalization.

Loads the finger_aligned_v2 model and tests cross-orientation accuracy
to prove whether it achieves orientation invariance.
"""

import json
import numpy as np
from pathlib import Path
from collections import defaultdict
import tensorflow as tf


def load_model():
    """Load the deployed finger_aligned_v2 model."""
    model_dir = Path("public/models/finger_aligned_v2")

    # Load config
    with open(model_dir / "config.json") as f:
        config = json.load(f)

    # Load model
    model = tf.keras.models.load_model(model_dir / "model.keras")

    return model, config


def load_session_data(session_path: str = "data/GAMBIT/2025-12-31T14_06_18.270Z.json"):
    """Load labeled session data."""
    with open(session_path) as f:
        data = json.load(f)
    return data


def prepare_windows(data: dict, config: dict, window_size: int = 50):
    """
    Prepare windowed samples for the model.

    Returns dict mapping finger_code -> list of (window, label, pitch)
    """
    samples = data['samples']
    labels = data['labels']

    mean = np.array(config['stats']['mean'])
    std = np.array(config['stats']['std'])

    windows_by_code = defaultdict(list)

    for lbl in labels:
        fingers = lbl.get('labels', {}).get('fingers', lbl.get('fingers', {}))
        start = lbl.get('start_sample', lbl.get('startIndex', 0))
        end = lbl.get('end_sample', lbl.get('endIndex', 0))

        # Get finger states as binary (0=extended, 1=flexed)
        states = []
        code_parts = []
        for f in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            state = fingers.get(f, 'unknown')
            if state == 'extended':
                states.append(0)
                code_parts.append('0')
            elif state == 'flexed':
                states.append(1)
                code_parts.append('2')
            else:
                states.append(-1)
                code_parts.append('?')

        code = ''.join(code_parts)
        if '?' in code:
            continue

        label_array = np.array(states)

        # Extract windows within this label range
        for i in range(start, end - window_size + 1):
            window_samples = samples[i:i + window_size]

            # Check all samples have required fields
            if not all('ax_g' in s and 'mx_ut' in s and 'euler_pitch' in s
                      for s in window_samples):
                continue

            # Build feature array: [ax, ay, az, gx, gy, gz, mx, my, mz]
            features = []
            for s in window_samples:
                features.append([
                    s['ax_g'], s['ay_g'], s['az_g'],
                    s.get('gx_dps', 0), s.get('gy_dps', 0), s.get('gz_dps', 0),
                    s['mx_ut'], s['my_ut'], s['mz_ut']
                ])

            features = np.array(features)

            # Normalize
            features = (features - mean) / std

            # Get pitch from middle of window
            pitch = window_samples[window_size // 2].get('euler_pitch', 0)

            windows_by_code[code].append({
                'features': features,
                'label': label_array,
                'pitch': pitch,
            })

    return dict(windows_by_code)


def predict_batch(model, windows: list) -> np.ndarray:
    """Run prediction on batch of windows."""
    features = np.array([w['features'] for w in windows])
    predictions = model.predict(features, verbose=0)
    return predictions


def evaluate_accuracy(predictions: np.ndarray, labels: np.ndarray) -> dict:
    """Compute per-finger and overall accuracy."""
    # Predictions are (batch, 5) sigmoid outputs
    pred_binary = (predictions > 0.5).astype(int)

    # Per-finger accuracy
    per_finger = {}
    finger_names = ['thumb', 'index', 'middle', 'ring', 'pinky']
    for i, name in enumerate(finger_names):
        correct = np.sum(pred_binary[:, i] == labels[:, i])
        per_finger[name] = correct / len(labels)

    # Overall (all 5 fingers correct)
    all_correct = np.all(pred_binary == labels, axis=1)
    overall = np.mean(all_correct)

    return {
        'overall': overall,
        'per_finger': per_finger,
    }


def test_cross_orientation(model, windows_by_code: dict):
    """Test cross-orientation generalization."""

    # Collect all windows with pitch
    all_windows = []
    all_labels = []
    all_pitches = []
    all_codes = []

    for code, windows in windows_by_code.items():
        for w in windows:
            all_windows.append(w['features'])
            all_labels.append(w['label'])
            all_pitches.append(w['pitch'])
            all_codes.append(code)

    all_windows = np.array(all_windows)
    all_labels = np.array(all_labels)
    all_pitches = np.array(all_pitches)

    print(f"Total windows: {len(all_windows)}")
    print(f"Pitch range: {all_pitches.min():.1f}° to {all_pitches.max():.1f}°")

    # Split by pitch quartiles
    q1, q3 = np.percentile(all_pitches, [25, 75])
    print(f"Q1={q1:.1f}°, Q3={q3:.1f}°")

    # Q4 (high pitch) -> Q1 (low pitch)
    train_mask = all_pitches >= q3
    test_mask = all_pitches <= q1

    train_windows = all_windows[train_mask]
    train_labels = all_labels[train_mask]
    test_windows = all_windows[test_mask]
    test_labels = all_labels[test_mask]

    print(f"\nQ4→Q1 split:")
    print(f"  Train (pitch >= {q3:.1f}°): {len(train_windows)} windows")
    print(f"  Test (pitch <= {q1:.1f}°): {len(test_windows)} windows")

    # Since model is already trained, we just evaluate on test set
    # (The model was trained on different data, so this tests generalization)

    print("\n" + "="*60)
    print("MODEL EVALUATION ON CROSS-PITCH DATA")
    print("="*60)

    # Evaluate on all data first
    print("\n--- All data ---")
    all_preds = model.predict(all_windows, verbose=0)
    all_acc = evaluate_accuracy(all_preds, all_labels)
    print(f"  Overall: {all_acc['overall']:.1%}")
    for f, acc in all_acc['per_finger'].items():
        print(f"  {f}: {acc:.1%}")

    # Evaluate on high-pitch subset
    print(f"\n--- High pitch (>= {q3:.1f}°) ---")
    high_preds = model.predict(train_windows, verbose=0)
    high_acc = evaluate_accuracy(high_preds, train_labels)
    print(f"  Overall: {high_acc['overall']:.1%}")

    # Evaluate on low-pitch subset
    print(f"\n--- Low pitch (<= {q1:.1f}°) ---")
    low_preds = model.predict(test_windows, verbose=0)
    low_acc = evaluate_accuracy(low_preds, test_labels)
    print(f"  Overall: {low_acc['overall']:.1%}")

    # Compare
    print("\n" + "="*60)
    print("ORIENTATION INVARIANCE ANALYSIS")
    print("="*60)

    gap = high_acc['overall'] - low_acc['overall']
    print(f"\nHigh pitch accuracy: {high_acc['overall']:.1%}")
    print(f"Low pitch accuracy:  {low_acc['overall']:.1%}")
    print(f"Gap: {abs(gap):.1%}")

    if abs(gap) < 0.05:
        print("\n✓ Model is ORIENTATION INVARIANT")
        print("  Performance is consistent across pitch ranges")
    elif abs(gap) < 0.15:
        print("\n⚠ Model shows MODERATE orientation dependence")
    else:
        print("\n✗ Model shows SIGNIFICANT orientation dependence")

    # Compare with k-NN baseline
    print("\n" + "-"*60)
    print("Comparison with k-NN (from previous analysis):")
    print("  k-NN cross-pitch Q4→Q1: 61.6%")
    print(f"  CNN-LSTM low-pitch:    {low_acc['overall']:.1%}")

    improvement = low_acc['overall'] - 0.616
    if improvement > 0:
        print(f"\n  CNN-LSTM improves by: +{improvement:.1%}")
    else:
        print(f"\n  CNN-LSTM difference: {improvement:.1%}")

    return {
        'all': all_acc,
        'high_pitch': high_acc,
        'low_pitch': low_acc,
        'gap': gap,
    }


def main():
    print("="*60)
    print("TESTING DEPLOYED MODEL ORIENTATION INVARIANCE")
    print("="*60)

    # Load model
    print("\nLoading model...")
    model, config = load_model()
    print(f"Model: {config.get('description', 'unknown')}")
    print(f"Reported accuracy: {config.get('accuracy', {}).get('overall', 0):.1%}")

    # Load data
    print("\nLoading session data...")
    data = load_session_data()
    print(f"Samples: {len(data['samples'])}")
    print(f"Labels: {len(data['labels'])}")

    # Prepare windows
    print("\nPreparing windows...")
    windows_by_code = prepare_windows(data, config)

    total_windows = sum(len(v) for v in windows_by_code.values())
    print(f"Total windows: {total_windows}")
    print(f"Classes: {len(windows_by_code)}")

    for code in sorted(windows_by_code.keys()):
        print(f"  {code}: {len(windows_by_code[code])} windows")

    # Test cross-orientation
    results = test_cross_orientation(model, windows_by_code)

    return results


if __name__ == "__main__":
    results = main()
