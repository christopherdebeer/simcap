"""Quick orientation invariance test for best configurations."""

import numpy as np
import json
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Reuse functions from ablation study
from ml.ablation_small_windows import (
    load_labeled_data, prepare_windows, build_model,
    FEATURE_SETS
)


def orientation_test(samples, metadata, fs_name, window_size):
    """Test orientation invariance for a configuration."""
    import tensorflow as tf

    fs = FEATURE_SETS[fs_name]
    X, y, classes, pitches = prepare_windows(
        samples, metadata['index_to_code'],
        window_size, fs, stride=max(1, window_size // 3)
    )

    if len(X) < 50:
        return None

    n_features = X.shape[2]
    n_classes = len(classes)

    # Pitch quartile split
    q1 = np.percentile(pitches, 25)
    q3 = np.percentile(pitches, 75)

    high_mask = pitches >= q3
    low_mask = pitches <= q1

    X_train, y_train = X[high_mask], y[high_mask]
    X_test, y_test = X[low_mask], y[low_mask]

    if len(X_train) < 20 or len(X_test) < 20:
        return None

    # Normalize
    mean = X_train.reshape(-1, n_features).mean(axis=0)
    std = X_train.reshape(-1, n_features).std(axis=0) + 1e-8
    X_train_n = (X_train - mean) / std
    X_test_n = (X_test - mean) / std

    # Train
    model = build_model(window_size, n_features, n_classes)

    model.fit(
        X_train_n, y_train,
        validation_split=0.15,
        epochs=50,
        batch_size=min(32, len(X_train) // 4),
        verbose=0
    )

    _, train_acc = model.evaluate(X_train_n, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test_n, y_test, verbose=0)

    tf.keras.backend.clear_session()

    return {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': train_acc - test_acc
    }


def main():
    print("=" * 70)
    print("ORIENTATION INVARIANCE TEST")
    print("=" * 70)

    data_dir = Path("data/GAMBIT")
    if not data_dir.exists():
        data_dir = Path(".worktrees/data/GAMBIT")

    samples, metadata = load_labeled_data(data_dir)
    if not samples:
        print("No data!")
        return

    print(f"\nSession: {metadata['session']}")
    print(f"Pitch range: testing Q4 (high) → Q1 (low)\n")

    # Test configurations
    configs = [
        ('mag_only', 1),
        ('mag_only', 2),
        ('mag_only', 5),
        ('mag_only', 12),
        ('iron_mag', 2),
        ('iron_mag', 12),
        ('raw_9dof', 2),
        ('raw_9dof', 5),
        ('mag_euler', 2),
        ('mag_euler', 5),
        ('iron_euler', 2),
        ('iron_euler', 5),
    ]

    print(f"{'Config':<20} {'Train':>8} {'Test':>8} {'Gap':>8} {'Samples'}")
    print("-" * 60)

    results = {}
    for fs_name, ws in configs:
        key = f"{fs_name}_w{ws}"
        result = orientation_test(samples, metadata, fs_name, ws)

        if result:
            print(f"{key:<20} {result['train_acc']:>7.1%} {result['test_acc']:>7.1%} "
                  f"{result['gap']:>7.1%}   ({result['train_samples']}/{result['test_samples']})")
            results[key] = result
        else:
            print(f"{key:<20} insufficient data")

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    if results:
        best_oi = min(results.items(), key=lambda x: abs(x[1]['gap']))
        best_test = max(results.items(), key=lambda x: x[1]['test_acc'])

        print(f"\nBest orientation invariance: {best_oi[0]}")
        print(f"  Gap: {best_oi[1]['gap']:.1%}, Test: {best_oi[1]['test_acc']:.1%}")

        print(f"\nBest cross-orientation accuracy: {best_test[0]}")
        print(f"  Test: {best_test[1]['test_acc']:.1%}, Gap: {best_test[1]['gap']:.1%}")

        # Compare with/without Euler
        mag_only = results.get('mag_only_w5')
        mag_euler = results.get('mag_euler_w5')
        if mag_only and mag_euler:
            print(f"\nEffect of adding Euler angles (w=5):")
            print(f"  mag_only: gap={mag_only['gap']:.1%}, test={mag_only['test_acc']:.1%}")
            print(f"  mag_euler: gap={mag_euler['gap']:.1%}, test={mag_euler['test_acc']:.1%}")

    # Save
    with open("ml/orientation_test_results.json", 'w') as f:
        json.dump(results, f, indent=2)
    print("\nSaved to ml/orientation_test_results.json")


if __name__ == "__main__":
    main()
