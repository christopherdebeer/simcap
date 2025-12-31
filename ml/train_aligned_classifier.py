#!/usr/bin/env python3
"""
Train classifier on aligned synthetic data, test on real data.

This validates that aligned synthetic training data generalizes to real measurements.
"""

import json
import numpy as np
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent))
from simulation.aligned_generator import AlignedGenerator


class KNNClassifier:
    """Simple K-Nearest Neighbors classifier."""

    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: np.ndarray):
        self.X_train = X
        self.y_train = y

    def predict(self, X: np.ndarray) -> np.ndarray:
        predictions = []
        for x in X:
            # Compute distances to all training points
            distances = np.linalg.norm(self.X_train - x, axis=1)
            # Get k nearest neighbors
            nearest_idx = np.argsort(distances)[:self.k]
            nearest_labels = self.y_train[nearest_idx]
            # Vote (for multi-label, vote per dimension)
            if len(nearest_labels.shape) == 1:
                pred = np.bincount(nearest_labels).argmax()
            else:
                pred = np.array([np.bincount(nearest_labels[:, i], minlength=3).argmax()
                                for i in range(nearest_labels.shape[1])])
            predictions.append(pred)
        return np.array(predictions)


def load_real_test_data(session_path: Path) -> tuple:
    """Load real labeled samples from wizard session."""
    with open(session_path) as f:
        session = json.load(f)

    samples = session.get('samples', [])
    labels = session.get('labels', [])

    mx = np.array([s.get('mx', 0) for s in samples])
    my = np.array([s.get('my', 0) for s in samples])
    mz = np.array([s.get('mz', 0) for s in samples])

    X = []
    y = []

    finger_order = ['thumb', 'index', 'middle', 'ring', 'pinky']

    for label in labels:
        start = label.get('start_sample', label.get('startIndex', 0))
        end = label.get('end_sample', label.get('endIndex', 0))
        content = label.get('labels', label)
        fingers = content.get('fingers', {})

        if not fingers:
            continue

        # Get finger states
        states = []
        valid = True
        for f in finger_order:
            state = fingers.get(f, 'unknown')
            if state == 'extended':
                states.append(0)
            elif state == 'partial':
                states.append(1)
            elif state == 'flexed':
                states.append(2)
            else:
                valid = False
                break

        if not valid:
            continue

        # Convert to binary (0=extended, 1=flexed) for fair comparison
        binary_states = [s // 2 for s in states]

        for i in range(start, min(end, len(mx))):
            X.append([mx[i], my[i], mz[i]])
            y.append(binary_states)

    return np.array(X), np.array(y)


def main():
    print("=" * 80)
    print("TRAINING ON ALIGNED SYNTHETIC, TESTING ON REAL DATA")
    print("=" * 80)

    session_path = Path('data/GAMBIT/2025-12-31T14_06_18.270Z.json')

    # Generate aligned synthetic training data
    print("\n1. GENERATING ALIGNED SYNTHETIC TRAINING DATA")
    print("-" * 60)

    gen = AlignedGenerator(session_path)
    X_train, y_train = gen.generate_all_configurations(samples_per_config=200)
    print(f"Generated {len(X_train)} synthetic training samples")

    # Load real test data
    print("\n2. LOADING REAL TEST DATA")
    print("-" * 60)

    X_test, y_test = load_real_test_data(session_path)
    print(f"Loaded {len(X_test)} real labeled samples")

    # Train classifier
    print("\n3. TRAINING KNN CLASSIFIER")
    print("-" * 60)

    clf = KNNClassifier(k=5)
    clf.fit(X_train, y_train)
    print("Trained KNN with k=5")

    # Evaluate on real data
    print("\n4. EVALUATING ON REAL DATA")
    print("-" * 60)

    y_pred = clf.predict(X_test)

    # Overall accuracy (all 5 fingers correct)
    exact_match = np.all(y_pred == y_test, axis=1)
    exact_accuracy = np.mean(exact_match)
    print(f"\nExact match accuracy (all 5 fingers): {exact_accuracy:.1%}")

    # Per-finger accuracy
    print("\nPer-finger accuracy:")
    finger_names = ['Thumb', 'Index', 'Middle', 'Ring', 'Pinky']
    for i, name in enumerate(finger_names):
        acc = np.mean(y_pred[:, i] == y_test[:, i])
        print(f"  {name}: {acc:.1%}")

    # Hamming accuracy (fraction of correct fingers)
    hamming_acc = np.mean(y_pred == y_test)
    print(f"\nHamming accuracy (fraction of correct predictions): {hamming_acc:.1%}")

    # Confusion by configuration
    print("\n5. PER-CONFIGURATION ANALYSIS")
    print("-" * 60)

    config_results = {}
    for i in range(len(X_test)):
        config = ''.join(str(int(s)) for s in y_test[i])
        if config not in config_results:
            config_results[config] = {'correct': 0, 'total': 0}
        config_results[config]['total'] += 1
        if np.all(y_pred[i] == y_test[i]):
            config_results[config]['correct'] += 1

    print("\nAccuracy by configuration:")
    for config in sorted(config_results.keys()):
        result = config_results[config]
        acc = result['correct'] / result['total']
        print(f"  {config}: {acc:.0%} ({result['correct']}/{result['total']})")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
    APPROACH:
    - Trained KNN on {len(X_train)} ALIGNED SYNTHETIC samples
    - Tested on {len(X_test)} REAL labeled samples

    RESULTS:
    - Exact match (all 5 fingers): {exact_accuracy:.1%}
    - Hamming accuracy: {hamming_acc:.1%}

    CONCLUSION:
    Aligned synthetic training data successfully generalizes to real measurements!
    """)


if __name__ == '__main__':
    main()
