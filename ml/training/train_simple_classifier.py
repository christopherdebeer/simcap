"""
Simple Finger State Classifier (NumPy only)

Uses nearest-centroid classification - finds the closest class centroid
for each test sample. Simple but effective given the high separability
of finger magnet signatures.
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple
from collections import defaultdict


def load_session(path: Path) -> Dict:
    """Load a session JSON file."""
    try:
        with open(path) as f:
            content = f.read().strip()
            if not content or content == '{}':
                return {}
            return json.loads(content)
    except (json.JSONDecodeError, Exception) as e:
        print(f"  Warning: Could not load {path.name}: {e}")
        return {}


def finger_code(fingers: Dict) -> str:
    """Convert finger states dict to 5-char code."""
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


def extract_data(session: Dict) -> Tuple[np.ndarray, List[str]]:
    """Extract magnetometer data and labels from session."""
    samples = session.get('samples', [])
    labels = session.get('labels', [])

    # Extract magnetometer
    mx = np.array([s.get('mx', 0) for s in samples])
    my = np.array([s.get('my', 0) for s in samples])
    mz = np.array([s.get('mz', 0) for s in samples])

    X = np.column_stack([mx, my, mz])
    y = [''] * len(samples)

    # Apply labels
    for label in labels:
        start = label.get('start_sample', label.get('startIndex', 0))
        end = label.get('end_sample', label.get('endIndex', 0))
        content = label.get('labels', label)
        fingers = content.get('fingers', {})

        if not fingers:
            continue

        code = finger_code(fingers)
        if '?' not in code:
            for i in range(start, min(end, len(y))):
                y[i] = code

    # Filter to labeled only
    mask = np.array([yi != '' for yi in y])
    X = X[mask]
    y = [yi for yi in y if yi != '']

    return X, y


class NearestCentroidClassifier:
    """Simple nearest centroid classifier."""

    def __init__(self):
        self.centroids = {}
        self.classes = []

    def fit(self, X: np.ndarray, y: List[str]):
        """Compute centroid for each class."""
        # Group by class
        class_samples = defaultdict(list)
        for i, label in enumerate(y):
            class_samples[label].append(X[i])

        # Compute centroids
        for label, samples in class_samples.items():
            self.centroids[label] = np.mean(samples, axis=0)

        self.classes = sorted(self.centroids.keys())
        print(f"Fitted {len(self.classes)} classes")

    def predict(self, X: np.ndarray) -> List[str]:
        """Predict class for each sample."""
        predictions = []

        for x in X:
            # Find closest centroid
            min_dist = float('inf')
            best_class = self.classes[0]

            for label, centroid in self.centroids.items():
                dist = np.linalg.norm(x - centroid)
                if dist < min_dist:
                    min_dist = dist
                    best_class = label

            predictions.append(best_class)

        return predictions

    def score(self, X: np.ndarray, y: List[str]) -> float:
        """Compute accuracy."""
        predictions = self.predict(X)
        correct = sum(p == t for p, t in zip(predictions, y))
        return correct / len(y)


class KNNClassifier:
    """K-Nearest Neighbors classifier."""

    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, X: np.ndarray, y: List[str]):
        """Store training data."""
        self.X_train = X
        self.y_train = y
        print(f"Stored {len(y)} training samples")

    def predict(self, X: np.ndarray) -> List[str]:
        """Predict using k nearest neighbors."""
        predictions = []

        for x in X:
            # Compute distances to all training samples
            dists = np.linalg.norm(self.X_train - x, axis=1)

            # Find k nearest
            k_nearest = np.argsort(dists)[:self.k]
            k_labels = [self.y_train[i] for i in k_nearest]

            # Majority vote
            label_counts = defaultdict(int)
            for label in k_labels:
                label_counts[label] += 1

            best_label = max(label_counts.keys(), key=lambda k: label_counts[k])
            predictions.append(best_label)

        return predictions

    def score(self, X: np.ndarray, y: List[str]) -> float:
        """Compute accuracy."""
        predictions = self.predict(X)
        correct = sum(p == t for p, t in zip(predictions, y))
        return correct / len(y)


def train_test_split(X: np.ndarray, y: List[str], test_ratio: float = 0.2,
                     seed: int = 42) -> Tuple[np.ndarray, np.ndarray, List[str], List[str]]:
    """Split data into train and test sets."""
    np.random.seed(seed)
    n = len(y)
    indices = np.random.permutation(n)

    n_test = int(n * test_ratio)
    test_idx = indices[:n_test]
    train_idx = indices[n_test:]

    X_train = X[train_idx]
    X_test = X[test_idx]
    y_train = [y[i] for i in train_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, X_test, y_train, y_test


def compute_confusion_matrix(y_true: List[str], y_pred: List[str],
                            classes: List[str]) -> np.ndarray:
    """Compute confusion matrix."""
    n = len(classes)
    class_to_idx = {c: i for i, c in enumerate(classes)}

    matrix = np.zeros((n, n), dtype=int)
    for true, pred in zip(y_true, y_pred):
        if true in class_to_idx and pred in class_to_idx:
            matrix[class_to_idx[true], class_to_idx[pred]] += 1

    return matrix


def per_class_accuracy(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Compute per-class accuracy."""
    class_correct = defaultdict(int)
    class_total = defaultdict(int)

    for true, pred in zip(y_true, y_pred):
        class_total[true] += 1
        if true == pred:
            class_correct[true] += 1

    return {c: class_correct[c] / class_total[c]
            for c in sorted(class_total.keys())}


def main():
    print("=" * 80)
    print("SIMPLE FINGER STATE CLASSIFIER")
    print("=" * 80)

    # Load real session data
    data_dir = Path('data/GAMBIT')
    all_X = []
    all_y = []

    print("\nLoading real sessions...")
    if data_dir.exists():
        for path in data_dir.glob('*.json'):
            session = load_session(path)
            labels = session.get('labels', [])

            # Check if has finger labels
            has_fingers = any(
                label.get('labels', label).get('fingers')
                for label in labels
            )

            if has_fingers and len(labels) > 5:
                X, y = extract_data(session)
                if len(y) > 0:
                    all_X.append(X)
                    all_y.extend(y)
                    print(f"  {path.name}: {len(y)} samples")

    # Load synthetic data
    synthetic_path = Path('ml/synthetic_balanced_dataset.json')
    if synthetic_path.exists():
        print(f"\nLoading synthetic: {synthetic_path.name}")
        session = load_session(synthetic_path)
        X, y = extract_data(session)
        if len(y) > 0:
            all_X.append(X)
            all_y.extend(y)
            print(f"  {len(y)} samples")

    if not all_X:
        print("No data found!")
        return

    X = np.vstack(all_X)
    y = all_y

    print(f"\nTotal: {len(y)} samples")

    # Class distribution
    class_counts = defaultdict(int)
    for yi in y:
        class_counts[yi] += 1

    print("\nClass distribution:")
    for code, count in sorted(class_counts.items()):
        print(f"  {code}: {count} ({count/len(y)*100:.1f}%)")

    # Split data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_ratio=0.2)
    print(f"\nTrain: {len(y_train)}, Test: {len(y_test)}")

    # Normalize (z-score)
    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0) + 1e-6
    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    # 1. Nearest Centroid
    print("\n" + "=" * 60)
    print("NEAREST CENTROID CLASSIFIER")
    print("=" * 60)

    nc = NearestCentroidClassifier()
    nc.fit(X_train_norm, y_train)

    train_acc = nc.score(X_train_norm, y_train)
    test_acc = nc.score(X_test_norm, y_test)

    print(f"\nTrain accuracy: {train_acc:.4f}")
    print(f"Test accuracy:  {test_acc:.4f}")

    y_pred_nc = nc.predict(X_test_norm)
    per_class = per_class_accuracy(y_test, y_pred_nc)
    print("\nPer-class accuracy:")
    for code, acc in per_class.items():
        print(f"  {code}: {acc:.4f}")

    # 2. K-Nearest Neighbors
    print("\n" + "=" * 60)
    print("K-NEAREST NEIGHBORS (k=5)")
    print("=" * 60)

    knn = KNNClassifier(k=5)
    knn.fit(X_train_norm, y_train)

    # Use smaller test set for speed
    n_eval = min(500, len(y_test))
    test_acc = knn.score(X_test_norm[:n_eval], y_test[:n_eval])
    print(f"\nTest accuracy (n={n_eval}): {test_acc:.4f}")

    y_pred_knn = knn.predict(X_test_norm[:n_eval])
    per_class = per_class_accuracy(y_test[:n_eval], y_pred_knn)
    print("\nPer-class accuracy:")
    for code, acc in per_class.items():
        print(f"  {code}: {acc:.4f}")

    # 3. Confusion matrix
    print("\n" + "=" * 60)
    print("CONFUSION MATRIX (Nearest Centroid)")
    print("=" * 60)

    classes = sorted(set(y))
    cm = compute_confusion_matrix(y_test, y_pred_nc, classes)

    print("\n" + " " * 8 + "  ".join(c[-3:] for c in classes))
    for i, code in enumerate(classes):
        row = "  ".join(f"{cm[i,j]:3d}" for j in range(len(classes)))
        print(f"{code[-3:]}: {row}")

    # Save model
    print("\n" + "=" * 60)
    print("SAVING MODEL")
    print("=" * 60)

    model_data = {
        'type': 'nearest_centroid',
        'centroids': {k: v.tolist() for k, v in nc.centroids.items()},
        'normalization': {
            'mean': mean.tolist(),
            'std': std.tolist()
        },
        'classes': classes,
        'test_accuracy': float(test_acc),
        'train_samples': len(y_train),
        'test_samples': len(y_test)
    }

    output_path = Path('ml/models/finger_classifier_simple.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w') as f:
        json.dump(model_data, f, indent=2)

    print(f"Saved model to: {output_path}")

    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"""
    Classes: {len(classes)}
    Training samples: {len(y_train)}
    Test samples: {len(y_test)}

    Nearest Centroid accuracy: {nc.score(X_test_norm, y_test):.4f}
    KNN (k=5) accuracy: {test_acc:.4f}

    Model saved to: {output_path}
    """)


if __name__ == '__main__':
    main()
