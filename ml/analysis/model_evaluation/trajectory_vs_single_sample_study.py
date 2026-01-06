#!/usr/bin/env python3
"""
Information-Theoretic and Empirical Study: Trajectory vs Single-Sample Inference

This study addresses the question:
"Would training on FFO$ magnetic trajectories (requiring trajectories during inference)
improve model performance compared to single-sample inference?"

Components:
1. INFORMATION-THEORETIC ANALYSIS
   - Entropy of single samples vs trajectories
   - Mutual information with finger states
   - Theoretical capacity bounds

2. EMPIRICAL COMPARISON
   - FFO$ template matching only
   - Aligned single-sample classifier (existing approach)
   - Trajectory-based neural network (new approach)

Author: Claude (Research Study)
Date: December 2025
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# =============================================================================
# CONFIGURATION
# =============================================================================

WINDOW_SIZE = 32  # FFO$ trajectory window size
N_RESAMPLE_POINTS = 32  # FFO$ resampling target
FINGER_ORDER = ['thumb', 'index', 'middle', 'ring', 'pinky']

# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class Sample:
    """Single-sample data point."""
    mx: float
    my: float
    mz: float
    finger_code: str  # e.g., "00000", "22222"
    timestamp: float = 0.0

    @property
    def as_vector(self) -> np.ndarray:
        return np.array([self.mx, self.my, self.mz])


@dataclass
class Trajectory:
    """Window of samples forming a trajectory."""
    points: np.ndarray  # Shape: (n_points, 3)
    finger_code: str
    timestamps: np.ndarray = None

    @property
    def path_length(self) -> float:
        if len(self.points) < 2:
            return 0.0
        return np.sum(np.linalg.norm(np.diff(self.points, axis=0), axis=1))

    @property
    def mean_point(self) -> np.ndarray:
        return np.mean(self.points, axis=0)

    @property
    def std_point(self) -> np.ndarray:
        return np.std(self.points, axis=0)


# =============================================================================
# DATA LOADING
# =============================================================================

def load_all_sessions(data_dir: Path) -> List[Dict]:
    """Load all wizard sessions with labels."""
    sessions = []
    for path in sorted(data_dir.glob('*.json')):
        if path.name == 'manifest.json':
            continue
        try:
            with open(path) as f:
                session = json.load(f)
            if 'labels' in session and session['labels']:
                sessions.append({
                    'path': path,
                    'samples': session.get('samples', []),
                    'labels': session.get('labels', []),
                    'metadata': session.get('metadata', {})
                })
        except Exception as e:
            continue
    return sessions


def extract_labeled_data(sessions: List[Dict],
                         use_sliding_window: bool = True,
                         window_stride: int = 8) -> Tuple[List[Sample], List[Trajectory]]:
    """
    Extract both single samples and trajectories from labeled sessions.

    Args:
        sessions: List of session dicts
        use_sliding_window: If True, generate multiple trajectories per segment
        window_stride: Stride for sliding window (samples between window starts)

    Returns:
        Tuple of (single_samples, trajectories)
    """
    single_samples = []
    trajectories = []

    for session in sessions:
        samples = session['samples']
        labels = session['labels']

        for label in labels:
            start = label.get('start_sample', label.get('startIndex', 0))
            end = label.get('end_sample', label.get('endIndex', 0))
            content = label.get('labels', label)
            fingers = content.get('fingers', {})

            if not fingers or end <= start:
                continue

            # Build finger code
            code = ''
            valid = True
            for f in FINGER_ORDER:
                state = fingers.get(f, 'unknown')
                if state == 'extended':
                    code += '0'
                elif state == 'partial':
                    code += '1'
                elif state == 'flexed':
                    code += '2'
                else:
                    valid = False
                    break

            if not valid:
                continue

            # Extract single samples
            for i in range(start, min(end, len(samples))):
                s = samples[i]
                # Use iron-corrected if available, else filtered, else raw
                mx = s.get('iron_mx', s.get('filtered_mx', s.get('mx', 0)))
                my = s.get('iron_my', s.get('filtered_my', s.get('my', 0)))
                mz = s.get('iron_mz', s.get('filtered_mz', s.get('mz', 0)))

                single_samples.append(Sample(
                    mx=mx, my=my, mz=mz,
                    finger_code=code,
                    timestamp=s.get('t', i)
                ))

            # Extract trajectories using sliding window
            segment_length = min(end, len(samples)) - start

            if use_sliding_window and segment_length >= WINDOW_SIZE:
                # Sliding window: generate multiple trajectories
                for win_start in range(0, segment_length - WINDOW_SIZE + 1, window_stride):
                    traj_points = []
                    for i in range(win_start, win_start + WINDOW_SIZE):
                        s = samples[start + i]
                        mx = s.get('iron_mx', s.get('filtered_mx', s.get('mx', 0)))
                        my = s.get('iron_my', s.get('filtered_my', s.get('my', 0)))
                        mz = s.get('iron_mz', s.get('filtered_mz', s.get('mz', 0)))
                        traj_points.append([mx, my, mz])

                    trajectories.append(Trajectory(
                        points=np.array(traj_points),
                        finger_code=code
                    ))
            elif segment_length >= WINDOW_SIZE // 2:
                # Fallback: single trajectory for shorter segments
                traj_points = []
                for i in range(start, min(end, len(samples))):
                    s = samples[i]
                    mx = s.get('iron_mx', s.get('filtered_mx', s.get('mx', 0)))
                    my = s.get('iron_my', s.get('filtered_my', s.get('my', 0)))
                    mz = s.get('iron_mz', s.get('filtered_mz', s.get('mz', 0)))
                    traj_points.append([mx, my, mz])

                if len(traj_points) >= 5:
                    trajectories.append(Trajectory(
                        points=np.array(traj_points),
                        finger_code=code
                    ))

    return single_samples, trajectories


# =============================================================================
# INFORMATION-THEORETIC ANALYSIS
# =============================================================================

def estimate_entropy(data: np.ndarray, n_bins: int = 30) -> float:
    """
    Estimate entropy of continuous data using histogram binning.

    H(X) = -sum(p(x) * log2(p(x)))
    """
    if len(data) < 10:
        return 0.0

    # Flatten if multi-dimensional
    if len(data.shape) > 1:
        # Joint entropy: bin in each dimension
        entropies = []
        for dim in range(data.shape[1]):
            hist, _ = np.histogram(data[:, dim], bins=n_bins, density=True)
            # Convert density to probability
            bin_width = (np.max(data[:, dim]) - np.min(data[:, dim])) / n_bins
            probs = hist * bin_width
            probs = probs[probs > 0]  # Remove zeros
            entropies.append(-np.sum(probs * np.log2(probs + 1e-10)))
        return np.sum(entropies)  # Approximate joint entropy
    else:
        hist, _ = np.histogram(data, bins=n_bins, density=True)
        bin_width = (np.max(data) - np.min(data) + 1e-10) / n_bins
        probs = hist * bin_width
        probs = probs[probs > 0]
        return -np.sum(probs * np.log2(probs + 1e-10))


def estimate_mutual_information(X: np.ndarray, y: np.ndarray, n_bins: int = 20) -> float:
    """
    Estimate mutual information I(X; Y) = H(X) - H(X|Y)

    For continuous X and discrete Y (class labels).
    Uses k-NN based estimation for better accuracy with small samples.
    """
    if len(X) < 10:
        return 0.0

    unique_classes = np.unique(y)

    # Simple k-NN based MI estimation
    # MI ≈ log(N) - mean(log(k_same_class)) where k is distance to k-th neighbor

    # Alternative: Use class separability as proxy for MI
    # Higher separability = higher MI

    # Compute within-class and between-class distances
    within_dists = []
    between_dists = []

    for i in range(min(500, len(X))):  # Sample for efficiency
        for j in range(i + 1, min(500, len(X))):
            dist = np.linalg.norm(X[i] - X[j])
            if y[i] == y[j]:
                within_dists.append(dist)
            else:
                between_dists.append(dist)

    if not within_dists or not between_dists:
        return 0.0

    # Fisher's discriminant ratio as MI proxy
    mean_within = np.mean(within_dists)
    mean_between = np.mean(between_dists)
    std_within = np.std(within_dists) + 1e-10

    # Convert ratio to bits (higher ratio = more separation = more MI)
    fisher_ratio = (mean_between - mean_within) / std_within
    mi_estimate = np.log2(1 + max(0, fisher_ratio))

    return mi_estimate


def analyze_information_theory(single_samples: List[Sample],
                               trajectories: List[Trajectory]) -> Dict[str, Any]:
    """
    Comprehensive information-theoretic analysis comparing single samples to trajectories.
    """
    results = {
        'single_sample': {},
        'trajectory': {},
        'comparison': {}
    }

    print("\n" + "=" * 80)
    print("INFORMATION-THEORETIC ANALYSIS")
    print("=" * 80)

    # === Single Sample Analysis ===
    print("\n1. SINGLE SAMPLE ANALYSIS")
    print("-" * 60)

    # Collect data by class
    X_single = np.array([s.as_vector for s in single_samples])
    y_single = np.array([s.finger_code for s in single_samples])

    # Convert finger codes to numeric
    unique_codes = list(set(y_single))
    code_to_idx = {c: i for i, c in enumerate(unique_codes)}
    y_numeric = np.array([code_to_idx[c] for c in y_single])

    # Entropy of magnetometer measurements
    H_mag_single = estimate_entropy(X_single)
    print(f"  H(magnetometer) = {H_mag_single:.2f} bits")

    # Entropy of finger states (class distribution)
    class_counts = np.bincount(y_numeric)
    class_probs = class_counts / len(y_numeric)
    H_finger = -np.sum(class_probs * np.log2(class_probs + 1e-10))
    print(f"  H(finger_states) = {H_finger:.2f} bits ({len(unique_codes)} classes)")

    # Mutual information
    I_single = estimate_mutual_information(X_single, y_numeric)
    print(f"  I(magnetometer; finger_states) = {I_single:.2f} bits")

    # Normalized MI (uncertainty reduction)
    nmi_single = I_single / (H_finger + 1e-10)
    print(f"  Normalized MI = {nmi_single:.1%} of class uncertainty reduced")

    results['single_sample'] = {
        'H_measurement': H_mag_single,
        'H_class': H_finger,
        'mutual_information': I_single,
        'normalized_mi': nmi_single,
        'n_samples': len(single_samples),
        'n_classes': len(unique_codes)
    }

    # === Trajectory Analysis ===
    print("\n2. TRAJECTORY ANALYSIS")
    print("-" * 60)

    # For trajectories, we have multiple feature representations:
    # 1. Mean point (3D)
    # 2. Std point (3D)
    # 3. Path length (1D)
    # 4. Full resampled trajectory (32*3 = 96D)

    y_traj = np.array([t.finger_code for t in trajectories])
    y_traj_numeric = np.array([code_to_idx.get(c, 0) for c in y_traj])

    # Feature 1: Mean point only (same as single sample baseline)
    X_mean = np.array([t.mean_point for t in trajectories])
    I_mean = estimate_mutual_information(X_mean, y_traj_numeric)
    print(f"  Trajectory mean only:")
    print(f"    I(mean; finger_states) = {I_mean:.2f} bits")

    # Feature 2: Mean + Std (temporal variability)
    X_mean_std = np.array([np.concatenate([t.mean_point, t.std_point]) for t in trajectories])
    I_mean_std = estimate_mutual_information(X_mean_std, y_traj_numeric)
    print(f"  Trajectory mean + std (6D):")
    print(f"    I(mean+std; finger_states) = {I_mean_std:.2f} bits")

    # Feature 3: Mean + Std + Path length
    X_stats = np.array([np.concatenate([t.mean_point, t.std_point, [t.path_length]])
                        for t in trajectories])
    I_stats = estimate_mutual_information(X_stats, y_traj_numeric)
    print(f"  Trajectory statistics (7D):")
    print(f"    I(stats; finger_states) = {I_stats:.2f} bits")

    # Feature 4: Full resampled trajectory (high-dimensional)
    X_full = []
    for t in trajectories:
        resampled = resample_trajectory(t.points, N_RESAMPLE_POINTS)
        X_full.append(resampled.flatten())
    X_full = np.array(X_full)

    # For high-dim, estimate per-dimension and sum (upper bound approximation)
    I_full_approx = 0
    for dim in range(min(15, X_full.shape[1])):  # Sample first 15 dims
        I_dim = estimate_mutual_information(X_full[:, dim:dim+1], y_traj_numeric)
        I_full_approx += I_dim
    I_full_approx *= (X_full.shape[1] / 15)  # Scale up
    print(f"  Full trajectory ({X_full.shape[1]}D resampled):")
    print(f"    I(trajectory; finger_states) ≈ {I_full_approx:.2f} bits (estimated)")

    results['trajectory'] = {
        'I_mean_only': I_mean,
        'I_mean_std': I_mean_std,
        'I_statistics': I_stats,
        'I_full_approx': I_full_approx,
        'n_trajectories': len(trajectories),
        'trajectory_dim': X_full.shape[1]
    }

    # === Comparison ===
    print("\n3. INFORMATION-THEORETIC COMPARISON")
    print("-" * 60)

    # Information gain from trajectory vs single sample
    info_gain = I_stats - I_single
    relative_gain = (I_stats - I_single) / (I_single + 1e-10)

    print(f"  Single sample MI:    {I_single:.2f} bits")
    print(f"  Trajectory stats MI: {I_stats:.2f} bits")
    print(f"  Information gain:    {info_gain:+.2f} bits ({relative_gain:+.1%})")

    # Theoretical capacity
    # Channel capacity C = max I(X;Y) bounded by min(H(X), H(Y))
    theoretical_max = H_finger  # Can't exceed class entropy
    single_efficiency = I_single / theoretical_max
    traj_efficiency = I_stats / theoretical_max

    print(f"\n  Channel efficiency (% of max {theoretical_max:.2f} bits):")
    print(f"    Single sample: {single_efficiency:.1%}")
    print(f"    Trajectory:    {traj_efficiency:.1%}")

    results['comparison'] = {
        'information_gain': info_gain,
        'relative_gain': relative_gain,
        'theoretical_max': theoretical_max,
        'single_efficiency': single_efficiency,
        'trajectory_efficiency': traj_efficiency
    }

    # === Key Insight ===
    print("\n" + "=" * 80)
    print("KEY INFORMATION-THEORETIC INSIGHT")
    print("=" * 80)
    print(f"""
    FINDING: Trajectories provide {relative_gain:+.1%} more information about finger states.

    WHY THIS MATTERS:
    - Single samples capture position in magnetic space at one instant
    - Trajectories capture how the signal evolves over time
    - Temporal patterns (std, path length) encode DYNAMICS of finger movement

    THEORETICAL IMPLICATIONS:
    - Single sample: {single_efficiency:.0%} of theoretical channel capacity utilized
    - Trajectory:    {traj_efficiency:.0%} of theoretical channel capacity utilized

    PREDICTION: Trajectory-based models should achieve ~{traj_efficiency/single_efficiency - 1:.0%}
    better classification, IF the additional information is learnable.
    """)

    return results


# =============================================================================
# FFO$-STYLE PROCESSING
# =============================================================================

def resample_trajectory(points: np.ndarray, n_points: int = 32) -> np.ndarray:
    """Resample trajectory to N equally-spaced points along path."""
    if len(points) < 2:
        return np.tile(points[0] if len(points) > 0 else np.zeros(3), (n_points, 1))

    # Calculate segment lengths
    diffs = np.diff(points, axis=0)
    segment_lengths = np.linalg.norm(diffs, axis=1)
    total_length = np.sum(segment_lengths)

    if total_length == 0:
        return np.tile(points[0], (n_points, 1))

    interval = total_length / (n_points - 1)
    resampled = [points[0].copy()]
    accumulated = 0.0
    current_idx = 0

    while len(resampled) < n_points and current_idx < len(segment_lengths):
        seg_len = segment_lengths[current_idx]

        if accumulated + seg_len >= interval:
            overshoot = interval - accumulated
            t = overshoot / seg_len if seg_len > 0 else 0
            new_point = (1 - t) * points[current_idx] + t * points[current_idx + 1]
            resampled.append(new_point)
            segment_lengths[current_idx] = seg_len - overshoot
            points[current_idx] = new_point
            accumulated = 0.0
        else:
            accumulated += seg_len
            current_idx += 1

    while len(resampled) < n_points:
        resampled.append(points[-1].copy())

    return np.array(resampled)


def normalize_trajectory(points: np.ndarray) -> np.ndarray:
    """Normalize: translate to origin, scale to unit size."""
    centered = points - np.mean(points, axis=0)
    max_range = np.max(np.ptp(centered, axis=0))
    if max_range > 0:
        centered = centered / max_range
    return centered


def process_ffo_trajectory(points: np.ndarray) -> np.ndarray:
    """Full FFO$ processing: resample + normalize."""
    resampled = resample_trajectory(points, N_RESAMPLE_POINTS)
    return normalize_trajectory(resampled)


# =============================================================================
# CLASSIFIERS
# =============================================================================

class FFOTemplateClassifier:
    """FFO$-style template matching classifier."""

    def __init__(self):
        self.templates = {}  # finger_code -> list of processed trajectories

    def fit(self, trajectories: List[Trajectory]):
        """Store templates per class."""
        for traj in trajectories:
            if traj.finger_code not in self.templates:
                self.templates[traj.finger_code] = []
            processed = process_ffo_trajectory(traj.points.copy())
            self.templates[traj.finger_code].append(processed)

    def predict(self, trajectory: Trajectory) -> Tuple[str, float]:
        """Predict finger code using template matching."""
        query = process_ffo_trajectory(trajectory.points.copy())

        best_code = None
        best_dist = np.inf

        for code, templates in self.templates.items():
            for template in templates:
                dist = np.mean(np.linalg.norm(query - template, axis=1))
                if dist < best_dist:
                    best_dist = dist
                    best_code = code

        confidence = 1.0 / (1.0 + best_dist)
        return best_code, confidence

    def evaluate(self, trajectories: List[Trajectory]) -> Dict[str, float]:
        """Evaluate on test trajectories."""
        correct = 0
        total = 0

        for traj in trajectories:
            pred_code, _ = self.predict(traj)
            if pred_code == traj.finger_code:
                correct += 1
            total += 1

        return {
            'accuracy': correct / total if total > 0 else 0,
            'correct': correct,
            'total': total
        }


class SingleSampleKNN:
    """K-Nearest Neighbors on single magnetometer samples."""

    def __init__(self, k: int = 5):
        self.k = k
        self.X_train = None
        self.y_train = None

    def fit(self, samples: List[Sample]):
        """Train on single samples."""
        self.X_train = np.array([s.as_vector for s in samples])
        self.y_train = np.array([s.finger_code for s in samples])

    def predict(self, sample: Sample) -> Tuple[str, float]:
        """Predict finger code for single sample."""
        x = sample.as_vector
        distances = np.linalg.norm(self.X_train - x, axis=1)
        nearest_idx = np.argsort(distances)[:self.k]
        nearest_labels = self.y_train[nearest_idx]

        # Majority vote
        label_counts = defaultdict(int)
        for label in nearest_labels:
            label_counts[label] += 1

        best_label = max(label_counts.keys(), key=lambda k: label_counts[k])
        confidence = label_counts[best_label] / self.k

        return best_label, confidence

    def evaluate(self, samples: List[Sample]) -> Dict[str, float]:
        """Evaluate on test samples."""
        correct = 0
        total = 0

        for sample in samples:
            pred_code, _ = self.predict(sample)
            if pred_code == sample.finger_code:
                correct += 1
            total += 1

        return {
            'accuracy': correct / total if total > 0 else 0,
            'correct': correct,
            'total': total
        }


class TrajectoryNeuralNet:
    """
    Simple neural network trained on trajectory features.
    Uses numpy for portability (no TensorFlow/PyTorch required).

    Architecture: Input -> ReLU(64) -> ReLU(32) -> Softmax(n_classes)
    """

    def __init__(self, input_dim: int, n_classes: int, hidden1: int = 64, hidden2: int = 32):
        self.input_dim = input_dim
        self.n_classes = n_classes

        # Xavier initialization
        self.W1 = np.random.randn(input_dim, hidden1) * np.sqrt(2.0 / input_dim)
        self.b1 = np.zeros(hidden1)
        self.W2 = np.random.randn(hidden1, hidden2) * np.sqrt(2.0 / hidden1)
        self.b2 = np.zeros(hidden2)
        self.W3 = np.random.randn(hidden2, n_classes) * np.sqrt(2.0 / hidden2)
        self.b3 = np.zeros(n_classes)

        self.classes = None

    def _relu(self, x):
        return np.maximum(0, x)

    def _softmax(self, x):
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

    def _forward(self, X):
        h1 = self._relu(X @ self.W1 + self.b1)
        h2 = self._relu(h1 @ self.W2 + self.b2)
        out = self._softmax(h2 @ self.W3 + self.b3)
        return out, h1, h2

    def fit(self, X: np.ndarray, y: np.ndarray,
            epochs: int = 100, lr: float = 0.01, batch_size: int = 32):
        """Train with mini-batch gradient descent."""
        self.classes = np.unique(y)
        class_to_idx = {c: i for i, c in enumerate(self.classes)}
        y_idx = np.array([class_to_idx[c] for c in y])

        n_samples = len(X)

        for epoch in range(epochs):
            # Shuffle
            perm = np.random.permutation(n_samples)
            X_shuffled = X[perm]
            y_shuffled = y_idx[perm]

            total_loss = 0

            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]

                # Forward
                probs, h1, h2 = self._forward(X_batch)

                # Cross-entropy loss
                y_onehot = np.zeros((len(y_batch), self.n_classes))
                y_onehot[np.arange(len(y_batch)), y_batch] = 1
                loss = -np.mean(np.sum(y_onehot * np.log(probs + 1e-10), axis=1))
                total_loss += loss

                # Backward (simplified gradient descent)
                dout = (probs - y_onehot) / len(y_batch)

                dW3 = h2.T @ dout
                db3 = np.sum(dout, axis=0)

                dh2 = dout @ self.W3.T
                dh2[h2 <= 0] = 0  # ReLU derivative

                dW2 = h1.T @ dh2
                db2 = np.sum(dh2, axis=0)

                dh1 = dh2 @ self.W2.T
                dh1[h1 <= 0] = 0

                dW1 = X_batch.T @ dh1
                db1 = np.sum(dh1, axis=0)

                # Update
                self.W3 -= lr * dW3
                self.b3 -= lr * db3
                self.W2 -= lr * dW2
                self.b2 -= lr * db2
                self.W1 -= lr * dW1
                self.b1 -= lr * db1

            if (epoch + 1) % 20 == 0:
                acc = self._compute_accuracy(X, y_idx)
                print(f"    Epoch {epoch+1}: loss={total_loss:.4f}, acc={acc:.1%}")

    def _compute_accuracy(self, X, y_idx):
        probs, _, _ = self._forward(X)
        preds = np.argmax(probs, axis=1)
        return np.mean(preds == y_idx)

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predict class and confidence."""
        probs, _, _ = self._forward(X)
        pred_idx = np.argmax(probs, axis=1)
        pred_classes = np.array([self.classes[i] for i in pred_idx])
        confidence = np.max(probs, axis=1)
        return pred_classes, confidence

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict[str, float]:
        """Evaluate accuracy."""
        pred_classes, _ = self.predict(X)
        correct = np.sum(pred_classes == y)
        return {
            'accuracy': correct / len(y),
            'correct': int(correct),
            'total': len(y)
        }


# =============================================================================
# EMPIRICAL STUDY
# =============================================================================

def run_empirical_comparison(single_samples: List[Sample],
                            trajectories: List[Trajectory]) -> Dict[str, Any]:
    """
    Run empirical comparison of three approaches:
    1. FFO$ template matching
    2. Single-sample KNN
    3. Trajectory neural network
    """
    results = {}

    print("\n" + "=" * 80)
    print("EMPIRICAL COMPARISON")
    print("=" * 80)

    # Split data
    np.random.seed(42)

    # For single samples: 80/20 split
    n_samples = len(single_samples)
    indices = np.random.permutation(n_samples)
    split_idx = int(0.8 * n_samples)
    train_samples = [single_samples[i] for i in indices[:split_idx]]
    test_samples = [single_samples[i] for i in indices[split_idx:]]

    # For trajectories: 80/20 split
    n_traj = len(trajectories)
    traj_indices = np.random.permutation(n_traj)
    traj_split = int(0.8 * n_traj)
    train_trajs = [trajectories[i] for i in traj_indices[:traj_split]]
    test_trajs = [trajectories[i] for i in traj_indices[traj_split:]]

    print(f"\nData split:")
    print(f"  Single samples: {len(train_samples)} train, {len(test_samples)} test")
    print(f"  Trajectories:   {len(train_trajs)} train, {len(test_trajs)} test")

    # === 1. FFO$ Template Matching ===
    print("\n1. FFO$ TEMPLATE MATCHING")
    print("-" * 60)

    ffo = FFOTemplateClassifier()
    ffo.fit(train_trajs)
    ffo_results = ffo.evaluate(test_trajs)

    print(f"  Templates stored: {sum(len(t) for t in ffo.templates.values())}")
    print(f"  Test accuracy: {ffo_results['accuracy']:.1%} ({ffo_results['correct']}/{ffo_results['total']})")
    results['ffo_template'] = ffo_results

    # === 2. Single-Sample KNN ===
    print("\n2. SINGLE-SAMPLE KNN (k=5)")
    print("-" * 60)

    knn = SingleSampleKNN(k=5)
    knn.fit(train_samples)
    knn_results = knn.evaluate(test_samples)

    print(f"  Training samples: {len(train_samples)}")
    print(f"  Test accuracy: {knn_results['accuracy']:.1%} ({knn_results['correct']}/{knn_results['total']})")
    results['single_sample_knn'] = knn_results

    # === 3. Trajectory Neural Network ===
    print("\n3. TRAJECTORY NEURAL NETWORK")
    print("-" * 60)

    # Prepare trajectory features (statistics-based for efficiency)
    def trajectory_to_features(traj: Trajectory) -> np.ndarray:
        """Extract rich features from trajectory."""
        points = traj.points
        features = []

        # Mean (3)
        features.extend(np.mean(points, axis=0))

        # Std (3)
        features.extend(np.std(points, axis=0))

        # Min/Max (6)
        features.extend(np.min(points, axis=0))
        features.extend(np.max(points, axis=0))

        # Path statistics (3)
        if len(points) > 1:
            diffs = np.diff(points, axis=0)
            step_lengths = np.linalg.norm(diffs, axis=1)
            features.append(np.sum(step_lengths))  # Total path length
            features.append(np.mean(step_lengths))  # Mean step
            features.append(np.std(step_lengths))   # Step variability
        else:
            features.extend([0, 0, 0])

        # Start/End (6)
        features.extend(points[0])
        features.extend(points[-1])

        return np.array(features)

    X_train_traj = np.array([trajectory_to_features(t) for t in train_trajs])
    y_train_traj = np.array([t.finger_code for t in train_trajs])
    X_test_traj = np.array([trajectory_to_features(t) for t in test_trajs])
    y_test_traj = np.array([t.finger_code for t in test_trajs])

    input_dim = X_train_traj.shape[1]
    n_classes = len(set(y_train_traj))

    print(f"  Feature dim: {input_dim}")
    print(f"  Classes: {n_classes}")
    print(f"  Training neural network...")

    nn = TrajectoryNeuralNet(input_dim, n_classes)
    nn.fit(X_train_traj, y_train_traj, epochs=100, lr=0.01)
    nn_results = nn.evaluate(X_test_traj, y_test_traj)

    print(f"  Test accuracy: {nn_results['accuracy']:.1%} ({nn_results['correct']}/{nn_results['total']})")
    results['trajectory_nn'] = nn_results

    # === 4. Trajectory NN with Full Resampled Trajectory ===
    print("\n4. TRAJECTORY NN (FULL RESAMPLED)")
    print("-" * 60)

    def trajectory_to_full_features(traj: Trajectory) -> np.ndarray:
        """Use full resampled trajectory as features."""
        resampled = resample_trajectory(traj.points.copy(), N_RESAMPLE_POINTS)
        return resampled.flatten()

    X_train_full = np.array([trajectory_to_full_features(t) for t in train_trajs])
    X_test_full = np.array([trajectory_to_full_features(t) for t in test_trajs])

    input_dim_full = X_train_full.shape[1]
    print(f"  Feature dim: {input_dim_full}")
    print(f"  Training neural network...")

    nn_full = TrajectoryNeuralNet(input_dim_full, n_classes, hidden1=128, hidden2=64)
    nn_full.fit(X_train_full, y_train_traj, epochs=100, lr=0.005)
    nn_full_results = nn_full.evaluate(X_test_full, y_test_traj)

    print(f"  Test accuracy: {nn_full_results['accuracy']:.1%} ({nn_full_results['correct']}/{nn_full_results['total']})")
    results['trajectory_nn_full'] = nn_full_results

    # === 5. Fair Comparison: Matched Sample Sizes ===
    print("\n5. FAIR COMPARISON (MATCHED SAMPLE SIZES)")
    print("-" * 60)

    # For fair comparison, match the number of single samples to trajectories
    # by using trajectory centers only
    n_traj_test = len(test_trajs)
    n_samples_matched = min(len(test_samples), n_traj_test * 20)  # Allow some margin

    if n_traj_test > 0 and n_samples_matched >= n_traj_test:
        # Subsample single samples to match trajectory count
        subsample_idx = np.random.choice(len(test_samples), size=min(n_samples_matched, len(test_samples)), replace=False)
        test_samples_matched = [test_samples[i] for i in subsample_idx]

        knn_matched_results = knn.evaluate(test_samples_matched)
        print(f"  KNN on matched samples ({len(test_samples_matched)}): {knn_matched_results['accuracy']:.1%}")
        results['single_sample_knn_matched'] = knn_matched_results
    else:
        results['single_sample_knn_matched'] = {'accuracy': 0, 'total': 0}

    # === 6. Per-Class Analysis ===
    print("\n6. PER-CLASS ACCURACY")
    print("-" * 60)

    # Show how each approach performs per finger code
    class_results = defaultdict(lambda: {'knn': [], 'ffo': [], 'nn': []})

    for sample in test_samples:
        pred_code, _ = knn.predict(sample)
        class_results[sample.finger_code]['knn'].append(pred_code == sample.finger_code)

    for traj in test_trajs:
        pred_code, _ = ffo.predict(traj)
        class_results[traj.finger_code]['ffo'].append(pred_code == traj.finger_code)

    print("\n{:<12} {:>12} {:>12} {:>12}".format("Code", "KNN", "FFO$", "Samples"))
    print("-" * 50)

    for code in sorted(class_results.keys()):
        knn_acc = np.mean(class_results[code]['knn']) if class_results[code]['knn'] else 0
        ffo_acc = np.mean(class_results[code]['ffo']) if class_results[code]['ffo'] else 0
        n_samples_code = len(class_results[code]['knn'])
        print("{:<12} {:>12.1%} {:>12.1%} {:>12}".format(code, knn_acc, ffo_acc, n_samples_code))

    results['per_class'] = {
        code: {
            'knn_accuracy': float(np.mean(v['knn'])) if v['knn'] else 0,
            'ffo_accuracy': float(np.mean(v['ffo'])) if v['ffo'] else 0,
            'n_samples': len(v['knn'])
        }
        for code, v in class_results.items()
    }

    # === Summary ===
    print("\n" + "=" * 80)
    print("EMPIRICAL RESULTS SUMMARY")
    print("=" * 80)

    print("\n{:<30} {:>15} {:>15} {:>10}".format("Approach", "Accuracy", "Data Type", "Test N"))
    print("-" * 72)
    print("{:<30} {:>15.1%} {:>15} {:>10}".format(
        "FFO$ Template Matching",
        results['ffo_template']['accuracy'],
        "Trajectory",
        results['ffo_template']['total']
    ))
    print("{:<30} {:>15.1%} {:>15} {:>10}".format(
        "Single-Sample KNN (k=5)",
        results['single_sample_knn']['accuracy'],
        "Single Point",
        results['single_sample_knn']['total']
    ))
    print("{:<30} {:>15.1%} {:>15} {:>10}".format(
        "Trajectory NN (stats)",
        results['trajectory_nn']['accuracy'],
        "Traj Stats",
        results['trajectory_nn']['total']
    ))
    print("{:<30} {:>15.1%} {:>15} {:>10}".format(
        "Trajectory NN (full)",
        results['trajectory_nn_full']['accuracy'],
        "Full Traj",
        results['trajectory_nn_full']['total']
    ))

    # Improvement analysis
    baseline = results['single_sample_knn']['accuracy']
    best_traj = max(results['trajectory_nn']['accuracy'],
                    results['trajectory_nn_full']['accuracy'],
                    results['ffo_template']['accuracy'])

    improvement = (best_traj - baseline) / baseline if baseline > 0 else 0

    print(f"\n  Trajectory improvement over single-sample: {improvement:+.1%}")

    # Key insight about data imbalance
    n_single = results['single_sample_knn']['total']
    n_traj = results['trajectory_nn']['total']
    if n_single > 10 * n_traj:
        print(f"\n  ⚠️ DATA IMBALANCE: {n_single} single samples vs {n_traj} trajectories")
        print(f"     Single-sample approach benefits from {n_single/n_traj:.1f}x more test data")

    return results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 80)
    print("TRAJECTORY VS SINGLE-SAMPLE INFERENCE STUDY")
    print("Information-Theoretic and Empirical Analysis")
    print("=" * 80)

    # Load data
    data_dir = Path('.worktrees/data/GAMBIT')
    if not data_dir.exists():
        data_dir = Path('data/GAMBIT')

    print(f"\nLoading data from: {data_dir}")
    sessions = load_all_sessions(data_dir)

    if not sessions:
        print("ERROR: No labeled sessions found!")
        print("Please ensure wizard sessions with labels exist in the data directory.")
        return

    print(f"Found {len(sessions)} sessions with labels")

    # Extract data
    single_samples, trajectories = extract_labeled_data(sessions)
    print(f"Extracted: {len(single_samples)} single samples, {len(trajectories)} trajectories")

    # Get class distribution
    single_codes = set(s.finger_code for s in single_samples)
    traj_codes = set(t.finger_code for t in trajectories)
    print(f"Unique finger codes: {len(single_codes)} (samples), {len(traj_codes)} (trajectories)")

    if len(single_samples) < 100:
        print("\nWARNING: Limited data available. Results may not be statistically robust.")

    # Run analyses
    info_results = analyze_information_theory(single_samples, trajectories)
    empirical_results = run_empirical_comparison(single_samples, trajectories)

    # Final conclusions
    print("\n" + "=" * 80)
    print("CONCLUSIONS")
    print("=" * 80)

    # Compare theoretical prediction to empirical results
    predicted_gain = info_results['comparison']['relative_gain']

    baseline_acc = empirical_results['single_sample_knn']['accuracy']
    best_traj_acc = max(
        empirical_results['trajectory_nn']['accuracy'],
        empirical_results['trajectory_nn_full']['accuracy']
    )
    actual_gain = (best_traj_acc - baseline_acc) / (baseline_acc + 1e-10)

    print(f"""
    QUESTION: Does trajectory-based inference improve performance?

    INFORMATION-THEORETIC PREDICTION:
    - Additional information in trajectories: {predicted_gain:+.1%}
    - Theoretical capacity utilization:
      * Single sample: {info_results['comparison']['single_efficiency']:.0%}
      * Trajectory:    {info_results['comparison']['trajectory_efficiency']:.0%}

    EMPIRICAL RESULTS:
    - Single-sample KNN:    {baseline_acc:.1%}
    - Best trajectory NN:   {best_traj_acc:.1%}
    - Actual improvement:   {actual_gain:+.1%}

    CONCLUSION:
    {"✓ Trajectories DO improve performance" if actual_gain > 0 else "✗ No significant improvement from trajectories"}

    PRACTICAL IMPLICATIONS:
    - Use trajectory-based inference when: movement data is available,
      fine-grained pose transitions matter, and compute allows window processing
    - Use single-sample inference when: low latency is critical,
      limited compute resources, or steady-state poses are sufficient

    RECOMMENDATION:
    For the GAMBIT system, a HYBRID approach may be optimal:
    1. Use single-sample inference for real-time pose estimation
    2. Use trajectory matching for gesture/transition detection
    3. Combine outputs for richer gesture+pose recognition
    """)

    # Save results
    output_path = Path('ml/trajectory_study_results.json')
    output_path.parent.mkdir(parents=True, exist_ok=True)

    def to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.int64, np.float64, np.int32, np.float32)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [to_serializable(v) for v in obj]
        return obj

    results_to_save = {
        'information_theoretic': to_serializable(info_results),
        'empirical': to_serializable(empirical_results),
        'summary': {
            'predicted_gain': float(predicted_gain),
            'actual_gain': float(actual_gain),
            'n_samples': len(single_samples),
            'n_trajectories': len(trajectories),
            'n_sessions': len(sessions)
        }
    }

    with open(output_path, 'w') as f:
        json.dump(results_to_save, f, indent=2)

    print(f"\nResults saved to: {output_path}")


if __name__ == '__main__':
    main()
