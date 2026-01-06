#!/usr/bin/env python3
"""
Template Matching Analysis for Magnetometer-Based Finger State Classification

This script explores using $-family normalization techniques on magnetometer
data for finger state classification. Unlike traditional $ algorithms which
match trajectories over time, this treats each finger state as a point cloud
in magnetic field space.

Usage:
    python -m ml.template_analysis
    python -m ml.template_analysis --output results.json
"""

import json
import os
import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple, Optional, Callable
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Sample:
    """A single magnetometer sample with optional accelerometer and orientation data."""
    mx: float  # Magnetic field X (μT)
    my: float  # Magnetic field Y (μT)
    mz: float  # Magnetic field Z (μT)
    ax: float = 0.0  # Accelerometer X (g)
    ay: float = 0.0  # Accelerometer Y (g)
    az: float = 0.0  # Accelerometer Z (g)
    roll: float = 0.0  # Roll angle (degrees)
    pitch: float = 0.0  # Pitch angle (degrees)
    yaw: float = 0.0  # Yaw angle (degrees)

    def mag_vector(self) -> np.ndarray:
        return np.array([self.mx, self.my, self.mz])

    def acc_vector(self) -> np.ndarray:
        return np.array([self.ax, self.ay, self.az])

    def orientation_vector(self) -> np.ndarray:
        return np.array([self.roll, self.pitch, self.yaw])

    def full_vector(self) -> np.ndarray:
        return np.array([self.mx, self.my, self.mz, self.ax, self.ay, self.az])

    def mag_with_orientation(self) -> np.ndarray:
        return np.array([self.mx, self.my, self.mz, self.roll, self.pitch, self.yaw])


@dataclass
class Template:
    """A template representing a finger state class."""
    finger_code: str
    centroid: np.ndarray
    samples: List[np.ndarray]  # All samples used to create template
    std: np.ndarray  # Standard deviation per dimension


# =============================================================================
# DATA LOADING
# =============================================================================

def load_session_data(data_dir: str = "data/GAMBIT") -> Dict[str, List[Sample]]:
    """
    Load all labeled samples from GAMBIT session files.

    Returns:
        Dictionary mapping finger codes to lists of Sample objects.
    """
    samples_by_code: Dict[str, List[Sample]] = defaultdict(list)

    for fname in os.listdir(data_dir):
        if not fname.endswith('.json') or fname == 'manifest.json':
            continue

        fpath = os.path.join(data_dir, fname)
        with open(fpath) as f:
            try:
                data = json.load(f)
            except json.JSONDecodeError:
                continue

        samples = data.get('samples', [])
        labels = data.get('labels', [])

        for lbl in labels:
            # Handle both label formats
            if 'labels' in lbl:
                fingers = lbl['labels'].get('fingers', {})
                start = lbl.get('start_sample', 0)
                end = lbl.get('end_sample', 0)
            else:
                fingers = lbl.get('fingers', {})
                start = lbl.get('startIndex', 0)
                end = lbl.get('endIndex', 0)

            # Create finger state code
            states = []
            for f in ['thumb', 'index', 'middle', 'ring', 'pinky']:
                state = fingers.get(f, 'unknown')
                if state == 'extended':
                    states.append('0')
                elif state == 'flexed':
                    states.append('2')
                elif state == 'curled':
                    states.append('1')
                else:
                    states.append('?')

            code = ''.join(states)
            if '?' not in code:
                for idx in range(start, min(end + 1, len(samples))):
                    s = samples[idx]
                    if 'mx_ut' in s:
                        samples_by_code[code].append(Sample(
                            mx=s.get('mx_ut', 0),
                            my=s.get('my_ut', 0),
                            mz=s.get('mz_ut', 0),
                            ax=s.get('ax_g', 0),
                            ay=s.get('ay_g', 0),
                            az=s.get('az_g', 0),
                            roll=s.get('euler_roll', s.get('ahrs_roll_deg', 0)),
                            pitch=s.get('euler_pitch', s.get('ahrs_pitch_deg', 0)),
                            yaw=s.get('euler_yaw', s.get('ahrs_yaw_deg', 0)),
                        ))

    return dict(samples_by_code)


# =============================================================================
# NORMALIZATION METHODS (Borrowing from $-family)
# =============================================================================

def normalize_none(samples: np.ndarray) -> np.ndarray:
    """No normalization - use raw values."""
    return samples


def normalize_translate(samples: np.ndarray, global_centroid: np.ndarray) -> np.ndarray:
    """
    Translation normalization ($1-style).
    Subtract global centroid to remove DC offset (Earth's field).
    """
    return samples - global_centroid


def normalize_translate_scale(samples: np.ndarray, global_centroid: np.ndarray) -> np.ndarray:
    """
    Translation + scaling normalization ($1-style).
    Center at origin and scale to unit cube.
    """
    centered = samples - global_centroid
    max_abs = np.max(np.abs(centered))
    if max_abs > 0:
        return centered / max_abs
    return centered


def normalize_unit_vector(samples: np.ndarray) -> np.ndarray:
    """
    Direction-only normalization.
    Project to unit sphere, ignoring magnitude.
    """
    norms = np.linalg.norm(samples, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1)  # Avoid division by zero
    return samples / norms


def normalize_zscore(samples: np.ndarray, mean: np.ndarray, std: np.ndarray) -> np.ndarray:
    """
    Z-score normalization.
    Standardize each dimension independently.
    """
    std_safe = np.where(std > 0, std, 1)
    return (samples - mean) / std_safe


def normalize_per_class_center(sample: np.ndarray, class_centroid: np.ndarray) -> np.ndarray:
    """
    Per-class centering.
    Subtract class-specific centroid for relative position.
    """
    return sample - class_centroid


def rotate_to_gravity_frame(mag: np.ndarray, acc: np.ndarray) -> np.ndarray:
    """
    Rotate magnetometer reading into gravity-aligned frame.

    This attempts to make readings orientation-invariant by expressing
    the magnetic field relative to gravity (down direction).

    Args:
        mag: Magnetometer vector [mx, my, mz]
        acc: Accelerometer vector [ax, ay, az] (in g)

    Returns:
        Magnetometer vector in gravity-aligned frame
    """
    # Normalize gravity vector
    g_norm = np.linalg.norm(acc)
    if g_norm < 0.1:  # Free fall or invalid
        return mag

    g = acc / g_norm  # Unit gravity vector (points down in sensor frame)

    # Create rotation matrix to align gravity with -Z axis
    # This gives us a frame where Z points up (opposite gravity)
    target = np.array([0, 0, -1])

    # Rotation axis = cross product of g and target
    axis = np.cross(g, target)
    axis_norm = np.linalg.norm(axis)

    if axis_norm < 1e-6:
        # Already aligned (or opposite) - handle edge cases
        if np.dot(g, target) > 0:
            return mag  # Already aligned
        else:
            # 180° rotation around X axis
            return np.array([mag[0], -mag[1], -mag[2]])

    axis = axis / axis_norm
    angle = np.arccos(np.clip(np.dot(g, target), -1, 1))

    # Rodrigues' rotation formula
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    K = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0]
    ])

    R = np.eye(3) + sin_a * K + (1 - cos_a) * np.dot(K, K)

    return R @ mag


def get_gravity_aligned_vector(sample: Sample) -> np.ndarray:
    """Get magnetometer in gravity-aligned frame."""
    mag = sample.mag_vector()
    acc = sample.acc_vector()
    return rotate_to_gravity_frame(mag, acc)


# =============================================================================
# DISTANCE METRICS
# =============================================================================

def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean distance ($1-style path distance)."""
    return float(np.linalg.norm(a - b))


def squared_euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Squared Euclidean distance (faster for comparisons)."""
    diff = a - b
    return float(np.dot(diff, diff))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 - cosine similarity)."""
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 1.0
    return 1.0 - float(np.dot(a, b) / (norm_a * norm_b))


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan (L1) distance."""
    return float(np.sum(np.abs(a - b)))


def mahalanobis_distance(a: np.ndarray, b: np.ndarray, inv_cov: np.ndarray) -> float:
    """Mahalanobis distance (variance-aware)."""
    diff = a - b
    return float(np.sqrt(np.dot(np.dot(diff, inv_cov), diff)))


# =============================================================================
# TEMPLATE EXTRACTION
# =============================================================================

def extract_template(samples: List[Sample], method: str = 'centroid') -> Template:
    """
    Extract a template from labeled samples.

    Args:
        samples: List of Sample objects for a single class
        method: 'centroid' (mean), 'medoid' (closest to mean), 'first' (first sample)

    Returns:
        Template object with centroid and statistics
    """
    vectors = np.array([s.mag_vector() for s in samples])

    if method == 'centroid':
        centroid = np.mean(vectors, axis=0)
    elif method == 'medoid':
        mean = np.mean(vectors, axis=0)
        distances = [euclidean_distance(v, mean) for v in vectors]
        centroid = vectors[np.argmin(distances)]
    elif method == 'first':
        centroid = vectors[0]
    else:
        raise ValueError(f"Unknown method: {method}")

    return Template(
        finger_code="",  # Set by caller
        centroid=centroid,
        samples=list(vectors),
        std=np.std(vectors, axis=0),
    )


def extract_templates(
    samples_by_code: Dict[str, List[Sample]],
    method: str = 'centroid'
) -> Dict[str, Template]:
    """
    Extract templates for all finger state classes.
    """
    templates = {}
    for code, samples in samples_by_code.items():
        template = extract_template(samples, method)
        template.finger_code = code
        templates[code] = template
    return templates


# =============================================================================
# CLASSIFICATION
# =============================================================================

def classify_sample(
    sample: np.ndarray,
    templates: Dict[str, Template],
    distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean_distance,
) -> Tuple[str, float]:
    """
    Classify a sample by finding the nearest template.

    Returns:
        Tuple of (predicted_class, distance)
    """
    best_class = None
    best_distance = float('inf')

    for code, template in templates.items():
        dist = distance_fn(sample, template.centroid)
        if dist < best_distance:
            best_distance = dist
            best_class = code

    return best_class, best_distance


def classify_sample_knn(
    sample: np.ndarray,
    all_samples: Dict[str, np.ndarray],
    k: int = 5,
    distance_fn: Callable[[np.ndarray, np.ndarray], float] = euclidean_distance,
) -> Tuple[str, float]:
    """
    Classify using k-nearest neighbors from all training samples.
    """
    distances = []
    for code, samples in all_samples.items():
        for s in samples:
            d = distance_fn(sample, s)
            distances.append((d, code))

    distances.sort(key=lambda x: x[0])
    top_k = distances[:k]

    # Vote
    votes = defaultdict(int)
    for d, code in top_k:
        votes[code] += 1

    best_class = max(votes.keys(), key=lambda c: votes[c])
    best_distance = top_k[0][0]

    return best_class, best_distance


# =============================================================================
# EVALUATION
# =============================================================================

def evaluate_leave_one_out(
    samples_by_code: Dict[str, List[Sample]],
    norm_method: str = 'none',
    distance_fn: Callable = euclidean_distance,
    template_method: str = 'centroid',
) -> Dict:
    """
    Leave-one-out cross-validation.
    For each sample, train on all others and test on it.
    """
    correct = 0
    total = 0
    confusion = defaultdict(lambda: defaultdict(int))

    # Compute global statistics for normalization
    all_samples = []
    for samples in samples_by_code.values():
        all_samples.extend([s.mag_vector() for s in samples])
    all_samples = np.array(all_samples)
    global_mean = np.mean(all_samples, axis=0)
    global_std = np.std(all_samples, axis=0)

    for test_code, test_samples in samples_by_code.items():
        for i, test_sample in enumerate(test_samples):
            # Create training set (all samples except this one)
            train_samples = {}
            for code, samples in samples_by_code.items():
                if code == test_code:
                    # Exclude test sample
                    train_samples[code] = samples[:i] + samples[i + 1:]
                else:
                    train_samples[code] = samples

            # Skip if any class has no samples
            if any(len(s) == 0 for s in train_samples.values()):
                continue

            # Extract templates from training set
            templates = extract_templates(train_samples, method=template_method)

            # Normalize test sample
            test_vec = test_sample.mag_vector()
            if norm_method == 'translate':
                test_vec = normalize_translate(test_vec.reshape(1, -1), global_mean)[0]
                for t in templates.values():
                    t.centroid = normalize_translate(t.centroid.reshape(1, -1), global_mean)[0]
            elif norm_method == 'translate_scale':
                test_vec = normalize_translate_scale(test_vec.reshape(1, -1), global_mean)[0]
                for t in templates.values():
                    t.centroid = normalize_translate_scale(t.centroid.reshape(1, -1), global_mean)[0]
            elif norm_method == 'unit_vector':
                test_vec = normalize_unit_vector(test_vec.reshape(1, -1))[0]
                for t in templates.values():
                    t.centroid = normalize_unit_vector(t.centroid.reshape(1, -1))[0]
            elif norm_method == 'zscore':
                test_vec = normalize_zscore(test_vec.reshape(1, -1), global_mean, global_std)[0]
                for t in templates.values():
                    t.centroid = normalize_zscore(t.centroid.reshape(1, -1), global_mean, global_std)[0]

            # Classify
            pred_code, _ = classify_sample(test_vec, templates, distance_fn)

            if pred_code == test_code:
                correct += 1
            confusion[test_code][pred_code] += 1
            total += 1

    accuracy = correct / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'confusion': dict(confusion),
    }


def evaluate_split(
    samples_by_code: Dict[str, List[Sample]],
    train_ratio: float = 0.7,
    norm_method: str = 'none',
    distance_fn: Callable = euclidean_distance,
    template_method: str = 'centroid',
    seed: int = 42,
) -> Dict:
    """
    Train/test split evaluation.
    """
    np.random.seed(seed)

    train_samples = {}
    test_samples = {}

    for code, samples in samples_by_code.items():
        indices = np.random.permutation(len(samples))
        split_idx = int(len(samples) * train_ratio)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        train_samples[code] = [samples[i] for i in train_idx]
        test_samples[code] = [samples[i] for i in test_idx]

    # Compute global statistics from training set
    all_train = []
    for samples in train_samples.values():
        all_train.extend([s.mag_vector() for s in samples])
    all_train = np.array(all_train)
    global_mean = np.mean(all_train, axis=0)
    global_std = np.std(all_train, axis=0)

    # Extract templates from training set
    templates = extract_templates(train_samples, method=template_method)

    # Apply normalization to templates
    if norm_method == 'translate':
        for t in templates.values():
            t.centroid = normalize_translate(t.centroid.reshape(1, -1), global_mean)[0]
    elif norm_method == 'translate_scale':
        for t in templates.values():
            t.centroid = normalize_translate_scale(t.centroid.reshape(1, -1), global_mean)[0]
    elif norm_method == 'unit_vector':
        for t in templates.values():
            t.centroid = normalize_unit_vector(t.centroid.reshape(1, -1))[0]
    elif norm_method == 'zscore':
        for t in templates.values():
            t.centroid = normalize_zscore(t.centroid.reshape(1, -1), global_mean, global_std)[0]

    # Evaluate on test set
    correct = 0
    total = 0
    confusion = defaultdict(lambda: defaultdict(int))
    per_class_acc = {}

    for true_code, samples in test_samples.items():
        class_correct = 0
        for sample in samples:
            test_vec = sample.mag_vector()

            # Normalize
            if norm_method == 'translate':
                test_vec = normalize_translate(test_vec.reshape(1, -1), global_mean)[0]
            elif norm_method == 'translate_scale':
                test_vec = normalize_translate_scale(test_vec.reshape(1, -1), global_mean)[0]
            elif norm_method == 'unit_vector':
                test_vec = normalize_unit_vector(test_vec.reshape(1, -1))[0]
            elif norm_method == 'zscore':
                test_vec = normalize_zscore(test_vec.reshape(1, -1), global_mean, global_std)[0]

            pred_code, _ = classify_sample(test_vec, templates, distance_fn)

            if pred_code == true_code:
                correct += 1
                class_correct += 1
            confusion[true_code][pred_code] += 1
            total += 1

        per_class_acc[true_code] = class_correct / len(samples) if samples else 0

    accuracy = correct / total if total > 0 else 0

    return {
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'confusion': dict(confusion),
        'per_class_accuracy': per_class_acc,
    }


# =============================================================================
# CROSS-ORIENTATION EVALUATION (Generalization Testing)
# =============================================================================

def evaluate_cross_orientation(
    samples_by_code: Dict[str, List[Sample]],
    split_axis: str = 'pitch',
    split_value: float = 0.0,
    norm_method: str = 'zscore',
    use_knn: bool = True,
    k: int = 5,
) -> Dict:
    """
    Evaluate generalization across orientation splits.

    Trains on samples from one orientation range and tests on another.
    This simulates cross-session testing where orientations differ.

    Args:
        samples_by_code: Labeled samples
        split_axis: 'roll', 'pitch', or 'yaw'
        split_value: Split threshold (degrees)
        norm_method: Normalization method
        use_knn: Use k-NN instead of centroid matching
        k: Number of neighbors for k-NN
    """
    train_samples = defaultdict(list)
    test_samples = defaultdict(list)

    # Split by orientation
    for code, samples in samples_by_code.items():
        for sample in samples:
            if split_axis == 'roll':
                angle = sample.roll
            elif split_axis == 'pitch':
                angle = sample.pitch
            else:
                angle = sample.yaw

            if angle < split_value:
                train_samples[code].append(sample)
            else:
                test_samples[code].append(sample)

    # Check we have samples in both sets
    train_total = sum(len(v) for v in train_samples.values())
    test_total = sum(len(v) for v in test_samples.values())

    if train_total == 0 or test_total == 0:
        return {'error': f'Split {split_axis}<{split_value} produced empty set'}

    # Compute normalization stats from training set
    all_train = []
    for samples in train_samples.values():
        all_train.extend([s.mag_vector() for s in samples])
    all_train = np.array(all_train)
    train_mean = np.mean(all_train, axis=0)
    train_std = np.std(all_train, axis=0)

    # For k-NN, normalize all training samples
    if use_knn:
        train_vectors = {}
        for code, samples in train_samples.items():
            vectors = np.array([s.mag_vector() for s in samples])
            if norm_method == 'zscore':
                vectors = normalize_zscore(vectors, train_mean, train_std)
            elif norm_method == 'translate':
                vectors = normalize_translate(vectors, train_mean)
            train_vectors[code] = vectors
    else:
        # Extract templates from training set
        templates = extract_templates(dict(train_samples), method='centroid')
        if norm_method == 'zscore':
            for t in templates.values():
                t.centroid = normalize_zscore(t.centroid.reshape(1, -1), train_mean, train_std)[0]

    # Evaluate on test set
    correct = 0
    total = 0
    confusion = defaultdict(lambda: defaultdict(int))
    per_class_results = {}

    for true_code, samples in test_samples.items():
        class_correct = 0
        for sample in samples:
            test_vec = sample.mag_vector()

            if norm_method == 'zscore':
                test_vec = normalize_zscore(test_vec.reshape(1, -1), train_mean, train_std)[0]
            elif norm_method == 'translate':
                test_vec = normalize_translate(test_vec.reshape(1, -1), train_mean)[0]

            if use_knn:
                pred_code, _ = classify_sample_knn(test_vec, train_vectors, k=k)
            else:
                pred_code, _ = classify_sample(test_vec, templates)

            if pred_code == true_code:
                correct += 1
                class_correct += 1
            confusion[true_code][pred_code] += 1
            total += 1

        per_class_results[true_code] = {
            'correct': class_correct,
            'total': len(samples),
            'accuracy': class_correct / len(samples) if samples else 0
        }

    accuracy = correct / total if total > 0 else 0

    return {
        'split_axis': split_axis,
        'split_value': split_value,
        'train_samples': train_total,
        'test_samples': test_total,
        'accuracy': accuracy,
        'correct': correct,
        'total': total,
        'per_class': per_class_results,
        'confusion': dict(confusion),
    }


def evaluate_orientation_quartiles(
    samples_by_code: Dict[str, List[Sample]],
    axis: str = 'pitch',
    norm_method: str = 'zscore',
    use_knn: bool = True,
    k: int = 5,
) -> Dict:
    """
    Test generalization by training on one quartile and testing on others.
    """
    # Collect all angles for the specified axis
    all_angles = []
    for samples in samples_by_code.values():
        for s in samples:
            if axis == 'roll':
                all_angles.append(s.roll)
            elif axis == 'pitch':
                all_angles.append(s.pitch)
            else:
                all_angles.append(s.yaw)

    q1, q2, q3 = np.percentile(all_angles, [25, 50, 75])

    results = {
        'axis': axis,
        'quartile_boundaries': [float(q1), float(q2), float(q3)],
        'experiments': [],
    }

    # Test training on Q1 and testing on Q4 (extreme split)
    train_samples = defaultdict(list)
    test_samples = defaultdict(list)

    for code, samples in samples_by_code.items():
        for sample in samples:
            if axis == 'roll':
                angle = sample.roll
            elif axis == 'pitch':
                angle = sample.pitch
            else:
                angle = sample.yaw

            if angle <= q1:
                train_samples[code].append(sample)
            elif angle >= q3:
                test_samples[code].append(sample)

    result = _evaluate_prebuilt_split(
        dict(train_samples), dict(test_samples), norm_method, use_knn, k
    )
    result['description'] = f'Train Q1 ({axis}<={q1:.1f}°) → Test Q4 ({axis}>={q3:.1f}°)'
    results['experiments'].append(result)

    # Swap: Train on Q4, test on Q1
    result2 = _evaluate_prebuilt_split(
        dict(test_samples), dict(train_samples), norm_method, use_knn, k
    )
    result2['description'] = f'Train Q4 ({axis}>={q3:.1f}°) → Test Q1 ({axis}<={q1:.1f}°)'
    results['experiments'].append(result2)

    return results


def _evaluate_prebuilt_split(
    train_samples: Dict[str, List[Sample]],
    test_samples: Dict[str, List[Sample]],
    norm_method: str,
    use_knn: bool,
    k: int,
) -> Dict:
    """Helper: evaluate on pre-split train/test sets."""
    train_total = sum(len(v) for v in train_samples.values())
    test_total = sum(len(v) for v in test_samples.values())

    if train_total == 0 or test_total == 0:
        return {'error': 'Empty split', 'train': train_total, 'test': test_total}

    # Compute normalization stats
    all_train = []
    for samples in train_samples.values():
        all_train.extend([s.mag_vector() for s in samples])
    all_train = np.array(all_train)
    train_mean = np.mean(all_train, axis=0)
    train_std = np.std(all_train, axis=0)

    if use_knn:
        train_vectors = {}
        for code, samples in train_samples.items():
            vectors = np.array([s.mag_vector() for s in samples])
            if norm_method == 'zscore':
                vectors = normalize_zscore(vectors, train_mean, train_std)
            train_vectors[code] = vectors
    else:
        templates = extract_templates(train_samples, method='centroid')
        if norm_method == 'zscore':
            for t in templates.values():
                t.centroid = normalize_zscore(t.centroid.reshape(1, -1), train_mean, train_std)[0]

    correct = 0
    total = 0

    for true_code, samples in test_samples.items():
        for sample in samples:
            test_vec = sample.mag_vector()
            if norm_method == 'zscore':
                test_vec = normalize_zscore(test_vec.reshape(1, -1), train_mean, train_std)[0]

            if use_knn:
                pred_code, _ = classify_sample_knn(test_vec, train_vectors, k=k)
            else:
                pred_code, _ = classify_sample(test_vec, templates)

            if pred_code == true_code:
                correct += 1
            total += 1

    return {
        'train_samples': train_total,
        'test_samples': test_total,
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total,
    }


def evaluate_with_orientation_features(
    samples_by_code: Dict[str, List[Sample]],
    split_axis: str = 'pitch',
    use_orientation: bool = True,
    norm_method: str = 'zscore',
    k: int = 5,
) -> Dict:
    """
    Evaluate using magnetometer + orientation as features.

    Tests if including orientation helps cross-orientation generalization.
    """
    # Collect all samples
    all_angles = []
    for samples in samples_by_code.values():
        for s in samples:
            if split_axis == 'roll':
                all_angles.append(s.roll)
            elif split_axis == 'pitch':
                all_angles.append(s.pitch)
            else:
                all_angles.append(s.yaw)

    q1, q3 = np.percentile(all_angles, [25, 75])

    # Split train/test by quartile
    train_samples = defaultdict(list)
    test_samples = defaultdict(list)

    for code, samples in samples_by_code.items():
        for sample in samples:
            if split_axis == 'roll':
                angle = sample.roll
            elif split_axis == 'pitch':
                angle = sample.pitch
            else:
                angle = sample.yaw

            if angle >= q3:
                train_samples[code].append(sample)
            elif angle <= q1:
                test_samples[code].append(sample)

    # Prepare feature vectors
    if use_orientation:
        get_vector = lambda s: s.mag_with_orientation()
        dim = 6
    else:
        get_vector = lambda s: s.mag_vector()
        dim = 3

    # Compute normalization stats
    all_train = []
    for samples in train_samples.values():
        all_train.extend([get_vector(s) for s in samples])
    all_train = np.array(all_train)
    train_mean = np.mean(all_train, axis=0)
    train_std = np.std(all_train, axis=0)

    # Build k-NN reference
    train_vectors = {}
    for code, samples in train_samples.items():
        vectors = np.array([get_vector(s) for s in samples])
        if norm_method == 'zscore':
            vectors = normalize_zscore(vectors, train_mean, train_std)
        train_vectors[code] = vectors

    # Evaluate
    correct = 0
    total = 0

    for true_code, samples in test_samples.items():
        for sample in samples:
            test_vec = get_vector(sample)
            if norm_method == 'zscore':
                test_vec = normalize_zscore(test_vec.reshape(1, -1), train_mean, train_std)[0]

            pred_code, _ = classify_sample_knn(test_vec, train_vectors, k=k)
            if pred_code == true_code:
                correct += 1
            total += 1

    return {
        'use_orientation': use_orientation,
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total,
    }


def analyze_generalization(samples_by_code: Dict[str, List[Sample]]) -> Dict:
    """
    Comprehensive generalization analysis.

    Tests how well templates generalize across:
    1. Different orientations within the session
    2. Random vs orientation-based splits
    """
    print("\n" + "=" * 70)
    print("GENERALIZATION ANALYSIS: Cross-Orientation Testing")
    print("=" * 70)

    results = {
        'orientation_stats': {},
        'random_split': {},
        'cross_orientation': {},
    }

    # 1. Orientation statistics
    print("\n1. Orientation Distribution")
    print("-" * 50)

    for axis_name in ['roll', 'pitch', 'yaw']:
        angles = []
        for samples in samples_by_code.values():
            for s in samples:
                if axis_name == 'roll':
                    angles.append(s.roll)
                elif axis_name == 'pitch':
                    angles.append(s.pitch)
                else:
                    angles.append(s.yaw)

        angles = np.array(angles)
        stats = {
            'min': float(np.min(angles)),
            'max': float(np.max(angles)),
            'mean': float(np.mean(angles)),
            'std': float(np.std(angles)),
            'q25': float(np.percentile(angles, 25)),
            'q75': float(np.percentile(angles, 75)),
        }
        results['orientation_stats'][axis_name] = stats
        print(f"  {axis_name.capitalize():6s}: range=[{stats['min']:.1f}°, {stats['max']:.1f}°], "
              f"μ={stats['mean']:.1f}°, σ={stats['std']:.1f}°")

    # 2. Random split baseline
    print("\n2. Random Split Baseline (70/30)")
    print("-" * 50)

    random_result = evaluate_split(
        samples_by_code, train_ratio=0.7, norm_method='zscore', seed=42
    )
    results['random_split'] = {
        'accuracy': random_result['accuracy'],
        'correct': random_result['correct'],
        'total': random_result['total'],
    }
    print(f"   k-NN (random split): {random_result['accuracy']:.1%}")

    # 3. Cross-orientation splits
    print("\n3. Cross-Orientation Splits")
    print("-" * 50)

    for axis in ['roll', 'pitch', 'yaw']:
        print(f"\n   {axis.upper()} axis:")
        quartile_results = evaluate_orientation_quartiles(
            samples_by_code, axis=axis, norm_method='zscore', use_knn=True, k=5
        )

        results['cross_orientation'][axis] = {
            'boundaries': quartile_results['quartile_boundaries'],
            'experiments': [],
        }

        for exp in quartile_results['experiments']:
            if 'error' not in exp:
                print(f"      {exp['description']}")
                print(f"         Accuracy: {exp['accuracy']:.1%} ({exp['correct']}/{exp['total']})")
                results['cross_orientation'][axis]['experiments'].append({
                    'description': exp['description'],
                    'accuracy': exp['accuracy'],
                    'train': exp['train_samples'],
                    'test': exp['test_samples'],
                })

    # 4. Summary
    print("\n" + "=" * 70)
    print("SUMMARY: Generalization Gap")
    print("=" * 70)

    random_acc = random_result['accuracy']

    worst_cross = 1.0
    worst_desc = ""
    for axis in ['roll', 'pitch', 'yaw']:
        for exp in results['cross_orientation'][axis]['experiments']:
            if exp['accuracy'] < worst_cross:
                worst_cross = exp['accuracy']
                worst_desc = exp['description']

    gap = random_acc - worst_cross
    print(f"\n   Random split accuracy:        {random_acc:.1%}")
    print(f"   Worst cross-orientation:      {worst_cross:.1%} ({worst_desc})")
    print(f"   Generalization gap:           {gap:.1%}")

    if gap > 0.1:
        print("\n   ⚠️  SIGNIFICANT GENERALIZATION GAP DETECTED")
        print("      Templates may not generalize to different orientations!")
    elif gap > 0.05:
        print("\n   ⚠️  Moderate generalization gap detected")
    else:
        print("\n   ✓  Templates appear orientation-robust within this session")

    results['summary'] = {
        'random_accuracy': random_acc,
        'worst_cross_orientation': worst_cross,
        'gap': gap,
    }

    # 5. Test orientation as additional feature
    print("\n4. Orientation as Additional Feature")
    print("-" * 50)
    print("   Testing if including orientation helps cross-orientation generalization:")

    orientation_results = {}
    for axis in ['pitch', 'roll', 'yaw']:
        res_mag = evaluate_with_orientation_features(
            samples_by_code, split_axis=axis, use_orientation=False
        )
        res_mag_orient = evaluate_with_orientation_features(
            samples_by_code, split_axis=axis, use_orientation=True
        )
        print(f"\n   {axis.upper()} axis (Train Q4 → Test Q1):")
        print(f"      Mag only:          {res_mag['accuracy']:.1%}")
        print(f"      Mag + Orientation: {res_mag_orient['accuracy']:.1%}")

        improvement = res_mag_orient['accuracy'] - res_mag['accuracy']
        if improvement > 0.01:
            print(f"      Improvement:       +{improvement:.1%} ✓")
        elif improvement < -0.01:
            print(f"      Degradation:       {improvement:.1%}")
        else:
            print(f"      No significant change")

        orientation_results[axis] = {
            'mag_only': res_mag['accuracy'],
            'mag_orientation': res_mag_orient['accuracy'],
            'improvement': improvement,
        }

    results['orientation_features'] = orientation_results

    # 6. Test gravity-aligned magnetometer
    print("\n5. Gravity-Aligned Magnetometer (Orientation-Invariant)")
    print("-" * 50)
    print("   Testing if rotating mag to gravity frame helps:")

    gravity_results = {}
    for axis in ['pitch', 'roll', 'yaw']:
        res_raw = evaluate_gravity_aligned(
            samples_by_code, split_axis=axis, use_gravity_frame=False
        )
        res_gravity = evaluate_gravity_aligned(
            samples_by_code, split_axis=axis, use_gravity_frame=True
        )

        print(f"\n   {axis.upper()} axis (Train Q4 → Test Q1):")
        print(f"      Raw mag:            {res_raw['accuracy']:.1%}")
        print(f"      Gravity-aligned:    {res_gravity['accuracy']:.1%}")

        improvement = res_gravity['accuracy'] - res_raw['accuracy']
        if improvement > 0.01:
            print(f"      Improvement:        +{improvement:.1%} ✓")
        elif improvement < -0.01:
            print(f"      Degradation:        {improvement:.1%}")
        else:
            print(f"      No significant change")

        gravity_results[axis] = {
            'raw_mag': res_raw['accuracy'],
            'gravity_aligned': res_gravity['accuracy'],
            'improvement': improvement,
        }

    results['gravity_aligned'] = gravity_results

    return results


def evaluate_gravity_aligned(
    samples_by_code: Dict[str, List[Sample]],
    split_axis: str = 'pitch',
    use_gravity_frame: bool = True,
    norm_method: str = 'zscore',
    k: int = 5,
) -> Dict:
    """
    Evaluate using gravity-aligned magnetometer.
    """
    # Collect all samples
    all_angles = []
    for samples in samples_by_code.values():
        for s in samples:
            if split_axis == 'roll':
                all_angles.append(s.roll)
            elif split_axis == 'pitch':
                all_angles.append(s.pitch)
            else:
                all_angles.append(s.yaw)

    q1, q3 = np.percentile(all_angles, [25, 75])

    # Split train/test by quartile
    train_samples = defaultdict(list)
    test_samples = defaultdict(list)

    for code, samples in samples_by_code.items():
        for sample in samples:
            if split_axis == 'roll':
                angle = sample.roll
            elif split_axis == 'pitch':
                angle = sample.pitch
            else:
                angle = sample.yaw

            if angle >= q3:
                train_samples[code].append(sample)
            elif angle <= q1:
                test_samples[code].append(sample)

    # Prepare feature vectors
    if use_gravity_frame:
        get_vector = get_gravity_aligned_vector
    else:
        get_vector = lambda s: s.mag_vector()

    # Compute normalization stats
    all_train = []
    for samples in train_samples.values():
        all_train.extend([get_vector(s) for s in samples])
    all_train = np.array(all_train)
    train_mean = np.mean(all_train, axis=0)
    train_std = np.std(all_train, axis=0)

    # Build k-NN reference
    train_vectors = {}
    for code, samples in train_samples.items():
        vectors = np.array([get_vector(s) for s in samples])
        if norm_method == 'zscore':
            vectors = normalize_zscore(vectors, train_mean, train_std)
        train_vectors[code] = vectors

    # Evaluate
    correct = 0
    total = 0

    for true_code, samples in test_samples.items():
        for sample in samples:
            test_vec = get_vector(sample)
            if norm_method == 'zscore':
                test_vec = normalize_zscore(test_vec.reshape(1, -1), train_mean, train_std)[0]

            pred_code, _ = classify_sample_knn(test_vec, train_vectors, k=k)
            if pred_code == true_code:
                correct += 1
            total += 1

    return {
        'use_gravity_frame': use_gravity_frame,
        'accuracy': correct / total if total > 0 else 0,
        'correct': correct,
        'total': total,
    }


# =============================================================================
# MAIN ANALYSIS
# =============================================================================

def run_full_analysis(data_dir: str = "data/GAMBIT") -> Dict:
    """
    Run comprehensive template matching analysis.
    """
    print("=" * 70)
    print("TEMPLATE MATCHING ANALYSIS FOR MAGNETOMETER FINGER CLASSIFICATION")
    print("=" * 70)
    print()

    # Load data
    print("Loading session data...")
    samples_by_code = load_session_data(data_dir)
    print(f"Loaded {sum(len(s) for s in samples_by_code.values())} samples across {len(samples_by_code)} classes")
    print()

    # Show data distribution
    print("Data Distribution:")
    print("-" * 40)
    for code in sorted(samples_by_code.keys()):
        desc = format_finger_code(code)
        print(f"  {code}: {len(samples_by_code[code]):4d} samples | {desc}")
    print()

    results = {
        'data_summary': {
            'total_samples': sum(len(s) for s in samples_by_code.values()),
            'num_classes': len(samples_by_code),
            'samples_per_class': {k: len(v) for k, v in samples_by_code.items()},
        },
        'experiments': [],
    }

    # Run experiments with different configurations
    norm_methods = ['none', 'translate', 'translate_scale', 'unit_vector', 'zscore']
    distance_fns = {
        'euclidean': euclidean_distance,
        'cosine': cosine_distance,
        'manhattan': manhattan_distance,
    }

    print("Running experiments...")
    print("-" * 70)
    print(f"{'Normalization':<20} {'Distance':<12} {'Accuracy':>10} {'Correct':>10}")
    print("-" * 70)

    best_acc = 0
    best_config = None

    for norm in norm_methods:
        for dist_name, dist_fn in distance_fns.items():
            result = evaluate_split(
                samples_by_code,
                train_ratio=0.7,
                norm_method=norm,
                distance_fn=dist_fn,
                template_method='centroid',
                seed=42,
            )

            acc = result['accuracy']
            correct = result['correct']

            print(f"{norm:<20} {dist_name:<12} {acc:>9.1%} {correct:>10}")

            results['experiments'].append({
                'normalization': norm,
                'distance': dist_name,
                'accuracy': acc,
                'correct': correct,
                'total': result['total'],
                'per_class_accuracy': result['per_class_accuracy'],
            })

            if acc > best_acc:
                best_acc = acc
                best_config = (norm, dist_name, result)

    print("-" * 70)
    print()

    # Best configuration details
    if best_config:
        norm, dist, result = best_config
        print(f"BEST CONFIGURATION: {norm} + {dist}")
        print(f"Overall Accuracy: {result['accuracy']:.1%}")
        print()
        print("Per-Class Accuracy:")
        for code in sorted(result['per_class_accuracy'].keys()):
            acc = result['per_class_accuracy'][code]
            desc = format_finger_code(code)
            bar = "█" * int(acc * 20)
            print(f"  {code} ({desc}): {acc:5.1%} {bar}")
        print()

        # Confusion matrix
        print("Confusion Matrix (rows=true, cols=predicted):")
        codes = sorted(samples_by_code.keys())
        print("       " + " ".join(f"{c[:5]:>5}" for c in codes))
        for true_code in codes:
            row = [result['confusion'].get(true_code, {}).get(pred_code, 0) for pred_code in codes]
            print(f"{true_code[:5]:>5}: " + " ".join(f"{x:>5}" for x in row))

        results['best'] = {
            'normalization': norm,
            'distance': dist,
            'accuracy': best_acc,
            'per_class_accuracy': result['per_class_accuracy'],
            'confusion': result['confusion'],
        }

    # Run generalization analysis
    gen_results = analyze_generalization(samples_by_code)
    results['generalization'] = gen_results

    return results


def format_finger_code(code: str) -> str:
    """Format finger code as human-readable description."""
    fingers = ['T', 'I', 'M', 'R', 'P']
    states = {'0': 'ext', '1': 'cur', '2': 'flx'}
    return ' '.join(f"{f}:{states.get(c, '?')}" for f, c in zip(fingers, code))


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Template Matching Analysis")
    parser.add_argument("--data-dir", default="data/GAMBIT", help="Data directory")
    parser.add_argument("--output", help="Output JSON file for results")
    args = parser.parse_args()

    results = run_full_analysis(args.data_dir)

    if args.output:
        with open(args.output, 'w') as f:
            # Convert numpy types to Python types
            def convert(obj):
                if isinstance(obj, np.ndarray):
                    return obj.tolist()
                if isinstance(obj, (np.float32, np.float64)):
                    return float(obj)
                if isinstance(obj, (np.int32, np.int64)):
                    return int(obj)
                if isinstance(obj, dict):
                    return {k: convert(v) for k, v in obj.items()}
                if isinstance(obj, list):
                    return [convert(v) for v in obj]
                return obj

            json.dump(convert(results), f, indent=2)
            print(f"\nResults saved to {args.output}")
