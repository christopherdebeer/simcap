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
    """A single magnetometer sample with optional accelerometer data."""
    mx: float  # Magnetic field X (μT)
    my: float  # Magnetic field Y (μT)
    mz: float  # Magnetic field Z (μT)
    ax: float = 0.0  # Accelerometer X (g)
    ay: float = 0.0  # Accelerometer Y (g)
    az: float = 0.0  # Accelerometer Z (g)

    def mag_vector(self) -> np.ndarray:
        return np.array([self.mx, self.my, self.mz])

    def acc_vector(self) -> np.ndarray:
        return np.array([self.ax, self.ay, self.az])

    def full_vector(self) -> np.ndarray:
        return np.array([self.mx, self.my, self.mz, self.ax, self.ay, self.az])


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
