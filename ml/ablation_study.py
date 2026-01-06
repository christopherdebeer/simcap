"""
CNN-LSTM Ablation Study: Window Sizes and Feature Sets

This script systematically tests:
1. Different window sizes (25, 50, 100, 150 samples)
2. Different feature combinations:
   - Raw 9-DoF (baseline)
   - Magnetometer only
   - Accelerometer + Gyroscope only
   - Iron-corrected magnetometer
   - With orientation (quaternion/euler)
   - With derived features (magnitudes, etc.)

Author: Claude
Date: January 2026
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class FeatureSet:
    """Definition of a feature set for ablation study."""
    name: str
    features: List[str]
    description: str

    @property
    def n_features(self) -> int:
        return len(self.features)


# Define feature sets for ablation study
FEATURE_SETS = {
    'raw_9dof': FeatureSet(
        name='raw_9dof',
        features=['ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps', 'mx_ut', 'my_ut', 'mz_ut'],
        description='Standard 9-DoF: accel + gyro + mag'
    ),
    'mag_only': FeatureSet(
        name='mag_only',
        features=['mx_ut', 'my_ut', 'mz_ut'],
        description='Magnetometer only (3 features)'
    ),
    'iron_mag': FeatureSet(
        name='iron_mag',
        features=['iron_mx', 'iron_my', 'iron_mz'],
        description='Iron-corrected magnetometer (3 features)'
    ),
    'filtered_mag': FeatureSet(
        name='filtered_mag',
        features=['filtered_mx', 'filtered_my', 'filtered_mz'],
        description='Filtered magnetometer (3 features)'
    ),
    'accel_gyro': FeatureSet(
        name='accel_gyro',
        features=['ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps'],
        description='Accelerometer + Gyroscope only (6 features)'
    ),
    'accel_only': FeatureSet(
        name='accel_only',
        features=['ax_g', 'ay_g', 'az_g'],
        description='Accelerometer only (3 features)'
    ),
    'mag_orientation': FeatureSet(
        name='mag_orientation',
        features=['mx_ut', 'my_ut', 'mz_ut',
                  'orientation_w', 'orientation_x', 'orientation_y', 'orientation_z'],
        description='Magnetometer + quaternion orientation (7 features)'
    ),
    'mag_euler': FeatureSet(
        name='mag_euler',
        features=['mx_ut', 'my_ut', 'mz_ut',
                  'euler_pitch', 'euler_roll', 'euler_yaw'],
        description='Magnetometer + Euler angles (6 features)'
    ),
    'iron_mag_euler': FeatureSet(
        name='iron_mag_euler',
        features=['iron_mx', 'iron_my', 'iron_mz',
                  'euler_pitch', 'euler_roll', 'euler_yaw'],
        description='Iron-corrected mag + Euler angles (6 features)'
    ),
    'full_sensor': FeatureSet(
        name='full_sensor',
        features=['ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps',
                  'iron_mx', 'iron_my', 'iron_mz',
                  'euler_pitch', 'euler_roll', 'euler_yaw'],
        description='All calibrated sensors + orientation (12 features)'
    ),
    'motion_aware': FeatureSet(
        name='motion_aware',
        features=['ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps',
                  'mx_ut', 'my_ut', 'mz_ut', 'dt'],
        description='9-DoF + time delta (10 features)'
    ),
}

# Window sizes to test
WINDOW_SIZES = [25, 50, 100, 150]


def load_labeled_data(data_dir: Path) -> Tuple[List[Dict], Dict]:
    """Load all labeled samples from session data."""
    FINGER_STATE_MAP = {'extended': '0', 'flexed': '2', 'unknown': None}
    FINGER_ORDER = ['thumb', 'index', 'middle', 'ring', 'pinky']

    session_files = sorted(data_dir.glob("*.json"),
                          key=lambda x: x.stat().st_size, reverse=True)

    for session_file in session_files:
        if session_file.name == 'manifest.json' or session_file.stat().st_size < 1000:
            continue

        try:
            with open(session_file) as f:
                data = json.load(f)

            labels = data.get('labels', [])
            samples = data.get('samples', [])

            # Build index mapping
            index_to_code = {}
            for lbl in labels:
                start = lbl.get('startIndex', lbl.get('start_sample', 0))
                end = lbl.get('endIndex', lbl.get('end_sample', 0))

                fingers = lbl.get('fingers', {})
                if not fingers and 'labels' in lbl:
                    fingers = lbl['labels'].get('fingers', {})

                codes = []
                for fn in FINGER_ORDER:
                    state = fingers.get(fn, 'unknown')
                    code = FINGER_STATE_MAP.get(state)
                    if code is None:
                        break
                    codes.append(code)

                if len(codes) == 5:
                    code = ''.join(codes)
                    for i in range(start, end):
                        index_to_code[i] = code

            if len(index_to_code) > 100:
                # Add labels to samples
                labeled_samples = []
                for i, sample in enumerate(samples):
                    if i in index_to_code:
                        sample['label'] = index_to_code[i]
                        labeled_samples.append(sample)

                metadata = {
                    'session': session_file.name,
                    'total_samples': len(labeled_samples),
                    'all_samples': samples,
                    'index_to_code': index_to_code,
                }

                return labeled_samples, metadata

        except Exception as e:
            print(f"Error loading {session_file}: {e}")
            continue

    return [], {}


def extract_features(sample: Dict, feature_set: FeatureSet) -> Optional[np.ndarray]:
    """Extract features from a sample based on feature set definition."""
    values = []
    for feat in feature_set.features:
        val = sample.get(feat)
        if val is None:
            return None
        values.append(float(val))
    return np.array(values)


def add_derived_features(samples: List[Dict]) -> List[Dict]:
    """Add derived features to samples."""
    for sample in samples:
        # Magnetometer magnitude
        mx = sample.get('mx_ut', 0)
        my = sample.get('my_ut', 0)
        mz = sample.get('mz_ut', 0)
        sample['mag_magnitude'] = np.sqrt(mx**2 + my**2 + mz**2)

        # Accelerometer magnitude
        ax = sample.get('ax_g', 0)
        ay = sample.get('ay_g', 0)
        az = sample.get('az_g', 0)
        sample['accel_magnitude'] = np.sqrt(ax**2 + ay**2 + az**2)

        # Gyroscope magnitude
        gx = sample.get('gx_dps', 0)
        gy = sample.get('gy_dps', 0)
        gz = sample.get('gz_dps', 0)
        sample['gyro_magnitude'] = np.sqrt(gx**2 + gy**2 + gz**2)

        # Iron-corrected magnitude
        if 'iron_mx' in sample:
            imx = sample['iron_mx']
            imy = sample['iron_my']
            imz = sample['iron_mz']
            sample['iron_mag_magnitude'] = np.sqrt(imx**2 + imy**2 + imz**2)

    return samples


def prepare_windows(
    samples: List[Dict],
    metadata: Dict,
    window_size: int,
    feature_set: FeatureSet,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    """
    Prepare windowed data for training.

    Returns:
        X: (n_windows, window_size, n_features)
        y: (n_windows,) - class indices
        classes: List of class names
    """
    all_samples = metadata['all_samples']
    index_to_code = metadata['index_to_code']

    # Collect windows
    windows = []
    labels = []

    # Find contiguous labeled regions
    labeled_indices = sorted(index_to_code.keys())

    for i in range(0, len(labeled_indices) - window_size + 1, stride):
        # Check if window is contiguous and has same label
        start_idx = labeled_indices[i]
        end_idx = labeled_indices[i + window_size - 1]

        # Skip if not contiguous
        if end_idx - start_idx != window_size - 1:
            continue

        # Get label (use middle of window)
        mid_idx = labeled_indices[i + window_size // 2]
        label = index_to_code[mid_idx]

        # Extract features for window
        window_features = []
        valid = True
        for j in range(window_size):
            idx = start_idx + j
            if idx >= len(all_samples):
                valid = False
                break
            feat = extract_features(all_samples[idx], feature_set)
            if feat is None:
                valid = False
                break
            window_features.append(feat)

        if valid and len(window_features) == window_size:
            windows.append(np.array(window_features))
            labels.append(label)

    if not windows:
        return np.array([]), np.array([]), []

    X = np.array(windows)

    # Convert labels to indices
    classes = sorted(set(labels))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_idx[l] for l in labels])

    return X, y, classes


def normalize_features(X_train: np.ndarray, X_test: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """Normalize features using training set statistics."""
    # Reshape to (n_samples * window_size, n_features)
    n_train, window_size, n_features = X_train.shape
    X_train_flat = X_train.reshape(-1, n_features)

    mean = X_train_flat.mean(axis=0)
    std = X_train_flat.std(axis=0) + 1e-8

    X_train_norm = (X_train - mean) / std
    X_test_norm = (X_test - mean) / std

    stats = {'mean': mean.tolist(), 'std': std.tolist()}

    return X_train_norm, X_test_norm, stats


def build_cnn_lstm_model(window_size: int, n_features: int, n_classes: int):
    """Build CNN-LSTM model with configurable input shape."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        # CNN layers
        layers.Conv1D(64, 3, activation='relu', padding='same',
                     input_shape=(window_size, n_features)),
        layers.BatchNormalization(),
        layers.Conv1D(64, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        layers.Conv1D(128, 3, activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),
        layers.Dropout(0.2),

        # LSTM layers
        layers.LSTM(64, return_sequences=True),
        layers.Dropout(0.2),
        layers.LSTM(32),
        layers.Dropout(0.2),

        # Dense layers
        layers.Dense(64, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def split_by_pitch(
    X: np.ndarray,
    y: np.ndarray,
    samples: List[Dict],
    metadata: Dict,
    window_size: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Split data by pitch angle for cross-orientation testing."""
    all_samples = metadata['all_samples']
    index_to_code = metadata['index_to_code']

    # Get pitch for each window (use middle sample)
    pitches = []
    labeled_indices = sorted(index_to_code.keys())

    for i in range(0, len(labeled_indices) - window_size + 1):
        start_idx = labeled_indices[i]
        end_idx = labeled_indices[i + window_size - 1]

        if end_idx - start_idx != window_size - 1:
            continue

        mid_idx = start_idx + window_size // 2
        if mid_idx < len(all_samples):
            pitch = all_samples[mid_idx].get('euler_pitch', 0)
            pitches.append(pitch)

    pitches = np.array(pitches[:len(X)])

    # Split by quartiles
    q1 = np.percentile(pitches, 25)
    q3 = np.percentile(pitches, 75)

    high_pitch_mask = pitches >= q3
    low_pitch_mask = pitches <= q1

    X_train = X[high_pitch_mask]
    y_train = y[high_pitch_mask]
    X_test = X[low_pitch_mask]
    y_test = y[low_pitch_mask]

    return X_train, y_train, X_test, y_test


def train_and_evaluate(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_classes: int,
    epochs: int = 30,
    batch_size: int = 32,
    verbose: int = 0
) -> Dict:
    """Train model and return evaluation metrics."""
    import tensorflow as tf

    window_size = X_train.shape[1]
    n_features = X_train.shape[2]

    # Build and train model
    model = build_cnn_lstm_model(window_size, n_features, n_classes)

    # Early stopping
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=verbose
    )

    # Evaluate
    train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)

    # Per-class accuracy
    y_pred = model.predict(X_test, verbose=0).argmax(axis=1)
    per_class_acc = {}
    for c in range(n_classes):
        mask = y_test == c
        if mask.sum() > 0:
            per_class_acc[c] = (y_pred[mask] == c).mean()

    return {
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'per_class_acc': per_class_acc,
        'epochs_trained': len(history.history['loss']),
        'final_train_loss': float(history.history['loss'][-1]),
        'final_val_loss': float(history.history['val_loss'][-1]),
    }


def run_ablation_study(
    feature_sets_to_test: List[str] = None,
    window_sizes_to_test: List[int] = None,
    n_runs: int = 3,
    verbose: bool = True
):
    """Run comprehensive ablation study."""

    if feature_sets_to_test is None:
        feature_sets_to_test = list(FEATURE_SETS.keys())
    if window_sizes_to_test is None:
        window_sizes_to_test = WINDOW_SIZES

    print("=" * 70)
    print("CNN-LSTM ABLATION STUDY")
    print("=" * 70)

    # Load data
    data_dir = Path("data/GAMBIT")
    if not data_dir.exists():
        data_dir = Path(".worktrees/data/GAMBIT")

    print("\nLoading data...")
    samples, metadata = load_labeled_data(data_dir)

    if not samples:
        print("ERROR: No labeled data found!")
        return {}

    print(f"Session: {metadata['session']}")
    print(f"Labeled samples: {metadata['total_samples']}")

    # Add derived features
    samples = add_derived_features(samples)
    metadata['all_samples'] = add_derived_features(metadata['all_samples'])

    # Results storage
    results = {}

    total_experiments = len(feature_sets_to_test) * len(window_sizes_to_test)
    experiment_num = 0

    for feature_set_name in feature_sets_to_test:
        feature_set = FEATURE_SETS[feature_set_name]

        for window_size in window_sizes_to_test:
            experiment_num += 1
            key = f"{feature_set_name}_w{window_size}"

            if verbose:
                print(f"\n{'=' * 70}")
                print(f"Experiment {experiment_num}/{total_experiments}: {key}")
                print(f"  Features: {feature_set.description}")
                print(f"  Window: {window_size} samples")
                print("=" * 70)

            # Prepare data
            X, y, classes = prepare_windows(
                samples, metadata, window_size, feature_set, stride=5
            )

            if len(X) < 100:
                print(f"  Skipping: insufficient data ({len(X)} windows)")
                continue

            if verbose:
                print(f"  Windows: {len(X)}, Classes: {len(classes)}")

            # Split by pitch for cross-orientation testing
            X_train, y_train, X_test, y_test = split_by_pitch(
                X, y, samples, metadata, window_size
            )

            if len(X_train) < 50 or len(X_test) < 50:
                print(f"  Skipping: insufficient train/test data")
                continue

            if verbose:
                print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

            # Normalize
            X_train_norm, X_test_norm, stats = normalize_features(X_train, X_test)

            # Run multiple times and average
            run_results = []
            for run in range(n_runs):
                if verbose:
                    print(f"  Run {run + 1}/{n_runs}...", end=" ", flush=True)

                try:
                    result = train_and_evaluate(
                        X_train_norm, y_train,
                        X_test_norm, y_test,
                        n_classes=len(classes),
                        epochs=30,
                        verbose=0
                    )
                    run_results.append(result)

                    if verbose:
                        print(f"train={result['train_acc']:.3f}, test={result['test_acc']:.3f}")
                except Exception as e:
                    print(f"ERROR: {e}")
                    continue

            if run_results:
                # Average results
                avg_train = np.mean([r['train_acc'] for r in run_results])
                avg_test = np.mean([r['test_acc'] for r in run_results])
                std_test = np.std([r['test_acc'] for r in run_results])

                results[key] = {
                    'feature_set': feature_set_name,
                    'window_size': window_size,
                    'n_features': feature_set.n_features,
                    'description': feature_set.description,
                    'n_windows': len(X),
                    'n_classes': len(classes),
                    'train_acc_mean': float(avg_train),
                    'test_acc_mean': float(avg_test),
                    'test_acc_std': float(std_test),
                    'runs': run_results,
                    'orientation_gap': float(avg_train - avg_test),
                }

                if verbose:
                    print(f"  Average: train={avg_train:.3f}, test={avg_test:.3f} ± {std_test:.3f}")
                    print(f"  Orientation gap: {avg_train - avg_test:.3f}")

    return results


def print_results_summary(results: Dict):
    """Print formatted summary of ablation results."""
    print("\n" + "=" * 80)
    print("ABLATION STUDY RESULTS SUMMARY")
    print("=" * 80)

    # Sort by test accuracy
    sorted_results = sorted(results.items(), key=lambda x: x[1]['test_acc_mean'], reverse=True)

    print(f"\n{'Config':<30} {'Features':>8} {'Window':>6} {'Train':>7} {'Test':>7} {'Gap':>6}")
    print("-" * 80)

    for key, r in sorted_results:
        print(f"{r['feature_set']:<30} {r['n_features']:>8} {r['window_size']:>6} "
              f"{r['train_acc_mean']:>7.1%} {r['test_acc_mean']:>7.1%} {r['orientation_gap']:>6.1%}")

    # Best by category
    print("\n" + "-" * 80)
    print("BEST CONFIGURATIONS:")
    print("-" * 80)

    # Best overall
    best = sorted_results[0]
    print(f"\nBest Overall: {best[0]}")
    print(f"  Test Accuracy: {best[1]['test_acc_mean']:.1%}")

    # Best per window size
    print("\nBest per Window Size:")
    for ws in WINDOW_SIZES:
        ws_results = [(k, v) for k, v in sorted_results if v['window_size'] == ws]
        if ws_results:
            best_ws = ws_results[0]
            print(f"  {ws} samples: {best_ws[1]['feature_set']} ({best_ws[1]['test_acc_mean']:.1%})")

    # Best per feature set
    print("\nBest per Feature Set:")
    for fs_name in FEATURE_SETS.keys():
        fs_results = [(k, v) for k, v in sorted_results if v['feature_set'] == fs_name]
        if fs_results:
            best_fs = fs_results[0]
            print(f"  {fs_name}: w={best_fs[1]['window_size']} ({best_fs[1]['test_acc_mean']:.1%})")


def run_quick_test():
    """Run a quick test with limited configurations."""
    print("Running quick ablation test...")

    results = run_ablation_study(
        feature_sets_to_test=['raw_9dof', 'mag_only', 'iron_mag', 'mag_euler'],
        window_sizes_to_test=[25, 50, 100],
        n_runs=2,
        verbose=True
    )

    if results:
        print_results_summary(results)

        # Save results
        output_path = Path("ml/ablation_results.json")
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    import sys

    # Suppress TF warnings
    import os
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    if len(sys.argv) > 1 and sys.argv[1] == '--full':
        # Full ablation study
        results = run_ablation_study(n_runs=3, verbose=True)
    else:
        # Quick test
        results = run_quick_test()
