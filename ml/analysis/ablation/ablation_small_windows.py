"""
Small Window Ablation Study

Test very small windows: 1, 2, 5, 10, 12, 20, 24, 30, 50
Focus on key feature sets: mag_only, iron_mag, raw_9dof

Author: Claude
Date: January 2026
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


@dataclass
class FeatureSet:
    name: str
    features: List[str]
    description: str

    @property
    def n_features(self) -> int:
        return len(self.features)


FEATURE_SETS = {
    'mag_only': FeatureSet(
        name='mag_only',
        features=['mx_ut', 'my_ut', 'mz_ut'],
        description='Magnetometer only'
    ),
    'iron_mag': FeatureSet(
        name='iron_mag',
        features=['iron_mx', 'iron_my', 'iron_mz'],
        description='Iron-corrected mag'
    ),
    'raw_9dof': FeatureSet(
        name='raw_9dof',
        features=['ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps', 'mx_ut', 'my_ut', 'mz_ut'],
        description='Full 9-DoF'
    ),
    'mag_euler': FeatureSet(
        name='mag_euler',
        features=['mx_ut', 'my_ut', 'mz_ut', 'euler_pitch', 'euler_roll', 'euler_yaw'],
        description='Mag + Euler'
    ),
    'iron_euler': FeatureSet(
        name='iron_euler',
        features=['iron_mx', 'iron_my', 'iron_mz', 'euler_pitch', 'euler_roll', 'euler_yaw'],
        description='Iron mag + Euler'
    ),
}

WINDOW_SIZES = [1, 2, 5, 10, 12, 20, 24, 30, 50]


def load_labeled_data(data_dir: Path) -> Tuple[List[Dict], Dict]:
    """Load labeled session data."""
    FINGER_STATE_MAP = {'extended': '0', 'flexed': '2', 'unknown': None}
    FINGER_ORDER = ['thumb', 'index', 'middle', 'ring', 'pinky']

    for session_file in sorted(data_dir.glob("*.json"),
                               key=lambda x: x.stat().st_size, reverse=True):
        if session_file.name == 'manifest.json' or session_file.stat().st_size < 1000:
            continue

        try:
            with open(session_file) as f:
                data = json.load(f)

            labels_list = data.get('labels', [])
            samples = data.get('samples', [])

            index_to_code = {}
            for lbl in labels_list:
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
                return samples, {'session': session_file.name, 'all_samples': samples,
                                'index_to_code': index_to_code}
        except Exception as e:
            continue

    return [], {}


def extract_features(sample: Dict, feature_set: FeatureSet) -> Optional[np.ndarray]:
    """Extract features from a sample."""
    values = []
    for feat in feature_set.features:
        val = sample.get(feat)
        if val is None:
            return None
        values.append(float(val))
    return np.array(values)


def prepare_windows(samples, index_to_code, window_size, feature_set, stride=1):
    """Prepare windowed data."""
    labeled_indices = sorted(index_to_code.keys())

    windows = []
    labels = []
    pitches = []

    i = 0
    while i <= len(labeled_indices) - window_size:
        start_idx = labeled_indices[i]

        # Check contiguity
        valid = True
        for j in range(1, window_size):
            if i + j >= len(labeled_indices) or labeled_indices[i + j] != start_idx + j:
                valid = False
                break

        if not valid:
            i += 1
            continue

        label = index_to_code[start_idx]
        same_label = all(index_to_code.get(start_idx + j) == label for j in range(window_size))

        if not same_label:
            i += stride
            continue

        window_features = []
        for j in range(window_size):
            idx = start_idx + j
            if idx >= len(samples):
                valid = False
                break
            feat = extract_features(samples[idx], feature_set)
            if feat is None:
                valid = False
                break
            window_features.append(feat)

        if valid and len(window_features) == window_size:
            windows.append(np.array(window_features))
            labels.append(label)
            mid_idx = start_idx + window_size // 2
            pitch = samples[mid_idx].get('euler_pitch', 0) if mid_idx < len(samples) else 0
            pitches.append(pitch)

        i += stride

    if not windows:
        return np.array([]), np.array([]), [], np.array([])

    X = np.array(windows)
    classes = sorted(set(labels))
    label_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.array([label_to_idx[l] for l in labels])

    return X, y, classes, np.array(pitches)


def build_model(window_size: int, n_features: int, n_classes: int):
    """Build appropriate model based on window size."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    if window_size == 1:
        # Single sample: just dense layers
        model = keras.Sequential([
            layers.Flatten(input_shape=(1, n_features)),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(n_classes, activation='softmax')
        ])
    elif window_size <= 5:
        # Very small: simple conv
        model = keras.Sequential([
            layers.Conv1D(16, min(3, window_size), activation='relu', padding='same',
                         input_shape=(window_size, n_features)),
            layers.GlobalAveragePooling1D(),
            layers.Dense(32, activation='relu'),
            layers.Dropout(0.3),
            layers.Dense(n_classes, activation='softmax')
        ])
    else:
        # Larger: CNN-LSTM
        model = keras.Sequential([
            layers.Conv1D(32, 3, activation='relu', padding='same',
                         input_shape=(window_size, n_features)),
            layers.BatchNormalization(),
            layers.MaxPooling1D(2) if window_size >= 10 else layers.GlobalAveragePooling1D(),
        ])
        if window_size >= 10:
            model.add(layers.LSTM(32))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dropout(0.3))
        model.add(layers.Dense(n_classes, activation='softmax'))

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def train_and_evaluate(X, y, n_classes, n_runs=3):
    """Train and evaluate with random splits."""
    import tensorflow as tf
    from sklearn.model_selection import train_test_split

    window_size = X.shape[1]
    n_features = X.shape[2]

    results = []

    for run in range(n_runs):
        # Random stratified split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42 + run
        )

        # Normalize
        mean = X_train.reshape(-1, n_features).mean(axis=0)
        std = X_train.reshape(-1, n_features).std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        # Build and train
        model = build_model(window_size, n_features, n_classes)

        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )

        model.fit(
            X_train, y_train,
            validation_split=0.15,
            epochs=50,
            batch_size=min(32, len(X_train) // 4),
            callbacks=[early_stop],
            verbose=0
        )

        _, train_acc = model.evaluate(X_train, y_train, verbose=0)
        _, test_acc = model.evaluate(X_test, y_test, verbose=0)

        results.append({'train': train_acc, 'test': test_acc})
        tf.keras.backend.clear_session()

    return {
        'train_mean': np.mean([r['train'] for r in results]),
        'test_mean': np.mean([r['test'] for r in results]),
        'test_std': np.std([r['test'] for r in results]),
    }


def orientation_split_eval(X, y, pitches, n_classes):
    """Evaluate with pitch-based split."""
    import tensorflow as tf

    window_size = X.shape[1]
    n_features = X.shape[2]

    q1 = np.percentile(pitches, 25)
    q3 = np.percentile(pitches, 75)

    high_mask = pitches >= q3
    low_mask = pitches <= q1

    X_train, y_train = X[high_mask], y[high_mask]
    X_test, y_test = X[low_mask], y[low_mask]

    if len(X_train) < 20 or len(X_test) < 20:
        return None

    mean = X_train.reshape(-1, n_features).mean(axis=0)
    std = X_train.reshape(-1, n_features).std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    model = build_model(window_size, n_features, n_classes)

    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_split=0.15,
        epochs=50,
        batch_size=min(32, len(X_train) // 4),
        callbacks=[early_stop],
        verbose=0
    )

    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)

    tf.keras.backend.clear_session()

    return {
        'train_acc': train_acc,
        'test_acc': test_acc,
        'gap': train_acc - test_acc,
    }


def run_study():
    """Run the small window ablation study."""
    print("=" * 80)
    print("SMALL WINDOW ABLATION STUDY")
    print("Window sizes:", WINDOW_SIZES)
    print("=" * 80)

    data_dir = Path("data/GAMBIT")
    if not data_dir.exists():
        data_dir = Path(".worktrees/data/GAMBIT")

    print("\nLoading data...")
    samples, metadata = load_labeled_data(data_dir)

    if not samples:
        print("ERROR: No data found!")
        return {}

    print(f"Session: {metadata['session']}")
    print(f"Labeled: {len(metadata['index_to_code'])} samples")

    results = {}
    feature_sets_to_test = ['mag_only', 'iron_mag', 'raw_9dof', 'mag_euler', 'iron_euler']

    print("\n" + "=" * 80)
    print("PART 1: RANDOM SPLIT EVALUATION")
    print("=" * 80)

    for fs_name in feature_sets_to_test:
        fs = FEATURE_SETS[fs_name]
        print(f"\n### {fs_name}: {fs.description} ({fs.n_features} features)")
        print("-" * 60)

        for window_size in WINDOW_SIZES:
            X, y, classes, pitches = prepare_windows(
                samples, metadata['index_to_code'],
                window_size, fs, stride=max(1, window_size // 3)
            )

            if len(X) < 50:
                print(f"  w={window_size:2d}: skipped (only {len(X)} windows)")
                continue

            result = train_and_evaluate(X, y, len(classes), n_runs=3)
            key = f"{fs_name}_w{window_size}"
            results[key] = {
                'feature_set': fs_name,
                'window_size': window_size,
                'n_features': fs.n_features,
                'n_windows': len(X),
                **result
            }

            print(f"  w={window_size:2d}: {result['test_mean']:5.1%} ± {result['test_std']:4.1%} "
                  f"({len(X)} windows)")

    # Print summary table
    print("\n" + "=" * 80)
    print("SUMMARY TABLE: Test Accuracy by Window Size")
    print("=" * 80)

    print(f"\n{'Feature Set':<15}", end="")
    for ws in WINDOW_SIZES:
        print(f" w={ws:2d}", end="")
    print()
    print("-" * (15 + 6 * len(WINDOW_SIZES)))

    for fs_name in feature_sets_to_test:
        print(f"{fs_name:<15}", end="")
        for ws in WINDOW_SIZES:
            key = f"{fs_name}_w{ws}"
            if key in results:
                print(f" {results[key]['test_mean']:4.0%}", end=" ")
            else:
                print("   - ", end=" ")
        print()

    # Find best configuration
    best_key = max(results.keys(), key=lambda k: results[k]['test_mean'])
    best = results[best_key]
    print(f"\n✓ Best: {best_key} = {best['test_mean']:.1%}")

    # Orientation invariance for best configs
    print("\n" + "=" * 80)
    print("PART 2: ORIENTATION INVARIANCE (Train high pitch → Test low pitch)")
    print("=" * 80)

    top_configs = sorted(results.keys(), key=lambda k: results[k]['test_mean'], reverse=True)[:5]

    for key in top_configs:
        parts = key.rsplit('_w', 1)
        fs_name, ws = parts[0], int(parts[1])
        fs = FEATURE_SETS[fs_name]

        X, y, classes, pitches = prepare_windows(
            samples, metadata['index_to_code'], ws, fs, stride=max(1, ws // 3)
        )

        if len(X) < 50:
            continue

        oi_result = orientation_split_eval(X, y, pitches, len(classes))
        if oi_result:
            print(f"  {key}: train={oi_result['train_acc']:.1%}, "
                  f"test={oi_result['test_acc']:.1%}, gap={oi_result['gap']:.1%}")

    # Save results
    output_path = Path("ml/ablation_small_windows.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {output_path}")

    return results


if __name__ == "__main__":
    from sklearn.model_selection import train_test_split  # Import to trigger error early
    run_study()
