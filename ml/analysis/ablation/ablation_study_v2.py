"""
CNN-LSTM Ablation Study v2: Improved experimental design

Changes from v1:
1. Uses stratified random splits (not pitch-based) for fair comparison
2. Simpler model architecture for limited data
3. K-fold cross-validation for robust estimates
4. Separate orientation invariance test

Author: Claude
Date: January 2026
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from sklearn.model_selection import StratifiedKFold
import warnings
warnings.filterwarnings('ignore')

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


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
        description='Standard 9-DoF (accel+gyro+mag)'
    ),
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
    'accel_gyro': FeatureSet(
        name='accel_gyro',
        features=['ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps'],
        description='Accel+Gyro (no mag)'
    ),
    'accel_only': FeatureSet(
        name='accel_only',
        features=['ax_g', 'ay_g', 'az_g'],
        description='Accelerometer only'
    ),
    'gyro_only': FeatureSet(
        name='gyro_only',
        features=['gx_dps', 'gy_dps', 'gz_dps'],
        description='Gyroscope only'
    ),
    'mag_euler': FeatureSet(
        name='mag_euler',
        features=['mx_ut', 'my_ut', 'mz_ut', 'euler_pitch', 'euler_roll', 'euler_yaw'],
        description='Mag + Euler angles'
    ),
    'iron_euler': FeatureSet(
        name='iron_euler',
        features=['iron_mx', 'iron_my', 'iron_mz', 'euler_pitch', 'euler_roll', 'euler_yaw'],
        description='Iron mag + Euler'
    ),
    'mag_quat': FeatureSet(
        name='mag_quat',
        features=['mx_ut', 'my_ut', 'mz_ut', 'orientation_w', 'orientation_x', 'orientation_y', 'orientation_z'],
        description='Mag + Quaternion'
    ),
    '9dof_euler': FeatureSet(
        name='9dof_euler',
        features=['ax_g', 'ay_g', 'az_g', 'gx_dps', 'gy_dps', 'gz_dps',
                  'mx_ut', 'my_ut', 'mz_ut', 'euler_pitch', 'euler_roll', 'euler_yaw'],
        description='9-DoF + Euler (12 feat)'
    ),
}

# Window sizes to test
WINDOW_SIZES = [10, 25, 50]


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
                metadata = {
                    'session': session_file.name,
                    'all_samples': samples,
                    'index_to_code': index_to_code,
                }
                return samples, metadata

        except Exception as e:
            print(f"Error loading {session_file}: {e}")
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


def prepare_windows(
    samples: List[Dict],
    index_to_code: Dict[int, str],
    window_size: int,
    feature_set: FeatureSet,
    stride: int = 1
) -> Tuple[np.ndarray, np.ndarray, List[str], np.ndarray]:
    """Prepare windowed data with pitch information."""
    labeled_indices = sorted(index_to_code.keys())

    windows = []
    labels = []
    pitches = []

    i = 0
    while i <= len(labeled_indices) - window_size:
        start_idx = labeled_indices[i]

        # Check contiguity
        valid_window = True
        for j in range(1, window_size):
            if i + j >= len(labeled_indices):
                valid_window = False
                break
            if labeled_indices[i + j] != start_idx + j:
                valid_window = False
                break

        if not valid_window:
            i += 1
            continue

        # Check same label throughout window
        label = index_to_code[start_idx]
        same_label = all(index_to_code.get(start_idx + j) == label for j in range(window_size))

        if not same_label:
            i += stride
            continue

        # Extract features
        window_features = []
        valid = True
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

        if valid:
            windows.append(np.array(window_features))
            labels.append(label)
            # Get pitch from middle of window
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
    pitches = np.array(pitches)

    return X, y, classes, pitches


def build_simple_model(window_size: int, n_features: int, n_classes: int):
    """Build a simpler model for limited data."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        # Single Conv layer
        layers.Conv1D(32, 3, activation='relu', padding='same',
                     input_shape=(window_size, n_features)),
        layers.BatchNormalization(),
        layers.GlobalAveragePooling1D(),

        # Dense
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_lstm_model(window_size: int, n_features: int, n_classes: int):
    """Build LSTM model."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.LSTM(32, return_sequences=False, input_shape=(window_size, n_features)),
        layers.Dropout(0.3),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def build_cnn_lstm_model(window_size: int, n_features: int, n_classes: int):
    """Build CNN-LSTM hybrid model."""
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    model = keras.Sequential([
        layers.Conv1D(32, 3, activation='relu', padding='same',
                     input_shape=(window_size, n_features)),
        layers.BatchNormalization(),
        layers.MaxPooling1D(2),

        layers.LSTM(32, return_sequences=False),
        layers.Dropout(0.3),

        layers.Dense(32, activation='relu'),
        layers.Dropout(0.3),
        layers.Dense(n_classes, activation='softmax')
    ])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def cross_validate(
    X: np.ndarray,
    y: np.ndarray,
    n_classes: int,
    model_type: str = 'cnn_lstm',
    n_folds: int = 5,
    epochs: int = 30
) -> Dict:
    """Run k-fold cross-validation."""
    import tensorflow as tf

    window_size = X.shape[1]
    n_features = X.shape[2]

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=42)

    fold_results = []

    for fold, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        # Normalize
        mean = X_train.reshape(-1, n_features).mean(axis=0)
        std = X_train.reshape(-1, n_features).std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_test = (X_test - mean) / std

        # Build model
        if model_type == 'simple':
            model = build_simple_model(window_size, n_features, n_classes)
        elif model_type == 'lstm':
            model = build_lstm_model(window_size, n_features, n_classes)
        else:
            model = build_cnn_lstm_model(window_size, n_features, n_classes)

        # Train
        early_stop = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=5, restore_best_weights=True
        )

        model.fit(
            X_train, y_train,
            validation_split=0.2,
            epochs=epochs,
            batch_size=min(32, len(X_train) // 4),
            callbacks=[early_stop],
            verbose=0
        )

        # Evaluate
        _, train_acc = model.evaluate(X_train, y_train, verbose=0)
        _, test_acc = model.evaluate(X_test, y_test, verbose=0)

        fold_results.append({
            'train_acc': float(train_acc),
            'test_acc': float(test_acc),
        })

        tf.keras.backend.clear_session()

    return {
        'train_acc_mean': np.mean([r['train_acc'] for r in fold_results]),
        'train_acc_std': np.std([r['train_acc'] for r in fold_results]),
        'test_acc_mean': np.mean([r['test_acc'] for r in fold_results]),
        'test_acc_std': np.std([r['test_acc'] for r in fold_results]),
        'folds': fold_results,
    }


def evaluate_orientation_invariance(
    X: np.ndarray,
    y: np.ndarray,
    pitches: np.ndarray,
    n_classes: int,
    model_type: str = 'cnn_lstm',
    epochs: int = 30
) -> Dict:
    """Evaluate orientation invariance by training on high pitch, testing on low pitch."""
    import tensorflow as tf

    window_size = X.shape[1]
    n_features = X.shape[2]

    # Split by pitch quartiles
    q1 = np.percentile(pitches, 25)
    q3 = np.percentile(pitches, 75)

    high_mask = pitches >= q3
    low_mask = pitches <= q1

    X_train, y_train = X[high_mask], y[high_mask]
    X_test, y_test = X[low_mask], y[low_mask]

    if len(X_train) < 20 or len(X_test) < 20:
        return {'error': 'insufficient data'}

    # Normalize
    mean = X_train.reshape(-1, n_features).mean(axis=0)
    std = X_train.reshape(-1, n_features).std(axis=0) + 1e-8
    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    # Build model
    if model_type == 'simple':
        model = build_simple_model(window_size, n_features, n_classes)
    elif model_type == 'lstm':
        model = build_lstm_model(window_size, n_features, n_classes)
    else:
        model = build_cnn_lstm_model(window_size, n_features, n_classes)

    # Train
    early_stop = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=5, restore_best_weights=True
    )

    model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=min(32, len(X_train) // 4),
        callbacks=[early_stop],
        verbose=0
    )

    _, train_acc = model.evaluate(X_train, y_train, verbose=0)
    _, test_acc = model.evaluate(X_test, y_test, verbose=0)

    tf.keras.backend.clear_session()

    return {
        'train_samples': len(X_train),
        'test_samples': len(X_test),
        'train_acc': float(train_acc),
        'test_acc': float(test_acc),
        'orientation_gap': float(train_acc - test_acc),
    }


def run_ablation_study():
    """Run the full ablation study."""
    print("=" * 80)
    print("CNN-LSTM ABLATION STUDY v2")
    print("=" * 80)

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
    print(f"Total samples: {len(samples)}")
    print(f"Labeled samples: {len(metadata['index_to_code'])}")

    results = {
        'cross_validation': {},
        'orientation_invariance': {},
    }

    # Test each feature set and window size
    feature_sets_to_test = ['raw_9dof', 'mag_only', 'iron_mag', 'accel_gyro',
                            'accel_only', 'gyro_only', 'mag_euler', 'iron_euler']
    window_sizes_to_test = [10, 25, 50]

    print("\n" + "=" * 80)
    print("PART 1: CROSS-VALIDATION (Stratified Random Splits)")
    print("=" * 80)

    for fs_name in feature_sets_to_test:
        fs = FEATURE_SETS[fs_name]

        for window_size in window_sizes_to_test:
            key = f"{fs_name}_w{window_size}"
            print(f"\n{key}: {fs.description} ({fs.n_features} features, {window_size} window)")

            # Prepare data
            X, y, classes, pitches = prepare_windows(
                samples, metadata['index_to_code'],
                window_size, fs, stride=3
            )

            if len(X) < 50:
                print(f"  Skipping: only {len(X)} windows")
                continue

            print(f"  Windows: {len(X)}, Classes: {len(classes)}")

            # Cross-validation
            cv_result = cross_validate(X, y, len(classes), model_type='cnn_lstm', n_folds=5)

            results['cross_validation'][key] = {
                'feature_set': fs_name,
                'n_features': fs.n_features,
                'window_size': window_size,
                'n_windows': len(X),
                'n_classes': len(classes),
                **cv_result
            }

            print(f"  CV Accuracy: {cv_result['test_acc_mean']:.1%} ± {cv_result['test_acc_std']:.1%}")

    print("\n" + "=" * 80)
    print("PART 2: ORIENTATION INVARIANCE (Train high pitch → Test low pitch)")
    print("=" * 80)

    # Test best configurations for orientation invariance
    best_configs = ['raw_9dof_w25', 'mag_only_w25', 'iron_euler_w25', 'mag_euler_w25']

    for key in best_configs:
        parts = key.rsplit('_w', 1)
        fs_name = parts[0]
        window_size = int(parts[1])

        if fs_name not in FEATURE_SETS:
            continue

        fs = FEATURE_SETS[fs_name]
        print(f"\n{key}: {fs.description}")

        X, y, classes, pitches = prepare_windows(
            samples, metadata['index_to_code'],
            window_size, fs, stride=3
        )

        if len(X) < 50:
            print(f"  Skipping: insufficient data")
            continue

        print(f"  Pitch range: {pitches.min():.1f}° to {pitches.max():.1f}°")

        oi_result = evaluate_orientation_invariance(X, y, pitches, len(classes))

        if 'error' not in oi_result:
            results['orientation_invariance'][key] = {
                'feature_set': fs_name,
                'window_size': window_size,
                **oi_result
            }
            print(f"  Train: {oi_result['train_acc']:.1%}, Test: {oi_result['test_acc']:.1%}")
            print(f"  Orientation Gap: {oi_result['orientation_gap']:.1%}")
        else:
            print(f"  {oi_result['error']}")

    # Print summary
    print_summary(results)

    # Save results
    output_path = Path("ml/ablation_results_v2.json")
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nResults saved to: {output_path}")

    return results


def print_summary(results: Dict):
    """Print formatted summary."""
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Cross-validation results
    if results['cross_validation']:
        print("\n### Cross-Validation Results (sorted by test accuracy)")
        print(f"{'Config':<25} {'Feat':>5} {'Win':>4} {'CV Acc':>10} {'± Std':>8}")
        print("-" * 60)

        sorted_cv = sorted(results['cross_validation'].items(),
                          key=lambda x: x[1]['test_acc_mean'], reverse=True)

        for key, r in sorted_cv[:15]:
            print(f"{r['feature_set']:<25} {r['n_features']:>5} {r['window_size']:>4} "
                  f"{r['test_acc_mean']:>10.1%} {r['test_acc_std']:>7.1%}")

    # Orientation invariance
    if results['orientation_invariance']:
        print("\n### Orientation Invariance Results")
        print(f"{'Config':<25} {'Train':>8} {'Test':>8} {'Gap':>8}")
        print("-" * 55)

        for key, r in results['orientation_invariance'].items():
            print(f"{r['feature_set']:<25} {r['train_acc']:>8.1%} {r['test_acc']:>8.1%} "
                  f"{r['orientation_gap']:>8.1%}")

    # Key insights
    print("\n### Key Insights")

    if results['cross_validation']:
        best = sorted_cv[0]
        print(f"1. Best overall: {best[0]} with {best[1]['test_acc_mean']:.1%} CV accuracy")

        # Check if mag-only beats full 9dof
        mag_results = {k: v for k, v in results['cross_validation'].items() if 'mag_only' in k}
        dof_results = {k: v for k, v in results['cross_validation'].items() if 'raw_9dof' in k}

        if mag_results and dof_results:
            best_mag = max(mag_results.values(), key=lambda x: x['test_acc_mean'])
            best_dof = max(dof_results.values(), key=lambda x: x['test_acc_mean'])

            if best_mag['test_acc_mean'] > best_dof['test_acc_mean']:
                print(f"2. Magnetometer alone ({best_mag['test_acc_mean']:.1%}) beats full 9-DoF ({best_dof['test_acc_mean']:.1%})")
            else:
                print(f"2. Full 9-DoF ({best_dof['test_acc_mean']:.1%}) beats magnetometer alone ({best_mag['test_acc_mean']:.1%})")


if __name__ == "__main__":
    run_ablation_study()
